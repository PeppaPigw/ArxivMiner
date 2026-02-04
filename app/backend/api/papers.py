"""
Paper API endpoints - Enhanced with export, recommendations, and more.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends, Header, Response
from sqlalchemy.orm import Session

from app.backend.db.models import (
    Paper, UserState, Tag, PaperTag, ReadingList, ReadingListItem,
    get_session
)
from app.backend.services.translator import get_translator
from app.backend.services.tagger import get_tagger
from app.backend.services.recommendation import get_recommendation_service
from app.backend.services.embedding import get_embedding_service
from app.backend.config import get_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/papers", tags=["papers"])


def get_db() -> Session:
    """Get database session."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


@router.get("")
def list_papers(
    q: Optional[str] = None,
    tag: Optional[str] = None,
    category: Optional[str] = None,
    author: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort: str = "published",
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    hide_hidden: bool = True,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """List papers with filtering and pagination."""
    
    query = db.query(Paper)
    
    # Text search
    if q:
        search_term = f"%{q}%"
        query = query.filter(
            (Paper.title.ilike(search_term)) |
            (Paper.abstract_en.ilike(search_term)) |
            (Paper.abstract_zh.ilike(search_term))
        )
    
    # Category filter
    if category:
        query = query.filter(Paper.primary_category == category)
    
    # Author filter
    if author:
        author_term = f"%{author}%"
        query = query.filter(Paper.authors_json.ilike(author_term))
    
    # Tag filter
    if tag:
        query = query.join(PaperTag).join(Tag).filter(Tag.name == tag)
    
    # Date range
    if date_from:
        try:
            date_from_dt = datetime.fromisoformat(date_from)
            query = query.filter(Paper.published_at >= date_from_dt)
        except ValueError:
            pass
    
    if date_to:
        try:
            date_to_dt = datetime.fromisoformat(date_to)
            query = query.filter(Paper.published_at <= date_to_dt)
        except ValueError:
            pass
    
    # Hide hidden papers
    if hide_hidden:
        query = query.outerjoin(UserState).filter(
            (UserState.is_hidden.is_(None)) | (UserState.is_hidden == False)
        )
    
    # Sorting
    if sort == "updated":
        query = query.order_by(Paper.updated_at.desc())
    elif sort == "views":
        query = query.order_by(Paper.view_count.desc())
    elif sort == "favorites":
        query = query.order_by(Paper.favorite_count.desc())
    else:
        query = query.order_by(Paper.published_at.desc())
    
    # Count total
    total = query.count()
    
    # Pagination
    offset = (page - 1) * page_size
    papers = query.offset(offset).limit(page_size).all()
    
    # Get user states and tags
    paper_ids = [p.id for p in papers]
    user_states = db.query(UserState).filter(
        UserState.paper_id.in_(paper_ids)
    ).all()
    states_dict = {s.paper_id: s for s in user_states}
    
    paper_tags = db.query(PaperTag).filter(
        PaperTag.paper_id.in_(paper_ids)
    ).all()
    tags_dict: Dict[int, List[str]] = {}
    for pt in paper_tags:
        if pt.paper_id not in tags_dict:
            tags_dict[pt.paper_id] = []
        tag_obj = db.query(Tag).filter(Tag.id == pt.tag_id).first()
        if tag_obj:
            tags_dict[pt.paper_id].append(tag_obj.name)
    
    # Build response
    items = []
    for p in papers:
        paper_dict = p.to_dict()
        state = states_dict.get(p.id)
        if state:
            paper_dict["user_state"] = {
                "is_read": state.is_read,
                "is_favorite": state.is_favorite,
                "is_hidden": state.is_hidden,
                "notes": state.notes,
                "read_progress": state.read_progress,
            }
        else:
            paper_dict["user_state"] = {
                "is_read": False,
                "is_favorite": False,
                "is_hidden": False,
                "notes": None,
                "read_progress": 0.0,
            }
        paper_dict["tags"] = tags_dict.get(p.id, [])
        items.append(paper_dict)
    
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.get("/{arxiv_id}")
def get_paper(arxiv_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get a single paper by arXiv ID."""
    paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    # Increment view count
    paper.view_count += 1
    db.commit()
    
    result = paper.to_dict()
    
    # Get user state
    state = db.query(UserState).filter(UserState.paper_id == paper.id).first()
    if state:
        result["user_state"] = {
            "is_read": state.is_read,
            "is_favorite": state.is_favorite,
            "is_hidden": state.is_hidden,
            "notes": state.notes,
            "read_progress": state.read_progress,
        }
    else:
        result["user_state"] = {
            "is_read": False,
            "is_favorite": False,
            "is_hidden": False,
            "notes": None,
            "read_progress": 0.0,
        }
    
    # Get tags
    paper_tags = db.query(PaperTag).filter(PaperTag.paper_id == paper.id).all()
    result["tags"] = [
        db.query(Tag).filter(Tag.id == pt.tag_id).first().name
        for pt in paper_tags
        if db.query(Tag).filter(Tag.id == pt.tag_id).first()
    ]
    
    return result


@router.get("/{arxiv_id}/similar")
def get_similar_papers(
    arxiv_id: str,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Get papers similar to a given paper."""
    paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    rec_service = get_recommendation_service()
    return rec_service.find_similar_papers(paper_id=paper.id, limit=limit, db=db)


@router.post("/{arxiv_id}/state")
def update_paper_state(
    arxiv_id: str,
    is_read: Optional[bool] = None,
    is_favorite: Optional[bool] = None,
    is_hidden: Optional[bool] = None,
    notes: Optional[str] = None,
    read_progress: Optional[float] = None,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Update user state for a paper."""
    
    paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    state = db.query(UserState).filter(UserState.paper_id == paper.id).first()
    if not state:
        state = UserState(paper_id=paper.id)
        db.add(state)
    
    if is_read is not None:
        state.is_read = is_read
        if is_read:
            state.last_read_at = datetime.utcnow()
    
    if is_favorite is not None:
        state.is_favorite = is_favorite
        # Update favorite count
        if is_favorite:
            paper.favorite_count += 1
        else:
            paper.favorite_count = max(0, paper.favorite_count - 1)
    
    if is_hidden is not None:
        state.is_hidden = is_hidden
    
    if notes is not None:
        state.notes = notes
    
    if read_progress is not None:
        state.read_progress = max(0.0, min(1.0, read_progress))
    
    state.updated_at = datetime.utcnow()
    db.commit()
    
    return {"success": True}


@router.get("/{arxiv_id}/translate")
def translate_paper(arxiv_id: str, db: Session = Depends(get_db)):
    """Trigger translation for a paper."""
    paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    if paper.translate_status == "success" and paper.abstract_zh:
        return {"success": True, "abstract_zh": paper.abstract_zh}
    
    translator = get_translator()
    translated = translator.translate(paper.abstract_en)
    
    if translated:
        paper.abstract_zh = translated
        paper.translate_status = "success"
    else:
        paper.translate_status = "failed"
    
    db.commit()
    
    return {"success": translated is not None, "abstract_zh": translated}


@router.get("/{arxiv_id}/tag")
def tag_paper(arxiv_id: str, db: Session = Depends(get_db)):
    """Trigger tagging for a paper."""
    paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    tagger = get_tagger()
    categories = json.loads(paper.categories_json)
    tags, timestamp = tagger.generate(paper.title, paper.abstract_en, categories)
    
    # Remove existing tags
    db.query(PaperTag).filter(PaperTag.paper_id == paper.id).delete()
    
    # Add new tags
    for tag_name in tags:
        tag = db.query(Tag).filter(Tag.name == tag_name).first()
        if not tag:
            tag = Tag(name=tag_name, kind="keyword")
            db.add(tag)
            db.flush()
        
        pt = PaperTag(paper_id=paper.id, tag_id=tag.id)
        db.add(pt)
    
    paper.tag_status = "success"
    db.commit()
    
    return {"success": True, "tags": tags}


@router.get("/{arxiv_id}/export/bibtex")
def export_paper_bibtex(arxiv_id: str, db: Session = Depends(get_db)) -> Response:
    """Export paper in BibTeX format."""
    paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    bibtex = paper.to_bibtex()
    
    return Response(
        content=bibtex,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{arxiv_id}.bib"'
        }
    )


@router.get("/{arxiv_id}/export/json")
def export_paper_json(
    arxiv_id: str,
    include_embedding: bool = False,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Export paper in JSON format."""
    paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    return paper.to_json(include_embedding=include_embedding)


@router.get("/export/all/bibtex")
def export_all_bibtex(
    category: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> Response:
    """Export multiple papers in BibTeX format."""
    query = db.query(Paper)
    
    if category:
        query = query.filter(Paper.primary_category == category)
    
    if date_from:
        try:
            date_from_dt = datetime.fromisoformat(date_from)
            query = query.filter(Paper.published_at >= date_from_dt)
        except ValueError:
            pass
    
    if date_to:
        try:
            date_to_dt = datetime.fromisoformat(date_to)
            query = query.filter(Paper.published_at <= date_to_dt)
        except ValueError:
            pass
    
    papers = query.order_by(Paper.published_at.desc()).limit(limit).all()
    
    bibtex_entries = [paper.to_bibtex() for paper in papers]
    bibtex_content = "\n\n".join(bibtex_entries)
    
    return Response(
        content=bibtex_content,
        media_type="text/plain",
        headers={
            "Content-Disposition": 'attachment; filename="papers.bib"'
        }
    )


@router.get("/export/all/json")
def export_all_json(
    category: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    include_embedding: bool = False,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Export multiple papers in JSON format."""
    query = db.query(Paper)
    
    if category:
        query = query.filter(Paper.primary_category == category)
    
    if date_from:
        try:
            date_from_dt = datetime.fromisoformat(date_from)
            query = query.filter(Paper.published_at >= date_from_dt)
        except ValueError:
            pass
    
    if date_to:
        try:
            date_to_dt = datetime.fromisoformat(date_to)
            query = query.filter(Paper.published_at <= date_to_dt)
        except ValueError:
            pass
    
    papers = query.order_by(Paper.published_at.desc()).limit(limit).all()
    
    return {
        "export_date": datetime.utcnow().isoformat(),
        "count": len(papers),
        "papers": [p.to_json(include_embedding=include_embedding) for p in papers],
    }


@router.get("/trending")
def get_trending_papers(
    days: int = Query(7, ge=1, le=30),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Get trending papers."""
    rec_service = get_recommendation_service()
    return rec_service.get_trending_papers(days=days, limit=limit, db=db)


@router.get("/recommendations")
def get_recommendations(
    strategy: str = "hybrid",
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Get personalized recommendations."""
    rec_service = get_recommendation_service()
    return rec_service.get_recommendations(
        user_preferences=None,
        limit=limit,
        strategy=strategy,
        db=db,
    )
