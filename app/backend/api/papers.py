"""
Paper API endpoints.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends, Header
from sqlalchemy.orm import Session

from app.backend.db.models import Paper, UserState, Tag, PaperTag, get_session
from app.backend.services.translator import get_translator
from app.backend.services.tagger import get_tagger
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
    else:
        query = query.order_by(Paper.published_at.desc())
    
    # Count total
    total = query.count()
    
    # Pagination
    offset = (page - 1) * page_size
    papers = query.offset(offset).limit(page_size).all()
    
    # Get user states for papers
    paper_ids = [p.id for p in papers]
    user_states = db.query(UserState).filter(
        UserState.paper_id.in_(paper_ids)
    ).all()
    states_dict = {s.paper_id: s for s in user_states}
    
    # Get tags for papers
    paper_tags = db.query(PaperTag).filter(
        PaperTag.paper_id.in_(paper_ids)
    ).all()
    tags_dict: Dict[int, List[str]] = {}
    for pt in paper_tags:
        if pt.paper_id not in tags_dict:
            tags_dict[pt.paper_id] = []
        # Get tag name
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
            }
        else:
            paper_dict["user_state"] = {
                "is_read": False,
                "is_favorite": False,
                "is_hidden": False,
                "notes": None,
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
    
    result = paper.to_dict()
    
    # Get user state
    state = db.query(UserState).filter(UserState.paper_id == paper.id).first()
    if state:
        result["user_state"] = {
            "is_read": state.is_read,
            "is_favorite": state.is_favorite,
            "is_hidden": state.is_hidden,
            "notes": state.notes,
        }
    else:
        result["user_state"] = {
            "is_read": False,
            "is_favorite": False,
            "is_hidden": False,
            "notes": None,
        }
    
    # Get tags
    paper_tags = db.query(PaperTag).filter(PaperTag.paper_id == paper.id).all()
    result["tags"] = [db.query(Tag).filter(Tag.id == pt.tag_id).first().name 
                      for pt in paper_tags if db.query(Tag).filter(Tag.id == pt.tag_id).first()]
    
    return result


@router.post("/{arxiv_id}/state")
def update_paper_state(
    arxiv_id: str,
    is_read: Optional[bool] = None,
    is_favorite: Optional[bool] = None,
    is_hidden: Optional[bool] = None,
    notes: Optional[str] = None,
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
    if is_favorite is not None:
        state.is_favorite = is_favorite
    if is_hidden is not None:
        state.is_hidden = is_hidden
    if notes is not None:
        state.notes = notes
    
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
