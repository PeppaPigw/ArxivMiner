"""
Authors API endpoints - Track authors and get their papers.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.backend.db.models import (
    Paper, AuthorFollow, UserState, Tag, PaperTag,
    get_session
)
from app.backend.services.arxiv_client import ArxivClient
from app.backend.config import get_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/authors", tags=["authors"])


def get_db() -> Session:
    """Get database session."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


@router.get("")
def search_authors(
    q: str,
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Search for authors by name."""
    search_term = f"%{q}%"
    
    # Find papers containing the author name
    papers = db.query(Paper).filter(
        Paper.authors_json.ilike(search_term)
    ).limit(limit * 3).all()
    
    # Extract unique authors
    author_papers: Dict[str, List[Paper]] = {}
    for paper in papers:
        authors = json.loads(paper.authors_json)
        for author in authors:
            if q.lower() in author.lower():
                if author not in author_papers:
                    author_papers[author] = []
                if paper not in author_papers[author]:
                    author_papers[author].append(paper)
    
    results = []
    for author_name, author_papers_list in author_papers.items():
        # Count total papers
        total_papers = len(author_papers_list)
        
        # Get most recent paper
        most_recent = max(author_papers_list, key=lambda p: p.published_at)
        
        # Get unique categories
        categories = set()
        for p in author_papers_list:
            categories.update(json.loads(p.categories_json))
        
        results.append({
            "name": author_name,
            "paper_count": total_papers,
            "most_recent_paper": {
                "arxiv_id": most_recent.arxiv_id,
                "title": most_recent.title,
                "published_at": most_recent.published_at.isoformat() if most_recent.published_at else None,
            },
            "categories": list(categories),
        })
    
    # Sort by paper count
    results.sort(key=lambda x: x["paper_count"], reverse=True)
    
    return results[:limit]


@router.get("/{author_name}")
def get_author(
    author_name: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get author details and their papers."""
    # Find papers by this author
    search_term = f"%{author_name}%"
    
    query = db.query(Paper).filter(Paper.authors_json.ilike(search_term))
    
    total = query.count()
    
    offset = (page - 1) * page_size
    papers = query.order_by(Paper.published_at.desc()).offset(offset).limit(page_size).all()
    
    # Get user states
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
        tag_obj = db.query(Tag).filter(Tag.id == pt.tag_id).first()
        if tag_obj:
            tags_dict[pt.paper_id].append(tag_obj.name)
    
    # Build paper list
    paper_list = []
    for p in papers:
        paper_dict = p.to_dict()
        state = states_dict.get(p.id)
        if state:
            paper_dict["user_state"] = {
                "is_read": state.is_read,
                "is_favorite": state.is_favorite,
                "notes": state.notes,
            }
        else:
            paper_dict["user_state"] = {
                "is_read": False,
                "is_favorite": False,
                "notes": None,
            }
        paper_dict["tags"] = tags_dict.get(p.id, [])
        paper_list.append(paper_dict)
    
    # Get unique categories this author has published in
    all_categories = set()
    all_papers = db.query(Paper).filter(Paper.authors_json.ilike(search_term)).all()
    for p in all_papers:
        all_categories.update(json.loads(p.categories_json))
    
    # Calculate statistics
    years = {}
    for p in all_papers:
        year = p.published_at.year if p.published_at else 0
        years[year] = years.get(year, 0) + 1
    
    return {
        "name": author_name,
        "total_papers": total,
        "categories": list(all_categories),
        "yearly_publication_count": dict(sorted(years.items())),
        "papers": paper_list,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.post("/follow")
def follow_author(
    author_name: str,
    arxiv_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Follow an author to track their new papers."""
    # Verify paper exists
    paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    # Check if already following
    existing = db.query(AuthorFollow).filter(
        AuthorFollow.author_name == author_name,
    ).first()
    
    if existing:
        return {"success": True, "message": "Already following this author"}
    
    follow = AuthorFollow(
        author_name=author_name,
        paper_id=paper.id,
    )
    db.add(follow)
    db.commit()
    
    return {
        "success": True,
        "author_name": author_name,
        "followed_at": follow.created_at.isoformat() if follow.created_at else None,
    }


@router.post("/unfollow")
def unfollow_author(
    author_name: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Stop following an author."""
    follow = db.query(AuthorFollow).filter(
        AuthorFollow.author_name == author_name
    ).first()
    
    if not follow:
        raise HTTPException(status_code=404, detail="Not following this author")
    
    db.delete(follow)
    db.commit()
    
    return {"success": True}


@router.get("/followed")
def list_followed_authors(
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """List all followed authors."""
    follows = db.query(AuthorFollow).distinct(AuthorFollow.author_name).all()
    
    results = []
    for follow in follows:
        # Get paper count
        search_term = f"%{follow.author_name}%"
        paper_count = db.query(Paper).filter(
            Paper.authors_json.ilike(search_term)
        ).count()
        
        # Get latest paper
        latest_paper = db.query(Paper).filter(
            Paper.authors_json.ilike(search_term)
        ).order_by(Paper.published_at.desc()).first()
        
        results.append({
            "name": follow.author_name,
            "paper_count": paper_count,
            "latest_paper": {
                "arxiv_id": latest_paper.arxiv_id if latest_paper else None,
                "title": latest_paper.title if latest_paper else None,
                "published_at": latest_paper.published_at.isoformat() if latest_paper and latest_paper.published_at else None,
            } if latest_paper else None,
            "followed_at": follow.created_at.isoformat() if follow.created_at else None,
        })
    
    # Sort by name
    results.sort(key=lambda x: x["name"])
    
    return results


@router.get("/followed/{author_name}/new-papers")
def get_new_papers_from_followed(
    author_name: str,
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Get new papers from a followed author."""
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    search_term = f"%{author_name}%"
    papers = db.query(Paper).filter(
        Paper.authors_json.ilike(search_term),
        Paper.published_at >= cutoff_date,
    ).order_by(Paper.published_at.desc()).all()
    
    return [p.to_dict() for p in papers]


@router.get("/new-papers")
def get_new_papers_from_all_followed(
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get new papers from all followed authors."""
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Get all followed authors
    follows = db.query(AuthorFollow.author_name).distinct().all()
    followed_authors = [f[0] for f in follows]
    
    if not followed_authors:
        return {
            "message": "No authors followed",
            "papers": [],
            "authors": [],
        }
    
    # Build query for papers from followed authors
    conditions = []
    for author in followed_authors:
        conditions.append(Paper.authors_json.ilike(f"%{author}%"))
    
    # Combine conditions with OR
    from sqlalchemy import or_
    query = db.query(Paper).filter(
        or_(*conditions),
        Paper.published_at >= cutoff_date,
    ).order_by(Paper.published_at.desc())
    
    papers = query.all()
    
    # Group papers by author
    author_papers: Dict[str, List[Paper]] = {}
    for paper in papers:
        authors = json.loads(paper.authors_json)
        for author in authors:
            if author in followed_authors:
                if author not in author_papers:
                    author_papers[author] = []
                if paper not in author_papers[author]:
                    author_papers[author].append(paper)
    
    return {
        "total_papers": len(papers),
        "authors_with_new_papers": len(author_papers),
        "papers": [p.to_dict() for p in papers],
        "by_author": {
            author: [p.to_dict() for p in papers]
            for author, papers in author_papers.items()
        },
    }
