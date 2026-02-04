"""
Reading lists API endpoints.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

from app.backend.db.models import (
    Paper, ReadingList, ReadingListItem, UserState,
    get_session
)
from app.backend.services.recommendation import get_recommendation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/lists", tags=["reading-lists"])


def get_db() -> Session:
    """Get database session."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


@router.get("")
def list_reading_lists(
    include_public: bool = True,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """List all reading lists."""
    query = db.query(ReadingList)
    
    if not include_public:
        query = query.filter(ReadingList.is_public == True)
    
    total = query.count()
    
    offset = (page - 1) * page_size
    lists = query.order_by(ReadingList.updated_at.desc()).offset(offset).limit(page_size).all()
    
    return {
        "items": [
            {
                "id": lst.id,
                "name": lst.name,
                "description": lst.description,
                "is_public": lst.is_public,
                "paper_count": len(lst.items),
                "created_at": lst.created_at.isoformat() if lst.created_at else None,
                "updated_at": lst.updated_at.isoformat() if lst.updated_at else None,
            }
            for lst in lists
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.post("")
def create_reading_list(
    name: str,
    description: Optional[str] = None,
    is_public: bool = False,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Create a new reading list."""
    reading_list = ReadingList(
        name=name,
        description=description,
        is_public=is_public,
    )
    db.add(reading_list)
    db.commit()
    db.refresh(reading_list)
    
    return {
        "id": reading_list.id,
        "name": reading_list.name,
        "description": reading_list.description,
        "is_public": reading_list.is_public,
        "created_at": reading_list.created_at.isoformat() if reading_list.created_at else None,
    }


@router.get("/{list_id}")
def get_reading_list(
    list_id: int,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get a reading list with its papers."""
    reading_list = db.query(ReadingList).filter(ReadingList.id == list_id).first()
    
    if not reading_list:
        raise HTTPException(status_code=404, detail="Reading list not found")
    
    items = db.query(ReadingListItem).filter(
        ReadingListItem.reading_list_id == list_id
    ).order_by(ReadingListItem.position).all()
    
    papers = []
    for item in items:
        paper = db.query(Paper).filter(Paper.id == item.paper_id).first()
        if paper:
            # Get user state for paper
            state = db.query(UserState).filter(UserState.paper_id == paper.id).first()
            user_state = {
                "is_read": state.is_read if state else False,
                "is_favorite": state.is_favorite if state else False,
                "notes": state.notes if state else None,
            }
            
            papers.append({
                "item_id": item.id,
                "position": item.position,
                "notes": item.notes,
                "paper": paper.to_dict(),
                "user_state": user_state,
            })
    
    return {
        "id": reading_list.id,
        "name": reading_list.name,
        "description": reading_list.description,
        "is_public": reading_list.is_public,
        "papers": papers,
        "paper_count": len(papers),
        "created_at": reading_list.created_at.isoformat() if reading_list.created_at else None,
        "updated_at": reading_list.updated_at.isoformat() if reading_list.updated_at else None,
    }


@router.put("/{list_id}")
def update_reading_list(
    list_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    is_public: Optional[bool] = None,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Update a reading list."""
    reading_list = db.query(ReadingList).filter(ReadingList.id == list_id).first()
    
    if not reading_list:
        raise HTTPException(status_code=404, detail="Reading list not found")
    
    if name is not None:
        reading_list.name = name
    if description is not None:
        reading_list.description = description
    if is_public is not None:
        reading_list.is_public = is_public
    
    reading_list.updated_at = datetime.utcnow()
    db.commit()
    
    return {
        "id": reading_list.id,
        "name": reading_list.name,
        "description": reading_list.description,
        "is_public": reading_list.is_public,
        "updated_at": reading_list.updated_at.isoformat() if reading_list.updated_at else None,
    }


@router.delete("/{list_id}")
def delete_reading_list(
    list_id: int,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Delete a reading list."""
    reading_list = db.query(ReadingList).filter(ReadingList.id == list_id).first()
    
    if not reading_list:
        raise HTTPException(status_code=404, detail="Reading list not found")
    
    db.delete(reading_list)
    db.commit()
    
    return {"success": True}


@router.post("/{list_id}/papers")
def add_paper_to_list(
    list_id: int,
    arxiv_id: str,
    notes: Optional[str] = None,
    position: Optional[int] = None,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Add a paper to a reading list."""
    reading_list = db.query(ReadingList).filter(ReadingList.id == list_id).first()
    
    if not reading_list:
        raise HTTPException(status_code=404, detail="Reading list not found")
    
    paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    # Check if already in list
    existing = db.query(ReadingListItem).filter(
        ReadingListItem.reading_list_id == list_id,
        ReadingListItem.paper_id == paper.id,
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Paper already in reading list")
    
    # Determine position
    if position is None:
        max_pos = db.query(func.max(ReadingListItem.position)).filter(
            ReadingListItem.reading_list_id == list_id
        ).scalar()
        position = (max_pos or 0) + 1
    
    item = ReadingListItem(
        reading_list_id=list_id,
        paper_id=paper.id,
        position=position,
        notes=notes,
    )
    db.add(item)
    
    reading_list.updated_at = datetime.utcnow()
    db.commit()
    
    return {
        "item_id": item.id,
        "paper_id": paper.id,
        "arxiv_id": arxiv_id,
        "position": position,
        "notes": notes,
    }


@router.delete("/{list_id}/papers/{paper_id}")
def remove_paper_from_list(
    list_id: int,
    paper_id: int,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Remove a paper from a reading list."""
    item = db.query(ReadingListItem).filter(
        ReadingListItem.reading_list_id == list_id,
        ReadingListItem.paper_id == paper_id,
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Paper not found in reading list")
    
    db.delete(item)
    
    reading_list = db.query(ReadingList).filter(ReadingList.id == list_id).first()
    if reading_list:
        reading_list.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {"success": True}


@router.put("/{list_id}/papers/{paper_id}/position")
def update_paper_position(
    list_id: int,
    paper_id: int,
    position: int,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Update the position of a paper in a reading list."""
    item = db.query(ReadingListItem).filter(
        ReadingListItem.reading_list_id == list_id,
        ReadingListItem.paper_id == paper_id,
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Paper not found in reading list")
    
    item.position = position
    db.commit()
    
    return {"success": True, "position": position}


@router.put("/{list_id}/papers/{paper_id}/notes")
def update_paper_notes(
    list_id: int,
    paper_id: int,
    notes: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Update notes for a paper in a reading list."""
    item = db.query(ReadingListItem).filter(
        ReadingListItem.reading_list_id == list_id,
        ReadingListItem.paper_id == paper_id,
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Paper not found in reading list")
    
    item.notes = notes
    db.commit()
    
    return {"success": True, "notes": notes}


@router.get("/{list_id}/export/bibtex")
def export_list_bibtex(
    list_id: int,
    db: Session = Depends(get_db),
) -> str:
    """Export reading list as BibTeX."""
    reading_list = db.query(ReadingList).filter(ReadingList.id == list_id).first()
    
    if not reading_list:
        raise HTTPException(status_code=404, detail="Reading list not found")
    
    items = db.query(ReadingListItem).filter(
        ReadingListItem.reading_list_id == list_id
    ).order_by(ReadingListItem.position).all()
    
    bibtex_entries = []
    for item in items:
        paper = db.query(Paper).filter(Paper.id == item.paper_id).first()
        if paper:
            bibtex_entries.append(paper.to_bibtex())
    
    return "\n\n".join(bibtex_entries)


@router.get("/{list_id}/recommendations")
def get_list_recommendations(
    list_id: int,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Get recommendations based on papers in a reading list."""
    items = db.query(ReadingListItem).filter(
        ReadingListItem.reading_list_id == list_id
    ).all()
    
    if not items:
        return []
    
    # Get papers in the list
    paper_ids = [item.paper_id for item in items]
    papers = db.query(Paper).filter(Paper.id.in_(paper_ids)).all()
    
    if not papers:
        return []
    
    # Find papers similar to papers in the list
    rec_service = get_recommendation_service()
    
    # Get recommendations based on the most recent paper in the list
    most_recent = max(papers, key=lambda p: p.published_at)
    return rec_service.find_similar_papers(
        paper_id=most_recent.id,
        limit=limit,
        db=db,
    )
