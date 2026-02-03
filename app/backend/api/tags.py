"""
Tags API endpoints.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.backend.db.models import Paper, Tag, PaperTag, get_session

router = APIRouter(prefix="/api/tags", tags=["tags"])


def get_db() -> Session:
    session = get_session()
    try:
        yield session
    finally:
        session.close()


@router.get("")
def list_tags(
    kind: str = None,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """List all tags with paper counts."""
    
    query = db.query(
        Tag.id,
        Tag.name,
        Tag.kind,
        func.count(PaperTag.id).label("count")
    ).outerjoin(PaperTag).group_by(Tag.id)
    
    if kind:
        query = query.filter(Tag.kind == kind)
    
    tags = query.order_by(func.count(PaperTag.id).desc()).limit(limit).all()
    
    return [
        {
            "id": t.id,
            "name": t.name,
            "kind": t.kind,
            "count": t.count,
        }
        for t in tags
    ]


@router.get("/popular")
def popular_tags(
    limit: int = 20,
    db: Session = Depends(get_db),
) -> List[str]:
    """Get most popular tag names."""
    
    results = db.query(
        Tag.name,
        func.count(PaperTag.id).label("count")
    ).join(PaperTag).group_by(Tag.id).order_by(
        func.count(PaperTag.id).desc()
    ).limit(limit).all()
    
    return [r.name for r in results]
