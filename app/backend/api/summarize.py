"""
Summarization API endpoints.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.backend.db.models import Paper, PaperSummary, get_session
from app.backend.services.summarizer import get_summarizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/summarize", tags=["summarization"])


def get_db() -> Session:
    """Get database session."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


@router.post("/paper/{arxiv_id}")
def summarize_paper(
    arxiv_id: str,
    summary_type: str = "comprehensive",
    max_length: int = 300,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Generate summary for a paper."""
    paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    summarizer = get_summarizer()
    
    result = summarizer.summarize(
        paper_id=paper.id,
        title=paper.title,
        abstract=paper.abstract_en,
        summary_type=summary_type,
        max_length=max_length,
    )
    
    return result


@router.post("/paper/{arxiv_id}/all")
def summarize_paper_all_types(
    arxiv_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Generate all summary types for a paper."""
    paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    summarizer = get_summarizer()
    
    result = summarizer.generate_all_summaries(
        paper_id=paper.id,
        title=paper.title,
        abstract=paper.abstract_en,
    )
    
    return result


@router.post("/compare")
def compare_papers(
    papers: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Compare multiple papers."""
    if len(papers) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 papers")
    
    summarizer = get_summarizer()
    comparison = summarizer.compare_papers(papers)
    
    return {
        "comparison": comparison,
        "papers_compared": len(papers),
    }


@router.get("/paper/{arxiv_id}/summaries")
def get_saved_summaries(
    arxiv_id: str,
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Get previously generated summaries."""
    paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    summaries = db.query(PaperSummary).filter(
        PaperSummary.paper_id == paper.id
    ).all()
    
    return [
        {
            "type": s.summary_type,
            "summary": s.summary_text,
            "model": s.model_used,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in summaries
    ]
