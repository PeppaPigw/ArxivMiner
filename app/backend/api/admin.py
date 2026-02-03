"""
Admin API endpoints.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Depends, Header
from sqlalchemy.orm import Session

from app.backend.db.models import Paper, Tag, PaperTag, get_session
from app.backend.services.arxiv_client import ArxivClient
from app.backend.services.translator import get_translator
from app.backend.services.tagger import get_tagger
from app.backend.config import get_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


def verify_admin_token(x_admin_token: str = Header(None)) -> bool:
    """Verify admin token."""
    config = get_config()
    if x_admin_token != config.admin_token:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return True


def get_db() -> Session:
    session = get_session()
    try:
        yield session
    finally:
        session.close()


@router.post("/fetch")
def trigger_fetch(
    categories: List[str] = None,
    hours: int = None,
    force: bool = False,
    _auth: bool = Depends(verify_admin_token),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Trigger manual paper fetch."""
    config = get_config()
    
    if hours is None:
        hours = config.fetch_window_hours
    if categories is None:
        categories = config.arxiv_categories
    
    from datetime import datetime, timedelta
    start_date = datetime.utcnow() - timedelta(hours=hours)
    
    client = ArxivClient(categories=categories)
    papers = client.fetch_all(start_date=start_date)
    
    new_count = 0
    updated_count = 0
    
    for paper_data in papers:
        arxiv_id = paper_data["arxiv_id"]
        
        # Check if paper exists
        existing = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
        
        if existing:
            # Update if newer
            if existing.updated_at < paper_data["updated_at"]:
                for key, value in paper_data.items():
                    if key not in ["arxiv_id"]:
                        setattr(existing, key, value)
                updated_count += 1
        else:
            # Create new paper
            paper = Paper(**paper_data)
            paper.translate_status = "pending"
            paper.tag_status = "pending"
            db.add(paper)
            new_count += 1
    
    db.commit()
    
    return {
        "success": True,
        "new_papers": new_count,
        "updated_papers": updated_count,
        "total_fetched": len(papers),
    }


@router.post("/retranslate")
def retranslate_all(
    status: str = "failed",
    limit: int = 50,
    _auth: bool = Depends(verify_admin_token),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Retry failed translations."""
    translator = get_translator()
    
    query = db.query(Paper).filter(Paper.translate_status == status)
    papers = query.limit(limit).all()
    
    success_count = 0
    
    for paper in papers:
        translated = translator.translate(paper.abstract_en)
        if translated:
            paper.abstract_zh = translated
            paper.translate_status = "success"
            success_count += 1
        else:
            paper.translate_status = "failed"
    
    db.commit()
    
    return {
        "success": True,
        "translated": success_count,
        "failed": len(papers) - success_count,
    }


@router.post("/retag")
def retag_all(
    status: str = "failed",
    limit: int = 50,
    _auth: bool = Depends(verify_admin_token),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Retry failed tagging."""
    tagger = get_tagger()
    
    query = db.query(Paper).filter(Paper.tag_status == status)
    papers = query.limit(limit).all()
    
    success_count = 0
    
    for paper in papers:
        categories = json.loads(paper.categories_json)
        tags, _ = tagger.generate(paper.title, paper.abstract_en, categories)
        
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
        success_count += 1
    
    db.commit()
    
    return {
        "success": True,
        "tagged": success_count,
        "failed": len(papers) - success_count,
    }


@router.post("/process-pending")
def process_pending(
    _auth: bool = Depends(verify_admin_token),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Process all pending translations and tags."""
    translator = get_translator()
    tagger = get_tagger()
    
    # Get papers needing translation
    pending_translate = db.query(Paper).filter(
        Paper.translate_status == "pending"
    ).all()
    
    # Get papers needing tagging
    pending_tag = db.query(Paper).filter(
        Paper.tag_status == "pending"
    ).all()
    
    # Process translations
    translate_success = 0
    for paper in pending_translate:
        if paper.abstract_zh:
            paper.translate_status = "success"
            translate_success += 1
        else:
            translated = translator.translate(paper.abstract_en)
            if translated:
                paper.abstract_zh = translated
                paper.translate_status = "success"
                translate_success += 1
            else:
                paper.translate_status = "failed"
    
    # Process tags
    tag_success = 0
    for paper in pending_tag:
        categories = json.loads(paper.categories_json)
        tags, _ = tagger.generate(paper.title, paper.abstract_en, categories)
        
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
        tag_success += 1
    
    db.commit()
    
    return {
        "success": True,
        "translations_processed": translate_success,
        "tags_processed": tag_success,
    }


@router.get("/stats")
def get_stats(
    _auth: bool = Depends(verify_admin_token),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get system statistics."""
    total_papers = db.query(Paper).count()
    pending_translate = db.query(Paper).filter(
        Paper.translate_status == "pending"
    ).count()
    success_translate = db.query(Paper).filter(
        Paper.translate_status == "success"
    ).count()
    failed_translate = db.query(Paper).filter(
        Paper.translate_status == "failed"
    ).count()
    
    pending_tag = db.query(Paper).filter(
        Paper.tag_status == "pending"
    ).count()
    success_tag = db.query(Paper).filter(
        Paper.tag_status == "success"
    ).count()
    
    total_tags = db.query(Tag).count()
    
    from sqlalchemy import func
    from datetime import datetime, timedelta
    
    papers_today = db.query(Paper).filter(
        Paper.published_at >= datetime.utcnow() - timedelta(days=1)
    ).count()
    
    return {
        "total_papers": total_papers,
        "translations": {
            "pending": pending_translate,
            "success": success_translate,
            "failed": failed_translate,
        },
        "tags": {
            "pending": pending_tag,
            "success": success_tag,
            "total": total_tags,
        },
        "papers_today": papers_today,
    }
