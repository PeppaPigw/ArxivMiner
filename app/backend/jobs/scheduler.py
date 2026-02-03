"""
Background job scheduler.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.backend.services.arxiv_client import ArxivClient
from app.backend.services.translator import get_translator
from app.backend.services.tagger import get_tagger
from app.backend.db.models import Paper, Tag, PaperTag, get_session
from app.backend.config import get_config
import json

logger = logging.getLogger(__name__)


def fetch_daily_papers():
    """Daily job to fetch new papers."""
    logger.info("Starting daily paper fetch...")
    config = get_config()
    
    try:
        from datetime import datetime, timedelta
        start_date = datetime.utcnow() - timedelta(hours=config.fetch_window_hours)
        
        client = ArxivClient(categories=config.arxiv_categories)
        papers = client.fetch_all(start_date=start_date)
        
        db = get_session()
        try:
            new_count = 0
            updated_count = 0
            
            for paper_data in papers:
                arxiv_id = paper_data["arxiv_id"]
                existing = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
                
                if existing:
                    if existing.updated_at < paper_data["updated_at"]:
                        for key, value in paper_data.items():
                            if key not in ["arxiv_id"]:
                                setattr(existing, key, value)
                        updated_count += 1
                else:
                    paper = Paper(**paper_data)
                    paper.translate_status = "pending"
                    paper.tag_status = "pending"
                    db.add(paper)
                    new_count += 1
            
            db.commit()
            logger.info(f"Daily fetch complete: {new_count} new, {updated_count} updated")
            
        finally:
            db.close()
    
    except Exception as e:
        logger.error(f"Daily fetch failed: {e}")


def process_pending_translations():
    """Process pending translations."""
    logger.info("Processing pending translations...")
    translator = get_translator()
    
    db = get_session()
    try:
        pending = db.query(Paper).filter(Paper.translate_status == "pending").all()
        
        for paper in pending:
            if paper.abstract_zh:
                paper.translate_status = "success"
                continue
            
            translated = translator.translate(paper.abstract_en)
            if translated:
                paper.abstract_zh = translated
                paper.translate_status = "success"
            else:
                paper.translate_status = "failed"
        
        db.commit()
        logger.info(f"Translation processing complete")
    
    finally:
        db.close()


def process_pending_tags():
    """Process pending tags."""
    logger.info("Processing pending tags...")
    tagger = get_tagger()
    
    db = get_session()
    try:
        pending = db.query(Paper).filter(Paper.tag_status == "pending").all()
        
        for paper in pending:
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
        
        db.commit()
        logger.info(f"Tag processing complete")
    
    finally:
        db.close()


# Global scheduler instance
_scheduler: BackgroundScheduler = None


def get_scheduler() -> BackgroundScheduler:
    """Get or create scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler()
        _scheduler.start()
    return _scheduler


def setup_scheduled_jobs():
    """Set up all scheduled jobs."""
    scheduler = get_scheduler()
    config = get_config()
    
    # Daily fetch at 8:00 AM
    scheduler.add_job(
        fetch_daily_papers,
        CronTrigger(hour=8, minute=0),
        id="daily_fetch",
        name="Daily arXiv paper fetch",
        replace_existing=True,
    )
    
    # Hourly translation processing
    scheduler.add_job(
        process_pending_translations,
        CronTrigger(minute=0),
        id="hourly_translate",
        name="Hourly translation processing",
        replace_existing=True,
    )
    
    # Hourly tag processing
    scheduler.add_job(
        process_pending_tags,
        CronTrigger(minute=30),
        id="hourly_tag",
        name="Hourly tag processing",
        replace_existing=True,
    )
    
    logger.info("Scheduled jobs configured")


def shutdown_scheduler():
    """Shut down the scheduler."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown()
        _scheduler = None
