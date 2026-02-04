"""
RSS feed endpoint.
"""
import sys
import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, Query, Depends
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.backend.db.models import Paper, Tag, PaperTag, get_session
from app.backend.config import get_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rss", tags=["rss"])


def get_db() -> Session:
    """Get database session."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def generate_rss_feed(
    title: str,
    description: str,
    link: str,
    items: List[Dict[str, Any]],
    language: str = "en",
) -> str:
    """Generate RSS feed XML."""
    
    rss_header = f'''<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
<channel>
  <title><![CDATA[{title}]]></title>
  <description><![CDATA[{description}]]></description>
  <link>{link}</link>
  <language>{language}</language>
  <lastBuildDate>{datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")}</lastBuildDate>
  <atom:link href="{link}" rel="self" type="application/rss+xml" />
'''
    
    item_template = '''
  <item>
    <title><![CDATA[{title}]]></title>
    <description><![CDATA[{description}]]></description>
    <link>{link}</link>
    <guid isPermaLink="false">{guid}</guid>
    <pubDate>{pubDate}</pubDate>
    <author><![CDATA[{author}]]></author>
  </item>
'''
    
    items_xml = ""
    for item in items:
        items_xml += item_template.format(
            title=item.get("title", "")[:500],
            description=item.get("description", "")[:2000],
            link=item.get("link", ""),
            guid=item.get("guid", ""),
            pubDate=item.get("pubDate", ""),
            author=item.get("author", "")[:500],
        )
    
    rss_footer = '''
</channel>
</rss>'''
    
    return rss_header + items_xml + rss_footer


@router.get("/papers")
def get_papers_rss(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> Response:
    """Get RSS feed of papers."""
    config = get_config()
    
    query = db.query(Paper)
    
    if category:
        query = query.filter(Paper.primary_category == category)
    
    if tag:
        query = query.join(PaperTag).join(Tag).filter(Tag.name == tag)
    
    papers = query.order_by(Paper.published_at.desc()).limit(limit).all()
    
    # Get tags for papers
    paper_ids = [p.id for p in papers]
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
    
    # Build RSS items
    items = []
    for paper in papers:
        # Create description with abstract and tags
        abstract = paper.abstract_en[:500] + "..." if len(paper.abstract_en) > 500 else paper.abstract_en
        tags = tags_dict.get(paper.id, [])
        description = f"Categories: {paper.primary_category}\nTags: {', '.join(tags)}\n\nAbstract:\n{abstract}"
        
        pub_date = paper.published_at.strftime("%a, %d %b %Y %H:%M:%S GMT") if paper.published_at else ""
        
        items.append({
            "title": paper.title,
            "description": description,
            "link": paper.abs_url,
            "guid": f"arxiv:{paper.arxiv_id}",
            "pubDate": pub_date,
            "author": ", ".join(json.loads(paper.authors_json)[:3]),
        })
    
    feed_title = f"ArxivMiner - {' '.join(config.arxiv_categories)}"
    if category:
        feed_title = f"ArxivMiner - {category}"
    if tag:
        feed_title = f"ArxivMiner - Tag: {tag}"
    
    rss_xml = generate_rss_feed(
        title=feed_title,
        description=f"Latest papers from arXiv in {', '.join(config.arxiv_categories)}",
        link="https://arxivminer.example.com/api/rss/papers",
        items=items,
    )
    
    return Response(
        content=rss_xml,
        media_type="application/rss+xml",
        headers={
            "Cache-Control": "public, max-age=300",  # Cache for 5 minutes
        }
    )


@router.get("/categories/{category}")
def get_category_rss(
    category: str,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> Response:
    """Get RSS feed for a specific category."""
    
    return get_papers_rss(category=category, limit=limit, db=db)


@router.get("/tags/{tag}")
def get_tag_rss(
    tag: str,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> Response:
    """Get RSS feed for a specific tag."""
    
    return get_papers_rss(tag=tag, limit=limit, db=db)


@router.get("/trending")
def get_trending_rss(
    days: int = Query(7, ge=1, le=30),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> Response:
    """Get RSS feed of trending papers."""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    papers = db.query(Paper).filter(
        Paper.published_at >= cutoff_date,
        Paper.view_count > 0,
    ).order_by(
        Paper.view_count.desc()
    ).limit(limit).all()
    
    # Get tags for papers
    paper_ids = [p.id for p in papers]
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
    
    # Build RSS items with view count in description
    items = []
    for paper in papers:
        abstract = paper.abstract_en[:500] + "..." if len(paper.abstract_en) > 500 else paper.abstract_en
        tags = tags_dict.get(paper.id, [])
        description = f"Views: {paper.view_count} | Categories: {paper.primary_category}\nTags: {', '.join(tags)}\n\nAbstract:\n{abstract}"
        
        pub_date = paper.published_at.strftime("%a, %d %b %Y %H:%M:%S GMT") if paper.published_at else ""
        
        items.append({
            "title": f"[{paper.view_count} views] {paper.title}",
            "description": description,
            "link": paper.abs_url,
            "guid": f"arxiv:{paper.arxiv_id}",
            "pubDate": pub_date,
            "author": ", ".join(json.loads(paper.authors_json)[:3]),
        })
    
    rss_xml = generate_rss_feed(
        title=f"ArxivMiner - Trending Papers (Last {days} days)",
        description=f"Most viewed papers in the last {days} days",
        link="https://arxivminer.example.com/api/rss/trending",
        items=items,
    )
    
    return Response(
        content=rss_xml,
        media_type="application/rss+xml",
        headers={
            "Cache-Control": "public, max-age=600",  # Cache for 10 minutes
        }
    )
