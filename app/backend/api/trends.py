"""
Trends API endpoints.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Query, Depends
from sqlalchemy.orm import Session

from app.backend.db.models import get_session
from app.backend.services.trends import get_trend_analyzer
from app.backend.services.topic_modeling import get_topic_modeler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trends", tags=["trends"])


def get_db() -> Session:
    """Get database session."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


@router.get("/tags")
def get_trending_tags(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(20, ge=1, le=100),
) -> List[Dict[str, Any]]:
    """Get trending tags."""
    analyzer = get_trend_analyzer()
    return analyzer.get_trending_tags(days=days, limit=limit)


@router.get("/keywords")
def get_keyword_trends(
    keywords: str = Query(..., description="Comma-separated keywords"),
    years: str = Query("2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025", description="Comma-separated years"),
) -> Dict[str, Any]:
    """Track keyword usage over time."""
    keyword_list = [k.strip() for k in keywords.split(",")]
    year_list = [int(y.strip()) for y in years.split(",")]
    
    analyzer = get_trend_analyzer()
    return analyzer.get_keyword_trends(keywords=keyword_list, years=year_list)


@router.get("/topics/emerging")
def get_emerging_topics(
    days: int = Query(90, ge=1, le=365),
    min_papers: int = Query(5, ge=1, le=50),
) -> List[Dict[str, Any]]:
    """Detect emerging topics."""
    analyzer = get_trend_analyzer()
    return analyzer.detect_merging_topics(days=days, min_papers=min_papers)


@router.get("/categories")
def get_category_trends(
    years: str = Query("2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025", description="Comma-separated years"),
) -> Dict[str, Any]:
    """Get category trends."""
    year_list = [int(y.strip()) for y in years.split(",")]
    
    analyzer = get_trend_analyzer()
    return analyzer.get_category_trends(years=year_list)


@router.get("/frontiers")
def get_research_frontiers(
    days: int = Query(180, ge=1, le=365),
    limit: int = Query(10, ge=1, le=50),
) -> List[Dict[str, Any]]:
    """Identify research frontiers."""
    analyzer = get_trend_analyzer()
    return analyzer.get_research_frontiers(days=days, limit=limit)


@router.get("/compare")
def compare_fields(
    fields: str = Query(..., description="Comma-separated fields (e.g., cs.AI,cs.LG,cs.CL)"),
    years: str = Query("2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025", description="Comma-separated years"),
) -> Dict[str, Any]:
    """Compare growth across research fields."""
    field_list = [f.strip() for f in fields.split(",")]
    year_list = [int(y.strip()) for y in years.split(",")]
    
    analyzer = get_trend_analyzer()
    return analyzer.compare_fields(fields=field_list, years=year_list)


# ============ Topics ============)


# ============points ============

@router.get("/topics")
def get_discovered_topics(
    num_topics: int = Query(10, ge=1, le=50),
    min_word_freq: int = Query(3, ge=1, le=20),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Discover topics from papers."""
    modeler = get_topic_modeler()
    return modeler.extract_topics_from_papers(
        num_topics=num_topics,
        min_word_freq=min_word_freq,
    )


@router.post("/topics/assign")
def assign_topics_to_papers(
    paper_ids: List[int] = None,
) -> Dict[str, Any]:
    """Assign papers to topics."""
    modeler = get_topic_modeler()
    return modeler.assign_topics_to_papers(paper_ids=paper_ids)


@router.get("/topics/{topic_id}")
def get_topic_distribution(
    topic_id: int,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get paper distribution for a topic."""
    modeler = get_topic_modeler()
    return modeler.get_topic_distribution(topic_id=topic_id)


@router.get("/topics/{topic_id}/similar")
def find_similar_topics(
    topic_id: int,
    limit: int = Query(5, ge=1, le=20),
) -> List[Dict[str, Any]]:
    """Find similar topics."""
    modeler = get_topic_modeler()
    return modeler.find_similar_topics(topic_id=topic_id, limit=limit)


@router.get("/topics/{topic_id}/evolution")
def track_topic_evolution(
    topic_id: int,
    years: str = Query("2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025", description="Comma-separated years"),
) -> Dict[str, Any]:
    """Track topic evolution over time."""
    year_list = [int(y.strip()) for y in years.split(",")]
    
    modeler = get_topic_modeler()
    return modeler.track_topic_evolution(topic_id=topic_id, years=year_list)
