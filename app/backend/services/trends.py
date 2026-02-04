"""
Trend Analysis Service.
Detect emerging topics and research trends.
"""
import json
import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.backend.config import get_config
from app.backend.db.models import Paper, Tag, PaperTag, get_session

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Analyze research trends and emerging topics."""
    
    def __init__(self):
        """Initialize trend analyzer."""
        self.config = get_config()
    
    def get_trending_tags(
        self,
        days: int = 30,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get trending tags based on recent paper usage."""
        db = get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Get papers from cutoff date
            papers = db.query(Paper).filter(
                Paper.published_at >= cutoff_date,
            ).all()
            
            # Count tag usage
            tag_counts = Counter()
            year_counts = Counter()
            
            for paper in papers:
                # Get tags
                paper_tags = db.query(PaperTag).filter(
                    PaperTag.paper_id == paper.id
                ).all()
                
                for pt in paper_tags:
                    tag = db.query(Tag).filter(Tag.id == pt.tag_id).first()
                    if tag:
                        tag_counts[tag.name] += 1
                
                # Count by category
                year = paper.published_at.year if paper.published_at else 0
                year_counts[year] += 1
            
            # Calculate growth rate compared to previous period
            previous_cutoff = cutoff_date - timedelta(days=days)
            previous_papers = db.query(Paper).filter(
                Paper.published_at >= previous_cutoff,
                Paper.published_at < cutoff_date,
            ).count()
            
            current_count = len(papers)
            growth_rate = (current_count - previous_count) / max(1, previous_count)
            
            # Build results
            results = []
            for tag, count in tag_counts.most_common(limit):
                tag_obj = db.query(Tag).filter(Tag.name == tag).first()
                results.append({
                    "tag": tag,
                    "count": count,
                    "growth_rate": growth_rate,
                    "trend": "up" if growth_rate > 0.1 else "stable" if growth_rate > -0.1 else "down",
                })
            
            return results
        
        finally:
            db.close()
    
    def get_keyword_trends(
        self,
        keywords: List[str],
        years: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Track keyword usage over time."""
        db = get_session()
        try:
            if years is None:
                years = list(range(2015, 2026))
            
            trends = {}
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                yearly_counts = {}
                
                for year in years:
                    start = datetime(year, 1, 1)
                    end = datetime(year + 1, 1, 1)
                    
                    count = db.query(Paper).filter(
                        Paper.published_at >= start,
                        Paper.published_at < end,
                    ).filter(
                        Paper.title.ilike(f"%{keyword}%") |
                        Paper.abstract_en.ilike(f"%{keyword}%")
                    ).count()
                    
                    yearly_counts[str(year)] = count
                
                trends[keyword] = {
                    "keyword": keyword,
                    "yearly_counts": yearly_counts,
                    "total": sum(yearly_counts.values()),
                }
            
            return trends
        
        finally:
            db.close()
    
    def detect_merging_topics(
        self,
        days: int = 90,
        min_papers: int = 5,
    ) -> List[Dict[str, Any]]:
        """Detect emerging topics from recent papers."""
        db = get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Get recent papers
            papers = db.query(Paper).filter(
                Paper.published_at >= cutoff_date,
            ).all()
            
            # Analyze tag co-occurrences
            cooccurrence = {}
            
            for paper in papers:
                paper_tags = db.query(PaperTag).filter(
                    PaperTag.paper_id == paper.id
                ).all()
                
                tags = [pt.tag_id for pt in paper_tags]
                
                # Count co-occurrences
                for i, tag1 in enumerate(tags):
                    for tag2 in tags[i+1:]:
                        key = tuple(sorted([tag1, tag2]))
                        cooccurrence[key] = cooccurrence.get(key, 0) + 1
            
            # Filter significant co-occurrences
            emerging = []
            for (tag1, tag2), count in sorted(
                cooccurrence.items(), key=lambda x: x[1], reverse=True
            ):
                if count >= min_papers:
                    t1 = db.query(Tag).filter(Tag.id == tag1).first()
                    t2 = db.query(Tag).filter(Tag.id == tag2).first()
                    
                    if t1 and t2:
                        emerging.append({
                            "topic": f"{t1.name} + {t2.name}",
                            "tag_1": t1.name,
                            "tag_2": t2.name,
                            "paper_count": count,
                        })
            
            return emerging[:20]
        
        finally:
            db.close()
    
    def get_category_trends(
        self,
        years: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Get trends for paper categories."""
        db = get_session()
        try:
            if years is None:
                years = list(range(2015, 2026))
            
            category_trends = {}
            
            # Major categories
            categories = [
                "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NI",
                "cs.DS", "cs.OS", "cs.SE", "cs.PL", "cs.DB",
            ]
            
            for category in categories:
                yearly_counts = {}
                
                for year in years:
                    start = datetime(year, 1, 1)
                    end = datetime(year + 1, 1, 1)
                    
                    count = db.query(Paper).filter(
                        Paper.published_at >= start,
                        Paper.published_at < end,
                        Paper.primary_category == category,
                    ).count()
                    
                    yearly_counts[str(year)] = count
                
                # Calculate growth
                values = list(yearly_counts.values())
                if len(values) >= 2:
                    recent_avg = sum(values[-2:]) / 2
                    older_avg = sum(values[:-2]) / max(1, len(values) - 2)
                    growth = (recent_avg - older_avg) / max(1, older_avg)
                else:
                    growth = 0
                
                category_trends[category] = {
                    "category": category,
                    "yearly_counts": yearly_counts,
                    "total": sum(yearly_counts.values()),
                    "recent_growth": growth,
                }
            
            return category_trends
        
        finally:
            db.close()
    
    def get_research_frontiers(
        self,
        days: int = 180,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Identify research frontiers (hot areas)."""
        db = get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Get recent papers with high citation potential
            recent_papers = db.query(Paper).filter(
                Paper.published_at >= cutoff_date,
            ).order_by(Paper.view_count.desc()).limit(100).all()
            
            # Analyze tags
            tag_counts = Counter()
            tag_citations = {}
            
            for paper in recent_papers:
                paper_tags = db.query(PaperTag).filter(
                    PaperTag.paper_id == paper.id
                ).all()
                
                for pt in paper_tags:
                    tag = db.query(Tag).filter(Tag.id == pt.tag_id).first()
                    if tag:
                        tag_counts[tag.name] += 1
                        if tag.name not in tag_citations:
                            tag_citations[tag.name] = 0
                        tag_citations[tag.name] += paper.view_count
            
            # Score tags
            frontiers = []
            for tag, count in tag_counts.most_common(limit * 2):
                total_views = tag_citations.get(tag, 0)
                avg_views = total_views / count if count > 0 else 0
                
                frontiers.append({
                    "topic": tag,
                    "paper_count": count,
                    "total_views": total_views,
                    "avg_views_per_paper": round(avg_views, 2),
                    "hotness_score": count * 0.5 + total_views * 0.5,
                })
            
            # Sort by hotness
            frontiers.sort(key=lambda x: x["hotness_score"], reverse=True)
            
            return frontiers[:limit]
        
        finally:
            db.close()
    
    def compare_fields(
        self,
        fields: List[str],
        years: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Compare growth across research fields."""
        if years is None:
            years = list(range(2015, 2026))
        
        db = get_session()
        try:
            comparison = {}
            
            for field in fields:
                yearly_data = {}
                
                for year in years:
                    start = datetime(year, 1, 1)
                    end = datetime(year + 1, 1, 1)
                    
                    count = db.query(Paper).filter(
                        Paper.published_at >= start,
                        Paper.published_at < end,
                    ).filter(
                        Paper.primary_category.ilike(f"{field}%")
                    ).count()
                    
                    yearly_data[str(year)] = count
                
                # Calculate metrics
                values = list(yearly_data.values())
                if len(values) >= 2:
                    cagr = ((values[-1] / max(1, values[0])) ** (1 / (len(values) - 1)) - 1) * 100
                else:
                    cagr = 0
                
                comparison[field] = {
                    "field": field,
                    "yearly_counts": yearly_data,
                    "total_papers": sum(values),
                    "avg_per_year": sum(values) / max(1, len(values)),
                    "compound_annual_growth": round(cagr, 2),
                }
            
            return comparison
        
        finally:
            db.close()


# Singleton
_trend_analyzer: Optional[TrendAnalyzer] = None


def get_trend_analyzer() -> TrendAnalyzer:
    """Get trend analyzer instance."""
    global _trend_analyzer
    if _trend_analyzer is None:
        _trend_analyzer = TrendAnalyzer()
    return _trend_analyzer
