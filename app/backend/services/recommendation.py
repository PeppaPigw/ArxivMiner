"""
Recommendation service for personalized paper suggestions.
"""
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import func

from ..config import get_config
from ..db.models import Paper, Tag, PaperTag, UserState, UserPreference, get_session
from .embedding import get_embedding_service

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for generating personalized paper recommendations."""
    
    def __init__(self):
        """Initialize recommendation service."""
        self.config = get_config()
        self.embedding_service = get_embedding_service()
        self._paper_cache: Dict[int, Paper] = {}
        self._tag_cache: Dict[str, int] = {}  # tag_name -> paper_count
    
    def get_recommendations(
        self,
        user_preferences: Optional[UserPreference] = None,
        limit: int = 20,
        strategy: str = "hybrid",
        db: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        """Get personalized paper recommendations.
        
        Args:
            user_preferences: User preference settings
            limit: Number of recommendations to return
            strategy: Recommendation strategy (hybrid, popular, recent, similar)
            db: Database session
            
        Returns:
            List of recommended papers with scores
        """
        if db is None:
            db = get_session()
            close_session = True
        else:
            close_session = False
        
        try:
            # Get candidate papers
            candidates = self._get_candidate_papers(db, user_preferences, limit * 3)
            
            if not candidates:
                # Fallback to recent popular papers
                return self._get_popular_recent_papers(db, limit)
            
            if strategy == "popular":
                scored = self._score_by_popularity(candidates)
            elif strategy == "recent":
                scored = self._score_by_recency(candidates)
            elif strategy == "similar":
                scored = self._score_by_similarity(candidates, user_preferences)
            else:  # hybrid
                scored = self._score_hybrid(candidates, user_preferences)
            
            # Sort by score and limit
            scored.sort(key=lambda x: x[1], reverse=True)
            result = scored[:limit]
            
            return [
                {
                    "paper": paper.to_dict(),
                    "score": score,
                    "reason": reason,
                }
                for paper, score, reason in result
            ]
        
        finally:
            if close_session:
                db.close()
    
    def _get_candidate_papers(
        self,
        db: Session,
        user_preferences: Optional[UserPreference],
        limit: int = 100,
    ) -> List[Paper]:
        """Get candidate papers for recommendations."""
        # Get papers from last 30 days
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        query = db.query(Paper).filter(
            Paper.published_at >= cutoff_date,
            Paper.translate_status == "success",
        )
        
        # Filter by user preferences
        if user_preferences:
            pref_categories = []
            if user_preferences.preferred_categories:
                pref_categories = json.loads(user_preferences.preferred_categories)
            
            if pref_categories:
                query = query.filter(Paper.primary_category.in_(pref_categories))
            
            # Exclude ignored categories
            ignored = []
            if user_preferences.ignored_categories:
                ignored = json.loads(user_preferences.ignored_categories)
            
            if ignored:
                query = query.filter(~Paper.primary_category.in_(ignored))
        
        # Exclude already favorited papers
        query = query.outerjoin(UserState).filter(
            (UserState.is_favorite.is_(None)) | (UserState.is_favorite == False)
        )
        
        # Order by view count (popularity signal)
        query = query.order_by(Paper.view_count.desc())
        
        return query.limit(limit).all()
    
    def _score_by_popularity(
        self, papers: List[Paper]
    ) -> List[Tuple[Paper, float, str]]:
        """Score papers by popularity metrics."""
        scores = []
        max_views = max((p.view_count for p in papers), default=1)
        max_favs = max((p.favorite_count for p in papers), default=1)
        
        for paper in papers:
            # Normalize scores
            view_score = (paper.view_count / max_views) * 0.4 if max_views > 0 else 0
            fav_score = (paper.favorite_count / max_favs) * 0.6 if max_favs > 0 else 0
            
            total_score = view_score + fav_score
            scores.append((paper, total_score, "popular"))
        
        return scores
    
    def _score_by_recency(
        self, papers: List[Paper]
    ) -> List[Tuple[Paper, float, str]]:
        """Score papers by recency."""
        scores = []
        now = datetime.utcnow()
        max_age = max((now - p.published_at for p in papers), default=timedelta(days=1))
        
        for paper in papers:
            age = now - paper.published_at
            # Inverse of age (newer = higher score)
            recency_score = 1.0 - (age.total_seconds() / max_age.total_seconds())
            scores.append((paper, recency_score, "recent"))
        
        return scores
    
    def _score_by_similarity(
        self,
        papers: List[Paper],
        user_preferences: Optional[UserPreference],
    ) -> List[Tuple[Paper, float, str]]:
        """Score papers by similarity to user interests."""
        scores = []
        
        # Build user interest profile
        user_embedding = self._build_user_profile_embedding(user_preferences)
        
        if not user_embedding:
            # Fall back to recency if no profile
            return self._score_by_recency(papers)
        
        for paper in papers:
            if paper.embedding_vector:
                try:
                    paper_embedding = json.loads(paper.embedding_vector)
                    similarity = self.embedding_service.compute_similarity(
                        user_embedding, paper_embedding
                    )
                    scores.append((paper, similarity, "similar"))
                except json.JSONDecodeError:
                    scores.append((paper, 0.0, "similar"))
            else:
                scores.append((paper, 0.0, "similar"))
        
        return scores
    
    def _score_hybrid(
        self,
        papers: List[Paper],
        user_preferences: Optional[UserPreference],
    ) -> List[Tuple[Paper, float, str]]:
        """Combine multiple scoring strategies."""
        # Get scores from different strategies
        popularity_scores = self._score_by_popularity(papers)
        recency_scores = self._score_by_recency(papers)
        similarity_scores = self._score_by_similarity(papers, user_preferences)
        
        # Create score maps
        pop_map = {p.id: (s, r) for p, s, r in popularity_scores}
        rec_map = {p.id: (s, r) for p, s, r in recency_scores}
        sim_map = {p.id: (s, r) for p, s, r in similarity_scores}
        
        # Combine scores
        combined = []
        for paper in papers:
            pop_s, pop_r = pop_map.get(paper.id, (0.0, "popular"))
            rec_s, rec_r = rec_map.get(paper.id, (0.0, "recent"))
            sim_s, sim_r = sim_map.get(paper.id, (0.0, "similar"))
            
            # Weighted combination
            # 30% popularity, 30% recency, 40% similarity
            total_score = pop_s * 0.3 + rec_s * 0.3 + sim_s * 0.4
            
            # Use most significant reason
            reasons = {"popular": pop_s, "recent": rec_s, "similar": sim_s}
            best_reason = max(reasons, key=reasons.get)
            
            combined.append((paper, total_score, best_reason))
        
        return combined
    
    def _build_user_profile_embedding(
        self, user_preferences: Optional[UserPreference],
    ) -> Optional[List[float]]:
        """Build user interest profile embedding."""
        # Build text representation of interests
        interest_parts = []
        
        if user_preferences:
            if user_preferences.preferred_categories:
                categories = json.loads(user_preferences.preferred_categories)
                interest_parts.extend(categories)
            
            if user_preferences.preferred_tags:
                tags = json.loads(user_preferences.preferred_tags)
                interest_parts.extend(tags)
        
        if not interest_parts:
            return None
        
        # Encode interests
        interest_text = " ".join(interest_parts)
        return self.embedding_service.encode_text(interest_text)
    
    def _get_popular_recent_papers(
        self, db: Session, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get popular recent papers as fallback."""
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        
        papers = db.query(Paper).filter(
            Paper.published_at >= cutoff_date,
            Paper.translate_status == "success",
        ).order_by(
            Paper.view_count.desc()
        ).limit(limit).all()
        
        return [
            {
                "paper": p.to_dict(),
                "score": 1.0 - (i / limit),  # Decreasing score
                "reason": "popular",
            }
            for i, p in enumerate(papers)
        ]
    
    def find_similar_papers(
        self,
        paper_id: int,
        limit: int = 10,
        db: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        """Find papers similar to a given paper.
        
        Args:
            paper_id: ID of the reference paper
            limit: Number of similar papers to return
            db: Database session
            
        Returns:
            List of similar papers with similarity scores
        """
        if db is None:
            db = get_session()
            close_session = True
        else:
            close_session = False
        
        try:
            # Get the reference paper
            paper = db.query(Paper).filter(Paper.id == paper_id).first()
            if not paper:
                return []
            
            # Increment view count
            paper.view_count += 1
            db.commit()
            
            # Find papers with embeddings
            candidates = db.query(Paper).filter(
                Paper.id != paper_id,
                Paper.embedding_vector.isnot(None),
                Paper.translate_status == "success",
            ).limit(limit * 3).all()
            
            if not candidates:
                # Fallback: same category, recent
                return self._find_by_category(paper, limit, db)
            
            # Get reference embedding
            if paper.embedding_vector:
                try:
                    ref_embedding = json.loads(paper.embedding_vector)
                except json.JSONDecodeError:
                    return self._find_by_category(paper, limit, db)
            else:
                return self._find_by_category(paper, limit, db)
            
            # Compute similarities
            similarities = []
            for candidate in candidates:
                try:
                    cand_embedding = json.loads(candidate.embedding_vector)
                    score = self.embedding_service.compute_similarity(
                        ref_embedding, cand_embedding
                    )
                    similarities.append((candidate, score))
                except json.JSONDecodeError:
                    continue
            
            # Sort and return
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return [
                {
                    "paper": p.to_dict(),
                    "similarity": score,
                }
                for p, score in similarities[:limit]
            ]
        
        finally:
            if close_session:
                db.close()
    
    def _find_by_category(
        self, paper: Paper, limit: int, db: Session
    ) -> List[Dict[str, Any]]:
        """Find similar papers by category as fallback."""
        candidates = db.query(Paper).filter(
            Paper.id != paper.id,
            Paper.primary_category == paper.primary_category,
            Paper.translate_status == "success",
        ).order_by(
            Paper.published_at.desc()
        ).limit(limit).all()
        
        return [
            {
                "paper": p.to_dict(),
                "similarity": 0.8 - (i * 0.1),  # Decreasing similarity
            }
            for i, p in enumerate(candidates)
        ]
    
    def get_trending_papers(
        self,
        days: int = 7,
        limit: int = 20,
        db: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        """Get trending papers based on recent activity.
        
        Args:
            days: Number of days to look back
            limit: Number of papers to return
            db: Database session
            
        Returns:
            List of trending papers
        """
        if db is None:
            db = get_session()
            close_session = True
        else:
            close_session = False
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Papers with most views in the time window
            papers = db.query(Paper).filter(
                Paper.published_at >= cutoff_date,
                Paper.translate_status == "success",
            ).order_by(
                Paper.view_count.desc()
            ).limit(limit).all()
            
            # Also consider papers published recently with high favorite counts
            recent_high_favs = db.query(Paper).filter(
                Paper.published_at >= cutoff_date,
                Paper.favorite_count > 0,
            ).order_by(
                Paper.favorite_count.desc()
            ).limit(limit // 2).all()
            
            # Combine and deduplicate
            seen_ids = set()
            trending = []
            
            for paper in papers + recent_high_favs:
                if paper.id not in seen_ids:
                    seen_ids.add(paper.id)
                    trending.append({
                        "paper": paper.to_dict(),
                        "views": paper.view_count,
                        "favorites": paper.favorite_count,
                    })
            
            return sorted(trending, key=lambda x: x["views"], reverse=True)[:limit]
        
        finally:
            if close_session:
                db.close()
    
    def get_daily_digest(
        self,
        user_preferences: Optional[UserPreference] = None,
        limit: int = 10,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """Generate a daily digest of recommended papers.
        
        Args:
            user_preferences: User preferences
            limit: Number of papers in digest
            db: Database session
            
        Returns:
            Digest containing recommended papers and metadata
        """
        if db is None:
            db = get_session()
            close_session = True
        else:
            close_session = False
        
        try:
            recommendations = self.get_recommendations(
                user_preferences=user_preferences,
                limit=limit,
                strategy="hybrid",
                db=db,
            )
            
            return {
                "date": datetime.utcnow().isoformat(),
                "paper_count": len(recommendations),
                "papers": recommendations,
                "strategy": "hybrid",
            }
        
        finally:
            if close_session:
                db.close()


# Singleton instance
_recommendation_service: Optional[RecommendationService] = None


def get_recommendation_service() -> RecommendationService:
    """Get recommendation service instance."""
    global _recommendation_service
    if _recommendation_service is None:
        _recommendation_service = RecommendationService()
    return _recommendation_service
