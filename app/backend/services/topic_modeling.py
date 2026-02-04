"""
Topic Modeling Service.
Auto-discover research topics and cluster papers.
"""
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.backend.config import get_config
from app.backend.db.models import Paper, Tag, PaperTag, Topic, PaperTopic, get_session

logger = logging.getLogger(__name__)


class TopicModeler:
    """Topic modeling service for paper clustering."""
    
    def __init__(self):
        """Initialize topic modeler."""
        self.config = get_config()
    
    def extract_topics_from_papers(
        self,
        paper_ids: Optional[List[int]] = None,
        num_topics: int = 10,
        min_word_freq: int = 3,
    ) -> List[Dict[str, Any]]:
        """Extract topics from papers using keyword analysis.
        
        Args:
            paper_ids: Specific papers to analyze (None = all)
            num_topics: Number of topics to extract
            min_word_freq: Minimum word frequency
            
        Returns:
            List of discovered topics
        """
        db = get_session()
        try:
            # Get papers
            if paper_ids:
                papers = db.query(Paper).filter(Paper.id.in_(paper_ids)).all()
            else:
                papers = db.query(Paper).limit(500).all()
            
            # Extract all text
            all_text = []
            for paper in papers:
                text = f"{paper.title} {paper.abstract_en}"
                all_text.append(text.lower())
            
            # Tokenize and count
            word_counts = Counter()
            for text in all_text:
                words = self._tokenize(text)
                word_counts.update(words)
            
            # Filter by frequency
            frequent_words = {
                word for word, count in word_counts.items()
                if count >= min_word_freq
            }
            
            # Find topic keywords
            topics = self._discover_topics(
                all_text, frequent_words, num_topics
            )
            
            return topics
        
        finally:
            db.close()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        words = text.split()
        
        # Stopwords
        stopwords = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "also", "now",
            "this", "that", "these", "those", "what", "which", "who",
            "whom", "its", "their", "they", "we", "you", "he", "she",
            "it", "i", "me", "my", "your", "his", "her", "our", "any",
            "both", "down", "up", "out", "off", "over", "while", "if",
            "about", "against", "between", "into", "through", "because",
            "since", "until", "although", "though", "after", "before",
            "paper", "present", "propose", "proposed", "method", "approach",
            "show", "shows", "shown", "using", "use", "used", "based", "new",
            "results", "result", "work", "study", "research", "system",
        }
        
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def _discover_topics(
        self,
        texts: List[str],
        vocabulary: set,
        num_topics: int,
    ) -> List[Dict[str, Any]]:
        """Discover topics using co-occurrence analysis."""
        # Build word co-occurrence matrix
        cooccurrence = defaultdict(Counter)
        
        for text in texts:
            words = [w for w in self._tokenize(text) if w in vocabulary]
            unique_words = set(words)
            
            for word in unique_words:
                for other in unique_words:
                    if word != other:
                        cooccurrence[word][other] += 1
        
        # Find topic clusters
        topics = []
        used_words = set()
        
        # Get top words by frequency as seed words
        word_freq = Counter()
        for text in texts:
            words = [w for w in self._tokenize(text) if w in vocabulary]
            word_freq.update(set(words))
        
        top_words = [w for w, _ in word_freq.most_common(100)]
        
        for seed in top_words:
            if seed in used_words or len(topics) >= num_topics:
                continue
            
            # Find related words
            related = cooccurrence[seed].most_common(20)
            topic_words = [seed] + [w for w, _ in related[:9] if w not in used_words]
            
            if len(topic_words) >= 3:
                used_words.update(topic_words)
                
                topics.append({
                    "topic_id": len(topics) + 1,
                    "keywords": topic_words,
                    "topic_name": self._generate_topic_name(topic_words),
                    "size": len(topic_words),
                })
        
        return topics
    
    def _generate_topic_name(self, keywords: List[str]) -> str:
        """Generate a human-readable topic name from keywords."""
        # Use top 2-3 keywords
        name = " ".join(keywords[:3]).title()
        return name
    
    def assign_topics_to_papers(
        self,
        paper_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Assign papers to discovered topics."""
        db = get_session()
        try:
            # Get or create topics
            topics = db.query(Topic).all()
            
            if not topics:
                # Create topics from existing tags
                tags = db.query(Tag).all()
                tag_names = [t.name for t in tags]
                
                # Group tags by prefix
                topic_map = defaultdict(list)
                for tag in tag_names:
                    prefix = tag.split("_")[0] if "_" in tag else tag[:3]
                    topic_map[prefix].append(tag)
                
                for name, tag_list in topic_map.items():
                    topic = Topic(
                        name=name.title(),
                        keywords_json=json.dumps(tag_list),
                    )
                    db.add(topic)
                
                db.flush()
                topics = db.query(Topic).all()
            
            # Get papers
            if paper_ids:
                papers = db.query(Paper).filter(Paper.id.in_(paper_ids)).all()
            else:
                papers = db.query(Paper).limit(200).all()
            
            assigned = 0
            for paper in papers:
                # Get paper tags
                paper_tags = db.query(PaperTag).filter(
                    PaperTag.paper_id == paper.id
                ).all()
                
                paper_tag_names = [
                    db.query(Tag).filter(Tag.id == pt.tag_id).first().name
                    for pt in paper_tags
                    if db.query(Tag).filter(Tag.id == pt.tag_id).first()
                ]
                
                # Match to topic
                for topic in topics:
                    topic_keywords = json.loads(topic.keywords_json)
                    
                    for ptag in paper_tag_names:
                        if ptag in topic_keywords:
                            # Check if already assigned
                            existing = db.query(PaperTopic).filter(
                                PaperTopic.paper_id == paper.id,
                                PaperTopic.topic_id == topic.id,
                            ).first()
                            
                            if not existing:
                                pt = PaperTopic(
                                    paper_id=paper.id,
                                    topic_id=topic.id,
                                    confidence=0.5,
                                )
                                db.add(pt)
                                assigned += 1
            
            db.commit()
            
            return {
                "success": True,
                "papers_processed": len(papers),
                "assignments_created": assigned,
            }
        
        finally:
            db.close()
    
    def get_topic_distribution(
        self,
        topic_id: int,
    ) -> Dict[str, Any]:
        """Get paper distribution for a topic."""
        db = get_session()
        try:
            topic = db.query(Topic).filter(Topic.id == topic_id).first()
            if not topic:
                return {"error": "Topic not found"}
            
            # Get papers in topic
            paper_topics = db.query(PaperTopic).filter(
                PaperTopic.topic_id == topic_id
            ).all()
            
            # Get paper details
            papers = []
            for pt in paper_topics:
                paper = db.query(Paper).filter(Paper.id == pt.paper_id).first()
                if paper:
                    papers.append({
                        "paper_id": paper.id,
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "year": paper.published_at.year if paper.published_at else None,
                        "confidence": pt.confidence,
                    })
            
            # Sort by confidence
            papers.sort(key=lambda x: x["confidence"], reverse=True)
            
            return {
                "topic": {
                    "id": topic.id,
                    "name": topic.name,
                    "keywords": json.loads(topic.keywords_json),
                },
                "paper_count": len(papers),
                "papers": papers,
            }
        
        finally:
            db.close()
    
    def find_similar_topics(
        self,
        topic_id: int,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find topics with similar keywords."""
        db = get_session()
        try:
            topic = db.query(Topic).filter(Topic.id == topic_id).first()
            if not topic:
                return []
            
            topic_keywords = set(json.loads(topic.keywords_json))
            
            # Calculate similarity to other topics
            similarities = []
            other_topics = db.query(Topic).filter(Topic.id != topic_id).all()
            
            for other in other_topics:
                other_keywords = set(json.loads(other.keywords_json))
                intersection = topic_keywords & other_keywords
                union = topic_keywords | other_keywords
                
                jaccard = len(intersection) / max(1, len(union))
                
                similarities.append({
                    "topic_id": other.id,
                    "name": other.name,
                    "keywords": json.loads(other.keywords_json),
                    "jaccard_similarity": round(jaccard, 4),
                    "shared_keywords": list(intersection),
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["jaccard_similarity"], reverse=True)
            
            return similarities[:limit]
        
        finally:
            db.close()
    
    def track_topic_evolution(
        self,
        topic_id: int,
        years: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Track how a topic evolves over time."""
        db = get_session()
        try:
            if years is None:
                years = list(range(2015, 2026))
            
            topic = db.query(Topic).filter(Topic.id == topic_id).first()
            if not topic:
                return {"error": "Topic not found"}
            
            topic_keywords = set(json.loads(topic.keywords_json))
            
            yearly_stats = {}
            
            for year in years:
                start = datetime(year, 1, 1)
                end = datetime(year + 1, 1, 1)
                
                # Count papers related to topic
                papers = db.query(Paper).filter(
                    Paper.published_at >= start,
                    Paper.published_at < end,
                ).all()
                
                related_count = 0
                for paper in papers:
                    paper_tags = db.query(PaperTag).filter(
                        PaperTag.paper_id == paper.id
                    ).all()
                    
                    tag_names = [
                        db.query(Tag).filter(Tag.id == pt.tag_id).first().name
                        for pt in paper_tags
                        if db.query(Tag).filter(Tag.id == pt.tag_id).first()
                    ]
                    
                    if any(tag in topic_keywords for tag in tag_names):
                        related_count += 1
                
                yearly_stats[str(year)] = {
                    "paper_count": related_count,
                }
            
            # Calculate growth
            values = list(yearly_stats.values())
            if len(values) >= 2:
                recent = sum(v["paper_count"] for v in values[-2:])
                older = sum(v["paper_count"] for v in values[:-2])
                growth = (recent - older) / max(1, older)
            else:
                growth = 0
            
            return {
                "topic": {
                    "id": topic.id,
                    "name": topic.name,
                    "keywords": list(topic_keywords),
                },
                "evolution": yearly_stats,
                "total_papers": sum(v["paper_count"] for v in values),
                "growth_rate": round(growth * 100, 2),
            }
        
        finally:
            db.close()


# Singleton
_topic_modeler: Optional[TopicModeler] = None


def get_topic_modeler() -> TopicModeler:
    """Get topic modeler instance."""
    global _topic_modeler
    if _topic_modeler is None:
        _topic_modeler = TopicModeler()
    return _topic_modeler
