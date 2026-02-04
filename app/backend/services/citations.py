"""
Citation Network Service.
Builds and analyzes citation relationships between papers.
"""
import json
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.backend.config import get_config
from app.backend.db.models import Paper, Citation, get_session

logger = logging.getLogger(__name__)


class CitationNetwork:
    """Citation network analysis service."""
    
    def __init__(self):
        """Initialize citation network."""
        self.config = get_config()
    
    def build_citation_graph(
        self,
        paper_ids: Optional[List[int]] = None,
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        """Build citation graph for papers.
        
        Args:
            paper_ids: Specific papers to include (None = all)
            max_depth: Citation tree depth
            
        Returns:
            Graph data structure
        """
        db = get_session()
        try:
            # Build adjacency list
            graph = {
                "nodes": {},
                "edges": [],
                "stats": {},
            }
            
            if paper_ids is None:
                papers = db.query(Paper).limit(1000).all()
            else:
                papers = db.query(Paper).filter(Paper.id.in_(paper_ids)).all()
            
            # Create nodes
            paper_map = {p.id: p for p in papers}
            node_ids = set(p.id for p in papers)
            
            for paper in papers:
                graph["nodes"][paper.id] = {
                    "id": paper.id,
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "venue": paper.primary_category,
                    "year": paper.published_at.year if paper.published_at else None,
                    "citation_count": paper.favorite_count or 0,
                }
            
            # Build edges from citation data
            citations = db.query(Citation).filter(
                Citation.citing_paper_id.in_(node_ids) |
                Citation.cited_paper_id.in_(node_ids)
            ).all()
            
            edge_set = set()
            for citation in citations:
                if citation.citing_paper_id in node_ids and citation.cited_paper_id in node_ids:
                    edge = (citation.citing_paper_id, citation.cited_paper_id)
                    edge_set.add(edge)
            
            # Add edges
            for citing, cited in edge_set:
                graph["edges"].append({
                    "source": citing,
                    "target": cited,
                    "type": "cites",
                })
            
            # Calculate statistics
            graph["stats"] = {
                "total_papers": len(papers),
                "total_citations": len(graph["edges"]),
                "density": len(graph["edges"]) / max(1, len(papers) * (len(papers) - 1)),
            }
            
            return graph
        
        finally:
            db.close()
    
    def find_influential_papers(
        self,
        paper_id: int,
        depth: int = 2,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Find influential papers in the citation network.
        
        Args:
            paper_id: Source paper ID
            depth: How far to traverse
            limit: Maximum results
            
        Returns:
            List of influential papers with scores
        """
        db = get_session()
        try:
            # BFS to find all reachable papers
            visited = {paper_id}
            queue = deque([(paper_id, 0)])
            distances = {paper_id: 0}
            
            while queue:
                current, d = queue.popleft()
                if d >= depth:
                    continue
                
                # Get citations (both directions)
                citations = db.query(Citation).filter(
                    (Citation.citing_paper_id == current) |
                    (Citation.cited_paper_id == current)
                ).all()
                
                for c in citations:
                    neighbor = c.cited_paper_id if c.citing_paper_id == current else c.citing_paper_id
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = d + 1
                        queue.append((neighbor, d + 1))
            
            # Calculate PageRank-like scores
            scores = {}
            for pid in visited:
                if pid == paper_id:
                    continue
                
                paper = db.query(Paper).filter(Paper.id == pid).first()
                if paper:
                    # Score based on distance and citations
                    dist = distances.get(pid, depth)
                    citations = paper.favorite_count or 0
                    score = citations / (dist ** 2 + 1)
                    
                    scores[pid] = {
                        "paper_id": pid,
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "venue": paper.primary_category,
                        "year": paper.published_at.year if paper.published_at else None,
                        "citations": citations,
                        "distance": dist,
                        "influence_score": score,
                    }
            
            # Sort by score
            results = sorted(scores.values(), key=lambda x: x["influence_score"], reverse=True)
            return results[:limit]
        
        finally:
            db.close()
    
    def find_citation_path(
        self,
        source_id: int,
        target_id: int,
    ) -> Optional[List[int]]:
        """Find shortest citation path between two papers.
        
        Args:
            source_id: Source paper ID
            target_id: Target paper ID
            
        Returns:
            List of paper IDs forming the path, or None if no path
        """
        db = get_session()
        try:
            # Build adjacency list for papers in session scope
            papers = db.query(Paper).limit(500).all()
            paper_ids = {p.id for p in papers}
            
            # Build citation map
            citations = db.query(Citation).filter(
                Citation.citing_paper_id.in_(paper_ids),
                Citation.cited_paper_id.in_(paper_ids),
            ).all()
            
            adj = defaultdict(set)
            for c in citations:
                adj[c.citing_paper_id].add(c.cited_paper_id)
            
            # BFS for shortest path
            if source_id not in paper_ids or target_id not in paper_ids:
                return None
            
            visited = {source_id}
            queue = deque([[source_id]])
            
            while queue:
                path = queue.popleft()
                current = path[-1]
                
                if current == target_id:
                    return path
                
                for neighbor in adj[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(path + [neighbor])
            
            return None
        
        finally:
            db.close()
    
    def get_citation_statistics(
        self,
        paper_id: int,
    ) -> Dict[str, Any]:
        """Get citation statistics for a paper.
        
        Args:
            paper_id: Paper ID
            
        Returns:
            Dict with citation statistics
        """
        db = get_session()
        try:
            paper = db.query(Paper).filter(Paper.id == paper_id).first()
            if not paper:
                return {"error": "Paper not found"}
            
            # Get citations
            citations_in = db.query(Citation).filter(
                Citation.cited_paper_id == paper_id
            ).all()
            
            citations_out = db.query(Citation).filter(
                Citation.citing_paper_id == paper_id
            ).all()
            
            # Get citing papers
            citing_papers = []
            for c in citations_in:
                citing = db.query(Paper).filter(Paper.id == c.citing_paper_id).first()
                if citing:
                    citing_papers.append({
                        "paper_id": citing.id,
                        "arxiv_id": citing.arxiv_id,
                        "title": citing.title,
                        "year": citing.published_at.year if citing.published_at else None,
                    })
            
            # Get cited papers
            cited_papers = []
            for c in citations_out:
                cited = db.query(Paper).filter(Paper.id == c.cited_paper_id).first()
                if cited:
                    cited_papers.append({
                        "paper_id": cited.id,
                        "arxiv_id": cited.arxiv_id,
                        "title": cited.title,
                        "year": cited.published_at.year if cited.published_at else None,
                    })
            
            return {
                "paper_id": paper_id,
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "total_citations": len(citations_in),
                "total_cited": len(citations_out),
                "citation_count": paper.favorite_count or 0,
                "citing_papers": citing_papers,
                "cited_papers": cited_papers,
                "h_index_contribution": 1 if paper.favorite_count > 0 else 0,
            }
        
        finally:
            db.close()
    
    def find_co_citations(
        self,
        paper_id: int,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find papers that are frequently cited together with the given paper.
        
        Args:
            paper_id: Paper ID
            limit: Maximum results
            
        Returns:
            List of co-cited papers
        """
        db = get_session()
        try:
            # Get papers that cite the same papers
            cited_by_source = set()
            citations = db.query(Citation).filter(
                Citation.citing_paper_id == paper_id
            ).all()
            for c in citations:
                cited_by_source.add(c.cited_paper_id)
            
            # Find papers citing similar papers
            co_citation_count = defaultdict(int)
            
            for cited_id in cited_by_source:
                similar_citations = db.query(Citation).filter(
                    Citation.cited_paper_id == cited_id,
                    Citation.citing_paper_id != paper_id,
                ).limit(50).all()
                
                for c in similar_citations:
                    co_citation_count[c.citing_paper_id] += 1
            
            # Get paper details
            results = []
            for pid, count in sorted(
                co_citation_count.items(), key=lambda x: x[1], reverse=True
            )[:limit]:
                paper = db.query(Paper).filter(Paper.id == pid).first()
                if paper:
                    results.append({
                        "paper_id": pid,
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "co_citation_count": count,
                        "venue": paper.primary_category,
                    })
            
            return results
        
        finally:
            db.close()
    
    def calculate_h_index(
        self,
        paper_ids: Optional[List[int]] = None,
    ) -> int:
        """Calculate h-index for a collection of papers.
        
        Args:
            paper_ids: List of paper IDs (None = all papers)
            
        Returns:
            h-index value
        """
        db = get_session()
        try:
            if paper_ids is None:
                papers = db.query(Paper).all()
            else:
                papers = db.query(Paper).filter(Paper.id.in_(paper_ids)).all()
            
            # Get citation counts
            citations = sorted(
                [p.favorite_count or 0 for p in papers],
                reverse=True,
            )
            
            h_index = 0
            for i, c in enumerate(citations):
                if c >= i + 1:
                    h_index = i + 1
                else:
                    break
            
            return h_index
        
        finally:
            db.close()
    
    def get_citation_trends(
        self,
        years: List[int] = None,
    ) -> Dict[str, Any]:
        """Get citation trends over time.
        
        Args:
            years: List of years to analyze
            
        Returns:
            Dict with yearly citation statistics
        """
        if years is None:
            years = list(range(2015, 2026))
        
        db = get_session()
        try:
            trends = {}
            
            for year in years:
                start = datetime(year, 1, 1)
                end = datetime(year + 1, 1, 1)
                
                # Papers published this year
                papers = db.query(Paper).filter(
                    Paper.published_at >= start,
                    Paper.published_at < end,
                ).all()
                
                total_citations = sum(p.favorite_count or 0 for p in papers)
                avg_citations = total_citations / max(1, len(papers))
                
                trends[str(year)] = {
                    "paper_count": len(papers),
                    "total_citations": total_citations,
                    "avg_citations": avg_citations,
                }
            
            return trends
        
        finally:
            db.close()


# Singleton
_citation_network: Optional[CitationNetwork] = None


def get_citation_network() -> CitationNetwork:
    """Get citation network instance."""
    global _citation_network
    if _citation_network is None:
        _citation_network = CitationNetwork()
    return _citation_network
