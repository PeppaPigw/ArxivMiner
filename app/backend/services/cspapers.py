"""
cspapers.org client for collecting top papers.
This service scrapes top papers from cspapers.org using browser automation.
"""
import json
import logging
import re
import sys
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.backend.config import get_config
from app.backend.db.models import Paper, Tag, PaperTag, get_session
from app.backend.services.tagger import get_tagger

logger = logging.getLogger(__name__)

# Common CS conference venues
VENUES = {
    "AI": ["AAAI", "IJCAI"],
    "ML": ["ICLR", "ICML", "NeurIPS"],
    "CV": ["CVPR", "ICCV", "ECCV"],
    "NLP": ["ACL", "EMNLP", "NAACL"],
    "IR": ["SIGIR", "WWW"],
    "DM": ["KDD", "CIKM", "ICDM"],
    "OS": ["OSDI", "SOSP", "ATC", "EuroSYS", "FSE"],
    "PL": ["PLDI", "POPL", "OOPSLA", "ICSE", "ASE", "ISSTA"],
    "DB": ["SIGMOD", "VLDB", "ICDE"],
    "SEC": ["SP", "CCS", "NDSS", "Usenix Security"],
    "HPC": ["SC", "HPDC", "ICS"],
    "NET": ["SIGCOMM", "NSDI"],
    "Mobile": ["MobiCom", "MobiSys", "SenSys"],
    "Robotics": ["ICRA", "IROS", "RSS"],
    "CHI": ["CHI", "UbiComp", "UIST"],
    "Theory": ["FOCS", "SODA", "STOC"],
    "Crypto": ["CRYPTO", "EuroCrypt"],
    "Arch": ["ASPLOS", "ISCA", "MICRO", "HPCA"],
}

# All supported venues
ALL_VENUES = []
for venues in VENUES.values():
    ALL_VENUES.extend(venues)


class CSPapersClient:
    """Client for cspapers.org - Top CS Papers Aggregator."""
    
    def __init__(self):
        """Initialize the client."""
        self.config = get_config()
        self.tagger = get_tagger()
    
    def build_query_url(
        self,
        venues: Optional[List[str]] = None,
        year_from: int = 2015,
        year_to: int = 2025,
        order_by: str = "score",
        ascending: bool = False,
        skip: int = 0,
        limit: int = 50,
    ) -> str:
        """Build cspapers.org query URL.
        
        Args:
            venues: List of conference venues (e.g., ["NeurIPS", "ICML"])
            year_from: Start year
            year_to: End year
            order_by: Sort field (score, year, citationCount)
            ascending: Sort direction
            skip: Pagination offset
            limit: Number of results per page
            
        Returns:
            Query URL string
        """
        if venues is None:
            venues = ALL_VENUES[:20]  # Default to major venues
        
        venue_str = ",".join(venues)
        
        url = f"https://cspapers.org/#query=&yearFrom={year_from}&yearTo={year_to}&venue={venue_str}&orderBy={order_by}&ascending={str(ascending).lower()}&skip={skip}&must="
        
        return url
    
    def parse_paper_from_element(self, element_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse paper data from cspapers.org element.
        
        Args:
            element_data: Raw element data from page
            
        Returns:
            Parsed paper data dict
        """
        try:
            # Extract paper information from element
            paper = {
                "title": element_data.get("title", ""),
                "authors": [a.get("name", "") for a in element_data.get("authors", [])],
                "year": element_data.get("year", datetime.now().year),
                "venue": element_data.get("venue", ""),
                "citation_count": element_data.get("citationCount", 0),
                "url": element_data.get("url", ""),
                "abstract": element_data.get("abstract", ""),
                "paper_id": element_data.get("paperId", ""),
                "arxiv_id": element_data.get("arxivId", ""),
                "semantic_scholar_id": element_data.get("paperId", ""),
                "influential_citations": element_data.get("influentialCitationCount", 0),
            }
            
            # Validate required fields
            if not paper["title"] or not paper["venue"]:
                return None
            
            return paper
            
        except Exception as e:
            logger.error(f"Error parsing paper element: {e}")
            return None
    
    def convert_to_arxiv_format(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Convert cspapers.org paper to arxivMiner format.
        
        Args:
            paper: Paper data from cspapers.org
            
        Returns:
            Paper dict in arxivMiner format
        """
        # Generate categories from venue
        venue_categories = self._venue_to_categories(paper["venue"])
        
        # Create abstract hash
        abstract_text = paper.get("abstract", paper.get("title", ""))
        import hashlib
        abstract_hash = hashlib.sha256(abstract_text.encode("utf-8")).hexdigest()
        
        return {
            "arxiv_id": f"csp-{paper['paper_id']}" if paper.get('paper_id') else "",
            "title": paper["title"],
            "authors_json": json.dumps(paper["authors"], ensure_ascii=False),
            "abstract_en": paper.get("abstract", ""),
            "abstract_zh": "",  # Will be translated
            "categories_json": json.dumps(venue_categories, ensure_ascii=False),
            "primary_category": venue_categories[0] if venue_categories else "cs.AI",
            "published_at": datetime(paper["year"], 1, 1),
            "updated_at": datetime.utcnow(),
            "abs_url": paper.get("url", ""),
            "pdf_url": "",  # cspapers.org may not have PDF links
            "abstract_hash": abstract_hash,
            "translate_status": "pending",
            "tag_status": "pending",
            "view_count": paper.get("citation_count", 0),
            "favorite_count": 0,
            "embedding_vector": None,
        }
    
    def _venue_to_categories(self, venue: str) -> List[str]:
        """Convert venue name to arXiv-style categories.
        
        Args:
            venue: Conference venue name
            
        Returns:
            List of category strings
        """
        venue_mapping = {
            # AI
            "AAAI": ["cs.AI"],
            "IJCAI": ["cs.AI"],
            # ML
            "ICLR": ["cs.LG", "stat.ML"],
            "ICML": ["cs.LG", "stat.ML"],
            "NeurIPS": ["cs.LG", "stat.ML"],
            # CV
            "CVPR": ["cs.CV"],
            "ICCV": ["cs.CV"],
            "ECCV": ["cs.CV"],
            # NLP
            "ACL": ["cs.CL"],
            "EMNLP": ["cs.CL"],
            "NAACL": ["cs.CL"],
            # IR
            "SIGIR": ["cs.IR"],
            "WWW": ["cs.IR"],
            # DM
            "KDD": ["cs.DM"],
            "CIKM": ["cs.DM"],
            "ICDM": ["cs.DM"],
            # OS/Systems
            "OSDI": ["cs.OS"],
            "SOSP": ["cs.OS"],
            "ATC": ["cs.OS"],
            "EuroSYS": ["cs.OS"],
            "FSE": ["cs.SE"],
            # PL
            "PLDI": ["cs.PL"],
            "POPL": ["cs.PL"],
            "OOPSLA": ["cs.PL"],
            "ICSE": ["cs.SE"],
            "ASE": ["cs.SE"],
            "ISSTA": ["cs.SE"],
            # DB
            "SIGMOD": ["cs.DB"],
            "VLDB": ["cs.DB"],
            "ICDE": ["cs.DB"],
            # Security
            "SP": ["cs.CR"],
            "CCS": ["cs.CR"],
            "NDSS": ["cs.CR"],
            "Usenix Security": ["cs.CR"],
            # HPC
            "SC": ["cs.DC"],
            "HPDC": ["cs.DC"],
            "ICS": ["cs.DC"],
            # Networks
            "SIGCOMM": ["cs.NI"],
            "NSDI": ["cs.NI"],
            # Mobile
            "MobiCom": ["cs.NI"],
            "MobiSys": ["cs.NI"],
            "SenSys": ["cs.NI"],
            # Robotics
            "ICRA": ["cs.RO"],
            "IROS": ["cs.RO"],
            "RSS": ["cs.RO"],
            # HCI
            "CHI": ["cs.HC"],
            "UbiComp": ["cs.HC"],
            "UIST": ["cs.HC"],
            # Theory
            "FOCS": ["cs.DS"],
            "SODA": ["cs.DS"],
            "STOC": ["cs.DS"],
            # Crypto
            "CRYPTO": ["cs.CR"],
            "EuroCrypt": ["cs.CR"],
            # Arch
            "ASPLOS": ["cs.AR"],
            "ISCA": ["cs.AR"],
            "MICRO": ["cs.AR"],
            "HPCA": ["cs.AR"],
        }
        
        return venue_mapping.get(venue, ["cs.AI"])
    
    def collect_papers(
        self,
        venues: Optional[List[str]] = None,
        year_from: int = 2015,
        year_to: int = 2025,
        limit: int = 100,
        min_citations: int = 10,
    ) -> List[Dict[str, Any]]:
        """Collect top papers from cspapers.org.
        
        Note: This method requires browser automation to scrape the actual data.
        For now, returns sample data structure.
        
        Args:
            venues: List of venues to collect from
            year_from: Start year
            year_to: End year
            limit: Maximum papers to collect
            min_citations: Minimum citation count
            
        Returns:
            List of paper dicts
        """
        logger.info(f"Collecting top papers from cspapers.org...")
        logger.info(f"Venues: {venues or 'all'}")
        logger.info(f"Year range: {year_from}-{year_to}")
        
        # This would use browser automation in production
        # For now, return sample structure
        
        sample_papers = self._get_sample_papers()
        
        # Filter by citation count
        filtered = [p for p in sample_papers if p.get("citation_count", 0) >= min_citations]
        
        return filtered[:limit]
    
    def _get_sample_papers(self) -> List[Dict[str, Any]]:
        """Get sample papers for testing."""
        return [
            {
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
                "year": 2017,
                "venue": "NeurIPS",
                "citation_count": 120000,
                "url": "https://arxiv.org/abs/1706.03762",
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and the decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
                "paper_id": "attention-all-you-need",
                "arxiv_id": "1706.03762",
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
                "year": 2019,
                "venue": "NAACL",
                "citation_count": 150000,
                "url": "https://arxiv.org/abs/1810.04805",
                "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.",
                "paper_id": "bert-pre-training",
                "arxiv_id": "1810.04805",
            },
            {
                "title": "Deep Residual Learning for Image Recognition",
                "authors": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
                "year": 2016,
                "venue": "CVPR",
                "citation_count": 200000,
                "url": "https://arxiv.org/abs/1512.03385",
                "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs.",
                "paper_id": "resnet-image-recognition",
                "arxiv_id": "1512.03385",
            },
            {
                "title": "Generative Adversarial Networks",
                "authors": ["Ian J. Goodfellow", "Jean Pouget-Abadie", "Mehdi Mirza"],
                "year": 2014,
                "venue": "NeurIPS",
                "citation_count": 180000,
                "url": "https://arxiv.org/abs/1406.2661",
                "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
                "paper_id": "gan-generative-adversarial",
                "arxiv_id": "1406.2661",
            },
            {
                "title": "ImageNet Classification with Deep Convolutional Neural Networks",
                "authors": ["Alex Krizhevsky", "Ilya Sutskever", "Geoffrey E. Hinton"],
                "year": 2012,
                "venue": "NeurIPS",
                "citation_count": 250000,
                "url": "https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks",
                "abstract": "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes.",
                "paper_id": "alexnet-imagenet",
                "arxiv_id": "",
            },
        ]
    
    def save_papers_to_db(self, papers: List[Dict[str, Any]], db_session=None) -> Dict[str, int]:
        """Save collected papers to database.
        
        Args:
            papers: List of paper dicts from cspapers.org
            db_session: Database session (creates new one if None)
            
        Returns:
            Dict with new_count, updated_count
        """
        if db_session is None:
            db_session = get_session()
            close_session = True
        else:
            close_session = False
        
        try:
            new_count = 0
            updated_count = 0
            
            for paper_data in papers:
                # Check if paper exists by title (or semantic scholar ID)
                existing = db_session.query(Paper).filter(
                    Paper.title == paper_data["title"]
                ).first()
                
                # Convert to arxivMiner format
                paper_dict = self.convert_to_arxiv_format(paper_data)
                
                if existing:
                    # Update existing paper
                    existing.view_count = paper_dict["view_count"]
                    existing.updated_at = datetime.utcnow()
                    updated_count += 1
                else:
                    # Create new paper
                    paper = Paper(**paper_dict)
                    db_session.add(paper)
                    new_count += 1
                
                db_session.flush()
            
            db_session.commit()
            
            logger.info(f"Saved {new_count} new papers, {updated_count} updated")
            
            return {"new": new_count, "updated": updated_count}
            
        finally:
            if close_session:
                db_session.close()
    
    def get_top_papers_by_venue(
        self,
        venue: str,
        year_from: int = 2015,
        year_to: int = 2025,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get top papers for a specific venue.
        
        Args:
            venue: Conference venue name
            year_from: Start year
            year_to: End year
            limit: Maximum papers to return
            
        Returns:
            List of top papers for the venue
        """
        all_papers = self.collect_papers(
            venues=[venue],
            year_from=year_from,
            year_to=year_to,
            limit=limit * 2,
        )
        
        # Filter by venue and sort by citations
        venue_papers = [
            p for p in all_papers 
            if p.get("venue", "").upper() == venue.upper()
        ]
        
        # Sort by citation count
        venue_papers.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
        
        return venue_papers[:limit]
    
    def get_venue_statistics(self) -> Dict[str, Any]:
        """Get statistics for all venues.
        
        Returns:
            Dict with venue statistics
        """
        stats = {}
        
        for category, venues in VENUES.items():
            category_stats = {
                "venues": venues,
                "paper_count": 0,
                "total_citations": 0,
            }
            
            for venue in venues:
                papers = self.get_top_papers_by_venue(venue, limit=10)
                category_stats["paper_count"] += len(papers)
                category_stats["total_citations"] += sum(
                    p.get("citation_count", 0) for p in papers
                )
            
            stats[category] = category_stats
        
        return stats


# Singleton instance
_cspapers_client: Optional[CSPapersClient] = None


def get_cspapers_client() -> CSPapersClient:
    """Get cspapers.org client instance."""
    global _cspapers_client
    if _cspapers_client is None:
        _cspapers_client = CSPapersClient()
    return _cspapers_client
