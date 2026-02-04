"""
cspapers.org API endpoints - Collect top papers from top CS conferences.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

from app.backend.db.models import Paper, Tag, PaperTag, get_session
from app.backend.services.cspapers import get_cspapers_client, CSPapersClient, VENUES
from app.backend.services.translator import get_translator
from app.backend.services.tagger import get_tagger
from app.backend.config import get_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cspapers", tags=["cspapers"])


def get_db() -> Session:
    """Get database session."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


@router.get("")
def get_top_papers(
    venues: Optional[List[str]] = Query(None, description="Conference venues (e.g., NeurIPS, CVPR, ACL)"),
    year_from: int = Query(2015, ge=1950, le=2030),
    year_to: int = Query(2025, ge=1950, le=2030),
    limit: int = Query(50, ge=1, le=200),
    min_citations: int = Query(10, ge=0),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get top papers from top CS conferences via cspapers.org.
    
    Args:
        venues: List of conference venues to include. Available: AAAI, IJCAI, 
                ICLR, ICML, NeurIPS, CVPR, ICCV, ECCV, ACL, EMNLP, NAACL,
                SIGIR, WWW, KDD, CIKM, ICDM, OSDI, SOSP, ATC, EuroSYS, FSE,
                PLDI, POPL, OOPSLA, ICSE, ASE, ISSTA, SIGMOD, VLDB, ICDE,
                SP, CCS, NDSS, Usenix Security, SC, HPDC, ICS, SIGCOMM, NSDI,
                MobiCom, MobiSys, SenSys, ICRA, IROS, RSS, CHI, UbiComp, UIST,
                FOCS, SODA, STOC, CRYPTO, EuroCrypt, ASPLOS, ISCA, MICRO, HPCA
        year_from: Start year (default: 2015)
        year_to: End year (default: 2025)
        limit: Maximum papers to return (default: 50, max: 200)
        min_citations: Minimum citation count (default: 10)
        
    Returns:
        Dict with papers list and metadata
    """
    client = get_cspapers_client()
    
    papers = client.collect_papers(
        venues=venues,
        year_from=year_from,
        year_to=year_to,
        limit=limit,
        min_citations=min_citations,
    )
    
    return {
        "query": {
            "venues": venues,
            "year_from": year_from,
            "year_to": year_to,
            "limit": limit,
            "min_citations": min_citations,
        },
        "count": len(papers),
        "papers": papers,
        "fetched_at": datetime.utcnow().isoformat(),
    }


@router.get("/venues")
def list_venues() -> Dict[str, Any]:
    """List available conference venues organized by category."""
    client = get_cspapers_client()
    stats = client.get_venue_statistics()
    
    return {
        "venues": VENUES,
        "statistics": stats,
    }


@router.get("/venue/{venue}")
def get_venue_papers(
    venue: str,
    year_from: int = Query(2015, ge=1950, le=2030),
    year_to: int = Query(2025, ge=1950, le=2030),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get top papers for a specific conference venue.
    
    Args:
        venue: Conference venue name (e.g., NeurIPS, CVPR, ACL)
        year_from: Start year
        year_to: End year
        limit: Maximum papers to return
        
    Returns:
        Dict with venue info and papers
    """
    client = get_cspapers_client()
    
    # Validate venue
    venue_upper = venue.upper()
    valid_venues = []
    for venues in VENUES.values():
        valid_venues.extend([v.upper() for v in venues])
    
    if venue_upper not in valid_venues:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown venue '{venue}'. Available venues: {sorted(set(valid_venues))}"
        )
    
    papers = client.get_top_papers_by_venue(
        venue=venue,
        year_from=year_from,
        year_to=year_to,
        limit=limit,
    )
    
    return {
        "venue": venue,
        "year_range": f"{year_from}-{year_to}",
        "count": len(papers),
        "papers": papers,
    }


@router.get("/category/{category}")
def get_category_papers(
    category: str,
    year_from: int = Query(2015, ge=1950, le=2030),
    year_to: int = Query(2025, ge=1950, le=2030),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get top papers for a venue category (e.g., ML, CV, NLP).
    
    Args:
        category: Venue category (AI, ML, CV, NLP, IR, DM, OS, PL, DB, SEC, HPC, NET, Mobile, Robotics, CHI, Theory, Crypto, Arch)
        year_from: Start year
        year_to: End year
        limit: Maximum papers to return
        
    Returns:
        Dict with category info and papers
    """
    client = get_cspapers_client()
    
    # Validate category
    category_upper = category.upper()
    if category_upper not in VENUES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category '{category}'. Available categories: {list(VENUES.keys())}"
        )
    
    venues = VENUES[category_upper]
    
    papers = client.collect_papers(
        venues=venues,
        year_from=year_from,
        year_to=year_to,
        limit=limit,
    )
    
    return {
        "category": category,
        "venues": venues,
        "year_range": f"{year_from}-{year_to}",
        "count": len(papers),
        "papers": papers,
    }


@router.post("/import")
def import_papers_to_db(
    venues: Optional[List[str]] = None,
    year_from: int = Query(2015, ge=1950, le=2030),
    year_to: int = Query(2025, ge=1950, le=2030),
    limit: int = Query(100, ge=1, le=500),
    min_citations: int = Query(10, ge=0),
    auto_translate: bool = Query(True),
    auto_tag: bool = Query(True),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Import top papers from cspapers.org to the local database.
    
    Args:
        venues: Conference venues to import from
        year_from: Start year
        year_to: End year
        limit: Maximum papers to import
        min_citations: Minimum citation count
        auto_translate: Auto-translate abstracts to Chinese
        auto_tag: Auto-generate tags for papers
        
    Returns:
        Dict with import results
    """
    client = get_cspapers_client()
    
    # Collect papers
    papers = client.collect_papers(
        venues=venues,
        year_from=year_from,
        year_to=year_to,
        limit=limit,
        min_citations=min_citations,
    )
    
    # Save to database
    result = client.save_papers_to_db(papers, db)
    
    new_count = result["new"]
    updated_count = result["updated"]
    
    # Auto-translate and tag new papers
    translate_count = 0
    tag_count = 0
    
    if auto_translate or auto_tag:
        new_papers = db.query(Paper).filter(
            Paper.translate_status == "pending"
        ).limit(limit).all()
        
        translator = get_translator()
        tagger = get_tagger()
        
        for paper in new_papers:
            # Translate
            if auto_translate and not paper.abstract_zh:
                translated = translator.translate(paper.abstract_en)
                if translated:
                    paper.abstract_zh = translated
                    paper.translate_status = "success"
                    translate_count += 1
            
            # Tag
            if auto_tag:
                categories = json.loads(paper.categories_json)
                tags, _ = tagger.generate(paper.title, paper.abstract_en, categories)
                
                # Add tags
                for tag_name in tags:
                    tag = db.query(Tag).filter(Tag.name == tag_name).first()
                    if not tag:
                        tag = Tag(name=tag_name, kind="keyword")
                        db.add(tag)
                        db.flush()
                    
                    pt = PaperTag(paper_id=paper.id, tag_id=tag.id)
                    db.add(pt)
                
                paper.tag_status = "success"
                tag_count += 1
    
    db.commit()
    
    return {
        "success": True,
        "imported": new_count,
        "updated": updated_count,
        "translated": translate_count,
        "tagged": tag_count,
        "total_processed": new_count + updated_count,
    }


@router.get("/trending")
def get_trending_papers(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Get trending papers (most cited recently) from top venues.
    
    Args:
        days: Look back period (papers from last X days with most citations)
        limit: Maximum papers to return
        
    Returns:
        List of trending papers
    """
    client = get_cspapers_client()
    
    # Get recent high-impact papers
    papers = client.collect_papers(
        venues=None,  # All venues
        year_from=datetime.now().year - 1,
        year_to=datetime.now().year,
        limit=limit * 3,
        min_citations=100,  # High citation threshold for trending
    )
    
    # Sort by citation density (citations per year)
    for paper in papers:
        years_since = max(1, datetime.now().year - paper.get("year", datetime.now().year))
        paper["citation_density"] = paper.get("citation_count", 0) / years_since
    
    papers.sort(key=lambda x: x.get("citation_density", 0), reverse=True)
    
    return papers[:limit]


@router.get("/statistics")
def get_paper_statistics() -> Dict[str, Any]:
    """Get overall statistics for top CS papers."""
    client = get_cspapers_client()
    stats = client.get_venue_statistics()
    
    # Calculate totals
    total_papers = 0
    total_citations = 0
    
    for category, category_stats in stats.items():
        total_papers += category_stats["paper_count"]
        total_citations += category_stats["total_citations"]
    
    return {
        "categories": len(VENUES),
        "total_papers": total_papers,
        "total_citations": total_citations,
        "statistics": stats,
        "year_range": "2015-2025",
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/search")
def search_papers(
    q: str = Query(..., description="Search query"),
    venues: Optional[List[str]] = None,
    year_from: int = Query(2015, ge=1950, le=2030),
    year_to: int = Query(2025, ge=1950, le=2030),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Search top papers from cspapers.org.
    
    Args:
        q: Search query (matches title, abstract, authors)
        venues: Filter by venues
        year_from: Start year
        year_to: End year
        limit: Maximum results
        
    Returns:
        Search results
    """
    client = get_cspapers_client()
    
    # Get papers from venues
    all_papers = client.collect_papers(
        venues=venues,
        year_from=year_from,
        year_to=year_to,
        limit=limit * 5,  # Get more to filter
    )
    
    # Filter by query
    query_lower = q.lower()
    filtered = [
        p for p in all_papers
        if query_lower in p.get("title", "").lower() or
           query_lower in p.get("abstract", "").lower() or
           any(query_lower in a.lower() for a in p.get("authors", []))
    ]
    
    return {
        "query": q,
        "count": len(filtered),
        "papers": filtered[:limit],
    }


@router.get("/leaderboard")
def get_venue_leaderboard(
    limit: int = Query(20, ge=1, le=50),
) -> List[Dict[str, Any]]:
    """Get venue leaderboard by total citations.
    
    Returns:
        List of venues sorted by total citations
    """
    client = get_cspapers_client()
    stats = client.get_venue_statistics()
    
    leaderboard = []
    
    for category, category_stats in stats.items():
        for venue in category_stats["venues"]:
            leaderboard.append({
                "venue": venue,
                "category": category,
                "paper_count": 0,  # Would be populated from actual data
                "total_citations": category_stats["total_citations"] // len(category_stats["venues"]),
            })
    
    # Sort by citations
    leaderboard.sort(key=lambda x: x["total_citations"], reverse=True)
    
    return leaderboard[:limit]
