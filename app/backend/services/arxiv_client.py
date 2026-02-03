"""
arXiv API client for fetching papers.
"""
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional

import requests
from xml.etree import ElementTree as ET

from ..config import get_config

logger = logging.getLogger(__name__)

ARXIV_API_BASE = "http://export.arxiv.org/api/query"


def parse_arxiv_datetime(dt_str: str) -> datetime:
    """Parse arXiv datetime string to datetime object."""
    # Handle formats like "2024-01-15T12:34:56Z"
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str)


def generate_abstract_hash(abstract: str) -> str:
    """Generate SHA256 hash of abstract for deduplication."""
    return hashlib.sha256(abstract.encode("utf-8")).hexdigest()


class ArxivClient:
    """Client for fetching papers from arXiv API."""
    
    def __init__(self, categories: Optional[List[str]] = None):
        """Initialize arXiv client."""
        self.config = get_config()
        self.categories = categories or self.config.arxiv_categories
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ArxivMiner/1.0 (contact@example.com)"
        })
    
    def fetch_all(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch all papers from configured categories."""
        all_papers = []
        
        for category in self.categories:
            papers = self.fetch_category(
                category,
                start_date=start_date,
                end_date=end_date,
                max_results=max_results,
            )
            all_papers.extend(papers)
            logger.info(f"Fetched {len(papers)} papers from {category}")
        
        return all_papers
    
    def fetch_category(
        self,
        category: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch papers from a specific category."""
        config = get_config()
        
        # Build query
        if start_date:
            date_str = start_date.strftime("%Y%m%d%H%M%S")
        else:
            # Default to fetch_window_hours ago
            hours_ago = datetime.utcnow() - timedelta(hours=config.fetch_window_hours)
            date_str = hours_ago.strftime("%Y%m%d%H%M%S")
        
        # Search query: cat:category AND (submittedDate OR lastUpdatedDate) >= date
        if config.fetch_mode == "published":
            search_query = f"cat:{category}+AND+submittedDate:[{date_str}+TO+NOW]"
        else:
            search_query = f"cat:{category}+AND+lastUpdatedDate:[{date_str}+TO+NOW]"
        
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        
        all_entries = []
        total_fetched = 0
        
        while True:
            try:
                response = self._fetch_with_retry(params)
                entries = self._parse_feed(response.text)
                all_entries.extend(entries)
                total_fetched += len(entries)
                
                if len(entries) < max_results:
                    break
                
                params["start"] += max_results
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching category {category}: {e}")
                break
        
        logger.info(f"Total fetched {total_fetched} papers from {category}")
        return all_entries
    
    def _fetch_with_retry(
        self, params: Dict[str, Any], max_retries: int = 3
    ) -> requests.Response:
        """Fetch with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    ARXIV_API_BASE,
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                return response
            
            except requests.RequestException as e:
                wait_time = (2 ** attempt) * 1
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Waiting {wait_time}s...")
                time.sleep(wait_time)
        
        raise Exception(f"Failed after {max_retries} retries")
    
    def _parse_feed(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse arXiv Atom feed XML."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            return []
        
        entries = []
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        
        for entry in root.findall("atom:entry", ns):
            try:
                paper = self._parse_entry(entry, ns)
                entries.append(paper)
            except Exception as e:
                logger.error(f"Error parsing entry: {e}")
                continue
        
        return entries
    
    def _parse_entry(self, entry: ET.Element, ns: Dict[str, str]) -> Dict[str, Any]:
        """Parse a single arXiv entry."""
        # arXiv ID
        arxiv_id_elem = entry.find("atom:id", ns)
        arxiv_id = ""
        if arxiv_id_elem is not None:
            id_text = arxiv_id_elem.text or ""
            # Extract ID from URL like http://arxiv.org/abs/2401.01234v1
            if "/abs/" in id_text:
                arxiv_id = id_text.split("/abs/")[1].split("v")[0]
        
        # Title
        title_elem = entry.find("atom:title", ns)
        title = ""
        if title_elem is not None and title_elem.text:
            title = title_elem.text.strip()
            # Clean up newlines that arXiv uses for formatting
            title = " ".join(title.split())
        
        # Authors
        authors = []
        for author in entry.findall("atom:author", ns):
            name_elem = author.find("atom:name", ns)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())
        
        # Abstract
        summary_elem = entry.find("atom:summary", ns)
        abstract = ""
        if summary_elem is not None and summary_elem.text:
            abstract = summary_elem.text.strip()
            abstract = " ".join(abstract.split())
        
        # Categories
        categories = []
        primary_category = ""
        for category in entry.findall("atom:category", ns):
            cat_term = category.get("term")
            if cat_term:
                categories.append(cat_term)
                if not primary_category:
                    primary_category = cat_term
        
        # Published date
        published_elem = entry.find("atom:published", ns)
        published = datetime.utcnow()
        if published_elem is not None and published_elem.text:
            published = parse_arxiv_datetime(published_elem.text)
        
        # Updated date
        updated_elem = entry.find("atom:updated", ns)
        updated = datetime.utcnow()
        if updated_elem is not None and updated_elem.text:
            updated = parse_arxiv_datetime(updated_elem.text)
        
        # Links
        abs_url = ""
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            href = link.get("href", "")
            rel = link.get("rel", "")
            if "/abs/" in href:
                abs_url = href
            elif "/pdf/" in href:
                pdf_url = href.replace("/pdf/", "/abs/").replace(".pdf", "")
        
        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors_json": json.dumps(authors, ensure_ascii=False),
            "abstract_en": abstract,
            "categories_json": json.dumps(categories, ensure_ascii=False),
            "primary_category": primary_category,
            "published_at": published,
            "updated_at": updated,
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else "",
            "abstract_hash": generate_abstract_hash(abstract),
        }
