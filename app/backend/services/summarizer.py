"""
AI Paper Summarization Service.
Generates summaries, key points, and TL;DR for papers using LLMs.
"""
import json
import logging
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.backend.config import get_config
from app.backend.db.models import Paper, PaperSummary, get_session

logger = logging.getLogger(__name__)


class SummarizationCache:
    """Simple file-based cache for summaries."""
    
    def __init__(self, cache_file: str = "summary_cache.json"):
        """Initialize cache."""
        self.cache_file = cache_file
        self.cache: Dict[str, Dict] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from file."""
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.cache = {}
    
    def save(self):
        """Save cache to file."""
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def get(self, paper_id: int, summary_type: str) -> Optional[str]:
        """Get cached summary."""
        key = f"{paper_id}_{summary_type}"
        return self.cache.get(key)
    
    def set(self, paper_id: int, summary_type: str, summary: str):
        """Cache summary."""
        key = f"{paper_id}_{summary_type}"
        self.cache[key] = {
            "summary": summary,
            "created_at": datetime.utcnow().isoformat(),
        }
        self.save()


class PaperSummarizer:
    """AI-powered paper summarization service."""
    
    def __init__(self):
        """Initialize summarizer."""
        self.config = get_config()
        self.cache = SummarizationCache()
        self._llm_client = None
    
    def _get_llm_client(self):
        """Get LLM client (OpenAI or local)."""
        if self._llm_client is not None:
            return self._llm_client
        
        if self.config.llm_provider == "openai":
            try:
                import openai
                self._llm_client = openai.OpenAI(api_key=self.config.openai_api_key)
            except ImportError:
                logger.warning("OpenAI not installed")
                self._llm_client = "unavailable"
        elif self.config.llm_provider == "ollama":
            try:
                import requests
                self._llm_client = {"base_url": self.config.ollama_url}
            except Exception:
                self._llm_client = "unavailable"
        
        return self._llm_client
    
    def summarize(
        self,
        paper_id: int,
        title: str,
        abstract: str,
        summary_type: str = "comprehensive",
        max_length: int = 300,
    ) -> Dict[str, str]:
        """Generate summary for a paper.
        
        Args:
            paper_id: Paper ID
            title: Paper title
            abstract: Paper abstract
            summary_type: Type of summary (brief, comprehensive, key_points, tldr)
            max_length: Maximum summary length
            
        Returns:
            Dict with summary and metadata
        """
        # Check cache first
        cached = self.cache.get(paper_id, summary_type)
        if cached:
            logger.debug(f"Cache hit for paper {paper_id}")
            return {
                "summary": cached["summary"],
                "type": summary_type,
                "cached": True,
                "created_at": cached.get("created_at"),
            }
        
        # Generate summary using LLM or rule-based fallback
        client = self._get_llm_client()
        
        if client == "unavailable" or self.config.llm_provider == "none":
            # Use rule-based summarization
            summary = self._rule_based_summary(
                abstract, summary_type, max_length
            )
        else:
            # Use LLM
            summary = self._llm_summarize(
                client, title, abstract, summary_type, max_length
            )
        
        # Cache the result
        self.cache.set(paper_id, summary_type, summary)
        
        return {
            "summary": summary,
            "type": summary_type,
            "cached": False,
            "created_at": datetime.utcnow().isoformat(),
        }
    
    def _llm_summarize(
        self,
        client,
        title: str,
        abstract: str,
        summary_type: str,
        max_length: int,
    ) -> str:
        """Generate summary using LLM."""
        prompts = {
            "brief": f"""Summarize this paper in 2-3 sentences:

Title: {title}
Abstract: {abstract}

Brief Summary:""",
            
            "comprehensive": f"""Provide a comprehensive summary of this paper in 4-5 paragraphs:

Title: {title}
Abstract: {abstract}

Comprehensive Summary:""",
            
            "key_points": f"""Extract the key points from this paper as a bulleted list:

Title: {title}
Abstract: {abstract}

Key Points:
-""",
            
            "tldr": f"""Write a very short TL;DR (one sentence) for this paper:

Title: {title}
Abstract: {abstract}

TL;DR:""",
        }
        
        prompt = prompts.get(summary_type, prompts["comprehensive"])
        
        try:
            if self.config.llm_provider == "openai":
                response = client.chat.completions.create(
                    model=self.config.llm_model or "gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_length,
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()
            
            elif self.config.llm_provider == "ollama":
                response = requests.post(
                    f"{self.config.ollama_url}/api/generate",
                    json={
                        "model": self.config.llm_model or "llama3",
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=60,
                )
                data = response.json()
                return data.get("response", "").strip()
        
        except Exception as e:
            logger.error(f"LLM summarization error: {e}")
        
        # Fallback to rule-based
        return self._rule_based_summary(abstract, summary_type, max_length)
    
    def _rule_based_summary(
        self,
        abstract: str,
        summary_type: str,
        max_length: int,
    ) -> str:
        """Generate summary using rule-based methods."""
        sentences = abstract.split(". ")
        
        if summary_type == "tldr":
            # Return first meaningful sentence
            for sent in sentences:
                if len(sent) > 20:
                    return sent.strip() + "."
            return abstract[:100] + "..."
        
        elif summary_type == "brief":
            # First 2-3 sentences
            brief = ". ".join(sentences[:3])
            return brief.strip() + "."
        
        elif summary_type == "key_points":
            # Extract sentences as bullet points
            points = []
            for sent in sentences[:5]:
                if len(sent) > 20:
                    points.append(f"• {sent.strip()}")
            return "\n".join(points)
        
        else:  # comprehensive
            # Full abstract with minor condensation
            return abstract[:max_length * 5] if len(abstract) > max_length * 5 else abstract
    
    def generate_all_summaries(
        self,
        paper_id: int,
        title: str,
        abstract: str,
    ) -> Dict[str, Any]:
        """Generate all summary types for a paper."""
        return {
            "paper_id": paper_id,
            "summaries": {
                "brief": self.summarize(paper_id, title, abstract, "brief", 150),
                "comprehensive": self.summarize(paper_id, title, abstract, "comprehensive", 500),
                "key_points": self.summarize(paper_id, title, abstract, "key_points", 400),
                "tldr": self.summarize(paper_id, title, abstract, "tldr", 50),
            },
            "generated_at": datetime.utcnow().isoformat(),
        }
    
    def compare_papers(
        self,
        papers: List[Dict[str, str]],
    ) -> str:
        """Compare multiple papers and find similarities/differences.
        
        Args:
            papers: List of dicts with title, abstract, venue
            
        Returns:
            Comparison text
        """
        if len(papers) < 2:
            return "Need at least 2 papers to compare"
        
        prompt = f"""Compare these {len(papers)} papers and identify:
1. Common themes and methodologies
2. Key differences
3. How they relate to each other

Papers:
"""
        for i, p in enumerate(papers):
            prompt += f"\n{i+1}. {p.get('title', '')} ({p.get('venue', '')})"
            prompt += f"\nAbstract: {p.get('abstract', '')[:300]}...\n"
        
        prompt += "\n\nComparison:"
        
        client = self._get_llm_client()
        
        if client == "unavailable" or self.config.llm_provider == "none":
            return "LLM not configured for paper comparison"
        
        try:
            if self.config.llm_provider == "openai":
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.5,
                )
                return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Comparison error: {e}")
        
        return "Unable to generate comparison"


# Singleton
_summarizer: Optional[PaperSummarizer] = None


def get_summarizer() -> PaperSummarizer:
    """Get summarizer instance."""
    global _summarizer
    if _summarizer is None:
        _summarizer = PaperSummarizer()
    return _summarizer
