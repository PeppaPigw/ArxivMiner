"""
PDF Parser Service.
Extract text, tables, figures from PDF files.
"""
import io
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.backend.config import get_config

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract content from PDF files."""
    
    def __init__(self):
        """Initialize PDF extractor."""
        self.config = get_config()
        self._pdfminer_installed = None
        self._fitz_installed = None
    
    def _check_dependencies(self):
        """Check if PDF libraries are available."""
        if self._pdfminer_installed is None:
            try:
                from pdfminer.high_level import extract_text
                self._pdfminer_installed = True
            except ImportError:
                self._pdfminer_installed = False
        
        if self._fitz_installed is None:
            try:
                import fitz
                self._fitz_installed = True
            except ImportError:
                self._fitz_installed = False
        
        return self._pdfminer_installed or self._fitz_installed
    
    def extract_text(
        self,
        pdf_path: str = None,
        pdf_content: bytes = None,
        max_pages: int = 50,
    ) -> Dict[str, Any]:
        """Extract text from PDF.
        
        Args:
            pdf_path: Path to PDF file
            pdf_content: PDF content as bytes
            max_pages: Maximum pages to extract
            
        Returns:
            Dict with text and metadata
        """
        if not self._check_dependencies():
            logger.warning("PDF libraries not installed")
            return {
                "text": "",
                "error": "PDF libraries not installed. Install with: pip install pdfminer.six pymupdf",
                "pages": 0,
            }
        
        try:
            if pdf_content:
                with open("/tmp/temp.pdf", "wb") as f:
                    f.write(pdf_content)
                pdf_path = "/tmp/temp.pdf"
            
            if self._fitz_installed:
                return self._extract_with_fitz(pdf_path, max_pages)
            else:
                return self._extract_with_pdfminer(pdf_path, max_pages)
        
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return {
                "text": "",
                "error": str(e),
                "pages": 0,
            }
    
    def _extract_with_fitz(
        self,
        pdf_path: str,
        max_pages: int,
    ) -> Dict[str, Any]:
        """Extract text using PyMuPDF (fitz)."""
        import fitz
        
        doc = fitz.open(pdf_path)
        pages = min(len(doc), max_pages)
        
        full_text = []
        page_texts = []
        
        for i in range(pages):
            page = doc[i]
            text = page.get_text()
            page_texts.append({
                "page": i + 1,
                "text": text,
                "has_images": len(page.get_images()) > 0,
            })
            full_text.append(text)
        
        doc.close()
        
        return {
            "text": "\n\n".join(full_text),
            "pages": pages,
            "page_details": page_texts,
            "extraction_method": "fitz",
        }
    
    def _extract_with_pdfminer(
        self,
        pdf_path: str,
        max_pages: int,
    ) -> Dict[str, Any]:
        """Extract text using pdfminer."""
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams
        
        laparams = LAParams()
        text = extract_text(pdf_path, maxpages=max_pages, laparams=laparams)
        
        return {
            "text": text,
            "pages": min(max_pages, text.count("\f") + 1),
            "extraction_method": "pdfminer",
        }
    
    def extract_sections(
        self,
        pdf_path: str = None,
        pdf_content: bytes = None,
    ) -> Dict[str, Any]:
        """Extract paper sections (abstract, intro, methods, etc.).
        
        Args:
            pdf_path: Path to PDF file
            pdf_content: PDF content as bytes
            
        Returns:
            Dict with extracted sections
        """
        result = self.extract_text(pdf_path, pdf_content)
        
        if not result.get("text"):
            return result
        
        text = result["text"]
        
        # Section patterns
        section_patterns = {
            "abstract": r"(?i)(?:^|\n)\s*(abstract|摘要)\s*[:\n](.*?)(?=\n\s*(?:1\.?\s*(?:introduction|引言)|2\.?\s*))",
            "introduction": r"(?i)(?:^|\n)\s*1\.?\s*(?:introduction|引言)\s*[:\n](.*?)(?=\n\s*2\.?\s*)",
            "related_work": r"(?i)(?:^|\n)\s*(?:2|3|4)\.?\s*(?:related work|background|prior work)\s*[:\n](.*?)(?=\n\s*(?:\d+\.?\s*))",
            "methods": r"(?i)(?:^|\n)\s*(?:2|3|4)\.?\s*(?:methodology|methods|approach|proposed method)\s*[:\n](.*?)(?=\n\s*(?:\d+\.?\s*))",
            "experiments": r"(?i)(?:^|\n)\s*(?:4|5|6)\.?\s*(?:experiments|evaluation|results)\s*[:\n](.*?)(?=\n\s*(?:\d+\.?\s*))",
            "conclusion": r"(?i)(?:^|\n)\s*(?:conclusion|conclusions|discussion)\s*[:\n](.*?)(?=\n\s*(?:references|bibliography|$))",
            "references": r"(?i)(?:^|\n)\s*(?:references|bibliography)\s*[:\n](.*)",
        }
        
        sections = {}
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                # Clean up
                content = re.sub(r"\n+", "\n", content)
                content = re.sub(r"\s+", " ", content)
                sections[section_name] = content
        
        return {
            "sections": sections,
            "full_text": text,
            "page_count": result.get("pages", 0),
        }
    
    def extract_tables(
        self,
        pdf_path: str = None,
        pdf_content: bytes = None,
    ) -> List[Dict[str, Any]]:
        """Extract tables from PDF.
        
        Args:
            pdf_path: Path to PDF file
            pdf_content: PDF content as bytes
            
        Returns:
            List of extracted tables
        """
        if not self._check_dependencies():
            return []
        
        # PyMuPDF can extract tables
        if self._fitz_installed and pdf_path:
            import fitz
            
            try:
                doc = fitz.open(pdf_path)
                tables = []
                
                for page_num, page in enumerate(doc):
                    tables_on_page = page.find_tables()
                    
                    for table_num, table in enumerate(tables_on_page):
                        table_data = []
                        for row in table.extract():
                            table_data.append(row)
                        
                        if table_data:
                            tables.append({
                                "page": page_num + 1,
                                "table_number": table_num + 1,
                                "rows": len(table_data),
                                "columns": len(table_data[0]) if table_data else 0,
                                "data": table_data[:10],  # First 10 rows
                            })
                
                doc.close()
                return tables
            
            except Exception as e:
                logger.error(f"Table extraction error: {e}")
                return []
        
        # Fallback: try to find table-like patterns in text
        result = self.extract_text(pdf_path, pdf_content)
        text = result.get("text", "")
        
        # Look for table patterns
        table_pattern = r"(?:Table|Tab\.|表)\s*\.?\s*(\d+)[:\.]?\s*(.*?)(?=\n(?:Table|Tab\.|表)|\n\n|\Z)"
        matches = re.findall(table_pattern, text, re.DOTALL | re.IGNORECASE)
        
        tables = []
        for i, (table_num, content) in enumerate(matches):
            tables.append({
                "table_number": int(table_num),
                "page": 0,  # Unknown from text extraction
                "content": content.strip()[:500],
            })
        
        return tables
    
    def extract_figures(
        self,
        pdf_path: str = None,
        pdf_content: bytes = None,
    ) -> List[Dict[str, Any]]:
        """Extract figure references from PDF.
        
        Args:
            pdf_path: Path to PDF file
            pdf_content: PDF content as bytes
            
        Returns:
            List of figure references
        """
        result = self.extract_text(pdf_path, pdf_content)
        text = result.get("text", "")
        
        # Look for figure references
        figure_patterns = [
            r"(?:Figure|Fig\.|图)\s*\.?\s*(\d+)[:\.]?\s*([^\n]+)",
            r"(?:Figure|Fig\.|图)\s*(\d+)\s*[-–]\s*([^\n]+)",
        ]
        
        figures = []
        seen = set()
        
        for pattern in figure_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for fig_num, caption in matches:
                key = f"{fig_num}"
                if key not in seen:
                    seen.add(key)
                    figures.append({
                        "figure_number": fig_num,
                        "caption": caption.strip()[:200],
                        "page": 0,
                    })
        
        # Sort by figure number
        figures.sort(key=lambda x: int(x.get("figure_number", 0)))
        
        return figures
    
    def get_metadata(
        self,
        pdf_path: str = None,
        pdf_content: bytes = None,
    ) -> Dict[str, Any]:
        """Extract PDF metadata.
        
        Args:
            pdf_path: Path to PDF file
            pdf_content: PDF content as bytes
            
        Returns:
            Dict with metadata
        """
        if not self._check_dependencies():
            return {"error": "PDF libraries not installed"}
        
        if self._fitz_installed:
            import fitz
            
            if pdf_content:
                with open("/tmp/temp.pdf", "wb") as f:
                    f.write(pdf_content)
                pdf_path = "/tmp/temp.pdf"
            
            try:
                doc = fitz.open(pdf_path)
                meta = doc.metadata
                doc.close()
                
                return {
                    "title": meta.get("title", ""),
                    "author": meta.get("author", ""),
                    "subject": meta.get("subject", ""),
                    "keywords": meta.get("keywords", ""),
                    "creator": meta.get("creator", ""),
                    "producer": meta.get("producer", ""),
                    "creation_date": meta.get("creationDate", ""),
                    "modification_date": meta.get("modDate", ""),
                    "page_count": doc.page_count if doc else 0,
                }
            except Exception as e:
                return {"error": str(e)}
        
        return {"error": "PyMuPDF required for metadata extraction"}
    
    def calculate_readability(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """Calculate readability metrics for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with readability scores
        """
        if not text:
            return {"error": "No text provided"}
        
        # Basic metrics
        sentences = len(re.split(r"[.!?]+", text))
        words = len(text.split())
        syllables = sum(self._count_syllables(word) for word in text.split())
        
        avg_words_per_sentence = words / max(1, sentences)
        avg_syllables_per_word = syllables / max(1, words)
        
        # Flesch-Kincaid Grade Level
        fk_grade = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59
        
        # Flesch Reading Ease
        flesch_score = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
        
        return {
            "word_count": words,
            "sentence_count": sentences,
            "avg_words_per_sentence": round(avg_words_per_sentence, 2),
            "avg_syllables_per_word": round(avg_syllables_per_word, 2),
            "flesch_kincaid_grade": round(fk_grade, 2),
            "flesch_reading_ease": round(max(0, min(100, flesch_score)), 2),
            "reading_difficulty": self._get_readability_level(flesch_score),
        }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith("e") and count > 1:
            count -= 1
        
        return max(1, count)
    
    def _get_readability_level(self, score: float) -> str:
        """Get readability level description."""
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Difficult"
        elif score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"


# Singleton
_pdf_extractor: Optional[PDFExtractor] = None


def get_pdf_extractor() -> PDFExtractor:
    """Get PDF extractor instance."""
    global _pdf_extractor
    if _pdf_extractor is None:
        _pdf_extractor = PDFExtractor()
    return _pdf_extractor
