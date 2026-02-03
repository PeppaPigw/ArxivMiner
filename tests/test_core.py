"""
Unit tests for ArxivMiner.
"""
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.backend.config import Config, load_config
from app.backend.db.models import Paper, Tag, PaperTag, UserState
from app.backend.services.arxiv_client import ArxivClient, generate_abstract_hash, parse_arxiv_datetime
from app.backend.services.tagger import KeywordTagger


class TestConfig:
    """Test configuration module."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.arxiv_categories == ["cs.AI", "cs.CL", "stat.ML"]
        assert config.fetch_window_hours == 24
        assert config.database_url == "sqlite:///./data/arxivminer.db"
        assert config.tags_per_paper == 8
        assert config.app_port == 8000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = Config(
            arxiv_categories=["cs.CV", "cs.LG"],
            fetch_window_hours=48,
            deepl_api_key="test_key",
            database_url="sqlite:///test.db"
        )
        
        assert config.arxiv_categories == ["cs.CV", "cs.LG"]
        assert config.fetch_window_hours == 48
        assert config.deepl_api_key == "test_key"
        assert config.database_url == "sqlite:///test.db"


class TestArxivClient:
    """Test arXiv client functions."""
    
    def test_generate_abstract_hash(self):
        """Test abstract hash generation."""
        abstract = "This is a test abstract for hashing."
        hash1 = generate_abstract_hash(abstract)
        hash2 = generate_abstract_hash(abstract)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 hex characters
        assert hash1.startswith("a9")  # Consistent hash for this text
    
    def test_generate_abstract_hash_different(self):
        """Test that different abstracts produce different hashes."""
        hash1 = generate_abstract_hash("Abstract 1")
        hash2 = generate_abstract_hash("Abstract 2")
        
        assert hash1 != hash2
    
    def test_parse_arxiv_datetime(self):
        """Test datetime parsing."""
        dt_str = "2024-01-15T12:34:56Z"
        dt = parse_arxiv_datetime(dt_str)
        
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 12
        assert dt.minute == 34
        assert dt.second == 56


class TestKeywordTagger:
    """Test keyword tagger."""
    
    def test_generate_tags_basic(self):
        """Test basic tag generation."""
        tagger = KeywordTagger(tags_per_paper=5)
        
        title = "Attention Is All You Need: A Novel Approach for Neural Machine Translation"
        abstract = "We propose a novel transformer architecture based solely on attention mechanisms."
        categories = ["cs.CL", "cs.LG"]
        
        tags, timestamp = tagger.generate(title, abstract, categories)
        
        assert isinstance(tags, list)
        assert len(tags) <= 5
        assert timestamp is not None
        # Categories should be included
        assert "Cs_Cl" in tags or "Cs_Lg" in tags
    
    def test_generate_tags_technical_terms(self):
        """Test that technical terms are detected."""
        tagger = KeywordTagger(tags_per_paper=10)
        
        title = "GPT-4: Large Language Model for Reasoning"
        abstract = "We present GPT-4, a large language model trained with reinforcement learning from human feedback."
        categories = ["cs.AI"]
        
        tags, _ = tagger.generate(title, abstract, categories)
        
        # Should detect technical terms like LLM, GPT
        assert len(tags) > 0
        
    def test_normalize_tag(self):
        """Test tag normalization."""
        tagger = KeywordTagger()
        
        # Test single word normalization
        assert tagger._normalize_tag("llm") == "Llm"
        assert tagger._normalize_tag("CNN") == "Cnn"
        
        # Test multi-word normalization
        assert tagger._normalize_tag("large language model") == "Large Language Model"
        
    def test_tokenize(self):
        """Test text tokenization."""
        tagger = KeywordTagger()
        
        text = "This is a test with some technical terms like neural networks."
        words = tagger._tokenize(text)
        
        # Stopwords should be filtered
        assert "this" not in words
        assert "is" not in words
        assert "a" not in words
        assert "test" in words
        assert "neural" in words
        assert "networks" in words
    
    def test_extract_ngrams(self):
        """Test n-gram extraction."""
        tagger = KeywordTagger()
        
        text = "machine learning and deep learning are important"
        ngrams = tagger._extract_ngrams(text, n=[1, 2, 3])
        
        assert len(ngrams) > 0
        # Should contain unigrams
        assert any("machine" in ng for ng in ngrams)
        # Should contain bigrams
        assert any("machine learning" in ng for ng in ngrams)


class TestPaperModel:
    """Test paper database model."""
    
    def test_paper_to_dict(self):
        """Test paper serialization to dictionary."""
        paper = Paper(
            id=1,
            arxiv_id="2401.01234",
            title="Test Paper Title",
            authors_json='["Author One", "Author Two"]',
            abstract_en="This is a test abstract.",
            abstract_zh="这是测试摘要。",
            categories_json='["cs.AI", "cs.LG"]',
            primary_category="cs.AI",
            published_at=datetime(2024, 1, 15, 12, 0, 0),
            updated_at=datetime(2024, 1, 15, 12, 0, 0),
            abs_url="https://arxiv.org/abs/2401.01234",
            pdf_url="https://arxiv.org/pdf/2401.01234.pdf",
            abstract_hash="abc123",
            translate_status="success",
            tag_status="success",
        )
        
        paper.tags = []
        paper.user_states = []
        
        data = paper.to_dict()
        
        assert data["id"] == 1
        assert data["arxiv_id"] == "2401.01234"
        assert data["title"] == "Test Paper Title"
        assert data["authors"] == ["Author One", "Author Two"]
        assert data["abstract_en"] == "This is a test abstract."
        assert data["abstract_zh"] == "这是测试摘要。"
        assert data["categories"] == ["cs.AI", "cs.LG"]
        assert data["translate_status"] == "success"
        assert data["tag_status"] == "success"
    
    def test_is_translated_property(self):
        """Test is_translated property."""
        paper_success = Paper(
            id=1, arxiv_id="2401.01234", title="Test",
            authors_json="[]", abstract_en="test",
            categories_json="[]", primary_category="cs.AI",
            published_at=datetime.now(), updated_at=datetime.now(),
            abs_url="", abstract_hash="", translate_status="success"
        )
        paper_success.abstract_zh = "translated"
        
        paper_pending = Paper(
            id=2, arxiv_id="2401.01235", title="Test",
            authors_json="[]", abstract_en="test",
            categories_json="[]", primary_category="cs.AI",
            published_at=datetime.now(), updated_at=datetime.now(),
            abs_url="", abstract_hash="", translate_status="pending"
        )
        
        assert paper_success.is_translated == True
        assert paper_pending.is_translated == False


class TestTagNormalization:
    """Test tag normalization rules."""
    
    def test_duplicate_removal(self):
        """Test that duplicate tags are removed."""
        tagger = KeywordTagger(tags_per_paper=10)
        
        title = "Learning to Learn with Conditional Class Dependencies"
        abstract = "We explore learning to learn approaches."
        categories = ["cs.LG"]
        
        tags, _ = tagger.generate(title, abstract, categories)
        
        # Should not have duplicates
        assert len(tags) == len(set(tags))
    
    def test_tag_length_limit(self):
        """Test that tag count is limited."""
        tagger = KeywordTagger(tags_per_paper=3)
        
        title = "Deep Residual Learning for Image Recognition with Attention Mechanisms"
        abstract = "We develop deep networks with residual connections."
        categories = ["cs.CV", "cs.LG", "cs.AI"]
        
        tags, _ = tagger.generate(title, abstract, categories)
        
        assert len(tags) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
