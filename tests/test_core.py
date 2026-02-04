"""
Unit tests for ArxivMiner - Enhanced version with new features.
"""
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.backend.config import Config, load_config
from app.backend.db.models import (
    Paper, Tag, PaperTag, UserState, ReadingList, ReadingListItem,
    AuthorFollow, init_db, get_session
)
from app.backend.services.arxiv_client import ArxivClient, generate_abstract_hash, parse_arxiv_datetime
from app.backend.services.tagger import KeywordTagger
from app.backend.services.embedding import EmbeddingService
from app.backend.services.recommendation import RecommendationService


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
        assert config.max_papers_per_fetch == 100
        assert config.embedding_provider == "none"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = Config(
            arxiv_categories=["cs.CV", "cs.LG"],
            fetch_window_hours=48,
            deepl_api_key="test_key",
            database_url="sqlite:///test.db",
            embedding_provider="openai",
        )
        
        assert config.arxiv_categories == ["cs.CV", "cs.LG"]
        assert config.fetch_window_hours == 48
        assert config.deepl_api_key == "test_key"
        assert config.embedding_provider == "openai"


class TestArxivClient:
    """Test arXiv client functions."""
    
    def test_generate_abstract_hash(self):
        """Test abstract hash generation."""
        abstract = "This is a test abstract for hashing."
        hash1 = generate_abstract_hash(abstract)
        hash2 = generate_abstract_hash(abstract)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 hex characters
    
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
    
    def test_paper_to_bibtex(self):
        """Test BibTeX export."""
        paper = Paper(
            id=1,
            arxiv_id="2401.01234",
            title="Attention Is All You Need",
            authors_json='["Ashish Vaswani", "Noam Shazeer"]',
            abstract_en="This is a test abstract.",
            abstract_zh="这是测试摘要。",
            categories_json='["cs.CL", "cs.LG"]',
            primary_category="cs.CL",
            published_at=datetime(2024, 1, 15, 12, 0, 0),
            updated_at=datetime(2024, 1, 15, 12, 0, 0),
            abs_url="https://arxiv.org/abs/2401.01234",
            pdf_url="https://arxiv.org/pdf/2401.01234.pdf",
            abstract_hash="abc123",
            translate_status="success",
            tag_status="success",
        )
        
        bibtex = paper.to_bibtex()
        
        assert "@article{" in bibtex
        assert "Attention Is All You Need" in bibtex
        assert "Ashish Vaswani" in bibtex
        assert "arxiv:2401.01234" in bibtex
        assert "cs.CL" in bibtex
    
    def test_paper_to_json(self):
        """Test JSON export."""
        paper = Paper(
            id=1,
            arxiv_id="2401.01234",
            title="Test Paper",
            authors_json='["Author One"]',
            abstract_en="Test abstract.",
            abstract_zh="测试摘要。",
            categories_json='["cs.AI"]',
            primary_category="cs.AI",
            published_at=datetime(2024, 1, 15, 12, 0, 0),
            updated_at=datetime(2024, 1, 15, 12, 0, 0),
            abs_url="https://arxiv.org/abs/2401.01234",
            pdf_url="https://arxiv.org/pdf/2401.01234.pdf",
            abstract_hash="abc123",
            translate_status="success",
            tag_status="success",
            view_count=100,
            favorite_count=10,
        )
        
        json_data = paper.to_json()
        
        assert json_data["arxiv_id"] == "2401.01234"
        assert json_data["view_count"] == 100
        assert json_data["favorite_count"] == 10
    
    def test_paper_with_embedding(self):
        """Test paper with embedding vector."""
        paper = Paper(
            id=1,
            arxiv_id="2401.01234",
            title="Test Paper",
            authors_json='["Author One"]',
            abstract_en="Test abstract.",
            abstract_zh="",
            categories_json='["cs.AI"]',
            primary_category="cs.AI",
            published_at=datetime.now(),
            updated_at=datetime.now(),
            abs_url="https://arxiv.org/abs/2401.01234",
            pdf_url="https://arxiv.org/pdf/2401.01234.pdf",
            abstract_hash="abc123",
            translate_status="success",
            tag_status="success",
            embedding_vector="[0.1, 0.2, 0.3, 0.4]",
        )
        
        data = paper.to_dict()
        
        # Embedding should be parsed from JSON
        assert data.get("embedding") is None  # Not in to_dict by default
        
        json_data = paper.to_json(include_embedding=True)
        assert json_data["embedding"] == [0.1, 0.2, 0.3, 0.4]


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


class TestEmbeddingService:
    """Test embedding service."""
    
    def test_embedding_service_init(self):
        """Test embedding service initialization."""
        service = EmbeddingService()
        assert service.provider == "none"  # Default
        
    def test_compute_similarity(self):
        """Test cosine similarity computation."""
        service = EmbeddingService()
        
        # Identical vectors
        vec = [1.0, 0.0, 0.0]
        similarity = service.compute_similarity(vec, vec)
        assert similarity == 1.0
        
        # Orthogonal vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = service.compute_similarity(vec1, vec2)
        assert similarity == 0.0
        
        # Similar vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.9, 0.1, 0.0]
        similarity = service.compute_similarity(vec1, vec2)
        assert similarity > 0.9
        assert similarity < 1.0
    
    def test_compute_similarity_empty_vectors(self):
        """Test similarity with empty vectors."""
        service = EmbeddingService()
        
        similarity = service.compute_similarity([], [])
        assert similarity == 0.0
        
        similarity = service.compute_similarity([1.0], [])
        assert similarity == 0.0
    
    def test_find_similar(self):
        """Test finding similar papers."""
        service = EmbeddingService()
        
        query_embedding = [1.0, 0.0, 0.0]
        embeddings = {
            1: [0.9, 0.1, 0.0],
            2: [0.0, 1.0, 0.0],
            3: [0.95, 0.05, 0.0],
        }
        
        results = service.find_similar(query_embedding, embeddings, top_k=2)
        
        # Should return paper 3 (most similar) and paper 1
        assert len(results) == 2
        assert results[0][0] == 3  # Most similar
        assert results[0][1] > results[1][1]
    
    def test_find_similar_with_threshold(self):
        """Test finding similar papers with threshold."""
        service = EmbeddingService()
        
        query_embedding = [1.0, 0.0, 0.0]
        embeddings = {
            1: [0.9, 0.1, 0.0],  # similarity ~0.99
            2: [0.0, 1.0, 0.0],  # similarity 0.0
            3: [0.95, 0.05, 0.0],  # similarity ~0.999
        }
        
        results = service.find_similar(query_embedding, embeddings, top_k=10, threshold=0.5)
        
        # Should only return papers with similarity > 0.5
        assert len(results) == 2
        assert all(r[1] > 0.5 for r in results)


class TestReadingListModel:
    """Test reading list model."""
    
    def test_reading_list_creation(self):
        """Test reading list creation."""
        reading_list = ReadingList(
            name="My Reading List",
            description="Papers I want to read",
            is_public=False,
        )
        
        assert reading_list.name == "My Reading List"
        assert reading_list.description == "Papers I want to read"
        assert reading_list.is_public == False
    
    def test_reading_list_item(self):
        """Test reading list item."""
        item = ReadingListItem(
            reading_list_id=1,
            paper_id=1,
            position=0,
            notes="Important paper",
        )
        
        assert item.reading_list_id == 1
        assert item.paper_id == 1
        assert item.position == 0
        assert item.notes == "Important paper"


class TestAuthorFollowModel:
    """Test author follow model."""
    
    def test_author_follow_creation(self):
        """Test author follow creation."""
        follow = AuthorFollow(
            author_name="Yann LeCun",
            paper_id=1,
        )
        
        assert follow.author_name == "Yann LeCun"
        assert follow.paper_id == 1


class TestUserStateModel:
    """Test user state model."""
    
    def test_user_state_defaults(self):
        """Test user state default values."""
        state = UserState(paper_id=1)
        
        assert state.is_read == False
        assert state.is_favorite == False
        assert state.is_hidden == False
        assert state.notes is None
        assert state.read_progress == 0.0
    
    def test_user_state_with_values(self):
        """Test user state with values."""
        state = UserState(
            paper_id=1,
            is_read=True,
            is_favorite=True,
            is_hidden=False,
            notes="Great paper!",
            read_progress=0.75,
        )
        
        assert state.is_read == True
        assert state.is_favorite == True
        assert state.notes == "Great paper!"
        assert state.read_progress == 0.75


class TestRecommendationService:
    """Test recommendation service."""
    
    def test_recommendation_service_init(self):
        """Test recommendation service initialization."""
        service = RecommendationService()
        assert service.config is not None
    
    def test_score_by_popularity(self):
        """Test popularity scoring."""
        service = RecommendationService()
        
        papers = [
            create_mock_paper(1, view_count=100, favorite_count=10),
            create_mock_paper(2, view_count=50, favorite_count=5),
            create_mock_paper(3, view_count=200, favorite_count=20),
        ]
        
        scores = service._score_by_popularity(papers)
        
        # Paper 3 should have highest score
        assert len(scores) == 3
        paper_ids = [s[0].id for s in scores]
        assert 3 in paper_ids
    
    def test_score_by_recency(self):
        """Test recency scoring."""
        service = RecommendationService()
        
        now = datetime.utcnow()
        papers = [
            create_mock_paper(1, published_at=now - timedelta(days=1)),
            create_mock_paper(2, published_at=now - timedelta(days=7)),
            create_mock_paper(3, published_at=now),
        ]
        
        scores = service._score_by_recency(papers)
        
        # Paper 3 (most recent) should have highest score
        assert len(scores) == 3
        scores_dict = {s[0].id: s[1] for s in scores}
        assert scores_dict[3] > scores_dict[1] > scores_dict[2]


def create_mock_paper(id, **kwargs):
    """Helper to create mock paper."""
    paper = Paper(
        id=id,
        arxiv_id=f"2401.{id:05d}",
        title=f"Test Paper {id}",
        authors_json='["Author"]',
        abstract_en="Test abstract.",
        abstract_zh="",
        categories_json='["cs.AI"]',
        primary_category="cs.AI",
        published_at=kwargs.get("published_at", datetime.utcnow()),
        updated_at=datetime.utcnow(),
        abs_url=f"https://arxiv.org/abs/2401.{id:05d}",
        pdf_url=f"https://arxiv.org/pdf/2401.{id:05d}.pdf",
        abstract_hash=f"hash{id}",
        translate_status="success",
        tag_status="success",
        view_count=kwargs.get("view_count", 0),
        favorite_count=kwargs.get("favorite_count", 0),
    )
    return paper


class TestCSPapersClient:
    """Test cspapers.org client."""
    
    def test_cspapers_client_init(self):
        """Test cspapers client initialization."""
        from app.backend.services.cspapers import CSPapersClient
        
        client = CSPapersClient()
        assert client.config is not None
        assert client.tagger is not None
        print("  ✓ cspapers client initialized")
    
    def test_venue_mapping(self):
        """Test venue to category mapping."""
        from app.backend.services.cspapers import CSPapersClient
        
        client = CSPapersClient()
        
        # Test common venues
        assert "cs.AI" in client._venue_to_categories("AAAI")
        assert "cs.AI" in client._venue_to_categories("IJCAI")
        assert "cs.LG" in client._venue_to_categories("ICLR")
        assert "cs.LG" in client._venue_to_categories("NeurIPS")
        assert "cs.CV" in client._venue_to_categories("CVPR")
        assert "cs.CV" in client._venue_to_categories("ICCV")
        assert "cs.CL" in client._venue_to_categories("ACL")
        assert "cs.CL" in client._venue_to_categories("EMNLP")
        assert "cs.OS" in client._venue_to_categories("OSDI")
        assert "cs.OS" in client._venue_to_categories("SOSP")
        assert "cs.PL" in client._venue_to_categories("PLDI")
        assert "cs.DB" in client._venue_to_categories("SIGMOD")
        assert "cs.CR" in client._venue_to_categories("CRYPTO")
        assert "cs.HC" in client._venue_to_categories("CHI")
        print("  ✓ Venue to category mapping works")
    
    def test_sample_papers(self):
        """Test sample paper collection."""
        from app.backend.services.cspapers import CSPapersClient
        
        client = CSPapersClient()
        papers = client._get_sample_papers()
        
        assert len(papers) == 5
        assert papers[0]["title"] == "Attention Is All You Need"
        assert papers[0]["venue"] == "NeurIPS"
        assert papers[0]["citation_count"] == 120000
        assert "Ashish Vaswani" in papers[0]["authors"]
        print(f"  ✓ Sample papers loaded: {len(papers)} papers")
    
    def test_convert_to_arxiv_format(self):
        """Test converting cspapers format to arxivMiner format."""
        from app.backend.services.cspapers import CSPapersClient
        
        client = CSPapersClient()
        
        paper = {
            "title": "Test Paper",
            "authors": ["Author One", "Author Two"],
            "year": 2023,
            "venue": "NeurIPS",
            "citation_count": 100,
            "url": "https://arxiv.org/abs/1234.56789",
            "abstract": "Test abstract",
            "paper_id": "test-123",
        }
        
        arxiv_paper = client.convert_to_arxiv_format(paper)
        
        assert arxiv_paper["title"] == "Test Paper"
        assert arxiv_paper["authors_json"] == '["Author One", "Author Two"]'
        assert arxiv_paper["primary_category"] == "cs.LG"
        assert arxiv_paper["view_count"] == 100
        assert "cs.LG" in arxiv_paper["categories_json"]
        print("  ✓ Format conversion works")
    
    def test_build_query_url(self):
        """Test building cspapers.org query URL."""
        from app.backend.services.cspapers import CSPapersClient
        
        client = CSPapersClient()
        
        url = client.build_query_url(
            venues=["NeurIPS", "ICML"],
            year_from=2020,
            year_to=2024,
            order_by="score",
            skip=50,
        )
        
        assert "venue=NeurIPS" in url
        assert "venue=ICML" in url
        assert "yearFrom=2020" in url
        assert "yearTo=2024" in url
        assert "skip=50" in url
        print("  ✓ Query URL builder works")
    
    def test_venue_categories(self):
        """Test venue categories structure."""
        from app.backend.services.cspapers import VENUES
        
        # Check categories exist
        assert "AI" in VENUES
        assert "ML" in VENUES
        assert "CV" in VENUES
        assert "NLP" in VENUES
        
        # Check venues in categories
        assert "NeurIPS" in VENUES["ML"]
        assert "CVPR" in VENUES["CV"]
        assert "ACL" in VENUES["NLP"]
        
        print(f"  ✓ Venue categories loaded: {len(VENUES)} categories")
    
    def test_get_top_papers_by_venue(self):
        """Test getting top papers for a venue."""
        from app.backend.services.cspapers import CSPapersClient
        
        client = CSPapersClient()
        papers = client.get_top_papers_by_venue(
            venue="NeurIPS",
            year_from=2020,
            year_to=2024,
            limit=5,
        )
        
        assert len(papers) <= 5
        # All papers should be from NeurIPS
        for paper in papers:
            if paper.get("venue"):
                assert "NeurIPS" in paper["venue"] or paper["venue"] == "NeurIPS"
        
        print(f"  ✓ Top papers by venue works: {len(papers)} papers")
    
    def test_get_venue_statistics(self):
        """Test getting venue statistics."""
        from app.backend.services.cspapers import CSPapersClient
        
        client = CSPapersClient()
        stats = client.get_venue_statistics()
        
        assert isinstance(stats, dict)
        assert "ML" in stats
        assert "CV" in stats
        
        ml_stats = stats["ML"]
        assert "venues" in ml_stats
        assert "paper_count" in ml_stats
        assert "total_citations" in ml_stats
        
        print(f"  ✓ Venue statistics works: {len(stats)} categories")


class TestCSPapersAPI:
    """Test cspapers API endpoints."""
    
    def test_cspapers_router_exists(self):
        """Test that cspapers router can be imported."""
        from app.backend.api.cspapers import router
        assert router is not None
        assert router.prefix == "/api/cspapers"
        print("  ✓ cspapers router exists")
    
    def test_cspapers_endpoints(self):
        """Test cspapers endpoint definitions."""
        from app.backend.api.cspapers import router
        
        routes = [r.path for r in router.routes]
        
        assert "/api/cspapers" in routes
        assert "/api/cspapers/venues" in routes
        assert "/api/cspapers/venue/{venue}" in routes
        assert "/api/cspapers/category/{category}" in routes
        assert "/api/cspapers/import" in routes
        assert "/api/cspapers/trending" in routes
        assert "/api/cspapers/statistics" in routes
        assert "/api/cspapers/search" in routes
        assert "/api/cspapers/leaderboard" in routes
        
        print(f"  ✓ cspapers endpoints loaded: {len(routes)} routes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
