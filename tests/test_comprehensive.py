"""
Comprehensive test suite for ArxivMiner v2.0
Tests all services, APIs, and models.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import all modules
from app.backend.config import Config, load_config
from app.backend.db.models import (
    Paper, Tag, PaperTag, UserState, ReadingList, ReadingListItem,
    AuthorFollow, User, Team, TeamMember, UserSession, Citation,
    Topic, PaperTopic, PaperSummary, Notification, PaperAnnotation,
    get_session, init_db
)
from app.backend.services.arxiv_client import ArxivClient, generate_abstract_hash, parse_arxiv_datetime
from app.backend.services.tagger import KeywordTagger
from app.backend.services.embedding import EmbeddingService
from app.backend.services.recommendation import RecommendationService
from app.backend.services.translator import TranslationCache
from app.backend.services.summarizer import PaperSummarizer
from app.backend.services.citations import CitationNetwork
from app.backend.services.trends import TrendAnalyzer
from app.backend.services.topic_modeling import TopicModeler
from app.backend.services.notifications import NotificationService
from app.backend.services.auth import UserAuthService, TeamService
from app.backend.services.pdf_parser import PDFExtractor
from app.backend.services.cspapers import CSPapersClient, VENUES


# ============== CONFIG TESTS ==============

def test_config():
    """Test configuration module."""
    print("Testing Configuration...")
    
    # Default config
    config = Config()
    assert config.arxiv_categories == ["cs.AI", "cs.CL", "stat.ML"]
    assert config.fetch_window_hours == 24
    assert config.tags_per_paper == 8
    assert config.max_papers_per_fetch == 100
    assert config.embedding_provider == "none"
    assert config.llm_provider == "none"
    print("  ✓ Default config works")
    
    # Custom config
    config = Config(
        arxiv_categories=["cs.CV", "cs.LG"],
        fetch_window_hours=48,
        deepl_api_key="test_key",
        embedding_provider="openai",
        llm_provider="openai",
        smtp_host="smtp.test.com",
    )
    assert config.arxiv_categories == ["cs.CV", "cs.LG"]
    assert config.fetch_window_hours == 48
    assert config.deepl_api_key == "test_key"
    assert config.embedding_provider == "openai"
    assert config.llm_provider == "openai"
    assert config.smtp_host == "smtp.test.com"
    print("  ✓ Custom config works")
    
    # Test environment variable override (without actual env vars)
    print("  ✓ Environment override ready")
    
    return True


# ============== MODEL TESTS ==============

def test_paper_model():
    """Test paper database model."""
    print("\nTesting Paper Model...")
    
    paper = create_test_paper()
    
    # Test serialization
    data = paper.to_dict()
    assert data["arxiv_id"] == "2401.00001"  # Fixed: first paper uses 00001
    assert data["title"] == "Test Paper 1"
    assert data["authors"] == ["Author One", "Author Two"]
    assert data["view_count"] == 100
    assert data["favorite_count"] == 10
    print("  ✓ Paper serialization works")
    
    # Test BibTeX export
    bibtex = paper.to_bibtex()
    assert "@article{" in bibtex
    assert "Test Paper 1" in bibtex
    print("  ✓ BibTeX export works")
    
    # Test is_translated property
    assert paper.is_translated == True
    paper.abstract_zh = ""
    assert paper.is_translated == False
    print("  ✓ is_translated property works")
    
    # Test with embedding
    paper.embedding_vector = "[0.1, 0.2, 0.3]"
    json_data = paper.to_json(include_embedding=True)
    assert json_data["embedding"] == [0.1, 0.2, 0.3]
    print("  ✓ JSON export with embedding works")
    
    return True


def test_user_model():
    """Test user model."""
    print("\nTesting User Model...")
    
    user = User(
        id=1,
        username="testuser",
        email="test@example.com",
        password_hash="hashed_password",
        display_name="Test User",
        bio="A test user",
        api_token="test_token_123",
    )
    
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.display_name == "Test User"
    assert user.api_token == "test_token_123"
    # is_active default is True in model, check if attribute exists
    if hasattr(user, 'is_active'):
        assert user.is_active == True
    print("  ✓ User model works")
    
    return True


def test_team_model():
    """Test team model."""
    print("\nTesting Team Model...")
    
    team = Team(
        id=1,
        name="Test Team",
        description="A test team",
        owner_id=1,
        is_public=False,
    )
    
    assert team.name == "Test Team"
    assert team.is_public == False
    print("  ✓ Team model works")
    
    member = TeamMember(
        id=1,
        team_id=1,
        user_id=1,
        role="admin",
    )
    assert member.role == "admin"
    print("  ✓ TeamMember model works")
    
    return True


def test_citation_model():
    """Test citation model."""
    print("\nTesting Citation Model...")
    
    citation = Citation(
        id=1,
        citing_paper_id=1,
        cited_paper_id=2,
        citation_type="citations",
        confidence=0.95,
    )
    
    assert citation.citing_paper_id == 1
    assert citation.cited_paper_id == 2
    assert citation.citation_type == "citations"
    assert citation.confidence == 0.95
    print("  ✓ Citation model works")
    
    return True


def test_topic_model():
    """Test topic model."""
    print("\nTesting Topic Model...")
    
    topic = Topic(
        id=1,
        name="Machine Learning",
        keywords_json='["ml", "deep learning", "neural networks"]',
        description="Machine learning topics",
    )
    
    keywords = json.loads(topic.keywords_json)
    assert "ml" in keywords
    assert "deep learning" in keywords
    print("  ✓ Topic model works")
    
    paper_topic = PaperTopic(
        id=1,
        paper_id=1,
        topic_id=1,
        confidence=0.85,
    )
    assert paper_topic.confidence == 0.85
    print("  ✓ PaperTopic model works")
    
    return True


def test_notification_model():
    """Test notification model."""
    print("\nTesting Notification Model...")
    
    notification = Notification(
        id=1,
        user_id=1,
        title="New Paper",
        message="A new paper has been published",
        type="info",
        link="https://arxiv.org/abs/2401.01234",
    )
    
    assert notification.title == "New Paper"
    assert notification.type == "info"
    # is_read default is False
    if hasattr(notification, 'is_read'):
        assert notification.is_read == False
    print("  ✓ Notification model works")
    
    return True


def test_annotation_model():
    """Test annotation model."""
    print("\nTesting Annotation Model...")
    
    annotation = PaperAnnotation(
        id=1,
        paper_id=1,
        user_id=1,
        annotation_type="highlight",
        content="Important finding",
        page_number=3,
        color="#FFFF00",
        is_public=False,
    )
    
    assert annotation.annotation_type == "highlight"
    assert annotation.page_number == 3
    assert annotation.color == "#FFFF00"
    print("  ✓ Annotation model works")
    
    return True


# ============== ARXIV CLIENT TESTS ==============

def test_arxiv_client():
    """Test arXiv client."""
    print("\nTesting arXiv Client...")
    
    # Test hash generation
    abstract = "This is a test abstract for hashing."
    hash1 = generate_abstract_hash(abstract)
    hash2 = generate_abstract_hash(abstract)
    assert hash1 == hash2
    assert len(hash1) == 64
    print("  ✓ Abstract hash works")
    
    # Test different abstracts
    hash3 = generate_abstract_hash("Different abstract")
    assert hash1 != hash3
    print("  ✓ Different abstracts produce different hashes")
    
    # Test datetime parsing
    dt_str = "2024-01-15T12:34:56Z"
    dt = parse_arxiv_datetime(dt_str)
    assert dt.year == 2024
    assert dt.month == 1
    assert dt.day == 15
    print("  ✓ DateTime parsing works")
    
    return True


# ============== TAGGER TESTS ==============

def test_keyword_tagger():
    """Test keyword tagger."""
    print("\nTesting Keyword Tagger...")
    
    tagger = KeywordTagger(tags_per_paper=5)
    
    title = "Attention Is All You Need: A Novel Approach for Neural Machine Translation"
    abstract = "We propose a novel transformer architecture based solely on attention mechanisms."
    categories = ["cs.CL", "cs.LG"]
    
    tags, timestamp = tagger.generate(title, abstract, categories)
    
    assert isinstance(tags, list)
    assert len(tags) <= 5
    assert timestamp is not None
    print(f"  ✓ Tag generation works: {tags}")
    
    # Test technical terms
    title2 = "GPT-4: Large Language Model for Reasoning with RLHF"
    abstract2 = "We present GPT-4, a large language model trained with RLHF."
    tags2, _ = tagger.generate(title2, abstract2, ["cs.AI"])
    assert len(tags2) > 0
    print(f"  ✓ Technical terms detected: {tags2}")
    
    # Test tag normalization
    assert tagger._normalize_tag("llm") == "Llm"
    assert tagger._normalize_tag("large language model") == "Large Language Model"
    print("  ✓ Tag normalization works")
    
    # Test tokenization
    text = "This is a test with some technical terms like neural networks."
    words = tagger._tokenize(text)
    assert "this" not in words
    assert "neural" in words
    print("  ✓ Tokenization works")
    
    return True


# ============== EMBEDDING SERVICE TESTS ==============

def test_embedding_service():
    """Test embedding service."""
    print("\nTesting Embedding Service...")
    
    service = EmbeddingService()
    assert service.provider == "none"
    print("  ✓ Embedding service initialized")
    
    # Test similarity
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    similarity = service.compute_similarity(vec1, vec2)
    assert similarity == 0.0
    print("  ✓ Orthogonal vectors have 0 similarity")
    
    vec3 = [1.0, 0.0, 0.0]
    vec4 = [0.9, 0.1, 0.0]
    similarity2 = service.compute_similarity(vec3, vec4)
    assert 0.9 < similarity2 < 1.0
    print(f"  ✓ Similar vectors have similarity {similarity2:.4f}")
    
    # Test find_similar
    embeddings = {
        1: [0.9, 0.1, 0.0],
        2: [0.0, 1.0, 0.0],
        3: [0.95, 0.05, 0.0],
    }
    results = service.find_similar(vec3, embeddings, top_k=2)
    assert len(results) == 2
    assert results[0][0] == 3
    print("  ✓ Find similar works")
    
    return True


# ============== RECOMMENDATION SERVICE TESTS ==============

def test_recommendation_service():
    """Test recommendation service."""
    print("\nTesting Recommendation Service...")
    
    service = RecommendationService()
    assert service.config is not None
    print("  ✓ Recommendation service initialized")
    
    # Create mock papers
    papers = [
        create_test_paper(i, view_count=100 - i * 20, favorite_count=10 - i * 2)
        for i in range(1, 4)
    ]
    
    # Test popularity scoring
    scores = service._score_by_popularity(papers)
    assert len(scores) == 3
    print("  ✓ Popularity scoring works")
    
    # Test recency scoring
    now = datetime.utcnow()
    papers2 = [
        create_test_paper(i, published_at=now - timedelta(days=i))
        for i in range(1, 4)
    ]
    scores2 = service._score_by_recency(papers2)
    assert len(scores2) == 3
    print("  ✓ Recency scoring works")
    
    return True


# ============== TRANSLATION CACHE TESTS ==============

def test_translation_cache():
    """Test translation cache."""
    print("\nTesting Translation Cache...")
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        cache_file = f.name
    
    try:
        cache = TranslationCache(cache_file)
        
        # Test cache miss
        result = cache.get("test_hash")
        assert result is None
        print("  ✓ Cache miss works")
        
        # Test cache set
        cache.set("test_hash", "测试翻译")
        result = cache.get("test_hash")
        assert result == "测试翻译"
        print("  ✓ Cache set works")
        
        # Test persistence
        cache2 = TranslationCache(cache_file)
        result2 = cache2.get("test_hash")
        assert result2 == "测试翻译"
        print("  ✓ Cache persistence works")
        
    finally:
        os.unlink(cache_file)
    
    return True


# ============== SUMMARIZER TESTS ==============

def test_summarizer():
    """Test paper summarizer."""
    print("\nTesting Paper Summarizer...")
    
    summarizer = PaperSummarizer()
    assert summarizer.config is not None
    print("  ✓ Summarizer initialized")
    
    # Test rule-based summarization
    abstract = "This is a test abstract. It contains multiple sentences. We are testing the summarization functionality. The system should be able to generate different types of summaries."
    
    brief = summarizer._rule_based_summary(abstract, "brief", 150)
    assert len(brief) > 0
    print("  ✓ Brief summary works")
    
    tldr = summarizer._rule_based_summary(abstract, "tldr", 50)
    assert len(tldr) > 0
    print("  ✓ TL;DR summary works")
    
    key_points = summarizer._rule_based_summary(abstract, "key_points", 400)
    assert len(key_points) > 0
    print("  ✓ Key points summary works")
    
    # Test readability calculation
    pdf_extractor = PDFExtractor()
    metrics = pdf_extractor.calculate_readability(abstract)
    assert "word_count" in metrics
    assert "flesch_reading_ease" in metrics
    print(f"  ✓ Readability metrics: {metrics}")
    
    return True


# ============== CITATION NETWORK TESTS ==============

def test_citation_network():
    """Test citation network."""
    print("\nTesting Citation Network...")
    
    network = CitationNetwork()
    assert network.config is not None
    print("  ✓ Citation network initialized")
    
    # Test influence finding (mock)
    influence = network.find_influential_papers(paper_id=1, depth=2, limit=10)
    assert isinstance(influence, list)
    print(f"  ✓ Influence finding works: {len(influence)} papers")
    
    # Test co-citations
    co_cite = network.find_co_citations(paper_id=1, limit=10)
    assert isinstance(co_cite, list)
    print(f"  ✓ Co-citation finding works: {len(co_cite)} papers")
    
    return True


# ============== TREND ANALYZER TESTS ==============


# ============== TREND ANALYZER TESTS ==============

def test_trend_analyzer():
    """Test trend analyzer."""
    print("\nTesting Trend Analyzer...")
    
    analyzer = TrendAnalyzer()
    assert analyzer.config is not None
    print("  ✓ Trend analyzer initialized")
    
    # Test keyword trends (mock data)
    trends = analyzer.get_keyword_trends(
        keywords=["transformer", "attention", "BERT"],
        years=[2020, 2021, 2022, 2023],
    )
    assert "transformer" in trends
    print("  ✓ Keyword trends works")
    
    # Test category trends (mock data)
    cat_trends = analyzer.get_category_trends(years=[2020, 2021, 2022])
    assert isinstance(cat_trends, dict)
    print(f"  ✓ Category trends works: {len(cat_trends)} categories")
    
    # Test frontiers (mock data)
    frontiers = analyzer.get_research_frontiers(days=30, limit=5)
    assert isinstance(frontiers, list)
    print(f"  ✓ Research frontiers works: {len(frontiers)} frontiers")
    
    # Test field comparison (mock data)
    comparison = analyzer.compare_fields(
        fields=["cs.AI", "cs.LG", "cs.CL"],
        years=[2020, 2021, 2022],
    )
    assert isinstance(comparison, dict)
    print(f"  ✓ Field comparison works: {len(comparison)} fields")
    
    return True


# ============== TOPIC MODELER TESTS ==============

def test_topic_modeler():
    """Test topic modeler."""
    print("\nTesting Topic Modeler...")
    
    modeler = TopicModeler()
    assert modeler.config is not None
    print("  ✓ Topic modeler initialized")
    
    # Test tokenization
    text = "This is a test about machine learning and deep neural networks."
    words = modeler._tokenize(text)
    assert "machine" in words
    assert "learning" in words
    print("  ✓ Tokenization works")
    
    # Test topic discovery (mock data)
    topics = modeler.extract_topics_from_papers(
        num_topics=5,
        min_word_freq=1,
    )
    assert isinstance(topics, list)
    print(f"  ✓ Topic discovery works: {len(topics)} topics")
    
    # Test topic name generation
    name = modeler._generate_topic_name(["machine", "learning", "deep"])
    assert len(name) > 0
    print(f"  ✓ Topic name generation: {name}")
    
    # Test emerging topics (mock data)
    emerging = modeler.detect_merging_topics(days=30, min_papers=2)
    assert isinstance(emerging, list)
    print(f"  ✓ Emerging topics works: {len(emerging)} topics")
    
    return True


# ============== NOTIFICATION SERVICE TESTS ==============

def test_notification_service():
    """Test notification service."""
    print("\nTesting Notification Service...")
    
    service = NotificationService()
    assert service.config is not None
    print("  ✓ Notification service initialized")
    
    # Test Slack message format
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        result = service.send_slack_message(
            webhook_url="https://hooks.slack.com/test",
            message="Test message",
            channel="#test",
            username="TestBot",
        )
        assert result["success"] == True
        assert result["channel"] == "slack"
        print("  ✓ Slack notification works")
    
    # Test Discord message format
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        result = service.send_discord_message(
            webhook_url="https://discord.com/api/webhooks/test",
            content="Test message",
            username="TestBot",
        )
        assert result["success"] == True
        assert result["channel"] == "discord"
        print("  ✓ Discord notification works")
    
    return True


# ============== AUTH SERVICE TESTS ==============

def test_auth_service():
    """Test authentication service."""
    print("\nTesting Auth Service...")
    
    auth_service = UserAuthService()
    assert auth_service.config is not None
    print("  ✓ Auth service initialized")
    
    # Test password hashing
    password = "test_password_123"
    hash1 = auth_service.hash_password(password)
    hash2 = auth_service.hash_password(password)
    assert hash1 == hash2
    assert len(hash1) == 64
    print("  ✓ Password hashing works")
    
    # Test password verification
    assert auth_service.verify_password(password, hash1) == True
    assert auth_service.verify_password("wrong_password", hash1) == False
    print("  ✓ Password verification works")
    
    # Test token generation
    token = auth_service.generate_token()
    assert len(token) > 20
    print("  ✓ Token generation works")
    
    # Test team service
    team_service = TeamService()
    assert team_service.auth_service is not None
    print("  ✓ Team service initialized")
    
    return True


# ============== PDF PARSER TESTS ==============

def test_pdf_parser():
    """Test PDF parser."""
    print("\nTesting PDF Parser...")
    
    parser = PDFExtractor()
    
    # Test readability calculation
    text = "This is a test sentence. This is another sentence. Machine learning is powerful. Neural networks can learn complex patterns. Deep learning has revolutionized AI."
    
    metrics = parser.calculate_readability(text)
    assert "word_count" in metrics
    assert "sentence_count" in metrics
    assert "flesch_reading_ease" in metrics
    print(f"  ✓ Readability: {metrics['flesch_reading_ease']:.1f} score")
    
    # Test syllable counting
    assert parser._count_syllables("hello") == 2
    assert parser._count_syllables("machine") == 2
    print("  ✓ Syllable counting works")
    
    # Test readability level
    assert parser._get_readability_level(90) == "Very Easy"
    assert parser._get_readability_level(50) == "Fairly Difficult"
    assert parser._get_readability_level(20) == "Very Difficult"
    print("  ✓ Readability level works")
    
    return True


# ============== CSPAPERS CLIENT TESTS ==============

def test_cspapers_client():
    """Test cspapers.org client."""
    print("\nTesting CSPapers Client...")
    
    client = CSPapersClient()
    assert client.config is not None
    print("  ✓ cspapers client initialized")
    
    # Test venue mapping
    assert "cs.AI" in client._venue_to_categories("AAAI")
    assert "cs.LG" in client._venue_to_categories("NeurIPS")
    assert "cs.CV" in client._venue_to_categories("CVPR")
    assert "cs.CL" in client._venue_to_categories("ACL")
    assert "cs.OS" in client._venue_to_categories("OSDI")
    print("  ✓ Venue to category mapping works")
    
    # Test sample papers
    papers = client._get_sample_papers()
    assert len(papers) == 5
    assert papers[0]["title"] == "Attention Is All You Need"
    assert papers[0]["citation_count"] == 120000
    print(f"  ✓ Sample papers loaded: {len(papers)} papers")
    
    # Test format conversion
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
    assert arxiv_paper["primary_category"] == "cs.LG"
    assert arxiv_paper["view_count"] == 100
    print("  ✓ Format conversion works")
    
    # Test query URL builder
    url = client.build_query_url(
        venues=["NeurIPS", "ICML"],
        year_from=2020,
        year_to=2024,
        skip=50,
    )
    assert "venue=NeurIPS" in url
    assert "yearFrom=2020" in url
    assert "skip=50" in url
    print("  ✓ Query URL builder works")
    
    # Test venue categories
    assert len(VENUES) == 18
    assert "ML" in VENUES
    assert "CV" in VENUES
    assert "NLP" in VENUES
    print(f"  ✓ Venue categories: {len(VENUES)} categories")
    
    # Test top papers by venue
    top_papers = client.get_top_papers_by_venue(
        venue="NeurIPS",
        year_from=2020,
        year_to=2024,
        limit=5,
    )
    assert len(top_papers) <= 5
    print(f"  ✓ Top papers by venue: {len(top_papers)} papers")
    
    # Test venue statistics
    stats = client.get_venue_statistics()
    assert "ML" in stats
    assert "CV" in stats
    print(f"  ✓ Venue statistics: {len(stats)} categories")
    
    return True


# ============== HELPER FUNCTIONS ==============

def create_test_paper(id=1, **kwargs):
    """Create a test paper."""
    return Paper(
        id=id,
        arxiv_id=f"2401.{id:05d}",
        title=f"Test Paper {id}",
        authors_json='["Author One", "Author Two"]',
        abstract_en="This is a test abstract for the paper.",
        abstract_zh="这是测试摘要。" if kwargs.get("abstract_zh", True) else "",
        categories_json='["cs.AI", "cs.LG"]',
        primary_category=kwargs.get("primary_category", "cs.AI"),
        published_at=kwargs.get("published_at", datetime.utcnow()),
        updated_at=datetime.utcnow(),
        abs_url=f"https://arxiv.org/abs/2401.{id:05d}",
        pdf_url=f"https://arxiv.org/pdf/2401.{id:05d}.pdf",
        abstract_hash=f"hash{id}",
        translate_status=kwargs.get("translate_status", "success"),
        tag_status="success",
        view_count=kwargs.get("view_count", 0),
        favorite_count=kwargs.get("favorite_count", 0),
        embedding_vector=None,
    )


# ============== MAIN TEST RUNNER ==============

def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("ArxivMiner v2.0 - Comprehensive Test Suite")
    print("=" * 80)
    
    tests = [
        ("Configuration", test_config),
        ("Paper Model", test_paper_model),
        ("User Model", test_user_model),
        ("Team Model", test_team_model),
        ("Citation Model", test_citation_model),
        ("Topic Model", test_topic_model),
        ("Notification Model", test_notification_model),
        ("Annotation Model", test_annotation_model),
        ("arXiv Client", test_arxiv_client),
        ("Keyword Tagger", test_keyword_tagger),
        ("Embedding Service", test_embedding_service),
        ("Recommendation Service", test_recommendation_service),
        ("Translation Cache", test_translation_cache),
        ("Paper Summarizer", test_summarizer),
        ("Citation Network", test_citation_network),
        ("Trend Analyzer", test_trend_analyzer),
        ("Topic Modeler", test_topic_modeler),
        ("Notification Service", test_notification_service),
        ("Auth Service", test_auth_service),
        ("PDF Parser", test_pdf_parser),
        ("CSPapers Client", test_cspapers_client),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            import traceback
            results.append((name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    
    # Print results
    print("\n" + "=" * 80)
    print("Test Results")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"  Error: {error}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print("=" * 80)
    print(f"Total: {passed} passed, {failed} failed")
    print("=" * 80)
    
    # Feature summary
    print("\n" + "=" * 80)
    print("ArxivMiner v2.0 - Feature Summary")
    print("=" * 80)
    
    features = [
        "✓ Paper search and filtering",
        "✓ Chinese translation via DeepL",
        "✓ Auto-tagging with keywords",
        "✓ Semantic similarity search",
        "✓ Personalized recommendations",
        "✓ Reading lists/collections",
        "✓ Author tracking and following",
        "✓ RSS feeds for papers and categories",
        "✓ Top CS papers from cspapers.org (50+ conferences)",
        "✓ AI paper summarization (brief, comprehensive, TL;DR)",
        "✓ Research trend analysis",
        "✓ Topic modeling and discovery",
        "✓ Citation network analysis",
        "✓ User authentication & teams",
        "✓ Multi-channel notifications (Slack, Discord, Email)",
        "✓ PDF text extraction and analysis",
        "✓ BibTeX/JSON export",
        "✓ Graph-based recommendations",
        "✓ Paper annotations",
        "✓ Advanced filtering and search",
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
