"""
Database models and schema.
"""
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    Index,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

Base = declarative_base()


class Paper(Base):
    """Paper model."""
    
    __tablename__ = "papers"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    arxiv_id = Column(String(20), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=False)
    authors_json = Column(Text, nullable=False)
    abstract_en = Column(Text, nullable=False)
    abstract_zh = Column(Text, nullable=True)
    categories_json = Column(Text, nullable=False)
    primary_category = Column(String(20), nullable=False)
    published_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    abs_url = Column(String(200), nullable=False)
    pdf_url = Column(String(200), nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    abstract_hash = Column(String(64), nullable=False)
    translate_status = Column(String(20), default="pending")
    tag_status = Column(String(20), default="pending")
    
    # Semantic embedding for similarity search
    embedding_vector = Column(Text, nullable=True)  # JSON array of floats
    
    # View statistics
    view_count = Column(Integer, default=0)
    favorite_count = Column(Integer, default=0)
    
    # Relationships
    user_states = relationship("UserState", back_populates="paper", cascade="all, delete-orphan")
    tags = relationship("PaperTag", back_populates="paper", cascade="all, delete-orphan")
    reading_list_items = relationship("ReadingListItem", back_populates="paper", cascade="all, delete-orphan")
    author_follows = relationship("AuthorFollow", back_populates="paper", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        embedding = None
        if self.embedding_vector:
            try:
                embedding = json.loads(self.embedding_vector)
            except json.JSONDecodeError:
                embedding = None
        
        return {
            "id": self.id,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": json.loads(self.authors_json),
            "abstract_en": self.abstract_en,
            "abstract_zh": self.abstract_zh,
            "categories": json.loads(self.categories_json),
            "primary_category": self.primary_category,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "abs_url": self.abs_url,
            "pdf_url": self.pdf_url,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "translate_status": self.translate_status,
            "tag_status": self.tag_status,
            "view_count": self.view_count,
            "favorite_count": self.favorite_count,
            "tags": [pt.tag.name for pt in self.tags],
        }
    
    @property
    def is_translated(self) -> bool:
        return self.translate_status == "success" and self.abstract_zh is not None
    
    def to_bibtex(self) -> str:
        """Export to BibTeX format."""
        # Extract first author last name
        authors = json.loads(self.authors_json)
        if authors:
            first_author = authors[0].split()[-1] if " " in authors[0] else authors[0]
            author_lastname = "".join(c for c in first_author if c.isalnum())
        else:
            author_lastname = "Unknown"
        
        year = self.published_at.year if self.published_at else datetime.utcnow().year
        
        # Clean title for BibTeX
        title = self.title.replace("{", "").replace("}", "")
        
        # Create BibTeX entry
        bibtex = f"""@article{{{author_lastname}{year},
  title = {{{title}}},
  author = {{{" and ".join(authors)}}},
  journal = {{arXiv preprint arXiv:{self.arxiv_id}}},
  year = {{{year}}},
  url = {{{self.abs_url}}},
  abstract = {{{self.abstract_en[:500] if self.abstract_en else ""}...}}
}}"""
        return bibtex
    
    def to_json(self, include_embedding: bool = False) -> Dict[str, Any]:
        """Export to JSON format."""
        data = self.to_dict()
        if include_embedding and self.embedding_vector:
            data["embedding"] = json.loads(self.embedding_vector)
        return data


class Tag(Base):
    """Tag model for normalized tags."""
    
    __tablename__ = "tags"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    kind = Column(String(20), default="keyword")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Tag statistics
    paper_count = Column(Integer, default=0)
    
    # Relationships
    paper_tags = relationship("PaperTag", back_populates="tag")


class PaperTag(Base):
    """Many-to-many relationship between papers and tags."""
    
    __tablename__ = "paper_tags"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False)
    tag_id = Column(Integer, ForeignKey("tags.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    paper = relationship("Paper", back_populates="tags")
    tag = relationship("Tag", back_populates="paper_tags")
    
    __table_args__ = (Index("idx_paper_tag", "paper_id", "tag_id"),)


class UserState(Base):
    """User state for each paper (read, favorite, hidden)."""
    
    __tablename__ = "user_states"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False, unique=True)
    is_read = Column(Boolean, default=False)
    is_favorite = Column(Boolean, default=False)
    is_hidden = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    read_progress = Column(Float, default=0.0)  # 0.0 to 1.0
    last_read_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    paper = relationship("Paper", back_populates="user_states")


class AuthorFollow(Base):
    """Track authors that users follow."""
    
    __tablename__ = "author_follows"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    author_name = Column(String(200), nullable=False, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    paper = relationship("Paper", back_populates="author_follows")
    
    __table_args__ = (Index("idx_author_follow", "author_name", "paper_id"),)


class ReadingList(Base):
    """User reading lists/collections."""
    
    __tablename__ = "reading_lists"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    items = relationship("ReadingListItem", back_populates="reading_list", cascade="all, delete-orphan")


class ReadingListItem(Base):
    """Items in a reading list."""
    
    __tablename__ = "reading_list_items"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    reading_list_id = Column(Integer, ForeignKey("reading_lists.id"), nullable=False)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False)
    position = Column(Integer, default=0)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    reading_list = relationship("ReadingList", back_populates="items")
    paper = relationship("Paper", back_populates="reading_list_items")
    
    __table_args__ = (Index("idx_reading_list_paper", "reading_list_id", "paper_id"),)


class UserPreference(Base):
    """User preferences for recommendations and settings."""
    
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    preferred_categories = Column(Text, nullable=True)  # JSON array
    preferred_tags = Column(Text, nullable=True)  # JSON array
    ignored_categories = Column(Text, nullable=True)  # JSON array
    email_notifications = Column(Boolean, default=False)
    daily_digest = Column(Boolean, default=False)
    digest_time = Column(String(10), default="08:00")  # HH:MM format
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SearchHistory(Base):
    """Track user search history for recommendations."""
    
    __tablename__ = "search_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(String(500), nullable=False)
    result_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (Index("idx_search_query", "query"),)


# ============ User Authentication Models ============

class User(Base):
    """User model for authentication."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(64), nullable=False)
    display_name = Column(String(100), nullable=True)
    bio = Column(Text, nullable=True)
    avatar_url = Column(String(500), nullable=True)
    api_token = Column(String(64), unique=True, nullable=True)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Team(Base):
    """Team model for collaboration."""
    
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TeamMember(Base):
    """Team membership."""
    
    __tablename__ = "team_members"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(String(20), default="member")  # admin, member, viewer
    joined_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_team_user", "team_id", "user_id"),
        UniqueConstraint("team_id", "user_id", name="uq_team_user"),
    )


class UserSession(Base):
    """User session tokens."""
    
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (Index("idx_user_token", "user_id", "token"),)


# ============ Citation Models ============

class Citation(Base):
    """Paper citation relationships."""
    
    __tablename__ = "citations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    citing_paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False)
    cited_paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False)
    citation_type = Column(String(20), default="citations")  # citations, references, related
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_citing", "citing_paper_id"),
        Index("idx_cited", "cited_paper_id"),
        UniqueConstraint("citing_paper_id", "cited_paper_id", name="uq_citation"),
    )


# ============ Topic Models ============

class Topic(Base):
    """Research topics for clustering."""
    
    __tablename__ = "topics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    keywords_json = Column(Text, nullable=False)  # JSON array of keywords
    description = Column(Text, nullable=True)
    parent_topic_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PaperTopic(Base):
    """Paper-Topic many-to-many relationship."""
    
    __tablename__ = "paper_topics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False)
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=False)
    confidence = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (Index("idx_paper_topic", "paper_id", "topic_id"),)


# ============ Paper Summary Models ============

class PaperSummary(Base):
    """AI-generated paper summaries."""
    
    __tablename__ = "paper_summaries"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False)
    summary_type = Column(String(20), nullable=False)  # brief, comprehensive, key_points, tldr
    summary_text = Column(Text, nullable=False)
    model_used = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_paper_summary_type", "paper_id", "summary_type"),
        UniqueConstraint("paper_id", "summary_type", name="uq_paper_summary"),
    )


# ============ Notification Models ============

class Notification(Base):
    """User notifications."""
    
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=True)
    type = Column(String(20), default="info")  # info, warning, success, error
    link = Column(String(500), nullable=True)
    metadata_json = Column(Text, nullable=True)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserNotificationPreference(Base):
    """User notification preferences."""
    
    __tablename__ = "user_notification_preferences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    email_enabled = Column(Boolean, default=True)
    slack_enabled = Column(Boolean, default=False)
    slack_webhook_url = Column(String(500), nullable=True)
    discord_enabled = Column(Boolean, default=False)
    discord_webhook_url = Column(String(500), nullable=True)
    new_papers = Column(Boolean, default=True)
    followed_authors = Column(Boolean, default=True)
    trending_papers = Column(Boolean, default=False)
    weekly_digest = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============ Annotation Models ============

class PaperAnnotation(Base):
    """User annotations on papers."""
    
    __tablename__ = "paper_annotations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    annotation_type = Column(String(20), default="highlight")  # highlight, note, bookmark
    content = Column(Text, nullable=True)
    page_number = Column(Integer, nullable=True)
    position_json = Column(Text, nullable=True)  # JSON for highlight position
    color = Column(String(20), default="#FFFF00")
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (Index("idx_annotation_paper_user", "paper_id", "user_id"),)


# Database engine and session
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create database engine."""
    global _engine
    if _engine is None:
        config = get_config()
        _engine = create_engine(
            config.database_url,
            connect_args={"check_same_thread": False} if "sqlite" in config.database_url else {},
        )
    return _engine


def get_session():
    """Get database session."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal()


def init_db():
    """Initialize database tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
