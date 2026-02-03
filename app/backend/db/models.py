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
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
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
    authors_json = Column(Text, nullable=False)  # JSON array of author names
    abstract_en = Column(Text, nullable=False)
    abstract_zh = Column(Text, nullable=True)
    categories_json = Column(Text, nullable=False)  # JSON array of categories
    primary_category = Column(String(20), nullable=False)
    published_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    abs_url = Column(String(200), nullable=False)
    pdf_url = Column(String(200), nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    abstract_hash = Column(String(64), nullable=False)  # SHA256 of abstract_en
    translate_status = Column(
        String(20), default="pending"
    )  # pending, success, failed
    tag_status = Column(String(20), default="pending")  # pending, success, failed
    
    # User state relationships
    user_states = relationship("UserState", back_populates="paper", cascade="all, delete-orphan")
    tags = relationship("PaperTag", back_populates="paper", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
            "tags": [pt.tag.name for pt in self.tags],
        }
    
    @property
    def is_translated(self) -> bool:
        return self.translate_status == "success" and self.abstract_zh is not None


class Tag(Base):
    """Tag model for normalized tags."""
    
    __tablename__ = "tags"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    kind = Column(
        String(20), default="keyword"
    )  # category, keyword, llm
    created_at = Column(DateTime, default=datetime.utcnow)


class PaperTag(Base):
    """Many-to-many relationship between papers and tags."""
    
    __tablename__ = "paper_tags"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False)
    tag_id = Column(Integer, ForeignKey("tags.id"), nullable=False)
    
    paper = relationship("Paper", back_populates="tags")
    tag = relationship("Tag")


class UserState(Base):
    """User state for each paper (read, favorite, hidden)."""
    
    __tablename__ = "user_states"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False, unique=True)
    is_read = Column(Boolean, default=False)
    is_favorite = Column(Boolean, default=False)
    is_hidden = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    paper = relationship("Paper", back_populates="user_states")


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
