"""
Configuration management module.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """Application configuration."""
    
    # arXiv Settings
    arxiv_categories: List[str] = ["cs.AI", "cs.CL", "stat.ML"]
    fetch_window_hours: int = 24
    fetch_mode: str = "published"
    max_papers_per_fetch: int = 100
    
    # Database
    database_url: str = "sqlite:///./data/arxivminer.db"
    
    # DeepL Translation
    deepl_api_key: str = ""
    deepl_api_url: str = "https://api-free.deepl.com/v2/translate"
    
    # Tagging
    tags_per_paper: int = 8
    tagger_provider: str = "keyword"
    
    # Embeddings for semantic search
    embedding_provider: str = "none"  # none, openai, sentence-transformers
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_api_key: str = ""
    
    # LLM for summarization
    llm_provider: str = "none"  # none, openai, ollama
    llm_model: str = "gpt-3.5-turbo"
    openai_api_key: str = ""
    ollama_url: str = "http://localhost:11434"
    
    # App Settings
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    admin_token: str = "admin_secret_token"
    secret_key: str = "your-secret-key-change-in-production"
    
    # User Preferences
    default_page_size: int = 20
    max_page_size: int = 100
    
    # Logging
    log_level: str = "INFO"
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    
    # SMTP for email notifications
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_email: str = "noreply@arxivminer.com"
    
    # Slack/Discord webhooks (comma-separated for multiple)
    slack_webhooks: str = ""
    discord_webhooks: str = ""
    
    class Config:
        arbitrary_types_allowed = True


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.environ.get(
            "CONFIG_PATH", 
            str(Path(__file__).parent.parent / "config.yaml")
        )
    
    config_data: Dict[str, Any] = {}
    
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
    
    # Override with environment variables if set
    if os.environ.get("DATABASE_URL"):
        config_data["database_url"] = os.environ["DATABASE_URL"]
    if os.environ.get("DEEPL_API_KEY"):
        config_data["deepl_api_key"] = os.environ["DEEPL_API_KEY"]
    if os.environ.get("ADMIN_TOKEN"):
        config_data["admin_token"] = os.environ["ADMIN_TOKEN"]
    if os.environ.get("OPENAI_API_KEY"):
        config_data["embedding_api_key"] = os.environ["OPENAI_API_KEY"]
    
    return Config(**config_data)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config():
    """Reset config (useful for testing)."""
    global _config
    _config = None
