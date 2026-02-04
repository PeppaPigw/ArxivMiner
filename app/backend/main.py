"""
Main FastAPI application.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.backend.db.models import init_db
from app.backend.config import get_config
from app.backend.jobs.scheduler import setup_scheduled_jobs, shutdown_scheduler
from app.backend.api.papers import router as papers_router
from app.backend.api.tags import router as tags_router
from app.backend.api.admin import router as admin_router
from app.backend.api.lists import router as lists_router
from app.backend.api.authors import router as authors_router
from app.backend.api.rss import router as rss_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting ArxivMiner...")
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Set up scheduled jobs
    setup_scheduled_jobs()
    logger.info("Scheduled jobs configured")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ArxivMiner...")
    shutdown_scheduler()


# Create FastAPI app
app = FastAPI(
    title="ArxivMiner",
    description="Daily arXiv paper collector with Chinese translation, auto-tagging, recommendations, and reading lists",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(papers_router)
app.include_router(tags_router)
app.include_router(admin_router)
app.include_router(lists_router)
app.include_router(authors_router)
app.include_router(rss_router)


# Mount static frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
async def root():
    """Root endpoint - serve index.html."""
    return {
        "name": "ArxivMiner",
        "version": "2.0.0",
        "status": "running",
        "description": "ArXiv paper collector with translation, tagging, recommendations, and more",
        "endpoints": {
            "papers": "/api/papers",
            "tags": "/api/tags",
            "admin": "/api/admin",
            "lists": "/api/lists",
            "authors": "/api/authors",
            "rss": "/api/rss",
        },
        "features": [
            "Paper search and filtering",
            "Chinese translation via DeepL",
            "Auto-tagging with keywords",
            "Semantic similarity search",
            "Personalized recommendations",
            "Reading lists/collections",
            "Author tracking",
            "RSS feeds",
            "BibTeX/JSON export",
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/info")
async def app_info():
    """Get application information."""
    config = get_config()
    return {
        "name": "ArxivMiner",
        "version": "2.0.0",
        "categories": config.arxiv_categories,
        "embedding_provider": config.embedding_provider,
        "tagger_provider": config.tagger_provider,
    }
