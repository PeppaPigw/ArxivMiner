"""
Main FastAPI application.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .db.models import init_db
from .config import get_config
from .jobs.scheduler import setup_scheduled_jobs, shutdown_scheduler
from .api.papers import router as papers_router
from .api.tags import router as tags_router
from .api.admin import router as admin_router

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
    description="Daily arXiv paper collector with Chinese translation and auto-tagging",
    version="1.0.0",
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


# Mount static frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
async def root():
    """Root endpoint - serve index.html."""
    return {
        "name": "ArxivMiner",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "papers": "/api/papers",
            "tags": "/api/tags",
            "admin": "/api/admin",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
