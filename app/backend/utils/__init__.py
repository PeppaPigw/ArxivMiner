"""
Utility functions.
"""
import logging
import os
from datetime import datetime
from typing import Optional

# Setup logging
def setup_logging(name: str = "arxivminer", level: str = "INFO"):
    """Set up logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
    )
    
    return logging.getLogger(name)


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def now() -> datetime:
    """Get current UTC datetime."""
    return datetime.utcnow()


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def chunk_list(lst: list, size: int):
    """Split list into chunks."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]
