"""
Logging configuration using loguru.
"""
import sys
from loguru import logger
from config.settings import settings

def setup_logging():
    """Configure loguru to log to console and file."""
    logger.remove()  # Remove default handler
    
    # Console handler
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # File handler
    log_file = settings.LOG_DIR / "app.log"
    logger.add(
        str(log_file),
        level=settings.LOG_LEVEL,
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

def get_logger(name: str):
    """Get a logger instance for a specific module."""
    return logger.bind(name=name)
