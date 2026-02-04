"""
Notifications API endpoints.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session

from app.backend.db.models import User, Notification, get_session
from app.backend.services.notifications import get_notification_service
from app.backend.services.auth import get_auth_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


def get_db() -> Session:
    """Get database session."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def get_current_user(
    x_api_token: str = Header(None),
    db: Session = Depends(get_db),
) -> User:
    """Get current user from API token."""
    if not x_api_token:
        raise HTTPException(status_code=401, detail="API token required")
    
    user = db.query(User).filter(User.api_token == x_api_token).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API token")
    
    return user


@router.get("")
def get_notifications(
    unread_only: bool = False,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """Get user notifications."""
    service = get_notification_service()
    return service.get_user_notifications(
        user_id=current_user.id,
        unread_only=unread_only,
        limit=limit,
    )


@router.post("/{notification_id}/read")
def mark_notification_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Mark notification as read."""
    service = get_notification_service()
    return service.mark_notification_read(
        notification_id=notification_id,
        user_id=current_user.id,
    )


@router.post("/read-all")
def mark_all_read(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Mark all notifications as read."""
    service = get_notification_service()
    return service.mark_all_read(user_id=current_user.id)


@router.post("/slack/test")
def test_slack_notification(
    webhook_url: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Test Slack notification."""
    service = get_notification_service()
    return service.send_slack_message(
        webhook_url=webhook_url,
        message="🧪 Test notification from ArxivMiner!",
        channel="#test",
        username="ArxivMiner",
    )


@router.post("/discord/test")
def test_discord_notification(
    webhook_url: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Test Discord notification."""
    service = get_notification_service()
    return service.send_discord_message(
        webhook_url=webhook_url,
        content="🧪 Test notification from ArxivMiner!",
        username="ArxivMiner",
    )


@router.post("/email/test")
def test_email_notification(
    email: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Test email notification."""
    service = get_notification_service()
    
    html = """
    <html>
    <body style="font-family: Arial, sans-serif;">
        <h1 style="color: #4A90D9;">🧪 Test Email</h1>
        <p>This is a test email from ArxivMiner.</p>
        <p>If you received this, your email settings are working correctly!</p>
    </body>
    </html>
    """
    
    return service.send_email(
        to_email=email,
        subject="🧪 Test Email from ArxivMiner",
        html_body=html,
    )
