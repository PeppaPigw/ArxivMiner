"""
Authentication API endpoints.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session

from app.backend.db.models import User, get_session
from app.backend.services.auth import get_auth_service, get_team_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])


def get_db() -> Session:
    """Get database session."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


@router.post("/register")
def register(
    username: str,
    email: str,
    password: str,
    display_name: str = None,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Register a new user."""
    auth_service = get_auth_service()
    return auth_service.create_user(
        username=username,
        email=email,
        password=password,
        display_name=display_name,
    )


@router.post("/login")
def login(
    username: str,
    password: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Login and get session token."""
    auth_service = get_auth_service()
    result = auth_service.authenticate(username, password)
    
    if not result:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return result


@router.post("/logout")
def logout(
    x_session_token: str = Header(None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Logout and invalidate session."""
    auth_service = get_auth_service()
    
    if not x_session_token:
        return {"success": True}
    
    success = auth_service.logout(x_session_token)
    return {"success": success}


@router.post("/refresh")
def refresh_token(
    x_session_token: str = Header(None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Refresh session token."""
    auth_service = get_auth_service()
    
    if not x_session_token:
        raise HTTPException(status_code=401, detail="Session token required")
    
    result = auth_service.refresh_token(x_session_token)
    
    if not result:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    return result


@router.get("/me")
def get_current_user(
    x_api_token: str = Header(None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get current user info."""
    if not x_api_token:
        raise HTTPException(status_code=401, detail="API token required")
    
    auth_service = get_auth_service()
    user = auth_service.verify_session(x_api_token)
    
    if not user:
        # Try API token
        user = db.query(User).filter(User.api_token == x_api_token).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "display_name": user.display_name,
        "bio": user.bio,
        "last_login": user.last_login.isoformat() if user.last_login else None,
    }


@router.put("/profile")
def update_profile(
    display_name: str = None,
    email: str = None,
    bio: str = None,
    x_api_token: str = Header(None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Update user profile."""
    if not x_api_token:
        raise HTTPException(status_code=401, detail="API token required")
    
    auth_service = get_auth_service()
    user = auth_service.verify_session(x_api_token)
    
    if not user:
        user = db.query(User).filter(User.api_token == x_api_token).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return auth_service.update_profile(
        user_id=user.id,
        display_name=display_name,
        email=email,
        bio=bio,
    )


@router.post("/change-password")
def change_password(
    old_password: str,
    new_password: str,
    x_api_token: str = Header(None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Change password."""
    if not x_api_token:
        raise HTTPException(status_code=401, detail="API token required")
    
    auth_service = get_auth_service()
    user = auth_service.verify_session(x_api_token)
    
    if not user:
        user = db.query(User).filter(User.api_token == x_api_token).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return auth_service.change_password(
        user_id=user.id,
        old_password=old_password,
        new_password=new_password,
    )


@router.post("/regenerate-token")
def regenerate_api_token(
    x_api_token: str = Header(None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Regenerate API token."""
    if not x_api_token:
        raise HTTPException(status_code=401, detail="API token required")
    
    auth_service = get_auth_service()
    user = auth_service.verify_session(x_api_token)
    
    if not user:
        user = db.query(User).filter(User.api_token == x_api_token).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return auth_service.regenerate_api_token(user_id=user.id)


# ============ Teams ============

@router.post("/teams")
def create_team(
    name: str,
    description: str = None,
    x_api_token: str = Header(None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Create a new team."""
    if not x_api_token:
        raise HTTPException(status_code=401, detail="API token required")
    
    auth_service = get_auth_service()
    user = auth_service.verify_session(x_api_token)
    
    if not user:
        user = db.query(User).filter(User.api_token == x_api_token).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    team_service = get_team_service()
    return team_service.create_team(
        name=name,
        description=description,
        owner_id=user.id,
    )


@router.post("/teams/{team_id}/members")
def add_team_member(
    team_id: int,
    user_id: int,
    role: str = "member",
    x_api_token: str = Header(None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Add member to team."""
    if not x_api_token:
        raise HTTPException(status_code=401, detail="API token required")
    
    auth_service = get_auth_service()
    user = auth_service.verify_session(x_api_token)
    
    if not user:
        user = db.query(User).filter(User.api_token == x_api_token).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    team_service = get_team_service()
    return team_service.add_member(
        team_id=team_id,
        user_id=user_id,
        role=role,
    )


@router.get("/teams/{team_id}/members")
def get_team_members(
    team_id: int,
    x_api_token: str = Header(None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get team members."""
    if not x_api_token:
        raise HTTPException(status_code=401, detail="API token required")
    
    auth_service = get_auth_service()
    user = auth_service.verify_session(x_api_token)
    
    if not user:
        user = db.query(User).filter(User.api_token == x_api_token).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    team_service = get_team_service()
    return {
        "team_id": team_id,
        "members": team_service.get_team_members(team_id=team_id),
    }
