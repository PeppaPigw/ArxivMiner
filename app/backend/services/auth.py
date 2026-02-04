"""
User Authentication Service.
Multi-user support with teams and collaboration.
"""
import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.backend.config import get_config
from app.backend.db.models import (
    User, Team, TeamMember, UserSession, Paper, ReadingList,
    get_session
)
from app.backend.services.tagger import get_tagger

logger = logging.getLogger(__name__)


class UserAuthService:
    """User authentication and authorization service."""
    
    def __init__(self):
        """Initialize auth service."""
        self.config = get_config()
        self.token_expiry_hours = 24
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt."""
        salt = self.config.secret_key[:16]
        combined = salt + password + self.config.secret_key
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        return self.hash_password(password) == password_hash
    
    def generate_token(self) -> str:
        """Generate secure API token."""
        return secrets.token_urlsafe(32)
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new user."""
        db = get_session()
        try:
            # Check if username or email exists
            existing = db.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing:
                return {"success": False, "error": "Username or email already exists"}
            
            user = User(
                username=username,
                email=email,
                password_hash=self.hash_password(password),
                display_name=display_name or username,
                api_token=self.generate_token(),
            )
            
            db.add(user)
            db.commit()
            db.refresh(user)
            
            return {
                "success": True,
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "display_name": user.display_name,
                    "api_token": user.api_token,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                },
            }
        
        finally:
            db.close()
    
    def authenticate(
        self,
        username: str,
        password: str,
    ) -> Optional[Dict[str, Any]]:
        """Authenticate user and return token."""
        db = get_session()
        try:
            user = db.query(User).filter(User.username == username).first()
            
            if not user or not self.verify_password(password, user.password_hash):
                return None
            
            # Generate new session token
            session = UserSession(
                user_id=user.id,
                token=self.generate_token(),
                expires_at=datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            )
            db.add(session)
            
            user.last_login = datetime.utcnow()
            db.commit()
            
            return {
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "display_name": user.display_name,
                },
                "session_token": session.token,
                "expires_at": session.expires_at.isoformat(),
            }
        
        finally:
            db.close()
    
    def verify_session(self, token: str) -> Optional[User]:
        """Verify session token and return user."""
        db = get_session()
        try:
            session = db.query(UserSession).filter(
                UserSession.token == token,
                UserSession.expires_at > datetime.utcnow(),
            ).first()
            
            if not session:
                return None
            
            user = db.query(User).filter(User.id == session.user_id).first()
            return user
        
        finally:
            db.close()
    
    def refresh_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Refresh session token."""
        db = get_session()
        try:
            session = db.query(UserSession).filter(
                UserSession.token == token,
            ).first()
            
            if not session:
                return None
            
            # Generate new token
            session.token = self.generate_token()
            session.expires_at = datetime.utcnow() + timedelta(hours=self.token_expiry_hours)
            db.commit()
            
            return {
                "session_token": session.token,
                "expires_at": session.expires_at.isoformat(),
            }
        
        finally:
            db.close()
    
    def logout(self, token: str) -> bool:
        """Invalidate session token."""
        db = get_session()
        try:
            session = db.query(UserSession).filter(
                UserSession.token == token,
            ).first()
            
            if session:
                db.delete(session)
                db.commit()
                return True
            
            return False
        
        finally:
            db.close()
    
    def update_profile(
        self,
        user_id: int,
        display_name: Optional[str] = None,
        email: Optional[str] = None,
        bio: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update user profile."""
        db = get_session()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                return {"success": False, "error": "User not found"}
            
            if display_name:
                user.display_name = display_name
            if email:
                user.email = email
            if bio is not None:
                user.bio = bio
            
            user.updated_at = datetime.utcnow()
            db.commit()
            
            return {
                "success": True,
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "display_name": user.display_name,
                    "bio": user.bio,
                },
            }
        
        finally:
            db.close()
    
    def change_password(
        self,
        user_id: int,
        old_password: str,
        new_password: str,
    ) -> Dict[str, Any]:
        """Change user password."""
        db = get_session()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                return {"success": False, "error": "User not found"}
            
            if not self.verify_password(old_password, user.password_hash):
                return {"success": False, "error": "Invalid current password"}
            
            user.password_hash = self.hash_password(new_password)
            db.commit()
            
            return {"success": True}
        
        finally:
            db.close()
    
    def regenerate_api_token(self, user_id: int) -> Dict[str, Any]:
        """Regenerate user API token."""
        db = get_session()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                return {"success": False, "error": "User not found"}
            
            user.api_token = self.generate_token()
            db.commit()
            
            return {
                "success": True,
                "api_token": user.api_token,
            }
        
        finally:
            db.close()


class TeamService:
    """Team management service."""
    
    def __init__(self):
        """Initialize team service."""
        self.auth_service = UserAuthService()
    
    def create_team(
        self,
        name: str,
        description: Optional[str] = None,
        owner_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a new team."""
        db = get_session()
        try:
            team = Team(
                name=name,
                description=description,
                owner_id=owner_id,
            )
            db.add(team)
            
            # Add owner as admin
            if owner_id:
                member = TeamMember(
                    team_id=team.id,
                    user_id=owner_id,
                    role="admin",
                )
                db.add(member)
            
            db.commit()
            db.refresh(team)
            
            return {
                "success": True,
                "team": {
                    "id": team.id,
                    "name": team.name,
                    "description": team.description,
                    "owner_id": team.owner_id,
                    "created_at": team.created_at.isoformat() if team.created_at else None,
                },
            }
        
        finally:
            db.close()
    
    def add_member(
        self,
        team_id: int,
        user_id: int,
        role: str = "member",
    ) -> Dict[str, Any]:
        """Add member to team."""
        db = get_session()
        try:
            # Check if already member
            existing = db.query(TeamMember).filter(
                TeamMember.team_id == team_id,
                TeamMember.user_id == user_id,
            ).first()
            
            if existing:
                return {"success": False, "error": "User already a member"}
            
            member = TeamMember(
                team_id=team_id,
                user_id=user_id,
                role=role,
            )
            db.add(member)
            db.commit()
            
            return {"success": True}
        
        finally:
            db.close()
    
    def remove_member(
        self,
        team_id: int,
        user_id: int,
    ) -> Dict[str, Any]:
        """Remove member from team."""
        db = get_session()
        try:
            member = db.query(TeamMember).filter(
                TeamMember.team_id == team_id,
                TeamMember.user_id == user_id,
            ).first()
            
            if not member:
                return {"success": False, "error": "User not a member"}
            
            db.delete(member)
            db.commit()
            
            return {"success": True}
        
        finally:
            db.close()
    
    def get_team_members(self, team_id: int) -> List[Dict[str, Any]]:
        """Get team members."""
        db = get_session()
        try:
            members = db.query(TeamMember).filter(
                TeamMember.team_id == team_id
            ).all()
            
            results = []
            for m in members:
                user = db.query(User).filter(User.id == m.user_id).first()
                if user:
                    results.append({
                        "user_id": user.id,
                        "username": user.username,
                        "display_name": user.display_name,
                        "role": m.role,
                        "joined_at": m.joined_at.isoformat() if m.joined_at else None,
                    })
            
            return results
        
        finally:
            db.close()
    
    def share_reading_list(
        self,
        list_id: int,
        team_id: int,
        permission: str = "view",
    ) -> Dict[str, Any]:
        """Share reading list with team."""
        db = get_session()
        try:
            reading_list = db.query(ReadingList).filter(
                ReadingList.id == list_id
            ).first()
            
            if not reading_list:
                return {"success": False, "error": "Reading list not found"}
            
            reading_list.team_id = team_id
            reading_list.shared_with_team = True
            reading_list.team_permission = permission
            db.commit()
            
            return {"success": True}
        
        finally:
            db.close()


# Singletons
_auth_service: Optional[UserAuthService] = None
_team_service: Optional[TeamService] = None


def get_auth_service() -> UserAuthService:
    """Get auth service instance."""
    global _auth_service
    if _auth_service is None:
        _auth_service = UserAuthService()
    return _auth_service


def get_team_service() -> TeamService:
    """Get team service instance."""
    global _team_service
    if _team_service is None:
        _team_service = TeamService()
    return _team_service
