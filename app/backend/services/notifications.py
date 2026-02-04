"""
Notification Service.
Send alerts via Slack, Discord, and Email.
"""
import json
import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import requests

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.backend.config import get_config
from app.backend.db.models import Notification, UserNotificationPreference, get_session

logger = logging.getLogger(__name__)


class NotificationService:
    """Multi-channel notification service."""
    
    def __init__(self):
        """Initialize notification service."""
        self.config = get_config()
    
    # ============ Slack Integration ============
    
    def send_slack_message(
        self,
        webhook_url: str,
        message: str,
        channel: Optional[str] = None,
        username: Optional[str] = None,
        blocks: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Send message to Slack."""
        try:
            payload = {
                "text": message,
            }
            
            if channel:
                payload["channel"] = channel
            
            if username:
                payload["username"] = username
            
            if blocks:
                payload["blocks"] = blocks
            
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=10,
            )
            
            response.raise_for_status()
            
            return {
                "success": True,
                "channel": "slack",
            }
        
        except Exception as e:
            logger.error(f"Slack notification error: {e}")
            return {
                "success": False,
                "error": str(e),
                "channel": "slack",
            }
    
    def send_slack_paper_alert(
        self,
        webhook_url: str,
        paper: Dict[str, Any],
        channel: str = "#papers",
    ) -> Dict[str, Any]:
        """Send paper alert to Slack."""
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "📄 New Paper Alert",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{paper.get('title', 'Unknown Paper')}*\n\n{paper.get('abstract_en', '')[:200]}...",
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Venue:*\n{paper.get('primary_category', 'N/A')}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Authors:*\n{', '.join(paper.get('authors', [])[:3])}",
                    },
                ],
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "View Paper",
                            "emoji": True,
                        },
                        "url": paper.get('abs_url', ''),
                        "style": "primary",
                    },
                ],
            },
        ]
        
        return self.send_slack_message(
            webhook_url=webhook_url,
            message=f"New paper: {paper.get('title', 'Unknown')}",
            channel=channel,
            username="ArxivMiner",
            blocks=blocks,
        )
    
    # ============ Discord Integration ============
    
    def send_discord_message(
        self,
        webhook_url: str,
        content: str,
        username: Optional[str] = None,
        avatar_url: Optional[str] = None,
        embeds: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Send message to Discord."""
        try:
            payload = {}
            
            if content:
                payload["content"] = content
            
            if username:
                payload["username"] = username
            
            if avatar_url:
                payload["avatar_url"] = avatar_url
            
            if embeds:
                payload["embeds"] = embeds
            
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=10,
            )
            
            response.raise_for_status()
            
            return {
                "success": True,
                "channel": "discord",
            }
        
        except Exception as e:
            logger.error(f"Discord notification error: {e}")
            return {
                "success": False,
                "error": str(e),
                "channel": "discord",
            }
    
    def send_discord_paper_alert(
        self,
        webhook_url: str,
        paper: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send paper alert to Discord."""
        embed = {
            "title": paper.get('title', 'Unknown Paper')[:256],
            "description": f"{paper.get('abstract_en', '')[:300]}...",
            "url": paper.get('abs_url', ''),
            "color": 0x5865F2,  # Discord blurple
            "fields": [
                {
                    "name": "Venue",
                    "value": paper.get('primary_category', 'N/A'),
                    "inline": True,
                },
                {
                    "name": "Authors",
                    "value": ", ".join(paper.get('authors', [])[:3]),
                    "inline": True,
                },
            ],
            "footer": {
                "text": "ArxivMiner",
            },
        }
        
        return self.send_discord_message(
            webhook_url=webhook_url,
            content="📄 **New Paper Alert**",
            username="ArxivMiner",
            embeds=[embed],
        )
    
    # ============ Email Integration ============
    
    def send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        smtp_config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Send email via SMTP."""
        if smtp_config is None:
            smtp_config = {
                "host": self.config.smtp_host,
                "port": self.config.smtp_port or 587,
                "username": self.config.smtp_username,
                "password": self.config.smtp_password,
                "from_email": self.config.smtp_from_email or "noreply@arxivminer.com",
            }
        
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = smtp_config["from_email"]
            msg["To"] = to_email
            
            # Attach both plain text and HTML
            text_body = self._html_to_text(html_body)
            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))
            
            # Connect and send
            server = smtplib.SMTP(smtp_config["host"], smtp_config["port"])
            server.starttls()
            server.login(smtp_config["username"], smtp_config["password"])
            server.send_message(msg)
            server.quit()
            
            return {
                "success": True,
                "channel": "email",
            }
        
        except Exception as e:
            logger.error(f"Email notification error: {e}")
            return {
                "success": False,
                "error": str(e),
                "channel": "email",
            }
    
    def send_email_paper_digest(
        self,
        to_email: str,
        papers: List[Dict[str, Any]],
        period: str = "daily",
    ) -> Dict[str, Any]:
        """Send email digest of papers."""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background: #4A90D9; color: white; padding: 20px; text-align: center; }}
                .paper {{ padding: 15px; border-bottom: 1px solid #eee; }}
                .paper h3 {{ margin: 0 0 10px; color: #4A90D9; }}
                .paper .meta {{ color: #666; font-size: 12px; }}
                .footer {{ text-align: center; padding: 20px; color: #999; font-size: 12px; }}
                .button {{ display: inline-block; padding: 10px 20px; background: #4A90D9; color: white; text-decoration: none; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📄 ArxivMiner {period.title()} Digest</h1>
                <p>{len(papers)} new papers</p>
            </div>
        """
        
        for paper in papers:
            html += f"""
            <div class="paper">
                <h3><a href="{paper.get('abs_url', '')}">{paper.get('title', 'Unknown')}</a></h3>
                <p>{paper.get('abstract_en', '')[:200]}...</p>
                <div class="meta">
                    📍 {paper.get('primary_category', 'N/A')} | 
                    ✍️ {', '.join(paper.get('authors', [])[:3])}
                </div>
            </div>
            """
        
        html += f"""
            <div class="footer">
                <p>Sent by ArxivMiner</p>
                <p><a href="https://arxivminer.example.com">Open Dashboard</a></p>
            </div>
        </body>
        </html>
        """
        
        subject = f"📄 {period.title()} Digest: {len(papers)} new papers"
        
        return self.send_email(to_email, subject, html)
    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text."""
        import re
        text = re.sub(r'<[^>]+>', '', html)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # ============ Database Integration ============
    
    def create_notification(
        self,
        user_id: int,
        title: str,
        message: str,
        notification_type: str = "info",
        link: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Create notification in database."""
        db = get_session()
        try:
            notification = Notification(
                user_id=user_id,
                title=title,
                message=message,
                type=notification_type,
                link=link,
                metadata_json=json.dumps(metadata) if metadata else None,
            )
            db.add(notification)
            db.commit()
            
            return {
                "success": True,
                "notification_id": notification.id,
            }
        
        finally:
            db.close()
    
    def get_user_notifications(
        self,
        user_id: int,
        unread_only: bool = False,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get user notifications."""
        db = get_session()
        try:
            query = db.query(Notification).filter(
                Notification.user_id == user_id
            )
            
            if unread_only:
                query = query.filter(Notification.is_read == False)
            
            notifications = query.order_by(
                Notification.created_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    "id": n.id,
                    "title": n.title,
                    "message": n.message,
                    "type": n.type,
                    "link": n.link,
                    "is_read": n.is_read,
                    "created_at": n.created_at.isoformat() if n.created_at else None,
                }
                for n in notifications
            ]
        
        finally:
            db.close()
    
    def mark_notification_read(
        self,
        notification_id: int,
        user_id: int,
    ) -> Dict[str, Any]:
        """Mark notification as read."""
        db = get_session()
        try:
            notification = db.query(Notification).filter(
                Notification.id == notification_id,
                Notification.user_id == user_id,
            ).first()
            
            if notification:
                notification.is_read = True
                db.commit()
                return {"success": True}
            
            return {"success": False, "error": "Notification not found"}
        
        finally:
            db.close()
    
    def mark_all_read(
        self,
        user_id: int,
    ) -> Dict[str, Any]:
        """Mark all notifications as read."""
        db = get_session()
        try:
            count = db.query(Notification).filter(
                Notification.user_id == user_id,
                Notification.is_read == False,
            ).update({"is_read": True})
            
            db.commit()
            
            return {
                "success": True,
                "marked_read": count,
            }
        
        finally:
            db.close()


# Singleton
_notification_service: Optional[NotificationService] = None


def get_notification_service() -> NotificationService:
    """Get notification service instance."""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service
