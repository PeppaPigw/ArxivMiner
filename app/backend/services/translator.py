"""
Translation service using DeepL API.
"""
import hashlib
import json
import logging
from typing import Dict, Optional

import requests

from ..config import get_config

logger = logging.getLogger(__name__)

# Translation cache file
TRANSLATION_CACHE_FILE = "translation_cache.json"


class TranslationCache:
    """Simple file-based cache for translations."""
    
    def __init__(self, cache_file: str = TRANSLATION_CACHE_FILE):
        """Initialize cache."""
        self.cache_file = cache_file
        self.cache: Dict[str, str] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from file."""
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            self.cache = {}
        except json.JSONDecodeError:
            logger.warning("Translation cache corrupted, starting fresh")
            self.cache = {}
    
    def save(self):
        """Save cache to file."""
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def get(self, text_hash: str) -> Optional[str]:
        """Get translation by text hash."""
        return self.cache.get(text_hash)
    
    def set(self, text_hash: str, translation: str):
        """Store translation."""
        self.cache[text_hash] = translation
        self.save()


class DeepLTranslator:
    """DeepL API translator."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize translator."""
        config = get_config()
        self.api_key = api_key or config.deepl_api_key
        self.api_url = config.deepl_api_url
        self.source_lang = "EN"
        self.target_lang = "ZH"
        self.cache = TranslationCache()
    
    def translate(self, text: str) -> Optional[str]:
        """Translate text from English to Chinese."""
        if not text or not text.strip():
            return None
        
        # Check cache first
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self.cache.get(text_hash)
        if cached:
            logger.debug(f"Cache hit for text hash: {text_hash[:16]}...")
            return cached
        
        # Make API request
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"DeepL-Auth-Key {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "text": [text],
                    "source_lang": self.source_lang,
                    "target_lang": self.target_lang,
                },
                timeout=30,
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("translations"):
                translated_text = data["translations"][0]["text"]
                
                # Cache the translation
                self.cache.set(text_hash, translated_text)
                
                return translated_text
            
        except requests.RequestException as e:
            logger.error(f"DeepL API error: {e}")
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing DeepL response: {e}")
        
        return None
    
    def translate_batch(self, texts: list, batch_size: int = 50) -> Dict[str, str]:
        """Translate multiple texts."""
        results = {}
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            
            # Check cache first
            uncached = []
            for text in batch:
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                cached = self.cache.get(text_hash)
                if cached:
                    results[text] = cached
                else:
                    uncached.append(text)
            
            # Translate uncached texts
            if uncached:
                try:
                    response = requests.post(
                        self.api_url,
                        headers={
                            "Authorization": f"DeepL-Auth-Key {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "text": uncached,
                            "source_lang": self.source_lang,
                            "target_lang": self.target_lang,
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    if data.get("translations"):
                        for j, translated in enumerate(data["translations"]):
                            original = uncached[j]
                            translated_text = translated["text"]
                            text_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()
                            self.cache.set(text_hash, translated_text)
                            results[original] = translated_text
                    
                except requests.RequestException as e:
                    logger.error(f"DeepL batch translation error: {e}")
            
            # Rate limiting
            if i + batch_size < len(texts):
                import time
                time.sleep(1)
        
        return results


# Singleton instance
_translator: Optional[DeepLTranslator] = None


def get_translator() -> DeepLTranslator:
    """Get translator instance."""
    global _translator
    if _translator is None:
        _translator = DeepLTranslator()
    return _translator
