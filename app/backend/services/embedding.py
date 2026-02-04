"""
Embedding service for semantic similarity search.
"""
import json
import logging
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

from ..config import get_config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and using paper embeddings."""
    
    def __init__(self):
        """Initialize embedding service."""
        self.config = get_config()
        self.provider = self.config.embedding_provider
        self.model = self.config.embedding_model
        self._model = None
        self._client = None
    
    def _load_model(self):
        """Load the embedding model."""
        if self._model is not None:
            return
        
        if self.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model)
                logger.info(f"Loaded sentence-transformers model: {self.model}")
            except ImportError:
                logger.warning("sentence-transformers not installed")
                self.provider = "none"
        elif self.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.config.embedding_api_key)
                logger.info("Initialized OpenAI embedding client")
            except ImportError:
                logger.warning("openai not installed")
                self.provider = "none"
    
    def encode(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Encode texts to embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            List of embedding vectors, or None if failed
        """
        if self.provider == "none":
            logger.debug("Embedding provider not configured")
            return None
        
        self._load_model()
        
        if self.provider == "sentence-transformers" and self._model:
            try:
                embeddings = self._model.encode(texts, convert_to_numpy=True)
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Error encoding with sentence-transformers: {e}")
                return None
        
        elif self.provider == "openai" and self._client:
            try:
                response = self._client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts[:100],  # Limit batch size
                )
                embeddings = [data.embedding for data in response.data]
                return embeddings
            except Exception as e:
                logger.error(f"Error encoding with OpenAI: {e}")
                return None
        
        return None
    
    def encode_text(self, text: str) -> Optional[List[float]]:
        """Encode a single text.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector, or None if failed
        """
        results = self.encode([text])
        return results[0] if results else None
    
    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        if not embedding1 or not embedding2:
            return 0.0
        
        if not NUMPY_AVAILABLE:
            # Fallback: simple implementation without numpy
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            norm1 = sum(a * a for a in embedding1) ** 0.5
            norm2 = sum(b * b for b in embedding2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_similar(
        self,
        query_embedding: List[float],
        embeddings: Dict[int, List[float]],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """Find most similar papers to a query.
        
        Args:
            query_embedding: Query embedding vector
            embeddings: Dict mapping paper_id to embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (paper_id, similarity_score) tuples
        """
        similarities = []
        
        for paper_id, embedding in embeddings.items():
            score = self.compute_similarity(query_embedding, embedding)
            if score >= threshold:
                similarities.append((paper_id, score))
        
        # Sort by score descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def truncate_text(self, text: str, max_tokens: int = 8000) -> str:
        """Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens (rough approximation)
            
        Returns:
            Truncated text
        """
        # Rough estimate: 4 characters per token
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
