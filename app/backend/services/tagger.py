"""
Tag generation service using keyword extraction.
"""
import hashlib
import json
import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from ..config import get_config

logger = logging.getLogger(__name__)

# Common stopwords to filter out
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare", "ought", "used",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "also", "now", "this", "that",
    "these", "those", "what", "which", "who", "whom", "its", "their", "they", "we", "you",
    "he", "she", "it", "i", "me", "my", "your", "his", "her", "our", "any", "both",
    "down", "up", "out", "off", "over", "while", "if", "about", "against", "between",
    "into", "through", "because", "since", "until", "although", "though", "after",
    "before", "when", "whenever", "where", "wherever", "whether", "which", "while",
    "proposed", "method", "approach", "technique", "paper", "work", "show", "showed",
    "shows", "present", "presents", "presented", "using", "use", "used", "based", "new",
    "novel", "state", "art", "results", "result", "experiment", "experiments", "data",
    "performance", "accuracy", "effectiveness", "efficient", "efficiently", "existing",
    "previous", "prior", "first", "well", "however", "therefore", "thus", "hence",
    "consequently", "resulting", "compared", "achieve", "achieves", "achieving",
}

# Technical term patterns (common ML/AI terms)
TECH_TERMS = {
    "llm", "large language model", "language model", "transformer", "attention",
    "diffusion", "gan", "vae", "autoencoder", "bert", "gpt", "clip", "stable diffusion",
    "reinforcement learning", "rl", "rlhf", "ppo", "actor critic", "deep learning",
    "neural network", "cnn", "rnn", "lstm", "gru", "recurrent", "convolutional",
    "generative", "classifier", "classification", "regression", "clustering",
    "semantic", "syntax", "parsing", "translation", "summarization", "generation",
    "retrieval", "embedding", "vector", "similarity", "retrieval augmented",
    "rag", "fine tuning", "finetuning", "pretraining", "pretrain", "pre training",
    "training", "inference", "optimization", "optimizer", "adam", "sgd", "momentum",
    "batch normalization", "dropout", "regularization", "loss", "objective",
    "gradient", "backpropagation", "backprop", "epoch", "batch", "iteration",
    "supervised", "unsupervised", "self supervised", "semi supervised",
    "multimodal", "vision", "image", "text", "audio", "video", "3d", "point cloud",
    "robotics", "control", "planning", "decision making", "agent", "environment",
    "reward", "policy", "value", "q learning", "model based", "model free",
    "few shot", "zero shot", "prompt", "prompting", "chain of thought", "cot",
    "reasoning", "knowledge", "memory", "retention", "compression", "quantization",
    "pruning", "distillation", "knowledge distillation", "efficient", "scalable",
    "scaling", "large scale", "benchmark", "dataset", "corpus", "annotation",
    "evaluation", "metric", "bleu", "rouge", "f1", "accuracy", "precision", "recall",
}


class KeywordTagger:
    """Keyword extraction based tagger."""
    
    def __init__(self, tags_per_paper: int = 8):
        """Initialize tagger."""
        self.config = get_config()
        self.tags_per_paper = tags_per_paper or self.config.tags_per_paper
    
    def generate(
        self,
        title: str,
        abstract: str,
        categories: List[str],
    ) -> Tuple[List[str], str]:
        """Generate tags for a paper.
        
        Returns:
            Tuple of (tags list, timestamp)
        """
        import time
        timestamp = time.strftime("%Y%m%d%H%M%S")
        
        # Combine title and abstract for analysis
        text = f"{title} {abstract}".lower()
        words = self._tokenize(text)
        
        # Extract n-grams (1-3 words)
        ngrams = self._extract_ngrams(text, n=[1, 2, 3])
        
        # Score words and n-grams
        scores = self._score_terms(words, ngrams)
        
        # Start with category tags
        tags = [cat.replace(".", "_") for cat in categories[:3]]
        
        # Add top-scoring terms
        sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for term, score in sorted_terms:
            if len(tags) >= self.tags_per_paper:
                break
            if term not in tags and score > 0:
                # Normalize term
                term_normalized = self._normalize_tag(term)
                if term_normalized and term_normalized not in tags:
                    tags.append(term_normalized)
        
        # Limit tags and ensure uniqueness
        tags = list(dict.fromkeys(tags))[:self.tags_per_paper]
        
        return tags, timestamp
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Remove special characters, keep alphanumerics and spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        words = text.split()
        
        # Filter stopwords and short words
        return [
            w for w in words
            if w not in STOPWORDS and len(w) > 2
        ]
    
    def _extract_ngrams(self, text: str, n: List[int] = [1, 2, 3]) -> Dict[str, int]:
        """Extract n-grams from text."""
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        words = text.split()
        ngrams = Counter()
        
        for ngram_len in n:
            for i in range(len(words) - ngram_len + 1):
                ngram = " ".join(words[i : i + ngram_len])
                if ngram_len == 1 and ngram in STOPWORDS:
                    continue
                if len(ngram) > 3:
                    ngrams[ngram] += 1
        
        return ngrams
    
    def _score_terms(
        self, words: List[str], ngrams: Dict[str, int]
    ) -> Dict[str, float]:
        """Score terms based on frequency and importance."""
        scores = {}
        word_counts = Counter(words)
        total_words = len(words) if words else 1
        
        # Score individual words (TF weighting)
        for word, count in word_counts.items():
            tf = count / total_words
            # Boost technical terms
            is_tech = 1 if any(tech in word for tech in TECH_TERMS) else 0
            scores[word] = tf * (1 + is_tech * 0.5)
        
        # Score n-grams
        for ngram, count in ngrams.items():
            tf = count / total_words
            # Boost known technical terms
            is_tech = 1 if any(tech in ngram for tech in TECH_TERMS) else 0
            # Boost longer n-grams slightly
            length_bonus = len(ngram.split()) * 0.1
            scores[ngram] = tf * (1 + is_tech * 0.5 + length_bonus)
        
        return scores
    
    def _normalize_tag(self, tag: str) -> str:
        """Normalize a tag string."""
        # Remove extra whitespace
        tag = " ".join(tag.split())
        
        # Title case for single words
        if " " not in tag:
            # Keep acronyms uppercase, title case others
            if tag.isupper() and len(tag) > 2:
                return tag
            return tag.title()
        
        # For multi-word tags, title case each word
        return " ".join(word.title() if not word.isupper() else word for word in tag.split())


# Singleton instance
_tagger: Optional[KeywordTagger] = None


def get_tagger() -> KeywordTagger:
    """Get tagger instance."""
    global _tagger
    if _tagger is None:
        _tagger = KeywordTagger()
    return _tagger
