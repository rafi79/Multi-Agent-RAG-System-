"""
Embedding generation using sentence-transformers
"""
import logging
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of the sentence-transformer model
        """
        if model_name is None:
            model_name = MODEL_CONFIG["embedding_model"]
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        logger.debug(f"Encoding {len(texts)} texts")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.embed(query)[0]
    
    def embed_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple documents
        
        Args:
            documents: List of document texts
            batch_size: Batch size for encoding
            
        Returns:
            Array of embeddings
        """
        return self.embed(documents, batch_size=batch_size)
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Normalize
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Compute dot product
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)


# Singleton instance
_embedding_generator = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create EmbeddingGenerator instance"""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator
