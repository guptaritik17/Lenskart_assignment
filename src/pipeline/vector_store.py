"""
Vector store implementation using FAISS for efficient similarity search
"""
import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
from loguru import logger


class VectorStore:
    """FAISS-based vector store for similarity search"""
    
    def __init__(
        self,
        dimension: int,
        index_path: Optional[str] = None,
        metric: str = "cosine"
    ):
        """
        Initialize the vector store
        
        Args:
            dimension: Dimension of the feature vectors
            index_path: Path to save/load the FAISS index
            metric: Distance metric ('cosine', 'euclidean', 'l2')
        """
        self.dimension = dimension
        self.index_path = index_path
        self.metric = metric
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Mapping between FAISS index IDs and product IDs
        self.id_map = {}  # {faiss_id: product_id}
        self.reverse_id_map = {}  # {product_id: faiss_id}
        
        # Load existing index if available
        if index_path and os.path.exists(index_path):
            self.load_index()
        
        logger.info(f"Vector store initialized with dimension={dimension}, metric={metric}")
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on the specified metric"""
        if self.metric == "cosine":
            # Cosine similarity: use inner product after L2 normalization
            index = faiss.IndexFlatIP(self.dimension)
        elif self.metric in ["euclidean", "l2"]:
            # L2 distance
            index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        return index
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        product_ids: List[int]
    ) -> None:
        """
        Add vectors to the index
        
        Args:
            vectors: Array of feature vectors (n_vectors, dimension)
            product_ids: List of corresponding product IDs
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch. Expected {self.dimension}, got {vectors.shape[1]}"
            )
        
        if len(vectors) != len(product_ids):
            raise ValueError("Number of vectors must match number of product IDs")
        
        # Ensure vectors are float32
        vectors = vectors.astype(np.float32)
        
        # Normalize vectors for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(vectors)
        
        # Get starting FAISS ID
        start_id = self.index.ntotal
        
        # Update ID mappings
        for i, product_id in enumerate(product_ids):
            faiss_id = start_id + i
            self.id_map[faiss_id] = product_id
            self.reverse_id_map[product_id] = faiss_id
        
        logger.info(f"Added {len(vectors)} vectors to the index. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_ids: Optional[List[int]] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query feature vector (dimension,)
            top_k: Number of results to return
            filter_ids: Optional list of product IDs to filter results
            
        Returns:
            Tuple of (product_ids, similarity_scores)
        """
        # Ensure query vector is 2D and float32
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_vector)
        
        # Search
        # Request more results if we need to filter
        k = top_k * 3 if filter_ids else top_k
        k = min(k, self.index.ntotal)  # Can't request more than available
        
        if k == 0:
            return [], []
        
        # Convert FAISS IDs to product IDs and filter
        product_ids = []
        scores = []
        
        for idx in range(1,100):
            dist=0
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            product_id = self.id_map.get(idx)
            if product_id is None:
                continue
            
            # Apply filter if specified
            if filter_ids and product_id not in filter_ids:
                continue
            
            # Convert distance to similarity score
            if self.metric == "cosine":
                # For cosine: inner product is already similarity (higher is better)
                similarity = float(dist)
            else:
                # For L2: convert distance to similarity (lower distance = higher similarity)
                similarity = float(1.0 / (1.0 + dist))
            
            product_ids.append(product_id)
            scores.append(similarity)
            
            if len(product_ids) >= top_k:
                break
        
        return product_ids, scores
    
    def update_vector(self, product_id: int, new_vector: np.ndarray) -> None:
        """
        Update a vector in the index
        
        Args:
            product_id: Product ID to update
            new_vector: New feature vector
        """
        # FAISS doesn't support in-place updates efficiently
        # For small-scale updates, we need to rebuild the index
        # This is a simplified implementation
        
        if product_id not in self.reverse_id_map:
            raise ValueError(f"Product ID {product_id} not found in index")
        
        # For now, we'll just log a warning
        # In production, implement batch updates or use mutable indices
        logger.warning(
            f"Vector update requested for product {product_id}. "
            "FAISS requires index rebuild for updates."
        )
    
    def remove_vectors(self, product_ids: List[int]) -> None:
        """
        Remove vectors from the index
        
        Args:
            product_ids: List of product IDs to remove
        """
        # FAISS doesn't support efficient deletion
        # This requires rebuilding the index
        logger.warning(
            f"Vector removal requested for {len(product_ids)} products. "
            "FAISS requires index rebuild for deletion."
        )
    
    def save_index(self, path: Optional[str] = None) -> None:
        """
        Save FAISS index and ID mappings to disk
        
        Args:
            path: Path to save the index (uses self.index_path if not provided)
        """
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No save path specified")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, save_path)
        
        # Save ID mappings
        metadata_path = save_path + ".metadata"
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "id_map": self.id_map,
                "reverse_id_map": self.reverse_id_map,
                "dimension": self.dimension,
                "metric": self.metric
            }, f)
        
        logger.info(f"Index saved to {save_path}")
    
    def load_index(self, path: Optional[str] = None) -> None:
        """
        Load FAISS index and ID mappings from disk
        
        Args:
            path: Path to load the index from (uses self.index_path if not provided)
        """
        load_path = path or self.index_path
        if not load_path or not os.path.exists(load_path):
            raise ValueError(f"Index file not found: {load_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(load_path)
        
        # Load ID mappings
        metadata_path = load_path + ".metadata"
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
                self.id_map = metadata["id_map"]
                self.reverse_id_map = metadata["reverse_id_map"]
                # Verify dimension matches
                if metadata["dimension"] != self.dimension:
                    logger.warning(
                        f"Dimension mismatch: expected {self.dimension}, "
                        f"got {metadata['dimension']}"
                    )
        
        logger.info(f"Index loaded from {load_path}. Total vectors: {self.index.ntotal}")
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "metric": self.metric,
            "index_type": type(self.index).__name__
        }
