"""
Visual search engine for finding similar eyewear products
"""
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np
from loguru import logger

from src.models.feature_extractor import FeatureExtractor
from src.models.attribute_classifier import AttributeClassifier
from src.database.models import DatabaseManager, EyewearProduct, SearchFeedback
from .vector_store import VectorStore
from ..config import settings


class SearchEngine:
    """Visual search engine for eyewear similarity search"""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        vector_store: VectorStore,
        feature_extractor: FeatureExtractor,
        attribute_classifier: Optional[AttributeClassifier] = None
    ):
        """
        Initialize the search engine
        
        Args:
            db_manager: Database manager instance
            vector_store: Vector store instance
            feature_extractor: Feature extraction model
            attribute_classifier: Optional attribute classifier
        """
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.feature_extractor = feature_extractor
        self.attribute_classifier = attribute_classifier
        
        logger.info("Search engine initialized")
    
    def search_by_image(
        self,
        query_image: Image.Image,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        apply_feedback_boost: bool = True
    ) -> List[Dict]:
        """
        Search for similar products using an image
        
        Args:
            query_image: Query image (PIL Image)
            top_k: Number of results to return
            filters: Optional filters (price_min, price_max, brand, material)
            apply_feedback_boost: Whether to boost results based on user feedback
            
        Returns:
            List of product dictionaries with similarity scores
        """
        try:
            # Extract features from query image
            logger.info("Extracting features from query image")
            query_features = self.feature_extractor.extract_features(query_image)
            
            # Get filter IDs if filters are specified
            filter_ids = None
            if filters:
                filter_ids = self._apply_filters(filters)
                if not filter_ids:
                    logger.warning("No products match the specified filters")
                    return []
            
            # Search in vector store (request more results for filtering)
            search_k = top_k * 2 if apply_feedback_boost else top_k
            product_ids, similarity_scores = self.vector_store.search(
                query_features,
                top_k=search_k,
                filter_ids=filter_ids
            )
            
            if not product_ids:
                logger.warning("No similar products found")
                return []
            
            # Fetch product details from database
            session = self.db_manager.get_session()
            try:
                products = session.query(EyewearProduct).filter(
                    EyewearProduct.id.in_(product_ids)
                ).all()
                
                # Create product dictionary with scores
                product_map = {p.id: p for p in products}
                results = []
                
                for product_id, score in zip(product_ids, similarity_scores):
                    if product_id in product_map:
                        
                        # Apply feedback boost if enabled
                        if apply_feedback_boost:
                            boost = self._get_feedback_boost(session, product_id)
                
                # Sort by boosted score
                results.sort(key=lambda x: x["boosted_score"], reverse=True)
                
                # Return top-k
                return results[:top_k]
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise
    
    def _apply_filters(self, filters: Dict) -> Optional[List[int]]:
        """
        Apply filters to get matching product IDs
        
        Args:
            filters: Dictionary of filters
            
        Returns:
            List of product IDs matching filters, or None if no filters
        """
        session = self.db_manager.get_session()
        try:
            query = session.query(EyewearProduct)
            
            # Price range filter
            if "price_min" in filters:
                query = query.filter(EyewearProduct.price >= filters["price_min"])
            if "price_max" in filters:
                query = query.filter(EyewearProduct.price <= filters["price_max"])
            
            # Brand filter
            if "brand" in filters:
                if isinstance(filters["brand"], list):
                    query = query.filter(EyewearProduct.brand.in_(filters["brand"]))
                else:
                    query = query.filter(EyewearProduct.brand == filters["brand"])
            
            # Material filter
            if "material" in filters:
                if isinstance(filters["material"], list):
                    query = query.filter(EyewearProduct.material.in_(filters["material"]))
                else:
                    query = query.filter(EyewearProduct.material == filters["material"])
            
            # Frame type filter
            if "frame_type" in filters:
                if isinstance(filters["frame_type"], list):
                    query = query.filter(EyewearProduct.frame_type.in_(filters["frame_type"]))
                else:
                    query = query.filter(EyewearProduct.frame_type == filters["frame_type"])
            
            # Color filter
            if "color" in filters:
                if isinstance(filters["color"], list):
                    query = query.filter(EyewearProduct.color.in_(filters["color"]))
                else:
                    query = query.filter(EyewearProduct.color == filters["color"])
            
            products = query.all()
            
        finally:
            session.close()
    
    def _get_feedback_boost(self, session, product_id: int) -> float:
        """
        Calculate feedback boost for a product
        
        Args:
            session: Database session
            product_id: Product ID
            
        Returns:
            Boost factor (0 to 1)
        """
        # Get feedback statistics
        feedbacks = session.query(SearchFeedback).filter(
            SearchFeedback.product_id == product_id
        ).all()
        
        if not feedbacks:
            return 0.0
        
        # Calculate boost based on positive feedback
        total_clicks = sum(f.clicked for f in feedbacks)
        relevant_count = sum(1 for f in feedbacks if f.is_relevant == 1)
        
        if total_clicks < settings.MIN_CLICKS_FOR_BOOST:
            return 0.0
        
        # Boost factor based on relevance ratio and click count
        relevance_ratio = relevant_count / len(feedbacks) if feedbacks else 0
        click_factor = min(total_clicks / 10, 1.0)  # Cap at 10 clicks
        
        boost = relevance_ratio * click_factor * settings.FEEDBACK_BOOST_FACTOR
        
        return boost
    
    def record_feedback(
        self,
        product_id: int,
        is_relevant: bool,
        similarity_score: float,
        query_image_hash: Optional[str] = None,
        clicked: bool = False
    ) -> None:
        """
        Record user feedback on search results
        
        Args:
            product_id: Product ID
            is_relevant: Whether the result was relevant
            similarity_score: Similarity score from search
            query_image_hash: Hash of query image
            clicked: Whether the user clicked on the result
        """
        session = self.db_manager.get_session()
        try:
            # Check if feedback already exists for this query-product pair
            existing = session.query(SearchFeedback).filter(
                SearchFeedback.product_id == product_id,
                SearchFeedback.query_image_hash == query_image_hash
            ).first()

            if existing:
                # update existing feedback
                feedback = SearchFeedback(
                    product_id=product_id,
                    query_image_hash=query_image_hash,
                    is_relevant=1 if is_relevant else 0,
                    similarity_score=similarity_score,
                    clicked=1 if clicked else 0
                )
                session.add(feedback)
            
            session.commit()
            logger.info(f"Feedback recorded for product {product_id}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error recording feedback: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_product_details(self, product_id: int) -> Optional[Dict]:
        """
        Get detailed information about a product
        
        Args:
            product_id: Product ID
            
        Returns:
            Product dictionary or None if not found
        """
        session = self.db_manager.get_session()
        try:
            product = session.query(EyewearProduct).filter(
                EyewearProduct.id == product_id
            ).first()
            
            if product:
                return product.to_dict()
            return None
            
        finally:
            session.close()
    
    def classify_query_image(self, query_image: Image.Image) -> Dict:
        """
        Classify attributes of the query image
        
        Args:
            query_image: Query image (PIL Image)
            
        Returns:
            Dictionary of attribute predictions
        """
        if not self.attribute_classifier:
            raise ValueError("Attribute classifier not available")
        
        try:
            predictions = self.attribute_classifier.get_top_predictions(
                query_image,
                top_k=3
            )
            return predictions
            
        except Exception as e:
            logger.error(f"Error classifying query image: {str(e)}")
            raise
    
    def get_search_stats(self) -> Dict:
        """Get statistics about the search system"""
        session = self.db_manager.get_session()
        try:
            total_products = session.query(EyewearProduct).count()
            total_feedback = session.query(SearchFeedback).count()
            
            return {
                "total_products": total_products,
                "total_feedback_entries": total_feedback,
                "vector_store_stats": self.vector_store.get_stats()
            }
            
        finally:
            session.close()
