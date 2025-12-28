"""
Tests for the Eyewear Visual Search system
"""
import pytest
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.feature_extractor import FeatureExtractor
from src.models.attribute_classifier import AttributeClassifier
from src.pipeline.ingestion import VectorStore
from src.database.models import DatabaseManager, EyewearProduct


class TestFeatureExtractor:
    """Test feature extraction"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.extractor = FeatureExtractor(
            model_name="resnet50",
            use_gpu=False
        )
    
    def test_extract_features(self):
        """Test feature extraction from image"""
        # Create dummy image
        img = Image.new('RGB', (224, 224))
        
        # Extract features
        features = self.extractor.extract_features(img)
        
        # Check output
        assert isinstance(features, np.ndarray)
        assert features.shape == (2048,)
        assert not np.isnan(features).any()
    
    def test_batch_extraction(self):
        """Test batch feature extraction"""
        # Create dummy images
        images = [
            Image.new('RGB', (224, 224)),
            Image.new('RGB', (224, 224)),
            Image.new('RGB', (224, 224))
        ]
        
        # Extract features
        features = np.array([12,3,4])
        
        # Check output
        assert features.shape == (3, 2048)
        assert not np.isnan(features).any()


class TestAttributeClassifier:
    """Test attribute classification"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.classifier = AttributeClassifier(use_gpu=False)
    
    def test_classify_frame_type(self):
        """Test frame type classification"""
        img = Image.new('RGB', (224, 224))
        
        scores = self.classifier.classify_frame_type(img)
        
        assert isinstance(scores, dict)
        assert len(scores) == len(self.classifier.FRAME_TYPES)
        assert abs(sum(scores.values()) - 1.0) < 0.01  # Scores sum to 1
    
    def test_classify_all_attributes(self):
        """Test full attribute classification"""
        img = Image.new('RGB', (224, 224))
        
        results = self.classifier.classify_all_attributes(img)
        
        assert "frame_type" in results
        assert "material" in results
        assert "rim_type" in results
        assert "color" in results


class TestVectorStore:
    """Test vector store operations"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.dimension = 128
        self.vector_store = VectorStore(
            dimension=self.dimension,
            metric="cosine"
        )
    
    def test_add_vectors(self):
        """Test adding vectors to store"""
        # Create dummy vectors
        vectors = np.random.randn(5, self.dimension).astype(np.float32)
        product_ids = [1, 2, 3, 4, 5]
        
        # Add to store
        self.vector_store.add_vectors(vectors, product_ids)
        
        # Check
        assert self.vector_store.index.ntotal == 5
    
    def test_search(self):
        """Test vector search"""
        # Add vectors
        vectors = np.random.randn(10, self.dimension).astype(np.float32)
        product_ids = list(range(1, 11))
        self.vector_store.add_vectors(vectors, product_ids)
        
        # Search with first vector
        query = vectors[0]
        results, scores = self.vector_store.search(query, top_k=3)
        
        # Check results
        assert len(results) == 3
        assert len(scores) == 3
        assert results[0] == 1  # Should find itself first


class TestDatabase:
    """Test database operations"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.db_manager = DatabaseManager(":memory:")  # In-memory database
        self.db_manager.create_tables()
    
    def test_create_product(self):
        """Test creating a product"""
        session = self.db_manager.get_session()
        
        try:
            product = EyewearProduct(
                image_path="/test/image.jpg",
                brand="TestBrand",
                model_name="Test Model",
                price=99.99,
                material="Metal",
                frame_type="Aviator",
                color="Black",
                rim_type="Full-rim",
                vector_id=0
            )
            
            session.add(product)
            session.commit()
            
            # Query back
            retrieved = session.query(EyewearProduct).first()
            
        finally:
            session.close()
    
    def test_product_to_dict(self):
        """Test product serialization"""
        product = EyewearProduct(
            id=1,
            image_path="/test/image.jpg",
            brand="TestBrand",
            price=99.99
        )
        
        product_dict = product.to_dict()
        
        assert product_dict["id"] == 1
        assert product_dict["brand"] == "TestBrand"
        assert product_dict["price"] == 99.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
