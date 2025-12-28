"""
Configuration settings for the Eyewear Visual Search system
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Eyewear Visual Search"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Paths
    DATA_DIR: str = "data"
    IMAGES_DIR: str = "data/images"
    VECTORS_DIR: str = "data/vectors"
    DB_PATH: str = "data/eyewear.db"
    FAISS_INDEX_PATH: str = "data/vectors/faiss_index.bin"
    
    # Model settings
    MODEL_NAME: str = "resnet50"  # Options: resnet50, efficientnet_b0, vit_base_patch16_224
    EMBEDDING_DIM: int = 2048  # ResNet50 output dimension
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 32
    
    # Search settings
    TOP_K_RESULTS: int = 10
    SIMILARITY_THRESHOLD: float = 0.5
    DISTANCE_METRIC: str = "cosine"  # Options: cosine, euclidean, l2
    
    # Feature extraction
    USE_GPU: bool = False
    NUM_WORKERS: int = 4
    
    # Feedback settings
    FEEDBACK_BOOST_FACTOR: float = 0.1
    MIN_CLICKS_FOR_BOOST: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
