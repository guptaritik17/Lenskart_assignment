"""
Main script to run the Eyewear Visual Search system
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings
from src.database.models import DatabaseManager
from src.models.feature_extractor import FeatureExtractor
from src.models.attribute_classifier import AttributeClassifier
from src.pipeline.ingestion import VectorStore, IngestionPipeline
from src.pipeline.search_engine import SearchEngine
from loguru import logger


def setup_system():
    """Initialize all system components"""
    logger.info("Setting up Eyewear Visual Search System...")
    
    # Create directories
    Path(settings.DATA_DIR).mkdir(exist_ok=True)
    Path(settings.IMAGES_DIR).mkdir(exist_ok=True)
    Path(settings.VECTORS_DIR).mkdir(exist_ok=True)
    
    # Initialize database
    db_manager = DatabaseManager(settings.DB_PATH)
    db_manager.create_tables()
    logger.info("Database initialized")
    
    # Initialize models
    feature_extractor = FeatureExtractor(
        model_name=settings.MODEL_NAME,
        use_gpu=settings.USE_GPU,
        image_size=settings.IMAGE_SIZE
    )
    
    attribute_classifier = AttributeClassifier(
        use_gpu=settings.USE_GPU,
        image_size=settings.IMAGE_SIZE
    )
    
    # Initialize vector store
    vector_store = VectorStore(
        dimension=settings.EMBEDDING_DIM,
        index_path=settings.FAISS_INDEX_PATH,
        metric=settings.DISTANCE_METRIC
    )
    
    # Initialize pipelines
    ingestion_pipeline = IngestionPipeline(
        db_manager=db_manager,
        vector_store=vector_store,
        feature_extractor=feature_extractor,
        attribute_classifier=attribute_classifier
    )
    
    search_engine = SearchEngine(
        db_manager=db_manager,
        vector_store=vector_store,
        feature_extractor=feature_extractor,
        attribute_classifier=attribute_classifier
    )
    
    logger.info("System setup complete!")
    
    return {
        "db_manager": db_manager,
        "vector_store": vector_store,
        "feature_extractor": feature_extractor,
        "attribute_classifier": attribute_classifier,
        "ingestion_pipeline": ingestion_pipeline,
        "search_engine": search_engine
    }


def ingest_from_directory(components, directory: str):
    """Ingest images from a directory"""
    logger.info(f"Ingesting images from {directory}")
    
    pipeline = components["ingestion_pipeline"]
    
    # Use default metadata for demo
    default_metadata = {
        "brand": "Sample Brand",
        "price": 99.99
    }
    
    product_ids = pipeline.ingest_from_directory(
        directory=directory,
        default_metadata=default_metadata
    )
    
    # Save vector store
    components["vector_store"].save_index()
    
    logger.info(f"Ingestion complete! Processed {len(product_ids)} images")
    return product_ids


def run_api_server():
    """Run the FastAPI server"""
    import uvicorn
    from src.api.app import app
    
    logger.info(f"Starting API server on {settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )


def main():
    parser = argparse.ArgumentParser(
        description="Eyewear Visual Search System"
    )
    
    parser.add_argument(
        "command",
        choices=["setup", "ingest", "serve", "stats"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--directory",
        type=str,
        help="Directory containing images (for ingest command)"
    )
    
    args = parser.parse_args()
    
    if args.command == "setup":
        # Just setup the system
        setup_system()
        logger.info("System setup complete. Ready to ingest data or start server.")
        
    elif args.command == "ingest":
        if not args.directory:
            logger.error("--directory argument required for ingest command")
            sys.exit(1)
        
        components = setup_system()
        ingest_from_directory(components, args.directory)
        
    elif args.command == "serve":
        # Start API server
        run_api_server()
        
    elif args.command == "stats":
        # Show system statistics
        components = setup_system()
        stats = components["search_engine"].get_search_stats()
        
        logger.info("System Statistics:")
        logger.info(f"  Total Products: {stats['total_products']}")
        logger.info(f"  Total Feedback: {stats['total_feedback_entries']}")
        logger.info(f"  Vector Store: {stats['vector_store_stats']}")


if __name__ == "__main__":
    main()
