"""
FastAPI application for Eyewear Visual Search API
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from PIL import Image
import io
from loguru import logger
import sys

from ..config import settings
from ..database import DatabaseManager
from ..models import FeatureExtractor, AttributeClassifier
from ..pipeline import VectorStore, IngestionPipeline, SearchEngine
from ..utils import calculate_image_hash

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="AI-powered visual similarity search for eyewear products"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
db_manager: Optional[DatabaseManager] = None
vector_store: Optional[VectorStore] = None
feature_extractor: Optional[FeatureExtractor] = None
attribute_classifier: Optional[AttributeClassifier] = None
ingestion_pipeline: Optional[IngestionPipeline] = None
search_engine: Optional[SearchEngine] = None


# Pydantic models for request/response
class ProductMetadata(BaseModel):
    """Product metadata model"""
    brand: str
    model_name: Optional[str] = None
    price: float
    material: Optional[str] = None
    frame_type: Optional[str] = None
    color: Optional[str] = None
    rim_type: Optional[str] = None


class SearchFilters(BaseModel):
    """Search filters model"""
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    brand: Optional[List[str]] = None
    material: Optional[List[str]] = None
    frame_type: Optional[List[str]] = None
    color: Optional[List[str]] = None


class FeedbackRequest(BaseModel):
    """Feedback request model"""
    product_id: int
    is_relevant: bool
    similarity_score: float
    query_image_hash: Optional[str] = None
    clicked: bool = False


class SearchResponse(BaseModel):
    """Search response model"""
    results: List[Dict]
    query_attributes: Optional[Dict] = None
    total_results: int
    search_time_ms: float


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global db_manager, vector_store, feature_extractor, attribute_classifier
    global ingestion_pipeline, search_engine
    
    try:
        logger.info("Initializing application components...")
        
        # Initialize database
        db_manager = DatabaseManager(settings.DB_PATH)
        db_manager.create_tables()
        logger.info("Database initialized")
        
        # Initialize feature extractor
        feature_extractor = FeatureExtractor(
            model_name=settings.MODEL_NAME,
            use_gpu=settings.USE_GPU,
            image_size=settings.IMAGE_SIZE
        )
        logger.info("Feature extractor initialized")
        
        # Initialize attribute classifier
        attribute_classifier = AttributeClassifier(
            use_gpu=settings.USE_GPU,
            image_size=settings.IMAGE_SIZE
        )
        logger.info("Attribute classifier initialized")
        
        # Initialize vector store
        vector_store = VectorStore(
            dimension=settings.EMBEDDING_DIM,
            index_path=settings.FAISS_INDEX_PATH,
            metric=settings.DISTANCE_METRIC
        )
        logger.info("Vector store initialized")
        
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
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        if vector_store:
            vector_store.save_index()
            logger.info("Vector store saved")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "components": {
            "database": db_manager is not None,
            "vector_store": vector_store is not None,
            "feature_extractor": feature_extractor is not None,
            "search_engine": search_engine is not None
        }
    }


# Ingest endpoint
@app.post("/ingest")
async def ingest_image(
    image: UploadFile = File(...),
    brand: str = Query(..., description="Brand name"),
    price: float = Query(..., description="Product price"),
    model_name: Optional[str] = Query(None, description="Model name"),
    material: Optional[str] = Query(None, description="Material type"),
    frame_type: Optional[str] = Query(None, description="Frame type"),
    color: Optional[str] = Query(None, description="Color"),
    rim_type: Optional[str] = Query(None, description="Rim type")
):
    """
    Ingest a new product image into the catalog
    """
    try:
        # Read and validate image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Save image temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Prepare metadata
            metadata = {
                "brand": brand,
                "price": price,
                "model_name": model_name,
                "material": material,
                "frame_type": frame_type,
                "color": color,
                "rim_type": rim_type
            }
            
            # Ingest image
            product_id = ingestion_pipeline.ingest_single_image(tmp_path, metadata)
            
            if product_id:
                # Save vector store
                vector_store.save_index()
                
                return {
                    "success": True,
                    "product_id": product_id,
                    "message": "Product ingested successfully"
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to ingest product")
                
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"Error in ingest endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Search endpoint
@app.post("/search", response_model=SearchResponse)
async def search_similar(
    image: UploadFile = File(...),
    top_k: int = Query(10, ge=1, le=50, description="Number of results"),
    price_min: Optional[float] = Query(None, description="Minimum price"),
    price_max: Optional[float] = Query(None, description="Maximum price"),
    brand: Optional[str] = Query(None, description="Filter by brand"),
    material: Optional[str] = Query(None, description="Filter by material"),
    frame_type: Optional[str] = Query(None, description="Filter by frame type"),
    color: Optional[str] = Query(None, description="Filter by color"),
    classify_query: bool = Query(False, description="Classify query image attributes")
):
    """
    Search for visually similar products
    """
    import time
    start_time = time.time()
    
    try:
        # Read and validate image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Prepare filters
        filters = {}
        if price_min is not None:
            filters["price_min"] = price_min
        if price_max is not None:
            filters["price_max"] = price_max
        if brand:
            filters["brand"] = brand
        if material:
            filters["material"] = material
        if frame_type:
            filters["frame_type"] = frame_type
        if color:
            filters["color"] = color
        
        # Search
        results = search_engine.search_by_image(
            query_image=img,
            top_k=top_k,
            filters=filters if filters else None,
            apply_feedback_boost=True
        )
        
        # Optionally classify query image
        query_attributes = None
        if classify_query:
            query_attributes = search_engine.classify_query_image(img)
        
        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=results,
            query_attributes=query_attributes,
            total_results=len(results),
            search_time_ms=search_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Classify endpoint
@app.post("/classify")
async def classify_attributes(image: UploadFile = File(...)):
    """
    Classify attributes of an eyewear image
    """
    try:
        # Read and validate image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Classify
        predictions = search_engine.classify_query_image(img)
        
        return {
            "success": True,
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Error in classify endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Feedback endpoint
@app.post("/feedback")
async def record_feedback(feedback: FeedbackRequest):
    """
    Record user feedback on search results
    """
    try:
        search_engine.record_feedback(
            product_id=feedback.product_id,
            is_relevant=feedback.is_relevant,
            similarity_score=feedback.similarity_score,
            query_image_hash=feedback.query_image_hash,
            clicked=feedback.clicked
        )
        
        return {
            "success": True,
            "message": "Feedback recorded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Get product details endpoint
@app.get("/product/{product_id}")
async def get_product(product_id: int):
    """
    Get details of a specific product
    """
    try:
        product = search_engine.get_product_details(product_id)
        
        if product:
            return product
        else:
            raise HTTPException(status_code=404, detail="Product not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_product endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Stats endpoint
@app.get("/stats")
async def get_stats():
    """
    Get system statistics
    """
    try:
        stats = search_engine.get_search_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error in stats endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
