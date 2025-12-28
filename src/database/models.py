"""
Database models for storing eyewear metadata and feedback
"""

from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# SQLAlchemy 2.x compatible base
Base = declarative_base()


class EyewearProduct(Base):
    """Model for storing eyewear product information"""

    __tablename__ = "eyewear_products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String, nullable=False, unique=True)
    brand = Column(String, nullable=False)
    model_name = Column(String)
    price = Column(Float, nullable=False)
    material = Column(String)
    frame_type = Column(String)
    color = Column(String)
    rim_type = Column(String)

    # Vector ID for FAISS index
    vector_id = Column(Integer, unique=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    feedbacks = relationship(
        "SearchFeedback",
        back_populates="product",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<EyewearProduct(id={self.id}, brand={self.brand}, model={self.model_name})>"

    def to_dict(self):
        """Convert model to serializable dict"""
        return {
            "id": self.id,
            "image_path": self.image_path,
            "brand": self.brand,
            "model_name": self.model_name,
            "price": self.price,
            "material": self.material,
            "frame_type": self.frame_type,
            "color": self.color,
            "rim_type": self.rim_type,
            "vector_id": self.vector_id,
            "created_at": (
                self.created_at.isoformat()
                if isinstance(self.created_at, datetime)
                else None
            ),
        }


class SearchFeedback(Base):
    """Model for storing user feedback on search results"""

    __tablename__ = "search_feedbacks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("eyewear_products.id"), nullable=False)
    query_image_hash = Column(String)
    is_relevant = Column(Integer)  # 1 = relevant, 0 = not relevant
    similarity_score = Column(Float)
    clicked = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    product = relationship("EyewearProduct", back_populates="feedbacks")

    def __repr__(self):
        return (
            f"<SearchFeedback(id={self.id}, "
            f"product_id={self.product_id}, "
            f"is_relevant={self.is_relevant})>"
        )


class DatabaseManager:
    """Manager class for database operations"""

    def __init__(self, db_path: str):
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            future=True,  # SQLAlchemy 2.x safe
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False
        )

    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)

    def drop_tables(self):
        """Drop all tables (use with caution)"""
        Base.metadata.drop_all(self.engine)

    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()
