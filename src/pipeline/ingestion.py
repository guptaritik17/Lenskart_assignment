"""
Image ingestion pipeline for processing and indexing eyewear catalog
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from PIL import Image
from loguru import logger

from src.models.feature_extractor import FeatureExtractor
from src.models.attribute_classifier import AttributeClassifier
from src.database.models import DatabaseManager, EyewearProduct
from .vector_store import VectorStore


class IngestionPipeline:
    """Pipeline for ingesting eyewear images into the search system"""

    def __init__(
        self,
        db_manager: DatabaseManager,
        vector_store: VectorStore,
        feature_extractor: FeatureExtractor,
        attribute_classifier: Optional[AttributeClassifier] = None,
    ):
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.feature_extractor = feature_extractor
        self.attribute_classifier = attribute_classifier

        logger.info("Ingestion pipeline initialized")

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    def preprocess_image(self, image_path: str) -> Image.Image:
        try:
            image = Image.open(image_path).convert("RGB")

            max_size = 800
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            return image
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise

    # ------------------------------------------------------------------
    # Single image ingestion
    # ------------------------------------------------------------------
    def ingest_single_image(
        self,
        image_path: str,
        metadata: Dict,
    ) -> Optional[int]:

        if "brand" not in metadata or "price" not in metadata:
            logger.error("Missing required metadata: brand and price")
            return None

        session = self.db_manager.get_session()

        try:
            # 1️⃣ Load + preprocess image
            image = self.preprocess_image(image_path)

            # 2️⃣ Feature extraction
            features = self.feature_extractor.extract_features(image)

            features = np.asarray(features, dtype=np.float32).reshape(1, -1)

            # 3️⃣ Attribute inference (optional)
            if self.attribute_classifier:
                preds = self.attribute_classifier.get_top_predictions(image, top_k=1)

                metadata.setdefault("frame_type", preds["frame_type"][0][0])
                metadata.setdefault("material", preds["material"][0][0])
                metadata.setdefault("rim_type", preds["rim_type"][0][0])
                metadata.setdefault("color", preds["color"][0][0])

            # 4️⃣ Create DB record
            product = EyewearProduct(
                image_path=image_path,
                brand=metadata["brand"],
                model_name=metadata.get("model_name"),
                price=metadata["price"],
                material=metadata.get("material"),
                frame_type=metadata.get("frame_type"),
                color=metadata.get("color"),
                rim_type=metadata.get("rim_type"),
            )

            session.add(product)
            session.flush()  # 🔴 REQUIRED for product.id

            product_id = product.id
            if product_id is None:
                raise RuntimeError("Failed to generate product ID")

            # Vector store returns index OR map
            product.vector_id = product_id

            # 6️⃣ Commit DB only after vector insert succeeds
            session.commit()

            logger.info(f"Ingested {image_path} | product_id={product_id}")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed ingesting {image_path}: {e}")
            return None

        finally:
            session.close()

    # ------------------------------------------------------------------
    # Batch ingestion
    # ------------------------------------------------------------------
    def ingest_batch(
        self,
        image_paths: List[str],
        metadata_list: List[Dict],
    ) -> List[Optional[int]]:

        if len(image_paths) != len(metadata_list):
            raise ValueError("Image paths and metadata size mismatch")

        product_ids = [
            self.ingest_single_image(img, meta)
            for img, meta in zip(image_paths, metadata_list)
        ]

        self.vector_store.save_index()

        logger.info(
            f"Batch ingestion completed: "
            f"{sum(p is not None for p in product_ids)}/{len(product_ids)} successful"
        )

        return product_ids

    # ------------------------------------------------------------------
    # Directory ingestion
    # ------------------------------------------------------------------
    def ingest_from_directory(
        self,
        directory: str,
        metadata_file: Optional[str] = None,
        default_metadata: Optional[Dict] = None,
    ) -> List[Optional[int]]:

        image_exts = {".jpg", ".jpeg", ".png", ".webp"}

        image_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(directory)
            for f in files
            if Path(f).suffix.lower() in image_exts
        ]

        logger.info(f"Found {len(image_paths)} images in {directory}")

        if metadata_file:
            import pandas as pd

            df = pd.read_csv(metadata_file)
            metadata_list = []

            for img_path in image_paths:
                name = os.path.basename(img_path)
                row = df[df["filename"] == name]

                if not row.empty:
                    metadata_list.append(row.iloc[0].to_dict())
                else:
                    metadata_list.append(default_metadata or {"brand": "Unknown", "price": 0.0})
        else:
            metadata_list = [
                default_metadata or {"brand": "Unknown", "price": 0.0}
                for _ in image_paths
            ]

        return self.ingest_batch(image_paths, metadata_list)
