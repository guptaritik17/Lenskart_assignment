"""
Utility functions for image processing and helpers
"""
import hashlib
from PIL import Image
import io
from typing import Union


def calculate_image_hash(image: Union[Image.Image, bytes, str]) -> str:
    """
    Calculate hash of an image for identification
    
    Args:
        image: PIL Image, bytes, or file path
        
    Returns:
        SHA256 hash string
    """
    if isinstance(image, str):
        # Load from file
        with open(image, "rb") as f:
            image_bytes = f.read()
    elif isinstance(image, Image.Image):
        # Convert PIL Image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
    else:
        image_bytes = image
    
    return hashlib.sha256(image_bytes).hexdigest()


def validate_image(image_path: str) -> bool:
    """
    Validate if file is a valid image
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def resize_image(image: Image.Image, max_size: int = 800) -> Image.Image:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: PIL Image
        max_size: Maximum dimension
        
    Returns:
        Resized PIL Image
    """
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image
