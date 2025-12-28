"""
Attribute recognition classifier for eyewear
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Dict, List
from loguru import logger


class AttributeClassifier:
    """Classify eyewear attributes like frame type, material, etc."""
    
    # Define attribute categories
    FRAME_TYPES = ["Aviator", "Wayfarer", "Round", "Square", "Cat-Eye", "Rectangle", "Oversized"]
    MATERIALS = ["Acetate", "Metal", "Plastic", "Titanium", "Mixed"]
    RIM_TYPES = ["Full-rim", "Semi-rimless", "Rimless"]
    COLORS = ["Black", "Brown", "Transparent", "Gold", "Silver", "Blue", "Tortoise"]
    
    def __init__(self, use_gpu: bool = False, image_size: int = 224):
        """
        Initialize the attribute classifier
        
        Args:
            use_gpu: Whether to use GPU for inference
            image_size: Input image size
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        
        # Load pre-trained model for classification
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Attribute classifier initialized on {self.device}")
    
    def _load_model(self) -> nn.Module:
      model = models.resnet18(pretrained=True)

      return model

    
    def _extract_features(self, image: Image.Image) -> torch.Tensor:
        """Extract features from image"""

        # ✅ Ensure PIL + RGB
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        # ✅ Apply transform
        img_tensor = self.transform(image)

        # 🔍 Safety check (prevents unsqueeze crash)
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError(
                f"Transform must return torch.Tensor, got {type(img_tensor)}"
            )

        # ✅ Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(img_tensor)

        return features.squeeze()

    
    def classify_frame_type(self, image: Image.Image) -> Dict[str, float]:
        """
        Classify the frame type using heuristic-based approach
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary of frame types with confidence scores
        """
        # Extract features
        features = self._extract_features(image)
        
        # Simple heuristic classification based on feature patterns
        # In production, this would be a trained classifier
        scores = {}
        
        # Convert features to numpy for analysis
        feat_np = features.cpu().numpy()
        
        # Generate pseudo-probabilities based on feature statistics
        # This is a simplified approach - in production, use a trained model
        mean_feat = np.mean(feat_np)
        std_feat = np.std(feat_np)
        
        # Heuristic scoring (simplified)
        for i, frame_type in enumerate(self.FRAME_TYPES):
            # Generate a score based on feature patterns
            score = abs(np.sin(mean_feat * (i + 1))) * (1 - std_feat * 0.1)
            scores[frame_type] = float(max(0.1, min(0.9, score)))
        
        # Normalize scores
        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def classify_material(self, image: Image.Image) -> Dict[str, float]:
        """
        Classify the material type
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary of materials with confidence scores
        """
        features = self._extract_features(image)
        feat_np = features.cpu().numpy()
        
        scores = {}
        mean_feat = np.mean(feat_np)
        
        for i, material in enumerate(self.MATERIALS):
            score = abs(np.cos(mean_feat * (i + 1.5))) * 0.8
            scores[material] = float(max(0.1, min(0.9, score)))
        
        # Normalize
        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def classify_rim_type(self, image: Image.Image) -> Dict[str, float]:
        """
        Classify the rim type
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary of rim types with confidence scores
        """
        features = self._extract_features(image)
        feat_np = features.cpu().numpy()
        
        scores = {}
        std_feat = np.std(feat_np)
        
        for i, rim_type in enumerate(self.RIM_TYPES):
            score = abs(np.sin(std_feat * (i + 2))) * 0.85
            scores[rim_type] = float(max(0.15, min(0.85, score)))
        
        # Normalize
        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def classify_color(self, image: Image.Image) -> Dict[str, float]:
        """
        Classify dominant color using color analysis
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary of colors with confidence scores
        """
        # Convert to numpy array
        img_array = np.array(image.resize((self.image_size, self.image_size)))
        
        # Calculate color statistics
        mean_color = np.mean(img_array, axis=(0, 1))
        
        scores = {}
        
        # Simple color classification based on RGB values
        r, g, b = mean_color
        
        # Heuristic color matching
        if r < 50 and g < 50 and b < 50:
            scores["Black"] = 0.7
        elif r > 200 and g > 200 and b > 200:
            scores["Transparent"] = 0.6
        elif r > g and r > b and r > 150:
            scores["Brown"] = 0.6
        elif r > 180 and g > 150 and b < 100:
            scores["Gold"] = 0.65
        elif r > 150 and g > 150 and b > 150:
            scores["Silver"] = 0.6
        elif b > r and b > g:
            scores["Blue"] = 0.65
        elif abs(r - g) < 30 and r > 100:
            scores["Tortoise"] = 0.55
        else:
            scores["Brown"] = 0.5
        
        # Fill in remaining colors with low scores
        for color in self.COLORS:
            if color not in scores:
                scores[color] = 0.1
        
        # Normalize
        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def classify_all_attributes(self, image: Image.Image) -> Dict[str, Dict[str, float]]:
        """
        Classify all attributes at once
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing all attribute classifications
        """
        try:
            return {
                "frame_type": self.classify_frame_type(image),
                "material": self.classify_material(image),
                "rim_type": self.classify_rim_type(image),
                "color": self.classify_color(image)
            }
        except Exception as e:
            logger.error(f"Error classifying attributes: {str(e)}")
            raise
    
    def get_top_predictions(
        self,
        image: Image.Image,
        top_k: int = 1
    ) -> Dict[str, List[tuple]]:
        """
        Get top-k predictions for each attribute
        
        Args:
            image: PIL Image object
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with top predictions for each attribute
        """
        all_attrs = self.classify_all_attributes(image)
        
        result = {}
        for attr_name, scores in all_attrs.items():
            # Sort by score and get top-k
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            result[attr_name] = sorted_items[:top_k]
        
        return result
