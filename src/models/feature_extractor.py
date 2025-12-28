"""
Feature extraction using pre-trained deep learning models
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import List, Union
from loguru import logger


class FeatureExtractor:
    """Extract image embeddings using pre-trained CNN models"""
    
    def __init__(
        self,
        model_name: str = "resnet50",
        use_gpu: bool = False,
        image_size: int = 224
    ):
        """
        Initialize the feature extractor
        
        Args:
            model_name: Name of the pre-trained model (resnet50, efficientnet_b0, etc.)
            use_gpu: Whether to use GPU for inference
            image_size: Input image size
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        
        # Load pre-trained model
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Feature extractor initialized with {model_name} on {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load and modify pre-trained model"""
        if self.model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            # Remove the final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
            
        elif self.model_name == "resnet18":
            model = models.resnet18(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
            
        elif self.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=True)
        
    
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model
    
    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocess image and return batched tensor (1, C, H, W)
        """

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        else:
            raise TypeError("Image must be a file path or PIL Image")

        img_tensor = self.transform(image)          # (C, H, W)
        img_tensor = img_tensor.unsqueeze(0)        # (1, C, H, W)

        return img_tensor


    
    def extract_features(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Extract feature vector from a single image
        
        Args:
            image: Path to image or PIL Image object
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Preprocess
            img_tensor = self.preprocess_image(image).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
            
            # Flatten and convert to numpy
            features = features.squeeze().cpu().numpy()
            
            # Normalize the feature vector
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise
    
    def extract_batch_features(self, images: List[Union[str, Image.Image]]) -> np.ndarray:
        """
        Extract features from a batch of images
        
        Args:
            images: List of image paths or PIL Image objects
            
        Returns:
            Array of feature vectors (n_images, feature_dim)
        """
        try:
            # Preprocess all images
            img_tensors = torch.cat([
                self.preprocess_image(img) for img in images
            ]).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensors)
            
            # Flatten and convert to numpy
            features = features.squeeze().cpu().numpy()
            
            # Normalize each feature vector
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / (norms + 1e-8)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting batch features: {str(e)}")
            raise
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of the embedding vector"""
        if self.model_name in ["resnet50"]:
            return 2048
        elif self.model_name in ["resnet18"]:
            return 512
        elif self.model_name == "efficientnet_b0":
            return 1280
        else:
            # Fallback: extract from a dummy image
            dummy_img = Image.new("RGB", (self.image_size, self.image_size))
            features = self.extract_features(dummy_img)
            return features.shape[0]
