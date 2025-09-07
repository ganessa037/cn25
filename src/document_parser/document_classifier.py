#!/usr/bin/env python3
"""
Document Classification Service

Classifies documents into different types (SPK, MyKad, etc.) using machine learning models.
Supports both image and text-based classification with confidence scoring.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO not available. Install ultralytics for YOLO support: pip install ultralytics")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentClassifier:
    """
    Document classification service for identifying document types.
    
    Supports multiple classification approaches:
    - Image-based classification using CNN models
    - Text-based classification using transformer models
    - Hybrid approach combining both methods
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize the document classifier.
        
        Args:
            model_path: Path to the trained classification model
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.device = self._setup_device(device)
        self.model_path = model_path or self._get_default_model_path()
        
        # Document type mappings
        self.document_types = {
            0: "SPK",
            1: "MyKad", 
            2: "Passport",
            3: "License",
            4: "Unknown"
        }
        
        # Load models
        self.text_model = None
        self.text_tokenizer = None
        self.image_model = None
        self.yolo_model = None
        
        self._load_models()
        
        logger.info(f"DocumentClassifier initialized on {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """
        Setup the computation device.
        
        Args:
            device: Device preference
            
        Returns:
            str: Selected device
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _get_default_model_path(self) -> str:
        """
        Get the default model path.
        
        Returns:
            str: Default model path
        """
        return str(Path(__file__).parent.parent.parent / "models" / "document_parser")
    
    def _load_models(self):
        """
        Load the classification models.
        """
        try:
            # Load YOLO document classification model
            yolo_model_path = Path(self.model_path) / "yolo_document_classifier_v1.pt"
            if yolo_model_path.exists() and YOLO_AVAILABLE:
                self.yolo_model = YOLO(str(yolo_model_path))
                logger.info("YOLO document classification model loaded successfully")
            
            # Load text classification model
            text_model_path = Path(self.model_path) / "text_classifier"
            if text_model_path.exists():
                self.text_tokenizer = AutoTokenizer.from_pretrained(str(text_model_path))
                self.text_model = AutoModelForSequenceClassification.from_pretrained(
                    str(text_model_path)
                ).to(self.device)
                logger.info("Text classification model loaded successfully")
            
            # Load image classification model
            image_model_path = Path(self.model_path) / "image_classifier.pkl"
            if image_model_path.exists():
                self.image_model = joblib.load(str(image_model_path))
                logger.info("Image classification model loaded successfully")
                
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            logger.info("Using rule-based classification fallback")
    
    def classify_document(self, 
                         image_path: Optional[str] = None,
                         text_content: Optional[str] = None,
                         image_array: Optional[np.ndarray] = None) -> Dict:
        """
        Classify a document using available inputs.
        
        Args:
            image_path: Path to document image
            text_content: Extracted text content
            image_array: Image as numpy array
            
        Returns:
            Dict: Classification results with confidence scores
        """
        results = {
            "document_type": "Unknown",
            "confidence": 0.0,
            "method": "rule_based",
            "details": {}
        }
        
        try:
            # Try text-based classification first
            if text_content and self.text_model:
                text_result = self._classify_by_text(text_content)
                if text_result["confidence"] > 0.7:
                    results.update(text_result)
                    results["method"] = "text_based"
                    return results
            
            # Try image-based classification
            if (image_path or image_array is not None) and self.image_model:
                image_result = self._classify_by_image(image_path, image_array)
                if image_result["confidence"] > 0.6:
                    results.update(image_result)
                    results["method"] = "image_based"
                    return results
            
            # Fallback to rule-based classification
            if text_content:
                rule_result = self._classify_by_rules(text_content)
                results.update(rule_result)
                results["method"] = "rule_based"
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            results["error"] = str(e)
        
        return results
    
    def _classify_by_text(self, text_content: str) -> Dict:
        """
        Classify document using text content.
        
        Args:
            text_content: Document text content
            
        Returns:
            Dict: Classification result
        """
        try:
            # Tokenize and encode text
            inputs = self.text_tokenizer(
                text_content,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Get top prediction
            confidence, predicted_class = torch.max(predictions, 1)
            document_type = self.document_types.get(predicted_class.item(), "Unknown")
            
            return {
                "document_type": document_type,
                "confidence": confidence.item(),
                "details": {
                    "all_scores": predictions.cpu().numpy().tolist()[0]
                }
            }
            
        except Exception as e:
            logger.error(f"Text classification error: {e}")
            return {"document_type": "Unknown", "confidence": 0.0, "details": {}}
    
    def _classify_by_image(self, 
                          image_path: Optional[str] = None,
                          image_array: Optional[np.ndarray] = None) -> Dict:
        """
        Classify document using image features.
        
        Args:
            image_path: Path to image file
            image_array: Image as numpy array
            
        Returns:
            Dict: Classification result
        """
        try:
            # Try YOLO model first if available
            if self.yolo_model is not None:
                if image_path:
                    results = self.yolo_model(image_path)
                elif image_array is not None:
                    results = self.yolo_model(image_array)
                else:
                    raise ValueError("No image provided")
                
                # Extract prediction from YOLO results
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'probs') and result.probs is not None:
                        # Classification result
                        top_class_idx = result.probs.top1
                        confidence = float(result.probs.top1conf)
                        
                        # Map class index to document type
                        class_names = result.names if hasattr(result, 'names') else {}
                        predicted_class = class_names.get(top_class_idx, "unknown")
                        
                        return {
                            "document_type": predicted_class,
                            "confidence": confidence,
                            "details": {
                                "model_type": "yolo",
                                "class_index": top_class_idx,
                                "all_classes": class_names
                            }
                        }
            
            # Fallback to traditional image classification if YOLO not available or failed
            if self.image_model is not None:
                # Load image
                if image_array is not None:
                    image = image_array
                elif image_path:
                    image = cv2.imread(image_path)
                else:
                    raise ValueError("No image provided")
                
                # Extract features (simplified - would use CNN features in practice)
                features = self._extract_image_features(image)
                
                # Predict using trained model
                prediction = self.image_model.predict([features])[0]
                confidence = max(self.image_model.predict_proba([features])[0])
                
                document_type = self.document_types.get(prediction, "Unknown")
                
                return {
                    "document_type": document_type,
                    "confidence": float(confidence),
                    "details": {
                        "model_type": "traditional",
                        "image_features_count": len(features)
                    }
                }
            
        except Exception as e:
            logger.error(f"Image classification error: {e}")
            return {"document_type": "Unknown", "confidence": 0.0, "details": {}}
    
    def _classify_by_rules(self, text_content: str) -> Dict:
        """
        Classify document using rule-based approach.
        
        Args:
            text_content: Document text content
            
        Returns:
            Dict: Classification result
        """
        text_lower = text_content.lower()
        confidence = 0.0
        document_type = "Unknown"
        
        # SPK detection rules
        spk_keywords = ["sijil pelajaran", "spk", "peperiksaan", "malaysia"]
        spk_score = sum(1 for keyword in spk_keywords if keyword in text_lower)
        
        # MyKad detection rules
        mykad_keywords = ["kad pengenalan", "mykad", "warganegara", "no. k/p"]
        mykad_score = sum(1 for keyword in mykad_keywords if keyword in text_lower)
        
        # Passport detection rules
        passport_keywords = ["passport", "pasport", "travel document", "immigration"]
        passport_score = sum(1 for keyword in passport_keywords if keyword in text_lower)
        
        # License detection rules
        license_keywords = ["lesen", "license", "memandu", "driving"]
        license_score = sum(1 for keyword in license_keywords if keyword in text_lower)
        
        # Determine document type based on highest score
        scores = {
            "SPK": spk_score / len(spk_keywords),
            "MyKad": mykad_score / len(mykad_keywords),
            "Passport": passport_score / len(passport_keywords),
            "License": license_score / len(license_keywords)
        }
        
        if max(scores.values()) > 0:
            document_type = max(scores, key=scores.get)
            confidence = scores[document_type]
        
        return {
            "document_type": document_type,
            "confidence": confidence,
            "details": {
                "rule_scores": scores
            }
        }
    
    def _extract_image_features(self, image: np.ndarray) -> List[float]:
        """
        Extract features from image for classification.
        
        Args:
            image: Input image
            
        Returns:
            List[float]: Extracted features
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Basic features
        features = []
        
        # Image dimensions
        height, width = gray.shape
        features.extend([height, width, height/width])
        
        # Intensity statistics
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.min(gray),
            np.max(gray)
        ])
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        features.append(edge_density)
        
        # Text region estimation (simplified)
        # In practice, would use more sophisticated text detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_regions = len([c for c in contours if cv2.contourArea(c) > 100])
        features.append(text_regions)
        
        return features
    
    def get_supported_types(self) -> List[str]:
        """
        Get list of supported document types.
        
        Returns:
            List[str]: Supported document types
        """
        return list(self.document_types.values())
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded models.
        
        Returns:
            Dict: Model information
        """
        return {
            "model_path": self.model_path,
            "device": self.device,
            "yolo_model_loaded": self.yolo_model is not None,
            "text_model_loaded": self.text_model is not None,
            "image_model_loaded": self.image_model is not None,
            "yolo_available": YOLO_AVAILABLE,
            "supported_types": self.get_supported_types()
        }