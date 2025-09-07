"""YOLOv8 License Plate Detector Module

This module provides a comprehensive wrapper around YOLOv8 for license plate detection,
including model loading, training, inference, and persistence capabilities.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from ..config.settings import (
    YOLO_CONFIG,
    TRAINED_MODELS_DIR,
    OUTPUTS_DIR,
    DETECTION_CONFIG,
    get_model_path,
    get_output_path
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LicensePlateDetector:
    """YOLOv8-based license plate detector with training and inference capabilities."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize the license plate detector.
        
        Args:
            model_path: Path to trained model. If None, uses default YOLOv8n
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.device = self._setup_device(device)
        self.model = None
        self.model_path = model_path
        self.is_trained = False
        
        # Load model
        self._load_model()
        
        logger.info(f"LicensePlateDetector initialized with device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate the computation device."""
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self) -> None:
        """Load YOLOv8 model from path or initialize with pretrained weights."""
        try:
            if self.model_path and Path(self.model_path).exists():
                # Load custom trained model
                self.model = YOLO(self.model_path)
                self.is_trained = True
                logger.info(f"Loaded trained model from: {self.model_path}")
            else:
                # Load pretrained YOLOv8 model
                model_name = YOLO_CONFIG["model_name"]
                self.model = YOLO(model_name)
                self.is_trained = False
                logger.info(f"Loaded pretrained model: {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Train the YOLOv8 model on license plate dataset.
        
        Args:
            data_yaml: Path to dataset YAML configuration
            epochs: Number of training epochs
            batch_size: Training batch size
            imgsz: Input image size
            save_dir: Directory to save training results
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        if not Path(data_yaml).exists():
            raise FileNotFoundError(f"Dataset configuration not found: {data_yaml}")
        
        # Set default save directory
        if save_dir is None:
            save_dir = str(OUTPUTS_DIR / "training")
        
        try:
            logger.info(f"Starting training with {epochs} epochs...")
            
            # Train the model
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                device=self.device,
                project=save_dir,
                name="license_plate_detection",
                save=True,
                save_period=10,  # Save checkpoint every 10 epochs
                **kwargs
            )
            
            # Update model path to best trained model
            best_model_path = Path(save_dir) / "license_plate_detection" / "weights" / "best.pt"
            if best_model_path.exists():
                self.model_path = str(best_model_path)
                self.is_trained = True
                logger.info(f"Training completed. Best model saved to: {best_model_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Training failed: {e}")
    
    def detect(
        self,
        image: Union[str, np.ndarray, Path],
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        save_results: bool = False,
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Detect license plates in an image.
        
        Args:
            image: Input image (path, numpy array, or Path object)
            confidence: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            save_results: Whether to save annotated results
            save_path: Path to save results (if save_results=True)
            
        Returns:
            List of detection dictionaries with bounding boxes and confidence scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please initialize the detector properly.")
        
        # Set default thresholds
        conf_thresh = confidence or YOLO_CONFIG["confidence_threshold"]
        iou_thresh = iou_threshold or YOLO_CONFIG["iou_threshold"]
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=conf_thresh,
                iou=iou_thresh,
                device=self.device,
                save=save_results,
                project=str(OUTPUTS_DIR) if save_results else None,
                name="detections" if save_results else None
            )
            
            # Process results
            detections = self._process_results(results)
            
            # Filter detections based on area and aspect ratio
            filtered_detections = self._filter_detections(detections)
            
            logger.info(f"Detected {len(filtered_detections)} license plates")
            
            return filtered_detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise RuntimeError(f"Detection failed: {e}")
    
    def detect_batch(
        self,
        images: List[Union[str, np.ndarray, Path]],
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> List[List[Dict]]:
        """
        Detect license plates in multiple images.
        
        Args:
            images: List of input images
            confidence: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of detection lists for each image
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                detections = self.detect(
                    image,
                    confidence=confidence,
                    iou_threshold=iou_threshold
                )
                results.append(detections)
                logger.info(f"Processed image {i+1}/{len(images)}")
                
            except Exception as e:
                logger.error(f"Failed to process image {i+1}: {e}")
                results.append([])
        
        return results
    
    def _process_results(self, results: List[Results]) -> List[Dict]:
        """Process YOLOv8 results into standardized format."""
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    detection = {
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "width": float(width),
                            "height": float(height)
                        },
                        "confidence": float(conf),
                        "class_id": int(cls),
                        "class_name": "license_plate",
                        "area": float(width * height),
                        "aspect_ratio": float(width / height) if height > 0 else 0.0
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections based on area and aspect ratio constraints."""
        filtered = []
        
        min_area = DETECTION_CONFIG["min_plate_area"]
        max_area = DETECTION_CONFIG["max_plate_area"]
        aspect_ratio_range = DETECTION_CONFIG["aspect_ratio_range"]
        min_conf = DETECTION_CONFIG["confidence_threshold"]
        
        for detection in detections:
            area = detection["area"]
            aspect_ratio = detection["aspect_ratio"]
            confidence = detection["confidence"]
            
            # Apply filters
            if (min_area <= area <= max_area and
                aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and
                confidence >= min_conf):
                filtered.append(detection)
            else:
                logger.debug(f"Filtered out detection: area={area}, aspect_ratio={aspect_ratio}, conf={confidence}")
        
        return filtered
    
    def extract_plate_regions(
        self,
        image: Union[str, np.ndarray, Path],
        detections: List[Dict],
        padding: int = 10
    ) -> List[np.ndarray]:
        """
        Extract license plate regions from image based on detections.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            padding: Padding around bounding box
            
        Returns:
            List of cropped plate region images
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img = image.copy()
        
        plate_regions = []
        
        for detection in detections:
            bbox = detection["bbox"]
            
            # Add padding and ensure bounds
            x1 = max(0, int(bbox["x1"]) - padding)
            y1 = max(0, int(bbox["y1"]) - padding)
            x2 = min(img.shape[1], int(bbox["x2"]) + padding)
            y2 = min(img.shape[0], int(bbox["y2"]) + padding)
            
            # Extract region
            plate_region = img[y1:y2, x1:x2]
            
            if plate_region.size > 0:
                plate_regions.append(plate_region)
            else:
                logger.warning(f"Empty plate region extracted for bbox: {bbox}")
        
        return plate_regions
    
    def save_model(self, save_path: Optional[str] = None) -> str:
        """
        Save the current model to disk.
        
        Args:
            save_path: Path to save the model. If None, uses default location
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        if save_path is None:
            save_path = str(get_model_path("license_plate_detector.pt"))
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model
            self.model.save(save_path)
            logger.info(f"Model saved to: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise RuntimeError(f"Model saving failed: {e}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the model file
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            self.is_trained = True
            logger.info(f"Model loaded from: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            "model_path": self.model_path,
            "is_trained": self.is_trained,
            "device": self.device,
            "model_type": "YOLOv8",
            "task": "object_detection",
            "classes": ["license_plate"]
        }
    
    def validate(
        self,
        data_yaml: str,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Validate the model on a dataset.
        
        Args:
            data_yaml: Path to validation dataset YAML
            save_dir: Directory to save validation results
            
        Returns:
            Validation results dictionary
        """
        if not self.is_trained:
            logger.warning("Validating with pretrained model, not custom trained model")
        
        if save_dir is None:
            save_dir = str(OUTPUTS_DIR / "validation")
        
        try:
            results = self.model.val(
                data=data_yaml,
                device=self.device,
                project=save_dir,
                name="license_plate_validation"
            )
            
            logger.info("Validation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise RuntimeError(f"Validation failed: {e}")

# Convenience function for quick detection
def detect_license_plates(
    image: Union[str, np.ndarray, Path],
    model_path: Optional[str] = None,
    confidence: float = 0.5
) -> List[Dict]:
    """
    Quick function to detect license plates in an image.
    
    Args:
        image: Input image
        model_path: Path to trained model (optional)
        confidence: Confidence threshold
        
    Returns:
        List of detection dictionaries
    """
    detector = LicensePlateDetector(model_path=model_path)
    return detector.detect(image, confidence=confidence)