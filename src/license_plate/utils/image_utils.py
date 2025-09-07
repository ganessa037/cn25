"""Image Processing Utilities for License Plate Detection

This module provides comprehensive image processing utilities including
OpenCV preprocessing, contour detection, image enhancement, and visualization
functions specifically optimized for license plate detection tasks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure

from ..config.settings import (
    IMAGE_CONFIG,
    LOGGING_CONFIG
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Comprehensive image processing utilities for license plate detection."""
    
    def __init__(self):
        """Initialize the image processor."""
        self.config = IMAGE_CONFIG
        logger.info("ImageProcessor initialized")
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load image from file path with error handling.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array or None if failed
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            logger.debug(f"Loaded image: {image_path} with shape {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def resize_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        maintain_aspect_ratio: bool = True
    ) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target (width, height) or None for default
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = (self.config["target_width"], self.config["target_height"])
        
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        if maintain_aspect_ratio:
            # Calculate scaling factor
            scale = min(target_width / width, target_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Create padded image
            padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Calculate padding
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            # Place resized image in center
            padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
            
            return padded
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def enhance_image(
        self,
        image: np.ndarray,
        enhance_contrast: bool = True,
        enhance_brightness: bool = True,
        denoise: bool = True
    ) -> np.ndarray:
        """
        Enhance image quality for better detection.
        
        Args:
            image: Input image
            enhance_contrast: Whether to enhance contrast
            enhance_brightness: Whether to enhance brightness
            denoise: Whether to apply denoising
            
        Returns:
            Enhanced image
        """
        enhanced = image.copy()
        
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        if enhance_contrast:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
        
        if enhance_brightness:
            # Adjust brightness based on image statistics
            mean_brightness = np.mean(l_channel)
            if mean_brightness < 100:  # Dark image
                l_channel = cv2.add(l_channel, 20)
            elif mean_brightness > 180:  # Bright image
                l_channel = cv2.subtract(l_channel, 10)
        
        # Merge channels back
        lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        if denoise:
            # Apply Non-local Means Denoising
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return enhanced
    
    def preprocess_for_detection(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Preprocess image for YOLO detection.
        
        Args:
            image: Input image
            target_size: Target size for detection model
            
        Returns:
            Preprocessed image
        """
        # Enhance image quality
        enhanced = self.enhance_image(image)
        
        # Resize for detection
        if target_size:
            processed = self.resize_image(enhanced, target_size)
        else:
            processed = enhanced
        
        return processed
    
    def detect_contours(
        self,
        image: np.ndarray,
        min_area: int = 500,
        max_area: int = 50000,
        aspect_ratio_range: Tuple[float, float] = (2.0, 6.0)
    ) -> List[Dict]:
        """
        Detect potential license plate contours.
        
        Args:
            image: Input image
            min_area: Minimum contour area
            max_area: Maximum contour area
            aspect_ratio_range: (min_ratio, max_ratio) for width/height
            
        Returns:
            List of contour information dictionaries
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_plates = []
        
        for i, contour in enumerate(contours):
            # Calculate contour properties
            area = cv2.contourArea(contour)
            
            if min_area <= area <= max_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Check aspect ratio
                if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                    # Calculate additional properties
                    perimeter = cv2.arcLength(contour, True)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    contour_info = {
                        "contour_id": i,
                        "contour": contour,
                        "bbox": {"x": x, "y": y, "width": w, "height": h},
                        "area": area,
                        "perimeter": perimeter,
                        "aspect_ratio": aspect_ratio,
                        "solidity": solidity,
                        "score": self._calculate_plate_score(area, aspect_ratio, solidity)
                    }
                    
                    potential_plates.append(contour_info)
        
        # Sort by score (best first)
        potential_plates.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"Found {len(potential_plates)} potential license plate contours")
        return potential_plates
    
    def _calculate_plate_score(
        self,
        area: float,
        aspect_ratio: float,
        solidity: float
    ) -> float:
        """
        Calculate a score for how likely a contour is a license plate.
        
        Args:
            area: Contour area
            aspect_ratio: Width/height ratio
            solidity: Contour solidity
            
        Returns:
            Score between 0 and 1
        """
        # Ideal values for Malaysian license plates
        ideal_aspect_ratio = 4.0
        ideal_solidity = 0.8
        
        # Calculate aspect ratio score
        aspect_score = 1.0 - abs(aspect_ratio - ideal_aspect_ratio) / ideal_aspect_ratio
        aspect_score = max(0, min(1, aspect_score))
        
        # Calculate solidity score
        solidity_score = min(solidity / ideal_solidity, 1.0)
        
        # Calculate area score (normalized)
        area_score = min(area / 10000, 1.0)  # Normalize to reasonable plate size
        
        # Weighted combination
        total_score = (0.4 * aspect_score + 0.3 * solidity_score + 0.3 * area_score)
        
        return total_score
    
    def extract_roi(
        self,
        image: np.ndarray,
        bbox: Dict,
        padding: int = 10
    ) -> np.ndarray:
        """
        Extract region of interest from image.
        
        Args:
            image: Input image
            bbox: Bounding box dictionary with x, y, width, height
            padding: Padding around the ROI
            
        Returns:
            Extracted ROI
        """
        height, width = image.shape[:2]
        
        # Calculate coordinates with padding
        x1 = max(0, bbox["x"] - padding)
        y1 = max(0, bbox["y"] - padding)
        x2 = min(width, bbox["x"] + bbox["width"] + padding)
        y2 = min(height, bbox["y"] + bbox["height"] + padding)
        
        return image[y1:y2, x1:x2]
    
    def create_visualization(
        self,
        image: np.ndarray,
        detections: List[Dict],
        title: str = "License Plate Detection",
        show_confidence: bool = True,
        show_text: bool = True
    ) -> Figure:
        """
        Create visualization of detection results.
        
        Args:
            image: Original image
            detections: List of detection dictionaries
            title: Plot title
            show_confidence: Whether to show confidence scores
            show_text: Whether to show detected text
            
        Returns:
            Matplotlib figure
        """
        # Convert BGR to RGB for matplotlib
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(rgb_image)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Draw detections
        for i, detection in enumerate(detections):
            bbox = detection.get("bbox", {})
            confidence = detection.get("confidence", detection.get("detection_confidence", 0))
            plate_text = detection.get("plate_text", detection.get("text", ""))
            is_valid = detection.get("is_valid_plate", False)
            
            if bbox:
                # Determine color based on validity
                color = 'green' if is_valid else 'orange'
                
                # Create rectangle
                rect = patches.Rectangle(
                    (bbox.get("x1", bbox.get("x", 0)), bbox.get("y1", bbox.get("y", 0))),
                    bbox.get("width", bbox.get("x2", 0) - bbox.get("x1", bbox.get("x", 0))),
                    bbox.get("height", bbox.get("y2", 0) - bbox.get("y1", bbox.get("y", 0))),
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add text label
                label_parts = []
                if show_confidence:
                    label_parts.append(f"Conf: {confidence:.2f}")
                if show_text and plate_text:
                    label_parts.append(f"Text: {plate_text}")
                
                if label_parts:
                    label = " | ".join(label_parts)
                    ax.text(
                        bbox.get("x1", bbox.get("x", 0)),
                        bbox.get("y1", bbox.get("y", 0)) - 10,
                        label,
                        fontsize=10,
                        color=color,
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
                    )
        
        plt.tight_layout()
        return fig
    
    def create_comparison_plot(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        detections: Optional[List[Dict]] = None
    ) -> Figure:
        """
        Create side-by-side comparison of original and processed images.
        
        Args:
            original: Original image
            processed: Processed image
            detections: Optional detection results
            
        Returns:
            Matplotlib figure
        """
        # Convert BGR to RGB
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        axes[0].imshow(original_rgb)
        axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Processed image
        axes[1].imshow(processed_rgb)
        axes[1].set_title("Processed Image", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Add detections to processed image if provided
        if detections:
            for detection in detections:
                bbox = detection.get("bbox", {})
                if bbox:
                    rect = patches.Rectangle(
                        (bbox.get("x1", bbox.get("x", 0)), bbox.get("y1", bbox.get("y", 0))),
                        bbox.get("width", bbox.get("x2", 0) - bbox.get("x1", bbox.get("x", 0))),
                        bbox.get("height", bbox.get("y2", 0) - bbox.get("y1", bbox.get("y", 0))),
                        linewidth=2,
                        edgecolor='green',
                        facecolor='none'
                    )
                    axes[1].add_patch(rect)
        
        plt.tight_layout()
        return fig
    
    def save_annotated_image(
        self,
        image: np.ndarray,
        detections: List[Dict],
        output_path: Union[str, Path],
        draw_text: bool = True
    ) -> bool:
        """
        Save annotated image with detection results.
        
        Args:
            image: Original image
            detections: Detection results
            output_path: Path to save the annotated image
            draw_text: Whether to draw text labels
            
        Returns:
            True if successful, False otherwise
        """
        try:
            annotated = image.copy()
            
            for detection in detections:
                bbox = detection.get("bbox", {})
                confidence = detection.get("confidence", detection.get("detection_confidence", 0))
                plate_text = detection.get("plate_text", detection.get("text", ""))
                is_valid = detection.get("is_valid_plate", False)
                
                if bbox:
                    # Determine color
                    color = (0, 255, 0) if is_valid else (0, 165, 255)  # Green or orange
                    
                    # Draw rectangle
                    cv2.rectangle(
                        annotated,
                        (int(bbox.get("x1", bbox.get("x", 0))), int(bbox.get("y1", bbox.get("y", 0)))),
                        (int(bbox.get("x2", bbox.get("x", 0) + bbox.get("width", 0))), 
                         int(bbox.get("y2", bbox.get("y", 0) + bbox.get("height", 0)))),
                        color,
                        2
                    )
                    
                    if draw_text:
                        # Create label
                        label_parts = [f"Conf: {confidence:.2f}"]
                        if plate_text:
                            label_parts.append(f"Text: {plate_text}")
                        
                        label = " | ".join(label_parts)
                        
                        # Draw text background
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        
                        text_x = int(bbox.get("x1", bbox.get("x", 0)))
                        text_y = int(bbox.get("y1", bbox.get("y", 0))) - 10
                        
                        cv2.rectangle(
                            annotated,
                            (text_x, text_y - text_height - 5),
                            (text_x + text_width, text_y + 5),
                            color,
                            -1
                        )
                        
                        # Draw text
                        cv2.putText(
                            annotated,
                            label,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
            
            # Save image
            success = cv2.imwrite(str(output_path), annotated)
            
            if success:
                logger.info(f"Saved annotated image to {output_path}")
            else:
                logger.error(f"Failed to save annotated image to {output_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving annotated image: {e}")
            return False
    
    def batch_process_images(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        enhance: bool = True,
        detect_contours: bool = False
    ) -> Dict:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save processed images
            enhance: Whether to enhance images
            detect_contours: Whether to detect contours
            
        Returns:
            Processing summary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "total_images": len(image_paths),
            "processed_images": 0,
            "failed_images": 0,
            "processing_details": []
        }
        
        for i, image_path in enumerate(image_paths):
            try:
                # Load image
                image = self.load_image(image_path)
                if image is None:
                    results["failed_images"] += 1
                    continue
                
                # Process image
                if enhance:
                    processed = self.enhance_image(image)
                else:
                    processed = image
                
                # Detect contours if requested
                contours = []
                if detect_contours:
                    contours = self.detect_contours(processed)
                
                # Save processed image
                output_path = output_dir / f"processed_{Path(image_path).name}"
                cv2.imwrite(str(output_path), processed)
                
                # Save contour visualization if contours found
                if contours:
                    contour_vis_path = output_dir / f"contours_{Path(image_path).name}"
                    self.save_annotated_image(processed, contours, contour_vis_path)
                
                results["processed_images"] += 1
                results["processing_details"].append({
                    "image_path": str(image_path),
                    "output_path": str(output_path),
                    "contours_found": len(contours),
                    "status": "success"
                })
                
                logger.info(f"Processed image {i+1}/{len(image_paths)}: {image_path}")
                
            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {e}")
                results["failed_images"] += 1
                results["processing_details"].append({
                    "image_path": str(image_path),
                    "status": "failed",
                    "error": str(e)
                })
        
        logger.info(
            f"Batch processing completed: {results['processed_images']} successful, "
            f"{results['failed_images']} failed"
        )
        
        return results

# Convenience functions
def enhance_image_quick(image: np.ndarray) -> np.ndarray:
    """
    Quick image enhancement function.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    processor = ImageProcessor()
    return processor.enhance_image(image)

def detect_plate_contours(image: np.ndarray) -> List[Dict]:
    """
    Quick contour detection function.
    
    Args:
        image: Input image
        
    Returns:
        List of potential plate contours
    """
    processor = ImageProcessor()
    return processor.detect_contours(image)

def visualize_detections(
    image: np.ndarray,
    detections: List[Dict],
    save_path: Optional[Union[str, Path]] = None
) -> Figure:
    """
    Quick visualization function.
    
    Args:
        image: Input image
        detections: Detection results
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    processor = ImageProcessor()
    fig = processor.create_visualization(image, detections)
    
    if save_path:
        fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
    
    return fig