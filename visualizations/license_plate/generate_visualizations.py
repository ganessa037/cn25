#!/usr/bin/env python3
"""
License Plate Detection Visualization Generator

This script generates comprehensive visualizations for license plate detection models,
including training metrics, performance analysis, detection results, and comparison charts.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import cv2
from PIL import Image
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))

from license_plate.core.detector import LicensePlateDetector
from license_plate.core.processor import LicensePlateProcessor
from license_plate.config.settings import *

# Configure matplotlib and seaborn
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class LicensePlateVisualizer:
    """Comprehensive visualization generator for license plate detection."""
    
    def __init__(self, output_dir: str = None):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'info': '#6A994E'
        }
        
        print(f"📊 Visualizer initialized. Output directory: {self.output_dir}")
    
    def plot_training_metrics(self, results_path: str, save_name: str = "training_metrics.png"):
        """Plot comprehensive training metrics.
        
        Args:
            results_path: Path to training results directory
            save_name: Name of the output file
        """
        results_file = Path(results_path) / 'results.csv'
        
        if not results_file.exists():
            print(f"❌ Results file not found: {results_file}")
            return
        
        # Load results
        df = pd.read_csv(results_file)
        df.columns = df.columns.str.strip()
        
        # Create comprehensive training plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('License Plate Detection Training Metrics', fontsize=16, fontweight='bold')
        
        # Loss curves
        if 'train/box_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], 
                           label='Train Box Loss', color=self.colors['primary'], linewidth=2)
            axes[0, 0].plot(df['epoch'], df['val/box_loss'], 
                           label='Val Box Loss', color=self.colors['secondary'], linewidth=2)
            axes[0, 0].set_title('Box Loss', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Class loss
        if 'train/cls_loss' in df.columns:
            axes[0, 1].plot(df['epoch'], df['train/cls_loss'], 
                           label='Train Class Loss', color=self.colors['primary'], linewidth=2)
            axes[0, 1].plot(df['epoch'], df['val/cls_loss'], 
                           label='Val Class Loss', color=self.colors['secondary'], linewidth=2)
            axes[0, 1].set_title('Classification Loss', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Precision and Recall
        if 'metrics/precision(B)' in df.columns:
            axes[0, 2].plot(df['epoch'], df['metrics/precision(B)'], 
                           label='Precision', color=self.colors['success'], linewidth=2)
            axes[0, 2].plot(df['epoch'], df['metrics/recall(B)'], 
                           label='Recall', color=self.colors['warning'], linewidth=2)
            axes[0, 2].set_title('Precision & Recall', fontweight='bold')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Score')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # mAP scores
        if 'metrics/mAP50(B)' in df.columns:
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], 
                           label='mAP@0.5', color=self.colors['info'], linewidth=2)
            if 'metrics/mAP50-95(B)' in df.columns:
                axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], 
                               label='mAP@0.5:0.95', color=self.colors['primary'], linewidth=2)
            axes[1, 0].set_title('Mean Average Precision', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mAP')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        if 'lr/pg0' in df.columns:
            axes[1, 1].plot(df['epoch'], df['lr/pg0'], 
                           color=self.colors['secondary'], linewidth=2)
            axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        # F1 Score
        if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
            precision = df['metrics/precision(B)']
            recall = df['metrics/recall(B)']
            f1_score = 2 * (precision * recall) / (precision + recall)
            axes[1, 2].plot(df['epoch'], f1_score, 
                           color=self.colors['warning'], linewidth=2)
            axes[1, 2].set_title('F1 Score', fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('F1 Score')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training metrics plot saved: {output_path}")
        return output_path
    
    def plot_detection_results(self, detector, image_dir: str, 
                             num_samples: int = 12, save_name: str = "detection_results.png"):
        """Plot detection results on sample images.
        
        Args:
            detector: LicensePlateDetector instance
            image_dir: Directory containing test images
            num_samples: Number of sample images to visualize
            save_name: Name of the output file
        """
        image_dir = Path(image_dir)
        images = list(image_dir.glob('*.jpg'))[:num_samples]
        
        if not images:
            print(f"❌ No images found in {image_dir}")
            return
        
        # Calculate grid dimensions
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        fig.suptitle('License Plate Detection Results', fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, img_path in enumerate(images):
            row, col = i // cols, i % cols
            
            # Load and process image
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get predictions
            results = detector.detect(img)
            
            # Draw predictions
            annotated_img = img_rgb.copy()
            detection_count = 0
            
            if results and len(results) > 0:
                for detection in results:
                    bbox = detection.get('bbox', [])
                    confidence = detection.get('confidence', 0)
                    
                    if len(bbox) >= 4 and confidence > 0.3:  # Confidence threshold
                        x1, y1, x2, y2 = map(int, bbox)
                        detection_count += 1
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Add confidence label
                        label = f'LP: {confidence:.2f}'
                        cv2.putText(annotated_img, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            axes[row, col].imshow(annotated_img)
            axes[row, col].set_title(f'{img_path.name}\nDetections: {detection_count}', 
                                   fontsize=10)
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(num_samples, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Detection results plot saved: {output_path}")
        return output_path
    
    def plot_performance_dashboard(self, metrics: Dict, save_name: str = "performance_dashboard.png"):
        """Create a comprehensive performance dashboard.
        
        Args:
            metrics: Dictionary containing performance metrics
            save_name: Name of the output file
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('License Plate Detection Performance Dashboard', 
                    fontsize=16, fontweight='bold')
        
        # Detection rate gauge
        detection_rate = metrics.get('detection_rate', 0)
        self._plot_gauge(axes[0, 0], detection_rate, 'Detection Rate', '%')
        
        # Confidence distribution
        confidences = metrics.get('confidences', [])
        if confidences:
            axes[0, 1].hist(confidences, bins=20, color=self.colors['primary'], 
                           alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(np.mean(confidences), color=self.colors['warning'], 
                              linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
            axes[0, 1].set_title('Confidence Score Distribution', fontweight='bold')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Processing time distribution
        processing_times = metrics.get('processing_times', [])
        if processing_times:
            axes[1, 0].hist(processing_times, bins=20, color=self.colors['success'], 
                           alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(np.mean(processing_times), color=self.colors['warning'], 
                              linestyle='--', linewidth=2, 
                              label=f'Mean: {np.mean(processing_times):.3f}s')
            axes[1, 0].set_title('Processing Time Distribution', fontweight='bold')
            axes[1, 0].set_xlabel('Processing Time (seconds)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Performance summary table
        axes[1, 1].axis('off')
        summary_data = [
            ['Metric', 'Value'],
            ['Total Images', f"{metrics.get('total_images', 0):,}"],
            ['Images with Detections', f"{metrics.get('images_with_detections', 0):,}"],
            ['Detection Rate', f"{detection_rate:.2%}"],
            ['Total Detections', f"{metrics.get('total_detections', 0):,}"],
            ['Avg Confidence', f"{metrics.get('avg_confidence', 0):.3f}"],
            ['Avg Processing Time', f"{metrics.get('avg_processing_time', 0):.3f}s"],
            ['Min Confidence', f"{metrics.get('min_confidence', 0):.3f}"],
            ['Max Confidence', f"{metrics.get('max_confidence', 0):.3f}"]
        ]
        
        table = axes[1, 1].table(cellText=summary_data[1:], colLabels=summary_data[0],
                                cellLoc='center', loc='center', 
                                colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_data)):
            if i == 0:  # Header
                table[(i, 0)].set_facecolor(self.colors['primary'])
                table[(i, 1)].set_facecolor(self.colors['primary'])
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            else:
                table[(i, 0)].set_facecolor('#f0f0f0')
                table[(i, 1)].set_facecolor('#ffffff')
        
        axes[1, 1].set_title('Performance Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Performance dashboard saved: {output_path}")
        return output_path
    
    def _plot_gauge(self, ax, value: float, title: str, unit: str = ''):
        """Plot a gauge chart for a single metric."""
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Background arc
        ax.plot(theta, r, color='lightgray', linewidth=10)
        
        # Value arc
        value_theta = np.linspace(0, np.pi * value, int(100 * value))
        value_r = np.ones_like(value_theta)
        
        if value < 0.5:
            color = self.colors['warning']
        elif value < 0.8:
            color = self.colors['success']
        else:
            color = self.colors['info']
        
        ax.plot(value_theta, value_r, color=color, linewidth=10)
        
        # Add value text
        ax.text(np.pi/2, 0.5, f'{value:.1%}' if unit == '%' else f'{value:.3f}{unit}', 
               ha='center', va='center', fontsize=16, fontweight='bold')
        
        ax.set_ylim(0, 1.2)
        ax.set_xlim(0, np.pi)
        ax.set_title(title, fontweight='bold', pad=20)
        ax.axis('off')
    
    def plot_dataset_analysis(self, data_dir: str, save_name: str = "dataset_analysis.png"):
        """Analyze and visualize dataset characteristics.
        
        Args:
            data_dir: Path to dataset directory
            save_name: Name of the output file
        """
        data_dir = Path(data_dir)
        train_images_dir = data_dir / 'images' / 'train'
        val_images_dir = data_dir / 'images' / 'val'
        train_labels_dir = data_dir / 'labels' / 'train'
        val_labels_dir = data_dir / 'labels' / 'val'
        
        # Analyze image dimensions
        train_dims = self._analyze_image_dimensions(train_images_dir)
        val_dims = self._analyze_image_dimensions(val_images_dir)
        
        # Analyze annotations
        train_annotations = self._analyze_annotations(train_labels_dir)
        val_annotations = self._analyze_annotations(val_labels_dir)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Analysis', fontsize=16, fontweight='bold')
        
        # Image dimensions
        axes[0, 0].hist(train_dims['width'], bins=20, alpha=0.7, 
                       label='Train', color=self.colors['primary'])
        axes[0, 0].hist(val_dims['width'], bins=20, alpha=0.7, 
                       label='Val', color=self.colors['secondary'])
        axes[0, 0].set_title('Image Width Distribution')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(train_dims['height'], bins=20, alpha=0.7, 
                       label='Train', color=self.colors['primary'])
        axes[0, 1].hist(val_dims['height'], bins=20, alpha=0.7, 
                       label='Val', color=self.colors['secondary'])
        axes[0, 1].set_title('Image Height Distribution')
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Aspect ratios
        axes[0, 2].hist(train_dims['aspect_ratio'], bins=20, alpha=0.7, 
                       label='Train', color=self.colors['primary'])
        axes[0, 2].hist(val_dims['aspect_ratio'], bins=20, alpha=0.7, 
                       label='Val', color=self.colors['secondary'])
        axes[0, 2].set_title('Aspect Ratio Distribution')
        axes[0, 2].set_xlabel('Aspect Ratio (W/H)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Bounding box sizes
        if train_annotations['bbox_widths']:
            axes[1, 0].hist(train_annotations['bbox_widths'], bins=20, alpha=0.7, 
                           label='Train', color=self.colors['success'])
        if val_annotations['bbox_widths']:
            axes[1, 0].hist(val_annotations['bbox_widths'], bins=20, alpha=0.7, 
                           label='Val', color=self.colors['warning'])
        axes[1, 0].set_title('Bounding Box Width Distribution')
        axes[1, 0].set_xlabel('Normalized Width')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        if train_annotations['bbox_heights']:
            axes[1, 1].hist(train_annotations['bbox_heights'], bins=20, alpha=0.7, 
                           label='Train', color=self.colors['success'])
        if val_annotations['bbox_heights']:
            axes[1, 1].hist(val_annotations['bbox_heights'], bins=20, alpha=0.7, 
                           label='Val', color=self.colors['warning'])
        axes[1, 1].set_title('Bounding Box Height Distribution')
        axes[1, 1].set_xlabel('Normalized Height')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Dataset summary
        axes[1, 2].axis('off')
        summary_data = [
            ['Dataset Split', 'Images', 'Labels'],
            ['Training', f"{len(list(train_images_dir.glob('*.jpg'))):,}", 
             f"{len(list(train_labels_dir.glob('*.txt'))):,}"],
            ['Validation', f"{len(list(val_images_dir.glob('*.jpg'))):,}", 
             f"{len(list(val_labels_dir.glob('*.txt'))):,}"],
            ['Total', f"{len(list(train_images_dir.glob('*.jpg'))) + len(list(val_images_dir.glob('*.jpg'))):,}", 
             f"{len(list(train_labels_dir.glob('*.txt'))) + len(list(val_labels_dir.glob('*.txt'))):,}"]
        ]
        
        table = axes[1, 2].table(cellText=summary_data[1:], colLabels=summary_data[0],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        axes[1, 2].set_title('Dataset Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Dataset analysis plot saved: {output_path}")
        return output_path
    
    def _analyze_image_dimensions(self, image_dir: Path) -> Dict:
        """Analyze image dimensions in a directory."""
        dimensions = {'width': [], 'height': [], 'aspect_ratio': []}
        
        for img_path in image_dir.glob('*.jpg'):
            try:
                img = Image.open(img_path)
                w, h = img.size
                dimensions['width'].append(w)
                dimensions['height'].append(h)
                dimensions['aspect_ratio'].append(w / h)
            except Exception as e:
                print(f"Warning: Could not process {img_path}: {e}")
        
        return dimensions
    
    def _analyze_annotations(self, label_dir: Path) -> Dict:
        """Analyze annotation characteristics."""
        annotations = {
            'bbox_widths': [],
            'bbox_heights': [],
            'bbox_areas': [],
            'objects_per_image': []
        }
        
        for label_path in label_dir.glob('*.txt'):
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                objects_count = 0
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, x_center, y_center, width, height = map(float, parts[:5])
                        annotations['bbox_widths'].append(width)
                        annotations['bbox_heights'].append(height)
                        annotations['bbox_areas'].append(width * height)
                        objects_count += 1
                
                annotations['objects_per_image'].append(objects_count)
                
            except Exception as e:
                print(f"Warning: Could not process {label_path}: {e}")
        
        return annotations
    
    def generate_all_visualizations(self, data_dir: str, model_path: str = None, 
                                  results_path: str = None):
        """Generate all available visualizations.
        
        Args:
            data_dir: Path to dataset directory
            model_path: Path to trained model (optional)
            results_path: Path to training results (optional)
        """
        print("🎨 Generating comprehensive visualizations...")
        
        generated_files = []
        
        # Dataset analysis
        try:
            output_path = self.plot_dataset_analysis(data_dir)
            generated_files.append(output_path)
        except Exception as e:
            print(f"❌ Failed to generate dataset analysis: {e}")
        
        # Training metrics (if available)
        if results_path and Path(results_path).exists():
            try:
                output_path = self.plot_training_metrics(results_path)
                generated_files.append(output_path)
            except Exception as e:
                print(f"❌ Failed to generate training metrics: {e}")
        
        # Detection results (if model available)
        if model_path and Path(model_path).exists():
            try:
                detector = LicensePlateDetector(model_path=model_path)
                val_images_dir = Path(data_dir) / 'images' / 'val'
                output_path = self.plot_detection_results(detector, val_images_dir)
                generated_files.append(output_path)
                
                # Performance dashboard
                metrics = self._calculate_performance_metrics(detector, val_images_dir)
                output_path = self.plot_performance_dashboard(metrics)
                generated_files.append(output_path)
                
            except Exception as e:
                print(f"❌ Failed to generate detection visualizations: {e}")
        
        print(f"\n✅ Generated {len(generated_files)} visualization files:")
        for file_path in generated_files:
            print(f"   📊 {file_path}")
        
        return generated_files
    
    def _calculate_performance_metrics(self, detector, image_dir: Path) -> Dict:
        """Calculate performance metrics for visualization."""
        metrics = {
            'total_images': 0,
            'images_with_detections': 0,
            'total_detections': 0,
            'confidences': [],
            'processing_times': []
        }
        
        for img_path in list(image_dir.glob('*.jpg'))[:50]:  # Limit for performance
            metrics['total_images'] += 1
            
            img = cv2.imread(str(img_path))
            start_time = datetime.now()
            results = detector.detect(img)
            processing_time = (datetime.now() - start_time).total_seconds()
            metrics['processing_times'].append(processing_time)
            
            if results and len(results) > 0:
                metrics['images_with_detections'] += 1
                metrics['total_detections'] += len(results)
                
                for detection in results:
                    conf = detection.get('confidence', 0)
                    metrics['confidences'].append(conf)
        
        # Calculate derived metrics
        if metrics['confidences']:
            metrics['avg_confidence'] = np.mean(metrics['confidences'])
            metrics['min_confidence'] = np.min(metrics['confidences'])
            metrics['max_confidence'] = np.max(metrics['confidences'])
        else:
            metrics['avg_confidence'] = 0
            metrics['min_confidence'] = 0
            metrics['max_confidence'] = 0
        
        metrics['avg_processing_time'] = np.mean(metrics['processing_times'])
        metrics['detection_rate'] = metrics['images_with_detections'] / metrics['total_images'] if metrics['total_images'] > 0 else 0
        
        return metrics

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Generate license plate detection visualizations')
    parser.add_argument('--data-dir', required=True, help='Path to dataset directory')
    parser.add_argument('--model-path', help='Path to trained model')
    parser.add_argument('--results-path', help='Path to training results directory')
    parser.add_argument('--output-dir', help='Output directory for visualizations')
    parser.add_argument('--type', choices=['all', 'dataset', 'training', 'detection', 'performance'],
                       default='all', help='Type of visualization to generate')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    output_dir = args.output_dir or Path(__file__).parent
    visualizer = LicensePlateVisualizer(output_dir)
    
    # Generate visualizations based on type
    if args.type == 'all':
        visualizer.generate_all_visualizations(args.data_dir, args.model_path, args.results_path)
    elif args.type == 'dataset':
        visualizer.plot_dataset_analysis(args.data_dir)
    elif args.type == 'training' and args.results_path:
        visualizer.plot_training_metrics(args.results_path)
    elif args.type == 'detection' and args.model_path:
        detector = LicensePlateDetector(model_path=args.model_path)
        val_images_dir = Path(args.data_dir) / 'images' / 'val'
        visualizer.plot_detection_results(detector, val_images_dir)
    elif args.type == 'performance' and args.model_path:
        detector = LicensePlateDetector(model_path=args.model_path)
        val_images_dir = Path(args.data_dir) / 'images' / 'val'
        metrics = visualizer._calculate_performance_metrics(detector, val_images_dir)
        visualizer.plot_performance_dashboard(metrics)
    else:
        print("❌ Invalid combination of arguments")
        return
    
    print("\n🎉 Visualization generation completed!")

if __name__ == '__main__':
    main()