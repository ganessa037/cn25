#!/usr/bin/env python3
"""
Document Parser Results Visualization

Provides visualization tools for document parsing results,
model performance analysis, and debugging assistance.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.document_parser.models.document_models import DocumentType
from src.document_parser.utils import ImageProcessor


class DocumentParserVisualizer:
    """Visualization tools for document parser results."""
    
    def __init__(self, output_dir: str = "visualizations/document_parser/output"):
        """Initialize the visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Color scheme for different document types
        self.colors = {
            'mykad': '#2E86AB',
            'spk': '#A23B72',
            'unknown': '#F18F01',
            'field': '#C73E1D',
            'text': '#4CAF50',
            'confidence_high': '#4CAF50',
            'confidence_medium': '#FF9800',
            'confidence_low': '#F44336'
        }
    
    def visualize_document_analysis(self, image: np.ndarray, 
                                  results: Dict[str, Any], 
                                  save_path: Optional[str] = None) -> np.ndarray:
        """Visualize document analysis results on the image."""
        # Create a copy of the image for annotation
        annotated_image = image.copy()
        
        # Convert BGR to RGB for matplotlib
        if len(image.shape) == 3 and image.shape[2] == 3:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Document Analysis Results - {results.get('document_type', 'Unknown').upper()}", 
                    fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(display_image)
        axes[0, 0].set_title('Original Document')
        axes[0, 0].axis('off')
        
        # Annotated image with field extraction
        annotated_display = self._annotate_fields(display_image.copy(), results)
        axes[0, 1].imshow(annotated_display)
        axes[0, 1].set_title('Field Extraction Results')
        axes[0, 1].axis('off')
        
        # Confidence scores visualization
        self._plot_confidence_scores(axes[1, 0], results)
        
        # Field validation results
        self._plot_validation_results(axes[1, 1], results)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        
        # Convert matplotlib figure to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return buf
    
    def _annotate_fields(self, image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Annotate extracted fields on the image."""
        if 'extracted_fields' not in results:
            return image
        
        # Convert to PIL for better text rendering
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        try:
            # Try to load a font
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        for field_name, field_data in results['extracted_fields'].items():
            if 'coordinates' in field_data and field_data['coordinates']:
                coords = field_data['coordinates']
                confidence = field_data.get('confidence', 0)
                value = field_data.get('value', '')
                
                # Determine color based on confidence
                if confidence >= 0.8:
                    color = self.colors['confidence_high']
                elif confidence >= 0.5:
                    color = self.colors['confidence_medium']
                else:
                    color = self.colors['confidence_low']
                
                # Draw bounding box
                x1, y1, x2, y2 = coords
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # Draw field label
                label = f"{field_name}: {confidence:.2f}"
                label_bbox = draw.textbbox((x1, y1 - 25), label, font=small_font)
                draw.rectangle(label_bbox, fill=color)
                draw.text((x1, y1 - 25), label, fill='white', font=small_font)
                
                # Draw extracted value (truncated if too long)
                if value:
                    display_value = value[:30] + '...' if len(value) > 30 else value
                    draw.text((x1, y2 + 5), display_value, fill=color, font=small_font)
        
        return np.array(pil_image)
    
    def _plot_confidence_scores(self, ax, results: Dict[str, Any]) -> None:
        """Plot confidence scores for extracted fields."""
        if 'extracted_fields' not in results:
            ax.text(0.5, 0.5, 'No fields extracted', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Field Confidence Scores')
            return
        
        fields = []
        confidences = []
        colors = []
        
        for field_name, field_data in results['extracted_fields'].items():
            confidence = field_data.get('confidence', 0)
            fields.append(field_name.replace('_', ' ').title())
            confidences.append(confidence)
            
            # Color based on confidence level
            if confidence >= 0.8:
                colors.append(self.colors['confidence_high'])
            elif confidence >= 0.5:
                colors.append(self.colors['confidence_medium'])
            else:
                colors.append(self.colors['confidence_low'])
        
        if fields:
            bars = ax.barh(fields, confidences, color=colors)
            ax.set_xlabel('Confidence Score')
            ax.set_title('Field Confidence Scores')
            ax.set_xlim(0, 1)
            
            # Add confidence values on bars
            for bar, conf in zip(bars, confidences):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{conf:.3f}', ha='left', va='center', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No fields extracted', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Field Confidence Scores')
    
    def _plot_validation_results(self, ax, results: Dict[str, Any]) -> None:
        """Plot field validation results."""
        if 'extracted_fields' not in results:
            ax.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Field Validation Results')
            return
        
        valid_count = 0
        invalid_count = 0
        field_names = []
        validation_status = []
        
        for field_name, field_data in results['extracted_fields'].items():
            is_valid = field_data.get('is_valid', False)
            field_names.append(field_name.replace('_', ' ').title())
            validation_status.append('Valid' if is_valid else 'Invalid')
            
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
        
        if field_names:
            # Create validation status chart
            status_colors = ['green' if status == 'Valid' else 'red' for status in validation_status]
            
            bars = ax.barh(field_names, [1] * len(field_names), color=status_colors, alpha=0.7)
            ax.set_xlabel('Validation Status')
            ax.set_title(f'Field Validation Results ({valid_count} Valid, {invalid_count} Invalid)')
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            
            # Add status labels
            for bar, status in zip(bars, validation_status):
                ax.text(0.5, bar.get_y() + bar.get_height()/2, 
                       status, ha='center', va='center', fontweight='bold', color='white')
        else:
            ax.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Field Validation Results')
    
    def create_performance_dashboard(self, results_data: List[Dict[str, Any]], 
                                   save_path: Optional[str] = None) -> None:
        """Create a performance dashboard from multiple processing results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Document Parser Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Document type distribution
        self._plot_document_type_distribution(axes[0, 0], results_data)
        
        # Processing time distribution
        self._plot_processing_time_distribution(axes[0, 1], results_data)
        
        # Classification confidence distribution
        self._plot_classification_confidence_distribution(axes[0, 2], results_data)
        
        # Field extraction success rate
        self._plot_field_extraction_success_rate(axes[1, 0], results_data)
        
        # Validation success rate
        self._plot_validation_success_rate(axes[1, 1], results_data)
        
        # Overall performance metrics
        self._plot_overall_performance_metrics(axes[1, 2], results_data)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Performance dashboard saved to {save_path}")
        
        plt.show()
    
    def _plot_document_type_distribution(self, ax, results_data: List[Dict[str, Any]]) -> None:
        """Plot document type distribution."""
        doc_types = [result.get('document_type', 'unknown') for result in results_data]
        type_counts = {}
        for doc_type in doc_types:
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        colors = [self.colors.get(doc_type, self.colors['unknown']) for doc_type in type_counts.keys()]
        
        wedges, texts, autotexts = ax.pie(type_counts.values(), labels=type_counts.keys(), 
                                         autopct='%1.1f%%', colors=colors)
        ax.set_title('Document Type Distribution')
    
    def _plot_processing_time_distribution(self, ax, results_data: List[Dict[str, Any]]) -> None:
        """Plot processing time distribution."""
        processing_times = [result.get('processing_time', 0) for result in results_data if 'processing_time' in result]
        
        if processing_times:
            ax.hist(processing_times, bins=20, alpha=0.7, color=self.colors['mykad'])
            ax.set_xlabel('Processing Time (seconds)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Processing Time Distribution\n(Avg: {np.mean(processing_times):.2f}s)')
            ax.axvline(np.mean(processing_times), color='red', linestyle='--', label='Average')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No processing time data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Processing Time Distribution')
    
    def _plot_classification_confidence_distribution(self, ax, results_data: List[Dict[str, Any]]) -> None:
        """Plot classification confidence distribution."""
        confidences = [result.get('classification_confidence', 0) for result in results_data 
                      if 'classification_confidence' in result]
        
        if confidences:
            ax.hist(confidences, bins=20, alpha=0.7, color=self.colors['spk'])
            ax.set_xlabel('Classification Confidence')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Classification Confidence Distribution\n(Avg: {np.mean(confidences):.3f})')
            ax.axvline(np.mean(confidences), color='red', linestyle='--', label='Average')
            ax.set_xlim(0, 1)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No confidence data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Classification Confidence Distribution')
    
    def _plot_field_extraction_success_rate(self, ax, results_data: List[Dict[str, Any]]) -> None:
        """Plot field extraction success rate by document type."""
        doc_type_stats = {}
        
        for result in results_data:
            doc_type = result.get('document_type', 'unknown')
            extracted_fields = result.get('extracted_fields', {})
            
            if doc_type not in doc_type_stats:
                doc_type_stats[doc_type] = {'total': 0, 'with_fields': 0}
            
            doc_type_stats[doc_type]['total'] += 1
            if extracted_fields:
                doc_type_stats[doc_type]['with_fields'] += 1
        
        if doc_type_stats:
            doc_types = list(doc_type_stats.keys())
            success_rates = [stats['with_fields'] / stats['total'] * 100 
                           for stats in doc_type_stats.values()]
            
            colors = [self.colors.get(doc_type, self.colors['unknown']) for doc_type in doc_types]
            
            bars = ax.bar(doc_types, success_rates, color=colors, alpha=0.7)
            ax.set_ylabel('Success Rate (%)')
            ax.set_title('Field Extraction Success Rate')
            ax.set_ylim(0, 100)
            
            # Add percentage labels on bars
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{rate:.1f}%', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No extraction data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Field Extraction Success Rate')
    
    def _plot_validation_success_rate(self, ax, results_data: List[Dict[str, Any]]) -> None:
        """Plot validation success rate."""
        validation_stats = {'passed': 0, 'failed': 0}
        
        for result in results_data:
            if 'validation_passed' in result:
                if result['validation_passed']:
                    validation_stats['passed'] += 1
                else:
                    validation_stats['failed'] += 1
        
        if sum(validation_stats.values()) > 0:
            labels = ['Passed', 'Failed']
            sizes = [validation_stats['passed'], validation_stats['failed']]
            colors = [self.colors['confidence_high'], self.colors['confidence_low']]
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
            ax.set_title('Validation Success Rate')
        else:
            ax.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validation Success Rate')
    
    def _plot_overall_performance_metrics(self, ax, results_data: List[Dict[str, Any]]) -> None:
        """Plot overall performance metrics."""
        metrics = {
            'Total Documents': len(results_data),
            'Avg Processing Time': 0,
            'Avg Classification Conf': 0,
            'Field Extraction Rate': 0,
            'Validation Pass Rate': 0
        }
        
        if results_data:
            # Calculate averages
            processing_times = [r.get('processing_time', 0) for r in results_data if 'processing_time' in r]
            if processing_times:
                metrics['Avg Processing Time'] = np.mean(processing_times)
            
            confidences = [r.get('classification_confidence', 0) for r in results_data if 'classification_confidence' in r]
            if confidences:
                metrics['Avg Classification Conf'] = np.mean(confidences)
            
            # Field extraction rate
            with_fields = sum(1 for r in results_data if r.get('extracted_fields'))
            metrics['Field Extraction Rate'] = with_fields / len(results_data) * 100
            
            # Validation pass rate
            passed_validation = sum(1 for r in results_data if r.get('validation_passed', False))
            metrics['Validation Pass Rate'] = passed_validation / len(results_data) * 100
        
        # Create text display of metrics
        ax.axis('off')
        ax.set_title('Overall Performance Metrics', fontweight='bold')
        
        y_pos = 0.8
        for metric, value in metrics.items():
            if 'Time' in metric:
                text = f"{metric}: {value:.2f}s"
            elif 'Rate' in metric or 'Conf' in metric:
                if 'Rate' in metric:
                    text = f"{metric}: {value:.1f}%"
                else:
                    text = f"{metric}: {value:.3f}"
            else:
                text = f"{metric}: {int(value)}"
            
            ax.text(0.1, y_pos, text, transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            y_pos -= 0.15
    
    def compare_models(self, model_results: Dict[str, List[Dict[str, Any]]], 
                      save_path: Optional[str] = None) -> None:
        """Compare performance between different models."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Processing time comparison
        self._compare_processing_times(axes[0, 0], model_results)
        
        # Classification confidence comparison
        self._compare_classification_confidence(axes[0, 1], model_results)
        
        # Field extraction success rate comparison
        self._compare_field_extraction_rates(axes[1, 0], model_results)
        
        # Validation success rate comparison
        self._compare_validation_rates(axes[1, 1], model_results)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Model comparison saved to {save_path}")
        
        plt.show()
    
    def _compare_processing_times(self, ax, model_results: Dict[str, List[Dict[str, Any]]]) -> None:
        """Compare processing times between models."""
        model_names = []
        avg_times = []
        
        for model_name, results in model_results.items():
            processing_times = [r.get('processing_time', 0) for r in results if 'processing_time' in r]
            if processing_times:
                model_names.append(model_name)
                avg_times.append(np.mean(processing_times))
        
        if model_names:
            bars = ax.bar(model_names, avg_times, color=sns.color_palette("husl", len(model_names)))
            ax.set_ylabel('Average Processing Time (seconds)')
            ax.set_title('Processing Time Comparison')
            
            # Add value labels on bars
            for bar, time in zip(bars, avg_times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{time:.2f}s', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No processing time data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Processing Time Comparison')
    
    def _compare_classification_confidence(self, ax, model_results: Dict[str, List[Dict[str, Any]]]) -> None:
        """Compare classification confidence between models."""
        model_names = []
        avg_confidences = []
        
        for model_name, results in model_results.items():
            confidences = [r.get('classification_confidence', 0) for r in results if 'classification_confidence' in r]
            if confidences:
                model_names.append(model_name)
                avg_confidences.append(np.mean(confidences))
        
        if model_names:
            bars = ax.bar(model_names, avg_confidences, color=sns.color_palette("husl", len(model_names)))
            ax.set_ylabel('Average Classification Confidence')
            ax.set_title('Classification Confidence Comparison')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, conf in zip(bars, avg_confidences):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{conf:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No confidence data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Classification Confidence Comparison')
    
    def _compare_field_extraction_rates(self, ax, model_results: Dict[str, List[Dict[str, Any]]]) -> None:
        """Compare field extraction rates between models."""
        model_names = []
        extraction_rates = []
        
        for model_name, results in model_results.items():
            total = len(results)
            with_fields = sum(1 for r in results if r.get('extracted_fields'))
            if total > 0:
                model_names.append(model_name)
                extraction_rates.append(with_fields / total * 100)
        
        if model_names:
            bars = ax.bar(model_names, extraction_rates, color=sns.color_palette("husl", len(model_names)))
            ax.set_ylabel('Field Extraction Rate (%)')
            ax.set_title('Field Extraction Rate Comparison')
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, rate in zip(bars, extraction_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{rate:.1f}%', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No extraction data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Field Extraction Rate Comparison')
    
    def _compare_validation_rates(self, ax, model_results: Dict[str, List[Dict[str, Any]]]) -> None:
        """Compare validation rates between models."""
        model_names = []
        validation_rates = []
        
        for model_name, results in model_results.items():
            total = len(results)
            passed = sum(1 for r in results if r.get('validation_passed', False))
            if total > 0:
                model_names.append(model_name)
                validation_rates.append(passed / total * 100)
        
        if model_names:
            bars = ax.bar(model_names, validation_rates, color=sns.color_palette("husl", len(model_names)))
            ax.set_ylabel('Validation Pass Rate (%)')
            ax.set_title('Validation Pass Rate Comparison')
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, rate in zip(bars, validation_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{rate:.1f}%', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validation Pass Rate Comparison')


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Visualize Document Parser Results')
    parser.add_argument('--results-file', type=str, help='Path to results JSON file')
    parser.add_argument('--image-file', type=str, help='Path to image file')
    parser.add_argument('--output-dir', type=str, default='visualizations/document_parser/output',
                       help='Output directory for visualizations')
    parser.add_argument('--dashboard', action='store_true', help='Create performance dashboard')
    parser.add_argument('--compare', nargs='+', help='Compare multiple result files')
    
    args = parser.parse_args()
    
    visualizer = DocumentParserVisualizer(args.output_dir)
    
    if args.results_file and args.image_file:
        # Single document visualization
        with open(args.results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        image = cv2.imread(args.image_file)
        if image is None:
            print(f"❌ Could not load image: {args.image_file}")
            return 1
        
        output_path = visualizer.output_dir / f"analysis_{Path(args.image_file).stem}.png"
        visualizer.visualize_document_analysis(image, results, str(output_path))
        
    elif args.dashboard and args.results_file:
        # Performance dashboard
        with open(args.results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        if not isinstance(results_data, list):
            results_data = [results_data]
        
        output_path = visualizer.output_dir / "performance_dashboard.png"
        visualizer.create_performance_dashboard(results_data, str(output_path))
        
    elif args.compare:
        # Model comparison
        model_results = {}
        for result_file in args.compare:
            model_name = Path(result_file).stem
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            if not isinstance(results, list):
                results = [results]
            
            model_results[model_name] = results
        
        output_path = visualizer.output_dir / "model_comparison.png"
        visualizer.compare_models(model_results, str(output_path))
        
    else:
        print("Please provide either:")
        print("  --results-file and --image-file for single document analysis")
        print("  --dashboard and --results-file for performance dashboard")
        print("  --compare with multiple result files for model comparison")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())