#!/usr/bin/env python3
"""
Synthetic Document Generator for Document Parser Training

This module generates synthetic document images for training the document parser,
following the organizational patterns established by the autocorrect model.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import random
import string
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

class SyntheticDocumentGenerator:
    """Generate synthetic document images for training data"""
    
    def __init__(self, output_path: str = "./synthetic_documents"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Document templates and configurations
        self.document_types = ['mykad', 'spk', 'passport', 'license']
        self.image_sizes = [(800, 600), (1024, 768), (1200, 900)]
        self.noise_levels = ['low', 'medium', 'high']
        
        # Field configurations for different document types
        self.field_configs = self._load_field_configurations()
        
        # Synthetic data pools
        self.name_pool = self._generate_name_pool()
        self.ic_pool = self._generate_ic_pool()
        self.address_pool = self._generate_address_pool()
        
        # Image processing parameters
        self.blur_kernels = [(3, 3), (5, 5), (7, 7)]
        self.rotation_angles = [-5, -3, -1, 0, 1, 3, 5]
        self.brightness_factors = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        
    def _load_field_configurations(self) -> Dict[str, Dict]:
        """Load field configurations for different document types"""
        return {
            'mykad': {
                'fields': ['name', 'ic_number', 'address', 'birth_date', 'gender', 'religion'],
                'positions': {
                    'name': (100, 150),
                    'ic_number': (100, 200),
                    'address': (100, 250),
                    'birth_date': (100, 300),
                    'gender': (400, 300),
                    'religion': (100, 350)
                },
                'fonts': ['Arial', 'Times New Roman', 'Calibri'],
                'font_sizes': [12, 14, 16, 18]
            },
            'spk': {
                'fields': ['name', 'ic_number', 'address', 'occupation', 'employer'],
                'positions': {
                    'name': (120, 180),
                    'ic_number': (120, 230),
                    'address': (120, 280),
                    'occupation': (120, 330),
                    'employer': (120, 380)
                },
                'fonts': ['Arial', 'Helvetica', 'Verdana'],
                'font_sizes': [11, 13, 15, 17]
            }
        }
    
    def _generate_name_pool(self) -> List[str]:
        """Generate pool of synthetic names"""
        first_names = [
            'Ahmad', 'Ali', 'Aminah', 'Fatimah', 'Hassan', 'Ibrahim', 'Khadijah',
            'Lim', 'Tan', 'Wong', 'Lee', 'Chen', 'Ng', 'Ong', 'Teo',
            'Raj', 'Kumar', 'Devi', 'Priya', 'Siti', 'Nurul', 'Farah'
        ]
        last_names = [
            'Abdullah', 'Rahman', 'Ibrahim', 'Ismail', 'Omar', 'Yusof',
            'Wei Ming', 'Choon Huat', 'Siew Lan', 'Mei Ling', 'Kok Wai',
            'Krishnan', 'Murugan', 'Selvam', 'Kamala', 'Lakshmi'
        ]
        
        names = []
        for first in first_names:
            for last in last_names:
                names.append(f"{first} {last}")
        
        return names
    
    def _generate_ic_pool(self) -> List[str]:
        """Generate pool of synthetic IC numbers"""
        ic_numbers = []
        
        # Generate IC numbers with different birth years
        for year in range(1950, 2005):
            year_str = str(year)[-2:]
            for month in range(1, 13):
                for day in [1, 15]:
                    for state_code in ['01', '02', '03', '04', '05']:
                        for serial in ['001', '002', '003']:
                            ic = f"{year_str}{month:02d}{day:02d}{state_code}{serial}1"
                            ic_numbers.append(ic)
        
        return ic_numbers[:1000]  # Limit to 1000 samples
    
    def _generate_address_pool(self) -> List[str]:
        """Generate pool of synthetic addresses"""
        streets = ['Jalan Bukit Bintang', 'Jalan Ampang', 'Jalan Tun Razak', 'Jalan Raja Chulan']
        areas = ['Kuala Lumpur', 'Petaling Jaya', 'Shah Alam', 'Subang Jaya']
        postcodes = ['50000', '47400', '40000', '47500']
        states = ['Kuala Lumpur', 'Selangor', 'Johor', 'Penang']
        
        addresses = []
        for i in range(100):
            house_no = random.randint(1, 999)
            street = random.choice(streets)
            area = random.choice(areas)
            postcode = random.choice(postcodes)
            state = random.choice(states)
            
            address = f"{house_no}, {street}, {area}, {postcode} {state}"
            addresses.append(address)
        
        return addresses
    
    def generate_synthetic_document(self, doc_type: str, noise_level: str = 'medium') -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate a single synthetic document with annotations"""
        if doc_type not in self.document_types:
            raise ValueError(f"Unsupported document type: {doc_type}")
        
        # Create base image
        width, height = random.choice(self.image_sizes)
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Generate document data
        doc_data = self._generate_document_data(doc_type)
        
        # Draw document fields
        annotations = self._draw_document_fields(draw, doc_type, doc_data, width, height)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Apply noise and distortions
        image_array = self._apply_image_distortions(image_array, noise_level)
        
        return image_array, {
            'document_type': doc_type,
            'fields': doc_data,
            'annotations': annotations,
            'noise_level': noise_level,
            'image_size': (width, height)
        }
    
    def _generate_document_data(self, doc_type: str) -> Dict[str, str]:
        """Generate synthetic data for document fields"""
        data = {}
        
        if doc_type in ['mykad', 'spk']:
            data['name'] = random.choice(self.name_pool)
            data['ic_number'] = random.choice(self.ic_pool)
            data['address'] = random.choice(self.address_pool)
            
            if doc_type == 'mykad':
                data['birth_date'] = self._generate_birth_date()
                data['gender'] = random.choice(['LELAKI', 'PEREMPUAN'])
                data['religion'] = random.choice(['ISLAM', 'KRISTIAN', 'BUDDHA', 'HINDU'])
            elif doc_type == 'spk':
                data['occupation'] = random.choice(['ENGINEER', 'TEACHER', 'DOCTOR', 'CLERK'])
                data['employer'] = random.choice(['ABC SDN BHD', 'XYZ CORPORATION', 'GOVERNMENT'])
        
        return data
    
    def _generate_birth_date(self) -> str:
        """Generate synthetic birth date"""
        start_date = datetime(1950, 1, 1)
        end_date = datetime(2000, 12, 31)
        
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        
        birth_date = start_date + timedelta(days=random_days)
        return birth_date.strftime('%d/%m/%Y')
    
    def _draw_document_fields(self, draw: ImageDraw.Draw, doc_type: str, 
                            doc_data: Dict[str, str], width: int, height: int) -> Dict[str, Dict]:
        """Draw document fields on image and return annotations"""
        config = self.field_configs[doc_type]
        annotations = {}
        
        try:
            # Try to load a font, fallback to default if not available
            font_size = random.choice(config['font_sizes'])
            font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        for field_name, value in doc_data.items():
            if field_name in config['positions']:
                x, y = config['positions'][field_name]
                
                # Add some randomness to positions
                x += random.randint(-10, 10)
                y += random.randint(-5, 5)
                
                # Ensure positions are within image bounds
                x = max(10, min(x, width - 200))
                y = max(10, min(y, height - 50))
                
                # Draw text
                draw.text((x, y), value, fill='black', font=font)
                
                # Calculate bounding box
                bbox = draw.textbbox((x, y), value, font=font)
                
                annotations[field_name] = {
                    'value': value,
                    'bbox': bbox,
                    'position': (x, y)
                }
        
        return annotations
    
    def _apply_image_distortions(self, image: np.ndarray, noise_level: str) -> np.ndarray:
        """Apply various distortions to simulate real-world conditions"""
        distorted = image.copy()
        
        # Noise intensity based on level
        noise_intensity = {'low': 0.1, 'medium': 0.2, 'high': 0.3}[noise_level]
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_intensity * 255, distorted.shape)
        distorted = np.clip(distorted + noise, 0, 255).astype(np.uint8)
        
        # Apply blur
        if random.random() < 0.5:
            kernel = random.choice(self.blur_kernels)
            distorted = cv2.blur(distorted, kernel)
        
        # Adjust brightness
        if random.random() < 0.7:
            factor = random.choice(self.brightness_factors)
            distorted = np.clip(distorted * factor, 0, 255).astype(np.uint8)
        
        # Apply rotation
        if random.random() < 0.3:
            angle = random.choice(self.rotation_angles)
            center = (distorted.shape[1] // 2, distorted.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            distorted = cv2.warpAffine(distorted, rotation_matrix, 
                                     (distorted.shape[1], distorted.shape[0]))
        
        return distorted
    
    def generate_dataset(self, num_samples: int = 1000, 
                        distribution: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Generate a complete synthetic dataset"""
        if distribution is None:
            distribution = {'mykad': 0.6, 'spk': 0.4}
        
        dataset = {
            'images': [],
            'annotations': [],
            'metadata': {
                'total_samples': num_samples,
                'distribution': distribution,
                'generation_date': datetime.now().isoformat(),
                'noise_levels': self.noise_levels
            }
        }
        
        print(f"🔄 Generating {num_samples} synthetic documents...")
        
        for i in range(num_samples):
            # Select document type based on distribution
            doc_type = np.random.choice(
                list(distribution.keys()),
                p=list(distribution.values())
            )
            
            # Select noise level
            noise_level = random.choice(self.noise_levels)
            
            # Generate document
            image, annotation = self.generate_synthetic_document(doc_type, noise_level)
            
            # Save image
            image_filename = f"synthetic_{doc_type}_{i:06d}.png"
            image_path = self.output_path / 'images' / image_filename
            image_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy array back to PIL Image for saving
            pil_image = Image.fromarray(image)
            pil_image.save(image_path)
            
            # Store annotation
            annotation['image_path'] = str(image_path)
            annotation['image_filename'] = image_filename
            
            dataset['images'].append(str(image_path))
            dataset['annotations'].append(annotation)
            
            if (i + 1) % 100 == 0:
                print(f"   Generated {i + 1}/{num_samples} documents")
        
        # Save dataset metadata
        metadata_path = self.output_path / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
        print(f"✅ Dataset generation completed!")
        print(f"   Images saved to: {self.output_path / 'images'}")
        print(f"   Metadata saved to: {metadata_path}")
        
        return dataset
    
    def save_generation_config(self) -> str:
        """Save generation configuration for reproducibility"""
        config = {
            'document_types': self.document_types,
            'image_sizes': self.image_sizes,
            'noise_levels': self.noise_levels,
            'field_configs': self.field_configs,
            'generation_parameters': {
                'blur_kernels': self.blur_kernels,
                'rotation_angles': self.rotation_angles,
                'brightness_factors': self.brightness_factors
            },
            'data_pools': {
                'names_count': len(self.name_pool),
                'ic_numbers_count': len(self.ic_pool),
                'addresses_count': len(self.address_pool)
            }
        }
        
        config_path = self.output_path / 'generation_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(config_path)

def main():
    """Main function for standalone execution"""
    print("🎨 Synthetic Document Generator")
    print("=" * 40)
    
    # Initialize generator
    generator = SyntheticDocumentGenerator("./synthetic_documents")
    
    # Save configuration
    config_path = generator.save_generation_config()
    print(f"📋 Configuration saved to: {config_path}")
    
    # Generate sample dataset
    dataset = generator.generate_dataset(
        num_samples=100,
        distribution={'mykad': 0.7, 'spk': 0.3}
    )
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {dataset['metadata']['total_samples']}")
    print(f"   Document types: {list(dataset['metadata']['distribution'].keys())}")
    print(f"   Output directory: {generator.output_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())