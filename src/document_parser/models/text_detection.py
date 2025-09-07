#!/usr/bin/env python3
"""
Text Detection Models for Document Parser

This module provides text detection capabilities using EAST, CRAFT, and DBNet
architectures for locating text regions in documents, following the
organizational patterns established by the autocorrect model.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import math
from collections import OrderedDict

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

@dataclass
class TextDetectionConfig:
    """Configuration for text detection models"""
    model_type: str = "dbnet"  # east, craft, dbnet
    input_size: Tuple[int, int] = (640, 640)
    score_threshold: float = 0.7
    nms_threshold: float = 0.4
    polygon_threshold: float = 0.3
    max_candidates: int = 1000
    unclip_ratio: float = 1.5
    learning_rate: float = 0.001
    batch_size: int = 8
    num_epochs: int = 100
    device: str = "auto"
    mixed_precision: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextDetectionConfig':
        return cls(**data)

class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class EASTModel(nn.Module):
    """EAST (Efficient and Accurate Scene Text) Detection Model"""
    
    def __init__(self, config: TextDetectionConfig):
        super(EASTModel, self).__init__()
        self.config = config
        
        # Feature extractor (ResNet-like backbone)
        self.conv1 = ConvBNReLU(3, 64, 3, 1, 1)
        self.conv2 = ConvBNReLU(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = ConvBNReLU(64, 128, 3, 1, 1)
        self.conv4 = ConvBNReLU(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = ConvBNReLU(128, 256, 3, 1, 1)
        self.conv6 = ConvBNReLU(256, 256, 3, 1, 1)
        self.conv7 = ConvBNReLU(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv8 = ConvBNReLU(256, 512, 3, 1, 1)
        self.conv9 = ConvBNReLU(512, 512, 3, 1, 1)
        self.conv10 = ConvBNReLU(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv11 = ConvBNReLU(512, 512, 3, 1, 1)
        self.conv12 = ConvBNReLU(512, 512, 3, 1, 1)
        self.conv13 = ConvBNReLU(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # Feature merging branch
        self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_merge1 = ConvBNReLU(1024, 512, 1, 1, 0)
        self.conv_merge1_2 = ConvBNReLU(512, 512, 3, 1, 1)
        
        self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_merge2 = ConvBNReLU(768, 256, 1, 1, 0)
        self.conv_merge2_2 = ConvBNReLU(256, 256, 3, 1, 1)
        
        self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_merge3 = ConvBNReLU(384, 128, 1, 1, 0)
        self.conv_merge3_2 = ConvBNReLU(128, 128, 3, 1, 1)
        
        self.unpool4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_merge4 = ConvBNReLU(192, 64, 1, 1, 0)
        self.conv_merge4_2 = ConvBNReLU(64, 64, 3, 1, 1)
        
        # Output layers
        self.conv_output = ConvBNReLU(64, 32, 3, 1, 1)
        self.score_map = nn.Conv2d(32, 1, 1, 1, 0)  # Text/non-text score
        self.geo_map = nn.Conv2d(32, 4, 1, 1, 0)    # Geometry (RBOX)
        self.angle_map = nn.Conv2d(32, 1, 1, 1, 0)  # Rotation angle
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Feature extraction
        f1 = self.pool1(self.conv2(self.conv1(x)))
        f2 = self.pool2(self.conv4(self.conv3(f1)))
        f3 = self.pool3(self.conv7(self.conv6(self.conv5(f2))))
        f4 = self.pool4(self.conv10(self.conv9(self.conv8(f3))))
        f5 = self.pool5(self.conv13(self.conv12(self.conv11(f4))))
        
        # Feature merging
        g1 = self.unpool1(f5)
        c1 = torch.cat([g1, f4], dim=1)
        h1 = self.conv_merge1_2(self.conv_merge1(c1))
        
        g2 = self.unpool2(h1)
        c2 = torch.cat([g2, f3], dim=1)
        h2 = self.conv_merge2_2(self.conv_merge2(c2))
        
        g3 = self.unpool3(h2)
        c3 = torch.cat([g3, f2], dim=1)
        h3 = self.conv_merge3_2(self.conv_merge3(c3))
        
        g4 = self.unpool4(h3)
        c4 = torch.cat([g4, f1], dim=1)
        h4 = self.conv_merge4_2(self.conv_merge4(c4))
        
        # Output
        output_feature = self.conv_output(h4)
        score_map = torch.sigmoid(self.score_map(output_feature))
        geo_map = torch.sigmoid(self.geo_map(output_feature)) * 512
        angle_map = (torch.sigmoid(self.angle_map(output_feature)) - 0.5) * math.pi
        
        return {
            'score_map': score_map,
            'geo_map': geo_map,
            'angle_map': angle_map
        }

class CRAFTModel(nn.Module):
    """CRAFT (Character Region Awareness for Text) Detection Model"""
    
    def __init__(self, config: TextDetectionConfig):
        super(CRAFTModel, self).__init__()
        self.config = config
        
        # VGG-like backbone
        self.basenet = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # U-Net decoder
        self.upconv1 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        
        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        
        self.upconv5 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(96, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        
        # Output layers
        self.text_map = nn.Conv2d(32, 1, 1, 1, 0)  # Text region map
        self.link_map = nn.Conv2d(32, 1, 1, 1, 0)  # Link map
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features at different scales
        features = []
        for i, layer in enumerate(self.basenet):
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                features.append(x)
        
        # Decoder with skip connections
        x = self.upconv1(x)
        x = torch.cat([x, features[3]], dim=1)
        x = self.conv1(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, features[2]], dim=1)
        x = self.conv2(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, features[1]], dim=1)
        x = self.conv3(x)
        
        x = self.upconv4(x)
        x = torch.cat([x, features[0]], dim=1)
        x = self.conv4(x)
        
        x = self.upconv5(x)
        # Concatenate with original input (downsampled)
        input_down = F.interpolate(features[0], size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, input_down], dim=1)
        x = self.conv5(x)
        
        # Output maps
        text_map = torch.sigmoid(self.text_map(x))
        link_map = torch.sigmoid(self.link_map(x))
        
        return {
            'text_map': text_map,
            'link_map': link_map
        }

class DBNetModel(nn.Module):
    """DBNet (Differentiable Binarization) Text Detection Model"""
    
    def __init__(self, config: TextDetectionConfig):
        super(DBNetModel, self).__init__()
        self.config = config
        
        # ResNet-18 backbone
        import torchvision.models as models
        resnet = models.resnet18(pretrained=True)
        
        # Extract feature layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # Feature Pyramid Network (FPN)
        self.fpn_conv1 = nn.Conv2d(512, 256, 1, 1, 0)
        self.fpn_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.fpn_conv3 = nn.Conv2d(128, 256, 1, 1, 0)
        self.fpn_conv4 = nn.Conv2d(64, 256, 1, 1, 0)
        
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth4 = nn.Conv2d(256, 256, 3, 1, 1)
        
        # Detection head
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.det_conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.det_conv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid()
        )
        
        # Threshold head (for differentiable binarization)
        self.thresh_conv1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.thresh_conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.thresh_conv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid()
        )
    
    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Upsample and add two feature maps"""
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Backbone feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        c1 = self.layer1(x)  # 1/4
        c2 = self.layer2(c1)  # 1/8
        c3 = self.layer3(c2)  # 1/16
        c4 = self.layer4(c3)  # 1/32
        
        # FPN
        p4 = self.fpn_conv1(c4)
        p3 = self._upsample_add(p4, self.fpn_conv2(c3))
        p2 = self._upsample_add(p3, self.fpn_conv3(c2))
        p1 = self._upsample_add(p2, self.fpn_conv4(c1))
        
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        p1 = self.smooth4(p1)
        
        # Concatenate all pyramid features
        _, _, h, w = p1.size()
        p2 = F.interpolate(p2, size=(h, w), mode='bilinear', align_corners=True)
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=True)
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=True)
        
        fuse = torch.cat([p1, p2, p3, p4], dim=1)  # 1024 channels
        
        # Reduce channels
        fuse = nn.Conv2d(1024, 256, 1, 1, 0).to(fuse.device)(fuse)
        
        # Detection head
        det = self.det_conv1(fuse)
        det = self.det_conv2(det)
        prob_map = self.det_conv3(det)
        
        # Threshold head
        thresh = self.thresh_conv1(fuse)
        thresh = self.thresh_conv2(thresh)
        thresh_map = self.thresh_conv3(thresh)
        
        # Differentiable binarization
        if self.training:
            binary_map = self._differentiable_binarization(prob_map, thresh_map)
        else:
            binary_map = (prob_map > thresh_map).float()
        
        return {
            'prob_map': prob_map,
            'thresh_map': thresh_map,
            'binary_map': binary_map
        }
    
    def _differentiable_binarization(self, prob_map: torch.Tensor, thresh_map: torch.Tensor, k: float = 50.0) -> torch.Tensor:
        """Differentiable binarization function"""
        return torch.sigmoid(k * (prob_map - thresh_map))

class TextDetector:
    """Main text detection class with multiple model support"""
    
    def __init__(self, config: TextDetectionConfig = None):
        self.config = config or TextDetectionConfig()
        
        # Setup device
        if self.config.device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize transforms
        self.transform = self._create_transform()
        
        self.logger.info(f"Text detector initialized with {self.config.model_type} on {self.device}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('TextDetector')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration"""
        model_type = self.config.model_type.lower()
        
        if model_type == "east":
            return EASTModel(self.config)
        elif model_type == "craft":
            return CRAFTModel(self.config)
        elif model_type == "dbnet":
            return DBNetModel(self.config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _create_transform(self) -> transforms.Compose:
        """Create image transforms"""
        return transforms.Compose([
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess image for detection"""
        # Load image
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Convert to PIL for transforms
        pil_image = Image.fromarray(image)
        
        # Apply transforms
        tensor_image = self.transform(pil_image).unsqueeze(0)
        
        return tensor_image.to(self.device), original_size
    
    def detect(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Detect text regions in image"""
        self.model.eval()
        
        # Preprocess
        input_tensor, original_size = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_tensor)
            else:
                outputs = self.model(input_tensor)
        
        # Post-process based on model type
        if self.config.model_type.lower() == "east":
            boxes = self._postprocess_east(outputs, original_size)
        elif self.config.model_type.lower() == "craft":
            boxes = self._postprocess_craft(outputs, original_size)
        elif self.config.model_type.lower() == "dbnet":
            boxes = self._postprocess_dbnet(outputs, original_size)
        else:
            boxes = []
        
        result = {
            'boxes': boxes,
            'num_detections': len(boxes),
            'model_type': self.config.model_type,
            'detection_date': datetime.now().isoformat()
        }
        
        return result
    
    def _postprocess_east(self, outputs: Dict[str, torch.Tensor], original_size: Tuple[int, int]) -> List[List[int]]:
        """Post-process EAST model outputs"""
        score_map = outputs['score_map'].cpu().numpy()[0, 0]
        geo_map = outputs['geo_map'].cpu().numpy()[0]
        angle_map = outputs['angle_map'].cpu().numpy()[0, 0]
        
        # Find text regions
        boxes = []
        h, w = score_map.shape
        
        for y in range(h):
            for x in range(w):
                if score_map[y, x] > self.config.score_threshold:
                    # Extract geometry
                    d1, d2, d3, d4 = geo_map[:, y, x]
                    angle = angle_map[y, x]
                    
                    # Calculate box coordinates
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    
                    # Scale to original image size
                    scale_x = original_size[0] / w
                    scale_y = original_size[1] / h
                    
                    x_scaled = x * scale_x
                    y_scaled = y * scale_y
                    
                    # Simple box approximation
                    x1 = int(x_scaled - d1 * scale_x)
                    y1 = int(y_scaled - d2 * scale_y)
                    x2 = int(x_scaled + d3 * scale_x)
                    y2 = int(y_scaled + d4 * scale_y)
                    
                    boxes.append([x1, y1, x2, y2])
        
        return self._apply_nms(boxes)
    
    def _postprocess_craft(self, outputs: Dict[str, torch.Tensor], original_size: Tuple[int, int]) -> List[List[int]]:
        """Post-process CRAFT model outputs"""
        text_map = outputs['text_map'].cpu().numpy()[0, 0]
        link_map = outputs['link_map'].cpu().numpy()[0, 0]
        
        # Combine text and link maps
        combined_map = np.maximum(text_map, link_map)
        
        # Threshold
        binary_map = (combined_map > self.config.score_threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        h, w = binary_map.shape
        scale_x = original_size[0] / w
        scale_y = original_size[1] / h
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small regions
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # Scale to original image size
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w_box) * scale_x)
                y2 = int((y + h_box) * scale_y)
                
                boxes.append([x1, y1, x2, y2])
        
        return boxes
    
    def _postprocess_dbnet(self, outputs: Dict[str, torch.Tensor], original_size: Tuple[int, int]) -> List[List[int]]:
        """Post-process DBNet model outputs"""
        binary_map = outputs['binary_map'].cpu().numpy()[0, 0]
        
        # Threshold
        binary_map = (binary_map > self.config.score_threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        h, w = binary_map.shape
        scale_x = original_size[0] / w
        scale_y = original_size[1] / h
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small regions
                # Get minimum area rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Get bounding rectangle
                x, y, w_box, h_box = cv2.boundingRect(box)
                
                # Scale to original image size
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w_box) * scale_x)
                y2 = int((y + h_box) * scale_y)
                
                boxes.append([x1, y1, x2, y2])
        
        return boxes
    
    def _apply_nms(self, boxes: List[List[int]]) -> List[List[int]]:
        """Apply Non-Maximum Suppression"""
        if not boxes:
            return []
        
        boxes = np.array(boxes, dtype=np.float32)
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by bottom-right y-coordinate
        indices = np.argsort(boxes[:, 3])
        
        keep = []
        while len(indices) > 0:
            # Pick the last index
            last = len(indices) - 1
            i = indices[last]
            keep.append(i)
            
            # Find the largest coordinates for intersection rectangle
            xx1 = np.maximum(boxes[i, 0], boxes[indices[:last], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[indices[:last], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[indices[:last], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[indices[:last], 3])
            
            # Compute width and height of intersection rectangle
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            # Compute intersection over union
            intersection = w * h
            union = areas[i] + areas[indices[:last]] - intersection
            overlap = intersection / union
            
            # Delete all indices from the index list that have IoU greater than threshold
            indices = np.delete(indices, np.concatenate(([last], np.where(overlap > self.config.nms_threshold)[0])))
        
        return boxes[keep].astype(int).tolist()
    
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Update config if available
        if 'config' in checkpoint:
            self.config = TextDetectionConfig.from_dict(checkpoint['config'])
            # Recreate model with loaded config
            self.model = self._create_model()
            self.model.to(self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"Model loaded from {checkpoint_path}")
    
    def save_model(self, save_path: str):
        """Save current model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'save_date': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Model saved to {save_path}")

def main():
    """Main function for standalone execution"""
    print("📝 Text Detection Models")
    print("=" * 40)
    
    # Initialize detector
    config = TextDetectionConfig(
        model_type="dbnet",
        input_size=(640, 640),
        score_threshold=0.7
    )
    
    detector = TextDetector(config)
    
    # Display configuration
    print(f"\n⚙️ Configuration:")
    print(f"   Model: {config.model_type}")
    print(f"   Input size: {config.input_size}")
    print(f"   Score threshold: {config.score_threshold}")
    print(f"   Device: {detector.device}")
    
    # Example usage
    print("\n📋 Usage Examples:")
    print("1. result = detector.detect('/path/to/image.jpg')")
    print("2. detector.save_model('/path/to/model.pth')")
    print("3. detector.load_model('/path/to/model.pth')")
    
    return 0

if __name__ == "__main__":
    exit(main())