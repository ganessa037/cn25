"""Document Classifier Training Module

This module provides comprehensive training functionality for document classification models,
including transfer learning, hyperparameter tuning, and evaluation metrics following the
autocorrect model's organizational patterns.
"""

import os
import json
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision import models

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# Import from our modules
from .data_preparation import DataPreparationConfig, DataPreparationPipeline
from ..models.document_classifier import DocumentClassifier, ClassifierConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for classifier training"""
    
    # Model architecture
    model_name: str = "resnet50"
    num_classes: int = 5
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.5
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Optimization
    optimizer_type: str = "adam"  # adam, sgd, adamw
    scheduler_type: str = "plateau"  # plateau, cosine, step
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_strength: float = 0.5
    
    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Paths
    data_path: str = "model_artifacts/document_parser/datasets"
    output_path: str = "model_artifacts/document_parser/trained_models"
    checkpoint_path: str = "model_artifacts/document_parser/checkpoints"
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging
    log_interval: int = 10
    save_interval: int = 5
    
    # Hyperparameter tuning
    hyperparameter_search: bool = False
    search_iterations: int = 20
    
    # Cross-validation
    use_cross_validation: bool = True
    cv_folds: int = 5
    
    # Random seed
    random_seed: int = 42

class DocumentDataset(Dataset):
    """Custom dataset for document images"""
    
    def __init__(self, image_paths: List[str], labels: List[str], 
                 transform=None, label_encoder=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_encoder = label_encoder
        
        if self.label_encoder is None:
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            self.encoded_labels = self.label_encoder.transform(labels)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.encoded_labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class MixupCutmix:
    """Mixup and CutMix augmentation"""
    
    def __init__(self, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0, 
                 prob: float = 0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
    
    def __call__(self, batch):
        if np.random.rand() > self.prob:
            return batch
        
        images, labels = batch
        batch_size = images.size(0)
        
        if np.random.rand() < 0.5:  # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            index = torch.randperm(batch_size)
            mixed_images = lam * images + (1 - lam) * images[index]
            return mixed_images, labels, labels[index], lam
        else:  # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            index = torch.randperm(batch_size)
            
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
            
            # Adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            
            return images, labels, labels[index], lam
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

class ClassifierTrainer:
    """Main trainer class for document classification"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.setup_directories()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        
        # Training state
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        
        # Set random seeds
        self._set_random_seeds()
    
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def setup_directories(self):
        """Create necessary directories"""
        for path in [self.config.output_path, self.config.checkpoint_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
    
    def create_model(self, num_classes: int) -> nn.Module:
        """Create model based on configuration"""
        logger.info(f"Creating {self.config.model_name} model with {num_classes} classes")
        
        if self.config.model_name == "resnet50":
            model = models.resnet50(pretrained=self.config.pretrained)
            if self.config.freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False
            model.fc = nn.Sequential(
                nn.Dropout(self.config.dropout_rate),
                nn.Linear(model.fc.in_features, num_classes)
            )
        
        elif self.config.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=self.config.pretrained)
            if self.config.freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False
            model.classifier = nn.Sequential(
                nn.Dropout(self.config.dropout_rate),
                nn.Linear(model.classifier[1].in_features, num_classes)
            )
        
        elif self.config.model_name == "vit_b_16":
            model = models.vit_b_16(pretrained=self.config.pretrained)
            if self.config.freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False
            model.heads = nn.Sequential(
                nn.Dropout(self.config.dropout_rate),
                nn.Linear(model.heads.head.in_features, num_classes)
            )
        
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")
        
        return model.to(self.device)
    
    def create_data_loaders(self, dataset_metadata: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for training, validation, and testing"""
        logger.info("Creating data loaders...")
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if self.config.use_augmentation else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        splits = dataset_metadata['splits']
        
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        all_labels = splits['train']['labels'] + splits['val']['labels'] + splits['test']['labels']
        label_encoder.fit(all_labels)
        
        train_dataset = DocumentDataset(
            splits['train']['paths'], splits['train']['labels'],
            transform=train_transform, label_encoder=label_encoder
        )
        
        val_dataset = DocumentDataset(
            splits['val']['paths'], splits['val']['labels'],
            transform=val_transform, label_encoder=label_encoder
        )
        
        test_dataset = DocumentDataset(
            splits['test']['paths'], splits['test']['labels'],
            transform=val_transform, label_encoder=label_encoder
        )
        
        # Create weighted sampler for balanced training
        class_counts = np.bincount(train_dataset.encoded_labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_dataset.encoded_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            sampler=sampler, num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
        
        return train_loader, val_loader, test_loader, label_encoder
    
    def setup_training_components(self, model: nn.Module):
        """Setup optimizer, scheduler, and loss function"""
        # Optimizer
        if self.config.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                model.parameters(), lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(), lr=self.config.learning_rate,
                momentum=self.config.momentum, weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(), lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # Scheduler
        if self.config.scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor, verbose=True
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs
            )
        
        # Loss function with label smoothing
        if self.config.label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta
        )
        
        # Mixup/CutMix
        if self.config.mixup_alpha > 0 or self.config.cutmix_alpha > 0:
            self.mixup_cutmix = MixupCutmix(
                mixup_alpha=self.config.mixup_alpha,
                cutmix_alpha=self.config.cutmix_alpha
            )
        else:
            self.mixup_cutmix = None
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Apply mixup/cutmix if enabled
            if self.mixup_cutmix and np.random.rand() < 0.5:
                mixed_data = self.mixup_cutmix((images, labels))
                if len(mixed_data) == 4:  # Mixup/CutMix applied
                    images, labels_a, labels_b, lam = mixed_data
                    outputs = model(images)
                    loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    outputs = model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % self.config.log_interval == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train_model(self, dataset_metadata: Dict) -> Dict[str, Any]:
        """Main training loop"""
        logger.info("Starting model training...")
        
        # Create data loaders
        train_loader, val_loader, test_loader, label_encoder = self.create_data_loaders(dataset_metadata)
        
        # Create model
        num_classes = len(label_encoder.classes_)
        model = self.create_model(num_classes)
        
        # Setup training components
        self.setup_training_components(model)
        
        # Training loop
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(model, val_loader)
            
            # Update scheduler
            if self.config.scheduler_type == "plateau":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log progress
            epoch_time = time.time() - epoch_start_time
            logger.info(f'Epoch {epoch+1}/{self.config.num_epochs} ({epoch_time:.1f}s) - '
                       f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(model, epoch, val_acc, label_encoder, is_best=True)
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(model, epoch, val_acc, label_encoder, is_best=False)
            
            # Early stopping
            if self.early_stopping(val_loss, model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Final evaluation
        test_results = self.evaluate_model(model, test_loader, label_encoder)
        
        training_time = time.time() - start_time
        
        # Save final results
        results = {
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'best_val_accuracy': best_val_acc,
            'test_results': test_results,
            'training_time': training_time,
            'num_classes': num_classes,
            'class_names': label_encoder.classes_.tolist(),
            'model_path': str(Path(self.config.output_path) / 'best_model.pth'),
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = Path(self.config.output_path) / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training completed in {training_time:.1f}s. Best val accuracy: {best_val_acc:.2f}%")
        
        return results
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, 
                      label_encoder) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        logger.info("Evaluating model on test set...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=label_encoder.classes_,
            output_dict=True
        )
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': all_predictions,
            'true_labels': all_labels,
            'probabilities': all_probabilities
        }
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        return results
    
    def save_checkpoint(self, model: nn.Module, epoch: int, val_acc: float, 
                       label_encoder, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_accuracy': val_acc,
            'config': self.config.__dict__,
            'label_encoder': label_encoder,
            'training_history': self.training_history
        }
        
        if is_best:
            checkpoint_path = Path(self.config.output_path) / 'best_model.pth'
        else:
            checkpoint_path = Path(self.config.checkpoint_path) / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(self.training_history['train_loss'], label='Train Loss')
        ax1.plot(self.training_history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.training_history['train_acc'], label='Train Acc')
        ax2.plot(self.training_history['val_acc'], label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        ax3.plot(self.training_history['learning_rates'])
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Remove the fourth subplot
        ax4.remove()
        
        plt.tight_layout()
        plot_path = Path(self.config.output_path) / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved: {plot_path}")

def main():
    """Main function for standalone execution"""
    # Initialize configuration
    config = TrainingConfig()
    
    # Load dataset metadata
    data_path = Path(config.data_path) / 'metadata' / 'dataset_metadata.json'
    if not data_path.exists():
        logger.error(f"Dataset metadata not found: {data_path}")
        logger.info("Please run data preparation first")
        return
    
    with open(data_path, 'r') as f:
        dataset_metadata = json.load(f)
    
    # Create trainer
    trainer = ClassifierTrainer(config)
    
    # Train model
    results = trainer.train_model(dataset_metadata)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model: {config.model_name}")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"Test accuracy: {results['test_results']['accuracy']:.4f}")
    print(f"Training time: {results['training_time']:.1f}s")
    print(f"Number of classes: {results['num_classes']}")
    print(f"Model saved to: {results['model_path']}")
    print("="*50)

if __name__ == "__main__":
    main()