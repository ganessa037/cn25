#!/usr/bin/env python3
"""
Model Serving Infrastructure for Document Parser

This module provides comprehensive model serving capabilities with load balancing,
caching, and monitoring for document parser models, following the organizational
patterns established by the autocorrect model.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import yaml
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict, deque
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue, Empty
import uuid
from functools import wraps
import gc

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from flask import Flask, request, jsonify, Response
    from werkzeug.serving import WSGIRequestHandler
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import cv2
    import PIL.Image as Image
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

@dataclass
class ServingConfig:
    """Configuration for model serving"""
    # Server settings
    host: str = '0.0.0.0'
    port: int = 8080
    debug: bool = False
    threaded: bool = True
    
    # Model settings
    model_type: str = 'document_classifier'  # 'document_classifier', 'text_detection', 'ocr', 'information_extraction'
    model_path: str = ''
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'
    batch_size: int = 1
    max_batch_delay: float = 0.1  # seconds
    
    # Load balancing
    num_workers: int = 1
    worker_type: str = 'thread'  # 'thread', 'process'
    load_balancing_strategy: str = 'round_robin'  # 'round_robin', 'least_loaded', 'random'
    
    # Caching
    enable_cache: bool = True
    cache_type: str = 'memory'  # 'memory', 'redis'
    cache_ttl: int = 3600  # seconds
    max_cache_size: int = 1000
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_db: int = 0
    
    # Performance monitoring
    enable_monitoring: bool = True
    metrics_window: int = 300  # seconds
    log_requests: bool = True
    
    # Request limits
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: float = 30.0  # seconds
    rate_limit: int = 100  # requests per minute
    
    # Health check
    health_check_interval: int = 60  # seconds
    
    def __post_init__(self):
        if self.device == 'auto':
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    self.device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = 'mps'
                else:
                    self.device = 'cpu'
            else:
                self.device = 'cpu'
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServingConfig':
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ServingConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

@dataclass
class PredictionRequest:
    """Prediction request container"""
    request_id: str
    data: Any
    timestamp: datetime
    client_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'timestamp': self.timestamp.isoformat(),
            'client_info': self.client_info
        }

@dataclass
class PredictionResponse:
    """Prediction response container"""
    request_id: str
    predictions: Any
    confidence: Optional[float]
    processing_time: float
    model_version: str
    timestamp: datetime
    cached: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'predictions': self.predictions,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'model_version': self.model_version,
            'timestamp': self.timestamp.isoformat(),
            'cached': self.cached
        }

class CacheManager:
    """Cache manager for prediction results"""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        
        if config.cache_type == 'redis' and REDIS_AVAILABLE:
            self.redis_client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
                decode_responses=True
            )
            self.cache_type = 'redis'
        else:
            self.memory_cache = {}
            self.cache_timestamps = {}
            self.cache_type = 'memory'
        
        self.logger.info(f"Cache manager initialized with {self.cache_type} backend")
    
    def _generate_cache_key(self, data: Any) -> str:
        """Generate cache key from input data"""
        if isinstance(data, (str, bytes)):
            content = data
        elif isinstance(data, np.ndarray):
            content = data.tobytes()
        else:
            content = str(data)
        
        return hashlib.md5(content.encode() if isinstance(content, str) else content).hexdigest()
    
    def get(self, data: Any) -> Optional[PredictionResponse]:
        """Get cached prediction result"""
        if not self.config.enable_cache:
            return None
        
        cache_key = self._generate_cache_key(data)
        
        try:
            if self.cache_type == 'redis':
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result_dict = json.loads(cached_data)
                    result_dict['timestamp'] = datetime.fromisoformat(result_dict['timestamp'])
                    result_dict['cached'] = True
                    return PredictionResponse(**result_dict)
            else:
                if cache_key in self.memory_cache:
                    # Check TTL
                    if time.time() - self.cache_timestamps[cache_key] < self.config.cache_ttl:
                        result = self.memory_cache[cache_key]
                        result.cached = True
                        return result
                    else:
                        # Expired
                        del self.memory_cache[cache_key]
                        del self.cache_timestamps[cache_key]
        
        except Exception as e:
            self.logger.warning(f"Cache get error: {e}")
        
        return None
    
    def set(self, data: Any, response: PredictionResponse):
        """Cache prediction result"""
        if not self.config.enable_cache:
            return
        
        cache_key = self._generate_cache_key(data)
        
        try:
            if self.cache_type == 'redis':
                response_dict = response.to_dict()
                self.redis_client.setex(
                    cache_key,
                    self.config.cache_ttl,
                    json.dumps(response_dict)
                )
            else:
                # Memory cache size management
                if len(self.memory_cache) >= self.config.max_cache_size:
                    # Remove oldest entry
                    oldest_key = min(self.cache_timestamps.keys(), 
                                   key=lambda k: self.cache_timestamps[k])
                    del self.memory_cache[oldest_key]
                    del self.cache_timestamps[oldest_key]
                
                self.memory_cache[cache_key] = response
                self.cache_timestamps[cache_key] = time.time()
        
        except Exception as e:
            self.logger.warning(f"Cache set error: {e}")
    
    def clear(self):
        """Clear all cached data"""
        try:
            if self.cache_type == 'redis':
                self.redis_client.flushdb()
            else:
                self.memory_cache.clear()
                self.cache_timestamps.clear()
            
            self.logger.info("Cache cleared")
        
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")

class MetricsCollector:
    """Metrics collection and monitoring"""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        
        # Metrics storage
        self.request_times = deque(maxlen=10000)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Time windows
        self.window_start = time.time()
        
        self.logger.info("Metrics collector initialized")
    
    def record_request(self, processing_time: float, cached: bool = False, error: bool = False):
        """Record request metrics"""
        current_time = time.time()
        
        self.request_times.append((current_time, processing_time))
        
        # Count requests per minute
        minute_key = int(current_time // 60)
        self.request_counts[minute_key] += 1
        
        if error:
            self.error_counts[minute_key] += 1
        
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Clean old data
        self._cleanup_old_data(current_time)
    
    def _cleanup_old_data(self, current_time: float):
        """Remove old metrics data"""
        cutoff_time = current_time - self.config.metrics_window
        
        # Clean request times
        while self.request_times and self.request_times[0][0] < cutoff_time:
            self.request_times.popleft()
        
        # Clean request counts
        cutoff_minute = int(cutoff_time // 60)
        old_keys = [k for k in self.request_counts.keys() if k < cutoff_minute]
        for key in old_keys:
            del self.request_counts[key]
            if key in self.error_counts:
                del self.error_counts[key]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        current_time = time.time()
        self._cleanup_old_data(current_time)
        
        # Calculate metrics
        total_requests = sum(self.request_counts.values())
        total_errors = sum(self.error_counts.values())
        
        processing_times = [pt for _, pt in self.request_times]
        
        metrics = {
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate': total_errors / max(total_requests, 1),
            'cache_hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'p95_processing_time': np.percentile(processing_times, 95) if processing_times else 0,
            'p99_processing_time': np.percentile(processing_times, 99) if processing_times else 0,
            'requests_per_minute': total_requests / max(self.config.metrics_window / 60, 1),
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics

class ModelWorker:
    """Individual model worker for processing requests"""
    
    def __init__(self, worker_id: int, config: ServingConfig):
        self.worker_id = worker_id
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}_{worker_id}')
        
        self.model = None
        self.model_version = "1.0.0"
        self.is_healthy = False
        self.request_count = 0
        self.last_request_time = time.time()
        
        # Load model
        self._load_model()
        
        self.logger.info(f"Worker {worker_id} initialized")
    
    def _load_model(self):
        """Load the model"""
        try:
            if self.config.model_type == 'document_classifier':
                self._load_document_classifier()
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            self.is_healthy = True
            self.logger.info(f"Model loaded successfully on worker {self.worker_id}")
        
        except Exception as e:
            self.logger.error(f"Failed to load model on worker {self.worker_id}: {e}")
            self.is_healthy = False
    
    def _load_document_classifier(self):
        """Load document classification model"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        from document_parser.models.document_classifier import DocumentClassifier, ClassifierConfig
        
        # Load checkpoint
        checkpoint = torch.load(self.config.model_path, map_location=self.config.device)
        
        # Get config
        if 'config' in checkpoint:
            model_config = ClassifierConfig.from_dict(checkpoint['config'])
        else:
            model_config = ClassifierConfig()
        
        # Create and load model
        classifier = DocumentClassifier(model_config)
        self.model = classifier.model
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.config.device)
        self.model.eval()
        
        # Get model version
        if 'version' in checkpoint:
            self.model_version = checkpoint['version']
    
    def predict(self, data: Any) -> PredictionResponse:
        """Make prediction"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            if not self.is_healthy:
                raise RuntimeError("Worker is not healthy")
            
            # Process based on model type
            if self.config.model_type == 'document_classifier':
                predictions, confidence = self._predict_document_classifier(data)
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            processing_time = time.time() - start_time
            
            # Update worker stats
            self.request_count += 1
            self.last_request_time = time.time()
            
            response = PredictionResponse(
                request_id=request_id,
                predictions=predictions,
                confidence=confidence,
                processing_time=processing_time,
                model_version=self.model_version,
                timestamp=datetime.now()
            )
            
            return response
        
        except Exception as e:
            self.logger.error(f"Prediction error on worker {self.worker_id}: {e}")
            raise
    
    def _predict_document_classifier(self, image_data: bytes) -> Tuple[Any, float]:
        """Predict document class"""
        import torchvision.transforms as transforms
        from PIL import Image
        import io
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(self.config.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities.max().item()
        
        return predicted_class, confidence
    
    def health_check(self) -> bool:
        """Check worker health"""
        try:
            # Simple health check - ensure model is loaded and responsive
            if self.model is None:
                return False
            
            # Check if worker has been idle too long
            if time.time() - self.last_request_time > 3600:  # 1 hour
                self.logger.warning(f"Worker {self.worker_id} has been idle for too long")
            
            return self.is_healthy
        
        except Exception as e:
            self.logger.error(f"Health check failed for worker {self.worker_id}: {e}")
            return False

class LoadBalancer:
    """Load balancer for distributing requests across workers"""
    
    def __init__(self, workers: List[ModelWorker], strategy: str = 'round_robin'):
        self.workers = workers
        self.strategy = strategy
        self.current_worker = 0
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        
        self.logger.info(f"Load balancer initialized with {len(workers)} workers using {strategy} strategy")
    
    def get_worker(self) -> ModelWorker:
        """Get next available worker"""
        healthy_workers = [w for w in self.workers if w.health_check()]
        
        if not healthy_workers:
            raise RuntimeError("No healthy workers available")
        
        if self.strategy == 'round_robin':
            worker = healthy_workers[self.current_worker % len(healthy_workers)]
            self.current_worker += 1
        
        elif self.strategy == 'least_loaded':
            worker = min(healthy_workers, key=lambda w: w.request_count)
        
        elif self.strategy == 'random':
            import random
            worker = random.choice(healthy_workers)
        
        else:
            # Default to round robin
            worker = healthy_workers[self.current_worker % len(healthy_workers)]
            self.current_worker += 1
        
        return worker

class ModelServer:
    """Main model serving server"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config = ServingConfig.from_yaml(config_path)
        else:
            self.config = ServingConfig()
        
        self.logger = self._setup_logging()
        
        # Initialize components
        self.cache_manager = CacheManager(self.config)
        self.metrics_collector = MetricsCollector(self.config)
        
        # Initialize workers
        self.workers = []
        self._initialize_workers()
        
        # Initialize load balancer
        self.load_balancer = LoadBalancer(self.workers, self.config.load_balancing_strategy)
        
        # Initialize Flask app
        if FLASK_AVAILABLE:
            self.app = self._create_flask_app()
        else:
            raise RuntimeError("Flask not available for web serving")
        
        self.logger.info("Model server initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('ModelServer')
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
    
    def _initialize_workers(self):
        """Initialize model workers"""
        for i in range(self.config.num_workers):
            try:
                worker = ModelWorker(i, self.config)
                self.workers.append(worker)
            except Exception as e:
                self.logger.error(f"Failed to initialize worker {i}: {e}")
        
        if not self.workers:
            raise RuntimeError("No workers could be initialized")
        
        self.logger.info(f"Initialized {len(self.workers)} workers")
    
    def _create_flask_app(self) -> Flask:
        """Create Flask application"""
        app = Flask(__name__)
        app.config['MAX_CONTENT_LENGTH'] = self.config.max_request_size
        
        @app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            healthy_workers = sum(1 for w in self.workers if w.health_check())
            
            status = {
                'status': 'healthy' if healthy_workers > 0 else 'unhealthy',
                'healthy_workers': healthy_workers,
                'total_workers': len(self.workers),
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(status), 200 if healthy_workers > 0 else 503
        
        @app.route('/metrics', methods=['GET'])
        def get_metrics():
            """Metrics endpoint"""
            metrics = self.metrics_collector.get_metrics()
            return jsonify(metrics)
        
        @app.route('/predict', methods=['POST'])
        def predict():
            """Prediction endpoint"""
            try:
                # Validate request
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                # Read file data
                file_data = file.read()
                
                # Check cache
                cached_response = self.cache_manager.get(file_data)
                if cached_response:
                    self.metrics_collector.record_request(
                        cached_response.processing_time, 
                        cached=True
                    )
                    return jsonify(cached_response.to_dict())
                
                # Get worker and make prediction
                worker = self.load_balancer.get_worker()
                response = worker.predict(file_data)
                
                # Cache response
                self.cache_manager.set(file_data, response)
                
                # Record metrics
                self.metrics_collector.record_request(response.processing_time)
                
                return jsonify(response.to_dict())
            
            except Exception as e:
                self.logger.error(f"Prediction error: {e}")
                self.metrics_collector.record_request(0, error=True)
                return jsonify({'error': str(e)}), 500
        
        @app.route('/cache/clear', methods=['POST'])
        def clear_cache():
            """Clear cache endpoint"""
            try:
                self.cache_manager.clear()
                return jsonify({'message': 'Cache cleared successfully'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        return app
    
    def start(self):
        """Start the model server"""
        self.logger.info(f"Starting model server on {self.config.host}:{self.config.port}")
        
        # Start health check thread
        health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        health_thread.start()
        
        # Start Flask app
        self.app.run(
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug,
            threaded=self.config.threaded
        )
    
    def _health_check_loop(self):
        """Periodic health check for workers"""
        while True:
            try:
                unhealthy_workers = []
                for worker in self.workers:
                    if not worker.health_check():
                        unhealthy_workers.append(worker.worker_id)
                
                if unhealthy_workers:
                    self.logger.warning(f"Unhealthy workers detected: {unhealthy_workers}")
                
                time.sleep(self.config.health_check_interval)
            
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                time.sleep(self.config.health_check_interval)

def main():
    """Main function for standalone execution"""
    print("🚀 Document Parser Model Server")
    print("=" * 50)
    
    # Check dependencies
    print(f"\n📦 Dependencies:")
    print(f"   PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
    print(f"   Flask: {'✅' if FLASK_AVAILABLE else '❌'}")
    print(f"   Redis: {'✅' if REDIS_AVAILABLE else '❌'}")
    print(f"   OpenCV: {'✅' if CV2_AVAILABLE else '❌'}")
    
    if not TORCH_AVAILABLE:
        print("\n⚠️  PyTorch not available. Install with: pip install torch torchvision")
        return 1
    
    if not FLASK_AVAILABLE:
        print("\n⚠️  Flask not available. Install with: pip install flask")
        return 1
    
    # Example configuration
    config = ServingConfig(
        host='0.0.0.0',
        port=8080,
        model_type='document_classifier',
        model_path='models/document_classifier.pth',
        num_workers=2,
        enable_cache=True,
        enable_monitoring=True
    )
    
    print(f"\n⚙️ Configuration:")
    print(f"   Host: {config.host}")
    print(f"   Port: {config.port}")
    print(f"   Model type: {config.model_type}")
    print(f"   Workers: {config.num_workers}")
    print(f"   Cache enabled: {config.enable_cache}")
    print(f"   Monitoring enabled: {config.enable_monitoring}")
    
    print("\n📋 API Endpoints:")
    print("   POST /predict - Make predictions")
    print("   GET  /health  - Health check")
    print("   GET  /metrics - Performance metrics")
    print("   POST /cache/clear - Clear cache")
    
    print("\n🔧 Usage Examples:")
    print("1. server = ModelServer('config.yaml')")
    print("2. server.start()")
    print("3. curl -X POST -F 'file=@image.jpg' http://localhost:8080/predict")
    
    return 0

if __name__ == "__main__":
    exit(main())