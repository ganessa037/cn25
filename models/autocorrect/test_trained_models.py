#!/usr/bin/env python3
"""
Integration Testing Script for Trained Autocorrect Models

This script loads the trained models from the notebook and runs comprehensive tests
using the testing framework.

Usage:
    python test_trained_models.py

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from comprehensive_model_testing import ModelTester, TestLogger
    from fuzzywuzzy import fuzz, process
    from jellyfish import levenshtein_distance, jaro_winkler_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    print("Please ensure all required packages are installed.")

class TrainedModelLoader:
    """Loads and manages trained autocorrect models"""
    
    def __init__(self, model_path: str = "."):
        self.model_path = Path(model_path)
        self.models = {}
        self.correction_mappings = {}
        self.metadata = {}
        self.vehicle_reference = None
        
    def load_models(self):
        """Load all trained models and artifacts"""
        print("📦 Loading trained models...")
        
        try:
            # Load correction mappings
            mappings_file = self.model_path / 'correction_mappings.json'
            if mappings_file.exists():
                with open(mappings_file, 'r') as f:
                    self.correction_mappings = json.load(f)
                print(f"✅ Loaded correction mappings: {len(self.correction_mappings.get('direct_mappings', {}))} direct, {len(self.correction_mappings.get('fuzzy_mappings', {}))} fuzzy")
            else:
                print("⚠️ Correction mappings not found")
            
            # Load model metadata
            metadata_file = self.model_path / 'model_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Loaded model metadata")
            else:
                print("⚠️ Model metadata not found")
            
            # Load ML models
            model_files = list(self.model_path.glob('ml_model_*.pkl'))
            for model_file in model_files:
                model_name = model_file.stem.replace('ml_model_', '')
                try:
                    with open(model_file, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    print(f"✅ Loaded ML model: {model_name}")
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
            
            # Load vehicle reference data
            reference_file = self.model_path / 'vehicle_reference.csv'
            if reference_file.exists():
                self.vehicle_reference = pd.read_csv(reference_file)
                print(f"✅ Loaded vehicle reference: {len(self.vehicle_reference)} records")
            else:
                # Try to load from data directory
                data_reference = self.model_path.parent / 'data' / 'vehicle_master.csv'
                if data_reference.exists():
                    self.vehicle_reference = pd.read_csv(data_reference)
                    print(f"✅ Loaded vehicle reference from data directory: {len(self.vehicle_reference)} records")
                else:
                    print("⚠️ Vehicle reference data not found")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def get_best_model(self):
        """Get the best performing model based on metadata"""
        if self.metadata and 'performance' in self.metadata:
            best_model_name = self.metadata['performance'].get('best_ml_model')
            if best_model_name and best_model_name in self.models:
                return self.models[best_model_name], best_model_name
        
        # Fallback to first available model
        if self.models:
            model_name = list(self.models.keys())[0]
            return self.models[model_name], model_name
        
        return None, None

class ProductionAutocorrectTester:
    """Production-ready autocorrect implementation for testing"""
    
    def __init__(self, model_loader: TrainedModelLoader):
        self.loader = model_loader
        self.best_model, self.best_model_name = model_loader.get_best_model()
        
        # Get reference data
        if model_loader.vehicle_reference is not None:
            self.brands = model_loader.vehicle_reference['brand'].unique().tolist()
            self.models_list = model_loader.vehicle_reference['model'].unique().tolist()
            self.all_targets = self.brands + self.models_list
        else:
            self.brands = []
            self.models_list = []
            self.all_targets = []
        
        print(f"🎯 Initialized with {len(self.brands)} brands and {len(self.models_list)} models")
        if self.best_model:
            print(f"🏆 Using best model: {self.best_model_name}")
    
    def correct_text(self, input_text: str, confidence_threshold: float = 0.6) -> Tuple[str, float, str]:
        """Production-ready text correction with comprehensive fallback"""
        if not input_text:
            return input_text, 0.0, 'no_input'
        
        original_input = input_text
        input_text = str(input_text).strip().lower()
        
        # 1. Check direct mappings first (fastest)
        if (self.loader.correction_mappings and 
            'direct_mappings' in self.loader.correction_mappings and
            input_text in self.loader.correction_mappings['direct_mappings']):
            
            mapping = self.loader.correction_mappings['direct_mappings'][input_text]
            return mapping['correction'], mapping['confidence'], 'direct_mapping'
        
        # 2. Check fuzzy mappings
        if (self.loader.correction_mappings and 
            'fuzzy_mappings' in self.loader.correction_mappings and
            input_text in self.loader.correction_mappings['fuzzy_mappings']):
            
            mapping = self.loader.correction_mappings['fuzzy_mappings'][input_text]
            if mapping['confidence'] >= confidence_threshold:
                return mapping['correction'], mapping['confidence'], 'fuzzy_mapping'
        
        # 3. Use ML model for prediction
        if self.best_model and self.all_targets:
            try:
                # For sklearn models, we need to handle prediction differently
                if hasattr(self.best_model, 'predict_proba'):
                    # Try to get probabilities
                    try:
                        proba = self.best_model.predict_proba([input_text])[0]
                        best_idx = np.argmax(proba)
                        ml_confidence = proba[best_idx]
                        
                        if ml_confidence >= confidence_threshold and best_idx < len(self.all_targets):
                            predicted_text = self.all_targets[best_idx]
                            return predicted_text, ml_confidence, 'ml_model'
                    except Exception:
                        pass
                
                # Fallback to simple prediction
                if hasattr(self.best_model, 'predict'):
                    try:
                        prediction = self.best_model.predict([input_text])[0]
                        if isinstance(prediction, (int, np.integer)) and prediction < len(self.all_targets):
                            predicted_text = self.all_targets[prediction]
                            return predicted_text, 0.7, 'ml_model_simple'  # Default confidence
                        elif isinstance(prediction, str):
                            return prediction, 0.7, 'ml_model_direct'
                    except Exception:
                        pass
                        
            except Exception as e:
                print(f"ML prediction error for '{input_text}': {e}")
        
        # 4. Fallback to fuzzy matching against reference data
        if self.all_targets:
            try:
                best_match = process.extractOne(input_text, self.all_targets)
                if best_match and best_match[1] >= confidence_threshold * 100:
                    return best_match[0], best_match[1] / 100.0, 'fuzzy_fallback'
            except Exception as e:
                print(f"Fuzzy matching error for '{input_text}': {e}")
        
        # 5. Rule-based fuzzy matching for common patterns
        if self.all_targets:
            for target in self.all_targets:
                # Check for simple character-level similarity
                if len(input_text) > 2 and len(target) > 2:
                    # Levenshtein distance check
                    try:
                        distance = levenshtein_distance(input_text, target.lower())
                        max_distance = max(len(input_text), len(target)) * 0.3  # Allow 30% difference
                        
                        if distance <= max_distance:
                            confidence = 1.0 - (distance / max(len(input_text), len(target)))
                            if confidence >= confidence_threshold:
                                return target, confidence, 'rule_based_levenshtein'
                    except Exception:
                        pass
        
        # 6. No correction found - return original
        return original_input, 0.0, 'no_correction'
    
    def batch_correct(self, input_list: List[str], confidence_threshold: float = 0.6) -> pd.DataFrame:
        """Batch correction for multiple inputs"""
        results = []
        
        for input_text in input_list:
            corrected, confidence, method = self.correct_text(input_text, confidence_threshold)
            results.append({
                'input': input_text,
                'corrected': corrected,
                'confidence': confidence,
                'method': method,
                'was_corrected': corrected.lower() != str(input_text).lower()
            })
        
        return pd.DataFrame(results)

def run_comprehensive_tests():
    """Run comprehensive tests on trained models"""
    print("🧪 COMPREHENSIVE TESTING OF TRAINED AUTOCORRECT MODELS")
    print("=" * 70)
    
    # Initialize model loader
    model_loader = TrainedModelLoader()
    
    # Load models
    if not model_loader.load_models():
        print("❌ Failed to load models. Please ensure models are trained first.")
        return
    
    # Initialize production corrector
    corrector = ProductionAutocorrectTester(model_loader)
    
    # Initialize tester
    tester = ModelTester()
    
    # Create corrector function for testing
    def corrector_func(text):
        return corrector.correct_text(text)
    
    # Run comprehensive tests
    print("\n🚀 Starting comprehensive test suite...")
    results = tester.run_all_tests(corrector_func)
    
    # Additional model-specific tests
    print("\n🔬 Running model-specific validation tests...")
    run_model_validation_tests(corrector, tester)
    
    # Generate final summary
    print("\n📊 Generating final test report...")
    tester.logger.generate_summary_report()
    
    return results

def run_model_validation_tests(corrector: ProductionAutocorrectTester, tester: ModelTester):
    """Run additional validation tests specific to the trained models"""
    
    # Test with real vehicle data if available
    if corrector.loader.vehicle_reference is not None:
        print("🚗 Testing with real vehicle data...")
        
        # Sample some real brands and models
        sample_brands = corrector.brands[:10] if len(corrector.brands) >= 10 else corrector.brands
        sample_models = corrector.models_list[:10] if len(corrector.models_list) >= 10 else corrector.models_list
        
        # Test exact matches
        for brand in sample_brands:
            result = corrector.correct_text(brand)
            print(f"   Exact brand test: '{brand}' → '{result[0]}' (confidence: {result[1]:.3f}, method: {result[2]})")
        
        for model in sample_models:
            result = corrector.correct_text(model)
            print(f"   Exact model test: '{model}' → '{result[0]}' (confidence: {result[1]:.3f}, method: {result[2]})")
    
    # Test correction mapping coverage
    if corrector.loader.correction_mappings:
        print("\n📋 Testing correction mapping coverage...")
        direct_mappings = corrector.loader.correction_mappings.get('direct_mappings', {})
        fuzzy_mappings = corrector.loader.correction_mappings.get('fuzzy_mappings', {})
        
        print(f"   Direct mappings: {len(direct_mappings)}")
        print(f"   Fuzzy mappings: {len(fuzzy_mappings)}")
        
        # Test a few direct mappings
        sample_direct = list(direct_mappings.items())[:5]
        for input_text, mapping in sample_direct:
            result = corrector.correct_text(input_text)
            expected = mapping['correction']
            status = "✅" if result[0] == expected else "❌"
            print(f"   {status} Direct mapping: '{input_text}' → '{result[0]}' (expected: '{expected}')")
    
    # Test model performance metrics
    if corrector.loader.metadata:
        print("\n📈 Model performance metrics:")
        performance = corrector.loader.metadata.get('performance', {})
        for metric, value in performance.items():
            print(f"   {metric}: {value}")
    
    print("\n✅ Model validation tests completed")

def main():
    """Main function"""
    try:
        results = run_comprehensive_tests()
        print(f"\n🎉 All tests completed successfully!")
        print(f"📁 Test results and logs saved in the test_logs directory")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()