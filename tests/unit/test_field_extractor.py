#!/usr/bin/env python3
"""
Unit Tests for Field Extractor

Comprehensive testing of field extraction functionality including
template matching, regex patterns, NER, and validation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

from src.document_parser.field_extractor import FieldExtractor
from src.document_parser.models.document_models import (
    DocumentType, ExtractedField, FieldType, ConfidenceLevel
)


class TestFieldExtractor:
    """Test cases for FieldExtractor class."""
    
    @pytest.fixture
    def field_extractor(self):
        """Create a FieldExtractor instance for testing."""
        return FieldExtractor(
            template_dir="test_templates",
            confidence_threshold=0.7
        )
    
    @pytest.fixture
    def sample_mykad_text(self):
        """Sample MyKad OCR text."""
        return """
        MALAYSIA
        MYKAD
        AHMAD BIN ALI
        123456-78-9012
        LELAKI
        KUALA LUMPUR
        ISLAM
        """
    
    @pytest.fixture
    def sample_spk_text(self):
        """Sample SPK OCR text."""
        return """
        SURAT PERAKUAN KELAHIRAN
        NAMA: SITI AMINAH BINTI HASSAN
        NO. SIJIL: SPK123456789
        TARIKH LAHIR: 15/03/2020
        TEMPAT LAHIR: HOSPITAL KUALA LUMPUR
        JANTINA: PEREMPUAN
        """
    
    @pytest.fixture
    def sample_mykad_template(self):
        """Sample MyKad extraction template."""
        return {
            "document_type": "MYKAD",
            "fields": {
                "name": {
                    "type": "text",
                    "patterns": [r"([A-Z\s]+)\n\d{6}-\d{2}-\d{4}"],
                    "position": "above_ic",
                    "required": True
                },
                "ic_number": {
                    "type": "ic_number",
                    "patterns": [r"(\d{6}-\d{2}-\d{4})"],
                    "validation": "ic_format",
                    "required": True
                },
                "gender": {
                    "type": "text",
                    "patterns": [r"(LELAKI|PEREMPUAN)"],
                    "required": True
                }
            }
        }
    
    def test_field_extractor_initialization(self, field_extractor):
        """Test field extractor initialization."""
        assert field_extractor is not None
        assert field_extractor.confidence_threshold == 0.7
        assert hasattr(field_extractor, 'templates')
    
    def test_extract_mykad_fields(self, field_extractor, sample_mykad_text, sample_mykad_template):
        """Test MyKad field extraction."""
        with patch.object(field_extractor, '_load_template') as mock_template:
            mock_template.return_value = sample_mykad_template
            
            result = field_extractor.extract_fields(sample_mykad_text, DocumentType.MYKAD)
            
            assert result is not None
            assert 'name' in result.fields
            assert 'ic_number' in result.fields
            assert result.fields['name'].value == "AHMAD BIN ALI"
            assert result.fields['ic_number'].value == "123456-78-9012"
    
    def test_extract_spk_fields(self, field_extractor, sample_spk_text):
        """Test SPK field extraction."""
        spk_template = {
            "document_type": "SPK",
            "fields": {
                "name": {
                    "type": "text",
                    "patterns": [r"NAMA:\s*([A-Z\s]+)"],
                    "required": True
                },
                "certificate_number": {
                    "type": "text",
                    "patterns": [r"NO\. SIJIL:\s*(SPK\d+)"],
                    "required": True
                },
                "birth_date": {
                    "type": "date",
                    "patterns": [r"TARIKH LAHIR:\s*(\d{2}/\d{2}/\d{4})"],
                    "validation": "date_format",
                    "required": True
                }
            }
        }
        
        with patch.object(field_extractor, '_load_template') as mock_template:
            mock_template.return_value = spk_template
            
            result = field_extractor.extract_fields(sample_spk_text, DocumentType.SPK)
            
            assert result is not None
            assert 'name' in result.fields
            assert 'certificate_number' in result.fields
            assert 'birth_date' in result.fields
            assert result.fields['name'].value == "SITI AMINAH BINTI HASSAN"
            assert result.fields['certificate_number'].value == "SPK123456789"
            assert result.fields['birth_date'].value == "15/03/2020"
    
    def test_extract_fields_with_confidence(self, field_extractor, sample_mykad_text, sample_mykad_template):
        """Test field extraction with confidence scoring."""
        with patch.object(field_extractor, '_load_template') as mock_template, \
             patch.object(field_extractor, '_calculate_field_confidence') as mock_confidence:
            
            mock_template.return_value = sample_mykad_template
            mock_confidence.return_value = 0.85
            
            result = field_extractor.extract_fields(sample_mykad_text, DocumentType.MYKAD)
            
            assert result is not None
            for field in result.fields.values():
                assert field.confidence >= 0.7
                assert field.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]
    
    def test_extract_fields_invalid_input(self, field_extractor):
        """Test field extraction with invalid input."""
        with pytest.raises(ValueError):
            field_extractor.extract_fields(None, DocumentType.MYKAD)
        
        with pytest.raises(ValueError):
            field_extractor.extract_fields("", None)
    
    def test_extract_fields_empty_text(self, field_extractor, sample_mykad_template):
        """Test field extraction with empty text."""
        with patch.object(field_extractor, '_load_template') as mock_template:
            mock_template.return_value = sample_mykad_template
            
            result = field_extractor.extract_fields("", DocumentType.MYKAD)
            
            assert result is not None
            assert len(result.fields) == 0 or all(
                field.value == "" for field in result.fields.values()
            )
    
    def test_regex_pattern_matching(self, field_extractor):
        """Test regex pattern matching functionality."""
        text = "IC Number: 123456-78-9012"
        pattern = r"IC Number:\s*(\d{6}-\d{2}-\d{4})"
        
        match = field_extractor._apply_regex_pattern(text, pattern)
        
        assert match is not None
        assert match == "123456-78-9012"
    
    def test_multiple_pattern_matching(self, field_extractor):
        """Test matching with multiple patterns."""
        text = "Name: AHMAD BIN ALI"
        patterns = [
            r"Full Name:\s*([A-Z\s]+)",  # Won't match
            r"Name:\s*([A-Z\s]+)",       # Will match
            r"NAMA:\s*([A-Z\s]+)"        # Won't match
        ]
        
        match = field_extractor._apply_multiple_patterns(text, patterns)
        
        assert match is not None
        assert match == "AHMAD BIN ALI"
    
    def test_field_validation(self, field_extractor):
        """Test field validation functionality."""
        # Test IC number validation
        valid_ic = "123456-78-9012"
        invalid_ic = "123456789012"
        
        assert field_extractor._validate_field(valid_ic, "ic_format") is True
        assert field_extractor._validate_field(invalid_ic, "ic_format") is False
        
        # Test date validation
        valid_date = "15/03/2020"
        invalid_date = "32/13/2020"
        
        assert field_extractor._validate_field(valid_date, "date_format") is True
        assert field_extractor._validate_field(invalid_date, "date_format") is False
    
    def test_confidence_calculation(self, field_extractor):
        """Test confidence score calculation."""
        # High confidence case
        field_value = "AHMAD BIN ALI"
        pattern_match = True
        validation_passed = True
        
        confidence = field_extractor._calculate_field_confidence(
            field_value, pattern_match, validation_passed
        )
        
        assert confidence >= 0.8
        
        # Low confidence case
        field_value = "UNCLEAR TEXT"
        pattern_match = False
        validation_passed = False
        
        confidence = field_extractor._calculate_field_confidence(
            field_value, pattern_match, validation_passed
        )
        
        assert confidence < 0.5
    
    def test_template_loading(self, field_extractor):
        """Test template loading functionality."""
        with patch('builtins.open') as mock_open, \
             patch('json.load') as mock_json:
            
            mock_json.return_value = {
                "document_type": "MYKAD",
                "fields": {"name": {"type": "text"}}
            }
            
            template = field_extractor._load_template(DocumentType.MYKAD)
            
            assert template is not None
            assert template['document_type'] == "MYKAD"
            assert 'fields' in template
    
    def test_ner_extraction(self, field_extractor, sample_mykad_text):
        """Test Named Entity Recognition extraction."""
        with patch.object(field_extractor, '_extract_with_ner') as mock_ner:
            mock_ner.return_value = {
                'PERSON': ['AHMAD BIN ALI'],
                'DATE': ['123456-78-9012'],
                'LOCATION': ['KUALA LUMPUR']
            }
            
            entities = field_extractor._extract_entities(sample_mykad_text)
            
            assert entities is not None
            assert 'PERSON' in entities
            assert 'AHMAD BIN ALI' in entities['PERSON']
    
    def test_field_post_processing(self, field_extractor):
        """Test field value post-processing."""
        # Test text cleaning
        dirty_text = "  AHMAD   BIN  ALI  "
        cleaned = field_extractor._post_process_field(dirty_text, FieldType.TEXT)
        assert cleaned == "AHMAD BIN ALI"
        
        # Test IC number formatting
        ic_number = "123456789012"
        formatted = field_extractor._post_process_field(ic_number, FieldType.IC_NUMBER)
        assert formatted == "123456-78-9012"
        
        # Test date formatting
        date_value = "15/3/2020"
        formatted_date = field_extractor._post_process_field(date_value, FieldType.DATE)
        assert formatted_date == "15/03/2020"
    
    def test_batch_field_extraction(self, field_extractor, sample_mykad_template):
        """Test batch field extraction."""
        texts = [
            "AHMAD BIN ALI\n123456-78-9012",
            "SITI FATIMAH\n987654-32-1098"
        ]
        
        with patch.object(field_extractor, '_load_template') as mock_template:
            mock_template.return_value = sample_mykad_template
            
            results = field_extractor.extract_fields_batch(texts, DocumentType.MYKAD)
            
            assert len(results) == 2
            assert all(result.fields for result in results)
    
    def test_performance_metrics(self, field_extractor, sample_mykad_text, sample_mykad_template):
        """Test performance metrics collection."""
        import time
        
        start_time = time.time()
        
        with patch.object(field_extractor, '_load_template') as mock_template:
            mock_template.return_value = sample_mykad_template
            
            result = field_extractor.extract_fields(sample_mykad_text, DocumentType.MYKAD)
            
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert processing_time < 5.0  # Should process within 5 seconds
        assert result is not None
    
    def test_error_handling(self, field_extractor, sample_mykad_text):
        """Test error handling in field extraction."""
        with patch.object(field_extractor, '_load_template') as mock_template:
            mock_template.side_effect = Exception("Template loading failed")
            
            result = field_extractor.extract_fields(sample_mykad_text, DocumentType.MYKAD)
            
            # Should handle error gracefully
            assert result is not None
            assert len(result.fields) == 0
    
    def test_memory_usage(self, field_extractor, sample_mykad_text, sample_mykad_template):
        """Test memory usage during field extraction."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch.object(field_extractor, '_load_template') as mock_template:
            mock_template.return_value = sample_mykad_template
            
            # Process multiple extractions
            for _ in range(50):
                field_extractor.extract_fields(sample_mykad_text, DocumentType.MYKAD)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB


class TestFieldExtractorIntegration:
    """Integration tests for FieldExtractor."""
    
    @pytest.fixture
    def real_field_extractor(self):
        """Create a real FieldExtractor for integration testing."""
        return FieldExtractor()
    
    def test_real_template_loading(self, real_field_extractor):
        """Test loading of actual template files."""
        try:
            template = real_field_extractor._load_template(DocumentType.MYKAD)
            assert template is not None
            assert 'fields' in template
        except Exception:
            pytest.skip("Template files not available")
    
    def test_supported_document_types(self, real_field_extractor):
        """Test getting supported document types."""
        supported_types = real_field_extractor.get_supported_types()
        
        assert DocumentType.MYKAD in supported_types
        assert DocumentType.SPK in supported_types
        assert len(supported_types) > 0


if __name__ == '__main__':
    pytest.main([__file__])