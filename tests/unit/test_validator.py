#!/usr/bin/env python3
"""
Unit Tests for Document Validator

Comprehensive testing of document validation functionality including
business rules, data consistency, format validation, and error handling.
"""

import pytest
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.document_parser.validator import DocumentValidator
from src.document_parser.models.document_models import (
    DocumentType, ExtractedField, FieldType, ValidationResult, ValidationError
)


class TestDocumentValidator:
    """Test cases for DocumentValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a DocumentValidator instance for testing."""
        return DocumentValidator(
            rules_config="test_rules.json",
            strict_mode=True
        )
    
    @pytest.fixture
    def sample_mykad_data(self):
        """Sample MyKad extracted data."""
        return {
            'name': ExtractedField(
                value="AHMAD BIN ALI",
                field_type=FieldType.TEXT,
                confidence=0.95
            ),
            'ic_number': ExtractedField(
                value="123456-78-9012",
                field_type=FieldType.IC_NUMBER,
                confidence=0.98
            ),
            'gender': ExtractedField(
                value="LELAKI",
                field_type=FieldType.TEXT,
                confidence=0.92
            ),
            'birth_date': ExtractedField(
                value="12/34/56",
                field_type=FieldType.DATE,
                confidence=0.88
            ),
            'address': ExtractedField(
                value="KUALA LUMPUR",
                field_type=FieldType.TEXT,
                confidence=0.85
            )
        }
    
    @pytest.fixture
    def sample_spk_data(self):
        """Sample SPK extracted data."""
        return {
            'name': ExtractedField(
                value="SITI AMINAH BINTI HASSAN",
                field_type=FieldType.TEXT,
                confidence=0.94
            ),
            'certificate_number': ExtractedField(
                value="SPK123456789",
                field_type=FieldType.TEXT,
                confidence=0.96
            ),
            'birth_date': ExtractedField(
                value="15/03/2020",
                field_type=FieldType.DATE,
                confidence=0.91
            ),
            'birth_place': ExtractedField(
                value="HOSPITAL KUALA LUMPUR",
                field_type=FieldType.TEXT,
                confidence=0.87
            ),
            'gender': ExtractedField(
                value="PEREMPUAN",
                field_type=FieldType.TEXT,
                confidence=0.93
            )
        }
    
    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None
        assert validator.strict_mode is True
        assert hasattr(validator, 'validation_rules')
    
    def test_validate_mykad_valid_data(self, validator, sample_mykad_data):
        """Test validation of valid MyKad data."""
        # Fix the birth date to be valid
        sample_mykad_data['birth_date'].value = "12/03/1956"
        
        result = validator.validate(sample_mykad_data, DocumentType.MYKAD)
        
        assert result is not None
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_mykad_invalid_ic_format(self, validator, sample_mykad_data):
        """Test validation with invalid IC number format."""
        sample_mykad_data['ic_number'].value = "123456789012"  # Missing dashes
        
        result = validator.validate(sample_mykad_data, DocumentType.MYKAD)
        
        assert result is not None
        assert result.is_valid is False
        assert any(error.field == 'ic_number' for error in result.errors)
    
    def test_validate_mykad_invalid_date_format(self, validator, sample_mykad_data):
        """Test validation with invalid date format."""
        sample_mykad_data['birth_date'].value = "32/13/2020"  # Invalid date
        
        result = validator.validate(sample_mykad_data, DocumentType.MYKAD)
        
        assert result is not None
        assert result.is_valid is False
        assert any(error.field == 'birth_date' for error in result.errors)
    
    def test_validate_spk_valid_data(self, validator, sample_spk_data):
        """Test validation of valid SPK data."""
        result = validator.validate(sample_spk_data, DocumentType.SPK)
        
        assert result is not None
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_spk_invalid_certificate_format(self, validator, sample_spk_data):
        """Test validation with invalid SPK certificate format."""
        sample_spk_data['certificate_number'].value = "INVALID123"  # Wrong format
        
        result = validator.validate(sample_spk_data, DocumentType.SPK)
        
        assert result is not None
        assert result.is_valid is False
        assert any(error.field == 'certificate_number' for error in result.errors)
    
    def test_validate_missing_required_fields(self, validator):
        """Test validation with missing required fields."""
        incomplete_data = {
            'name': ExtractedField(
                value="AHMAD BIN ALI",
                field_type=FieldType.TEXT,
                confidence=0.95
            )
            # Missing ic_number which is required
        }
        
        result = validator.validate(incomplete_data, DocumentType.MYKAD)
        
        assert result is not None
        assert result.is_valid is False
        assert any(error.error_type == 'missing_required_field' for error in result.errors)
    
    def test_validate_low_confidence_fields(self, validator, sample_mykad_data):
        """Test validation with low confidence fields."""
        sample_mykad_data['name'].confidence = 0.3  # Very low confidence
        
        result = validator.validate(sample_mykad_data, DocumentType.MYKAD)
        
        assert result is not None
        if validator.strict_mode:
            assert result.is_valid is False
            assert any(error.error_type == 'low_confidence' for error in result.errors)
    
    def test_ic_number_validation(self, validator):
        """Test IC number format validation."""
        # Valid IC numbers
        valid_ics = [
            "123456-78-9012",
            "987654-32-1098",
            "111111-11-1111"
        ]
        
        for ic in valid_ics:
            assert validator._validate_ic_format(ic) is True
        
        # Invalid IC numbers
        invalid_ics = [
            "123456789012",      # No dashes
            "12345-78-9012",     # Wrong first part length
            "123456-7-9012",     # Wrong middle part length
            "123456-78-901",     # Wrong last part length
            "abcdef-78-9012",    # Non-numeric characters
            "",                   # Empty string
            None                  # None value
        ]
        
        for ic in invalid_ics:
            assert validator._validate_ic_format(ic) is False
    
    def test_date_validation(self, validator):
        """Test date format validation."""
        # Valid dates
        valid_dates = [
            "15/03/2020",
            "01/01/2000",
            "31/12/1999"
        ]
        
        for date_str in valid_dates:
            assert validator._validate_date_format(date_str) is True
        
        # Invalid dates
        invalid_dates = [
            "32/01/2020",    # Invalid day
            "15/13/2020",    # Invalid month
            "15/03/20",      # Wrong year format
            "2020/03/15",    # Wrong format
            "15-03-2020",    # Wrong separator
            "",               # Empty string
            None              # None value
        ]
        
        for date_str in invalid_dates:
            assert validator._validate_date_format(date_str) is False
    
    def test_name_validation(self, validator):
        """Test name format validation."""
        # Valid names
        valid_names = [
            "AHMAD BIN ALI",
            "SITI AMINAH BINTI HASSAN",
            "JOHN DOE",
            "MARY JANE WATSON"
        ]
        
        for name in valid_names:
            assert validator._validate_name_format(name) is True
        
        # Invalid names
        invalid_names = [
            "ahmad bin ali",     # Lowercase
            "AHMAD123",          # Contains numbers
            "AHMAD@ALI",         # Contains special characters
            "A",                 # Too short
            "",                  # Empty string
            None                 # None value
        ]
        
        for name in invalid_names:
            assert validator._validate_name_format(name) is False
    
    def test_gender_validation(self, validator):
        """Test gender validation."""
        # Valid genders
        valid_genders = ["LELAKI", "PEREMPUAN"]
        
        for gender in valid_genders:
            assert validator._validate_gender(gender) is True
        
        # Invalid genders
        invalid_genders = [
            "MALE",
            "FEMALE",
            "lelaki",
            "perempuan",
            "OTHER",
            "",
            None
        ]
        
        for gender in invalid_genders:
            assert validator._validate_gender(gender) is False
    
    def test_spk_certificate_validation(self, validator):
        """Test SPK certificate number validation."""
        # Valid certificate numbers
        valid_certs = [
            "SPK123456789",
            "SPK987654321",
            "SPK111111111"
        ]
        
        for cert in valid_certs:
            assert validator._validate_spk_certificate(cert) is True
        
        # Invalid certificate numbers
        invalid_certs = [
            "123456789",      # Missing SPK prefix
            "SPK12345",       # Too short
            "SPK12345678901", # Too long
            "spk123456789",   # Lowercase
            "SPK12345678A",   # Contains letters in number part
            "",               # Empty string
            None              # None value
        ]
        
        for cert in invalid_certs:
            assert validator._validate_spk_certificate(cert) is False
    
    def test_business_rules_validation(self, validator, sample_mykad_data):
        """Test business rules validation."""
        # Test age consistency (IC number vs birth date)
        sample_mykad_data['ic_number'].value = "200315-03-1234"  # Born 2020
        sample_mykad_data['birth_date'].value = "15/03/1990"     # Different year
        
        result = validator.validate(sample_mykad_data, DocumentType.MYKAD)
        
        assert result is not None
        assert result.is_valid is False
        assert any(error.error_type == 'business_rule_violation' for error in result.errors)
    
    def test_cross_field_validation(self, validator, sample_mykad_data):
        """Test cross-field validation."""
        # Test gender consistency (IC number vs gender field)
        sample_mykad_data['ic_number'].value = "123456-78-9012"  # Even last digit = female
        sample_mykad_data['gender'].value = "LELAKI"             # Male
        
        result = validator.validate(sample_mykad_data, DocumentType.MYKAD)
        
        assert result is not None
        assert result.is_valid is False
        assert any(error.error_type == 'cross_field_inconsistency' for error in result.errors)
    
    def test_confidence_threshold_validation(self, validator, sample_mykad_data):
        """Test confidence threshold validation."""
        # Set all fields to low confidence
        for field in sample_mykad_data.values():
            field.confidence = 0.4
        
        result = validator.validate(sample_mykad_data, DocumentType.MYKAD)
        
        assert result is not None
        if validator.strict_mode:
            assert result.is_valid is False
            assert len(result.errors) > 0
    
    def test_validate_invalid_input(self, validator):
        """Test validation with invalid input."""
        with pytest.raises(ValueError):
            validator.validate(None, DocumentType.MYKAD)
        
        with pytest.raises(ValueError):
            validator.validate({}, None)
    
    def test_validation_rules_loading(self, validator):
        """Test validation rules loading."""
        with patch('builtins.open') as mock_open, \
             patch('json.load') as mock_json:
            
            mock_json.return_value = {
                "MYKAD": {
                    "required_fields": ["name", "ic_number"],
                    "field_rules": {
                        "ic_number": {"pattern": r"\d{6}-\d{2}-\d{4}"}
                    }
                }
            }
            
            rules = validator._load_validation_rules()
            
            assert rules is not None
            assert "MYKAD" in rules
            assert "required_fields" in rules["MYKAD"]
    
    def test_batch_validation(self, validator, sample_mykad_data, sample_spk_data):
        """Test batch validation."""
        data_list = [
            (sample_mykad_data, DocumentType.MYKAD),
            (sample_spk_data, DocumentType.SPK)
        ]
        
        results = validator.validate_batch(data_list)
        
        assert len(results) == 2
        assert all(isinstance(result, ValidationResult) for result in results)
    
    def test_performance_metrics(self, validator, sample_mykad_data):
        """Test performance metrics collection."""
        import time
        
        start_time = time.time()
        
        result = validator.validate(sample_mykad_data, DocumentType.MYKAD)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert processing_time < 2.0  # Should validate within 2 seconds
        assert result is not None
    
    def test_error_handling(self, validator, sample_mykad_data):
        """Test error handling in validation."""
        with patch.object(validator, '_validate_ic_format') as mock_validate:
            mock_validate.side_effect = Exception("Validation error")
            
            result = validator.validate(sample_mykad_data, DocumentType.MYKAD)
            
            # Should handle error gracefully
            assert result is not None
            assert result.is_valid is False
    
    def test_memory_usage(self, validator, sample_mykad_data):
        """Test memory usage during validation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform multiple validations
        for _ in range(100):
            validator.validate(sample_mykad_data, DocumentType.MYKAD)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB


class TestDocumentValidatorIntegration:
    """Integration tests for DocumentValidator."""
    
    @pytest.fixture
    def real_validator(self):
        """Create a real DocumentValidator for integration testing."""
        return DocumentValidator()
    
    def test_real_rules_loading(self, real_validator):
        """Test loading of actual validation rules."""
        try:
            rules = real_validator._load_validation_rules()
            assert rules is not None
            assert len(rules) > 0
        except Exception:
            pytest.skip("Validation rules file not available")
    
    def test_supported_document_types(self, real_validator):
        """Test getting supported document types for validation."""
        supported_types = real_validator.get_supported_types()
        
        assert DocumentType.MYKAD in supported_types
        assert DocumentType.SPK in supported_types
        assert len(supported_types) > 0


if __name__ == '__main__':
    pytest.main([__file__])