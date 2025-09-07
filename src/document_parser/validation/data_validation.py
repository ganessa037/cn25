"""Data Validation Rules Module

This module provides comprehensive data validation rules for document processing,
including format validation, cross-field validation, business rules, and confidence scoring.
Follows the autocorrect model's organizational patterns.
"""

import re
import json
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import math

import pandas as pd
import numpy as np
from dateutil import parser as date_parser
from fuzzywuzzy import fuzz, process
import phonenumbers
from phonenumbers import NumberParseException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class FieldType(Enum):
    """Field types for validation"""
    IC_NUMBER = "ic_number"
    PASSPORT_NUMBER = "passport_number"
    REGISTRATION_NUMBER = "registration_number"
    PHONE_NUMBER = "phone_number"
    EMAIL = "email"
    DATE = "date"
    ADDRESS = "address"
    NAME = "name"
    AMOUNT = "amount"
    POSTCODE = "postcode"
    STATE = "state"
    GENDER = "gender"
    RACE = "race"
    RELIGION = "religion"
    MARITAL_STATUS = "marital_status"

class DocumentType(Enum):
    """Document types"""
    IDENTITY_CARD = "identity_card"
    PASSPORT = "passport"
    DRIVING_LICENSE = "driving_license"
    VEHICLE_REGISTRATION = "vehicle_registration"
    BANK_STATEMENT = "bank_statement"
    INVOICE = "invoice"
    RECEIPT = "receipt"
    UTILITY_BILL = "utility_bill"

@dataclass
class ValidationResult:
    """Result of a validation check"""
    
    field_name: str
    field_type: FieldType
    is_valid: bool
    confidence_score: float
    
    # Validation details
    validation_level: ValidationLevel = ValidationLevel.INFO
    error_message: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    
    # Format validation
    format_valid: bool = True
    format_confidence: float = 1.0
    
    # Business rule validation
    business_rule_valid: bool = True
    business_rule_confidence: float = 1.0
    
    # Cross-field validation
    cross_field_valid: bool = True
    cross_field_confidence: float = 1.0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate overall confidence score"""
        if not hasattr(self, '_confidence_calculated'):
            self.confidence_score = self._calculate_overall_confidence()
            self._confidence_calculated = True
    
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence score from component scores"""
        if not self.is_valid:
            return 0.0
        
        # Weighted average of component confidences
        weights = {
            'format': 0.4,
            'business_rule': 0.35,
            'cross_field': 0.25
        }
        
        weighted_score = (
            weights['format'] * self.format_confidence +
            weights['business_rule'] * self.business_rule_confidence +
            weights['cross_field'] * self.cross_field_confidence
        )
        
        return min(1.0, max(0.0, weighted_score))

@dataclass
class ValidationConfig:
    """Configuration for validation rules"""
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.7
    high_confidence_threshold: float = 0.9
    
    # Format validation settings
    strict_format_validation: bool = True
    allow_partial_matches: bool = False
    
    # Business rule settings
    enforce_business_rules: bool = True
    age_calculation_tolerance: int = 1  # years
    
    # Cross-field validation settings
    enable_cross_field_validation: bool = True
    address_consistency_threshold: float = 0.8
    
    # Malaysian specific settings
    malaysian_states: List[str] = field(default_factory=lambda: [
        "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan",
        "Pahang", "Perak", "Perlis", "Pulau Pinang", "Sabah",
        "Sarawak", "Selangor", "Terengganu", "Kuala Lumpur",
        "Labuan", "Putrajaya"
    ])
    
    malaysian_postcodes: Dict[str, List[str]] = field(default_factory=lambda: {
        "Johor": ["79000-86999"],
        "Kedah": ["05000-09999"],
        "Kelantan": ["15000-18999"],
        "Melaka": ["75000-78999"],
        "Negeri Sembilan": ["70000-73999"],
        "Pahang": ["25000-28999", "39000-39999", "49000-49999"],
        "Perak": ["30000-36999"],
        "Perlis": ["01000-02999"],
        "Pulau Pinang": ["10000-14999"],
        "Sabah": ["88000-91999"],
        "Sarawak": ["93000-98999"],
        "Selangor": ["40000-48999", "63000-68999"],
        "Terengganu": ["20000-24999"],
        "Kuala Lumpur": ["50000-60999"],
        "Labuan": ["87000-87999"],
        "Putrajaya": ["62000-62999"]
    })
    
    # Validation rules file paths
    custom_rules_path: Optional[str] = None
    blacklist_path: Optional[str] = None
    whitelist_path: Optional[str] = None

class MalaysianICValidator:
    """Validator for Malaysian IC numbers"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
        # IC number patterns
        self.ic_pattern = re.compile(r'^(\d{6})[-]?(\d{2})[-]?(\d{4})$')
        self.old_ic_pattern = re.compile(r'^[A-Z]\d{7}$')
        
        # State codes for IC numbers
        self.state_codes = {
            '01': 'Johor', '21': 'Johor', '22': 'Johor', '24': 'Johor',
            '02': 'Kedah', '25': 'Kedah', '26': 'Kedah', '27': 'Kedah',
            '03': 'Kelantan', '28': 'Kelantan', '29': 'Kelantan',
            '04': 'Melaka', '30': 'Melaka',
            '05': 'Negeri Sembilan', '31': 'Negeri Sembilan', '59': 'Negeri Sembilan',
            '06': 'Pahang', '32': 'Pahang', '33': 'Pahang',
            '07': 'Perak', '34': 'Perak', '35': 'Perak',
            '08': 'Perlis', '36': 'Perlis',
            '09': 'Pulau Pinang', '37': 'Pulau Pinang', '38': 'Pulau Pinang', '39': 'Pulau Pinang',
            '12': 'Sabah', '47': 'Sabah', '48': 'Sabah', '49': 'Sabah',
            '13': 'Sarawak', '50': 'Sarawak', '51': 'Sarawak', '52': 'Sarawak', '53': 'Sarawak',
            '10': 'Selangor', '41': 'Selangor', '42': 'Selangor', '43': 'Selangor', '44': 'Selangor',
            '11': 'Terengganu', '45': 'Terengganu', '46': 'Terengganu',
            '14': 'Kuala Lumpur', '54': 'Kuala Lumpur', '55': 'Kuala Lumpur', '56': 'Kuala Lumpur', '57': 'Kuala Lumpur',
            '15': 'Labuan', '58': 'Labuan',
            '16': 'Putrajaya', '82': 'Putrajaya'
        }
    
    def validate(self, ic_number: str) -> ValidationResult:
        """Validate Malaysian IC number"""
        result = ValidationResult(
            field_name="ic_number",
            field_type=FieldType.IC_NUMBER,
            is_valid=False,
            confidence_score=0.0
        )
        
        if not ic_number or not isinstance(ic_number, str):
            result.error_message = "IC number is required"
            result.validation_level = ValidationLevel.ERROR
            return result
        
        # Clean input
        clean_ic = ic_number.strip().replace('-', '').replace(' ', '')
        
        # Check format
        format_result = self._validate_format(clean_ic)
        result.format_valid = format_result['valid']
        result.format_confidence = format_result['confidence']
        
        if not result.format_valid:
            result.error_message = format_result['error']
            result.suggestions = format_result['suggestions']
            result.validation_level = ValidationLevel.ERROR
            return result
        
        # Extract components
        match = self.ic_pattern.match(clean_ic)
        if match:
            birth_date, state_code, sequence = match.groups()
            
            # Validate birth date
            birth_date_result = self._validate_birth_date(birth_date)
            
            # Validate state code
            state_result = self._validate_state_code(state_code)
            
            # Validate sequence number
            sequence_result = self._validate_sequence(sequence)
            
            # Calculate business rule confidence
            business_rule_scores = [birth_date_result['confidence'], state_result['confidence'], sequence_result['confidence']]
            result.business_rule_confidence = np.mean(business_rule_scores)
            result.business_rule_valid = all([birth_date_result['valid'], state_result['valid'], sequence_result['valid']])
            
            # Store metadata
            result.metadata = {
                'birth_date': birth_date_result['parsed_date'],
                'state': state_result['state'],
                'gender': 'Male' if int(sequence) % 2 == 1 else 'Female',
                'age': birth_date_result['age'],
                'state_code': state_code,
                'sequence': sequence
            }
            
            if result.business_rule_valid:
                result.is_valid = True
                result.validation_level = ValidationLevel.INFO
            else:
                result.error_message = "Business rule validation failed"
                result.validation_level = ValidationLevel.WARNING
                result.suggestions.extend([
                    birth_date_result.get('suggestion', ''),
                    state_result.get('suggestion', ''),
                    sequence_result.get('suggestion', '')
                ])
                result.suggestions = [s for s in result.suggestions if s]
        
        return result
    
    def _validate_format(self, ic_number: str) -> Dict[str, Any]:
        """Validate IC number format"""
        # Check new IC format (12 digits)
        if self.ic_pattern.match(ic_number):
            return {
                'valid': True,
                'confidence': 1.0,
                'format_type': 'new_ic'
            }
        
        # Check old IC format (1 letter + 7 digits)
        if self.old_ic_pattern.match(ic_number.upper()):
            return {
                'valid': True,
                'confidence': 0.8,  # Lower confidence for old format
                'format_type': 'old_ic',
                'suggestion': 'Consider updating to new IC format'
            }
        
        # Check for common format errors
        suggestions = []
        
        if len(ic_number) == 12 and ic_number.isdigit():
            suggestions.append("IC number format should be YYMMDD-SS-####")
        elif len(ic_number) < 12:
            suggestions.append("IC number appears to be incomplete")
        elif len(ic_number) > 12:
            suggestions.append("IC number appears to have extra characters")
        
        if not ic_number.isdigit() and not self.old_ic_pattern.match(ic_number.upper()):
            suggestions.append("IC number should contain only digits (new format) or 1 letter + 7 digits (old format)")
        
        return {
            'valid': False,
            'confidence': 0.0,
            'error': 'Invalid IC number format',
            'suggestions': suggestions
        }
    
    def _validate_birth_date(self, birth_date_str: str) -> Dict[str, Any]:
        """Validate birth date component of IC"""
        try:
            # Parse birth date (YYMMDD)
            year = int(birth_date_str[:2])
            month = int(birth_date_str[2:4])
            day = int(birth_date_str[4:6])
            
            # Determine century (assume 00-30 is 2000s, 31-99 is 1900s)
            if year <= 30:
                full_year = 2000 + year
            else:
                full_year = 1900 + year
            
            # Create date object
            birth_date = date(full_year, month, day)
            
            # Validate date is not in future
            today = date.today()
            if birth_date > today:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'error': 'Birth date cannot be in the future',
                    'suggestion': 'Check birth date format'
                }
            
            # Calculate age
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            
            # Validate reasonable age range
            if age < 0 or age > 150:
                return {
                    'valid': False,
                    'confidence': 0.2,
                    'error': f'Unrealistic age: {age} years',
                    'suggestion': 'Verify birth date accuracy'
                }
            
            return {
                'valid': True,
                'confidence': 1.0,
                'parsed_date': birth_date,
                'age': age
            }
            
        except ValueError as e:
            return {
                'valid': False,
                'confidence': 0.0,
                'error': f'Invalid birth date format: {e}',
                'suggestion': 'Birth date should be in YYMMDD format'
            }
    
    def _validate_state_code(self, state_code: str) -> Dict[str, Any]:
        """Validate state code component of IC"""
        if state_code in self.state_codes:
            return {
                'valid': True,
                'confidence': 1.0,
                'state': self.state_codes[state_code]
            }
        else:
            return {
                'valid': False,
                'confidence': 0.0,
                'error': f'Invalid state code: {state_code}',
                'suggestion': 'State code should be valid Malaysian state identifier'
            }
    
    def _validate_sequence(self, sequence: str) -> Dict[str, Any]:
        """Validate sequence number component of IC"""
        try:
            seq_num = int(sequence)
            
            # Sequence should be 4 digits
            if len(sequence) != 4:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'error': 'Sequence number should be 4 digits',
                    'suggestion': 'Check sequence number format'
                }
            
            # Sequence should not be 0000
            if seq_num == 0:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'error': 'Sequence number cannot be 0000',
                    'suggestion': 'Verify sequence number'
                }
            
            return {
                'valid': True,
                'confidence': 1.0
            }
            
        except ValueError:
            return {
                'valid': False,
                'confidence': 0.0,
                'error': 'Sequence number must be numeric',
                'suggestion': 'Sequence number should contain only digits'
            }

class RegistrationNumberValidator:
    """Validator for vehicle registration numbers"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
        # Malaysian vehicle registration patterns
        self.patterns = {
            'standard': re.compile(r'^[A-Z]{1,3}\s?\d{1,4}\s?[A-Z]?$'),
            'federal': re.compile(r'^W[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]?$'),
            'putrajaya': re.compile(r'^F\s?\d{1,4}\s?[A-Z]{1,3}$'),
            'classic': re.compile(r'^\d{1,4}\s?[A-Z]{1,3}$')
        }
        
        # State prefixes
        self.state_prefixes = {
            'A': 'Perak', 'B': 'Selangor', 'C': 'Pahang', 'D': 'Kelantan',
            'E': 'Terengganu', 'F': 'Putrajaya', 'G': 'Pahang', 'H': 'Selangor',
            'J': 'Johor', 'K': 'Kedah', 'L': 'Kuala Lumpur', 'M': 'Melaka',
            'N': 'Negeri Sembilan', 'P': 'Pulau Pinang', 'Q': 'Sarawak',
            'R': 'Perlis', 'S': 'Sabah', 'T': 'Terengganu', 'U': 'Selangor',
            'V': 'Kuala Lumpur', 'W': 'Federal Territory', 'X': 'Sabah',
            'Y': 'Sarawak', 'Z': 'Sarawak'
        }
    
    def validate(self, registration_number: str) -> ValidationResult:
        """Validate vehicle registration number"""
        result = ValidationResult(
            field_name="registration_number",
            field_type=FieldType.REGISTRATION_NUMBER,
            is_valid=False,
            confidence_score=0.0
        )
        
        if not registration_number or not isinstance(registration_number, str):
            result.error_message = "Registration number is required"
            result.validation_level = ValidationLevel.ERROR
            return result
        
        # Clean input
        clean_reg = registration_number.strip().upper().replace('-', ' ')
        
        # Check against patterns
        pattern_matches = []
        for pattern_name, pattern in self.patterns.items():
            if pattern.match(clean_reg):
                pattern_matches.append(pattern_name)
        
        if pattern_matches:
            result.format_valid = True
            result.format_confidence = 1.0
            result.is_valid = True
            result.validation_level = ValidationLevel.INFO
            
            # Extract state information
            state_info = self._extract_state_info(clean_reg)
            result.metadata = {
                'pattern_type': pattern_matches[0],
                'state': state_info.get('state'),
                'prefix': state_info.get('prefix')
            }
        else:
            result.format_valid = False
            result.format_confidence = 0.0
            result.error_message = "Invalid registration number format"
            result.validation_level = ValidationLevel.ERROR
            result.suggestions = [
                "Registration number should follow Malaysian format (e.g., ABC1234, W1A2345)",
                "Check for correct state prefix and number sequence"
            ]
        
        return result
    
    def _extract_state_info(self, registration_number: str) -> Dict[str, str]:
        """Extract state information from registration number"""
        # Get first letter as potential state prefix
        if registration_number and registration_number[0] in self.state_prefixes:
            return {
                'prefix': registration_number[0],
                'state': self.state_prefixes[registration_number[0]]
            }
        return {}

class PhoneNumberValidator:
    """Validator for phone numbers"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
        # Malaysian phone patterns
        self.mobile_pattern = re.compile(r'^(\+?6?0?1)[0-9]-?[0-9]{7,8}$')
        self.landline_pattern = re.compile(r'^(\+?6?0?[3-9])-?[0-9]{7,8}$')
    
    def validate(self, phone_number: str) -> ValidationResult:
        """Validate phone number"""
        result = ValidationResult(
            field_name="phone_number",
            field_type=FieldType.PHONE_NUMBER,
            is_valid=False,
            confidence_score=0.0
        )
        
        if not phone_number or not isinstance(phone_number, str):
            result.error_message = "Phone number is required"
            result.validation_level = ValidationLevel.ERROR
            return result
        
        # Clean input
        clean_phone = phone_number.strip().replace(' ', '').replace('-', '')
        
        try:
            # Use phonenumbers library for validation
            parsed_number = phonenumbers.parse(clean_phone, "MY")
            
            if phonenumbers.is_valid_number(parsed_number):
                result.format_valid = True
                result.format_confidence = 1.0
                result.business_rule_valid = True
                result.business_rule_confidence = 1.0
                result.is_valid = True
                result.validation_level = ValidationLevel.INFO
                
                # Determine phone type
                number_type = phonenumbers.number_type(parsed_number)
                phone_type = "unknown"
                
                if number_type == phonenumbers.PhoneNumberType.MOBILE:
                    phone_type = "mobile"
                elif number_type == phonenumbers.PhoneNumberType.FIXED_LINE:
                    phone_type = "landline"
                elif number_type == phonenumbers.PhoneNumberType.FIXED_LINE_OR_MOBILE:
                    phone_type = "fixed_line_or_mobile"
                
                result.metadata = {
                    'formatted_number': phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
                    'national_number': phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.NATIONAL),
                    'phone_type': phone_type,
                    'country_code': parsed_number.country_code
                }
            else:
                result.format_valid = False
                result.format_confidence = 0.0
                result.error_message = "Invalid phone number format"
                result.validation_level = ValidationLevel.ERROR
                result.suggestions = [
                    "Phone number should be in valid Malaysian format",
                    "Include country code +60 for international format",
                    "Mobile numbers start with 01, landlines with area codes 03-09"
                ]
                
        except NumberParseException as e:
            result.format_valid = False
            result.format_confidence = 0.0
            result.error_message = f"Phone number parsing error: {e}"
            result.validation_level = ValidationLevel.ERROR
            result.suggestions = [
                "Check phone number format",
                "Ensure proper country code and area code"
            ]
        
        return result

class DateValidator:
    """Validator for dates"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
        # Common date formats
        self.date_formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
            '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d',
            '%d %B %Y', '%d %b %Y',
            '%B %d, %Y', '%b %d, %Y'
        ]
    
    def validate(self, date_str: str, field_name: str = "date") -> ValidationResult:
        """Validate date string"""
        result = ValidationResult(
            field_name=field_name,
            field_type=FieldType.DATE,
            is_valid=False,
            confidence_score=0.0
        )
        
        if not date_str or not isinstance(date_str, str):
            result.error_message = "Date is required"
            result.validation_level = ValidationLevel.ERROR
            return result
        
        # Clean input
        clean_date = date_str.strip()
        
        # Try parsing with different formats
        parsed_date = None
        used_format = None
        
        # Try dateutil parser first (more flexible)
        try:
            parsed_date = date_parser.parse(clean_date, dayfirst=True)
            used_format = "dateutil_parser"
        except (ValueError, TypeError):
            # Try specific formats
            for fmt in self.date_formats:
                try:
                    parsed_date = datetime.strptime(clean_date, fmt)
                    used_format = fmt
                    break
                except ValueError:
                    continue
        
        if parsed_date:
            result.format_valid = True
            result.format_confidence = 1.0
            
            # Business rule validation
            business_result = self._validate_date_business_rules(parsed_date, field_name)
            result.business_rule_valid = business_result['valid']
            result.business_rule_confidence = business_result['confidence']
            
            if business_result['valid']:
                result.is_valid = True
                result.validation_level = ValidationLevel.INFO
            else:
                result.error_message = business_result['error']
                result.validation_level = ValidationLevel.WARNING
                result.suggestions = business_result.get('suggestions', [])
            
            result.metadata = {
                'parsed_date': parsed_date.date() if hasattr(parsed_date, 'date') else parsed_date,
                'format_used': used_format,
                'day_of_week': parsed_date.strftime('%A'),
                'is_weekend': parsed_date.weekday() >= 5
            }
        else:
            result.format_valid = False
            result.format_confidence = 0.0
            result.error_message = "Unable to parse date"
            result.validation_level = ValidationLevel.ERROR
            result.suggestions = [
                "Use common date formats like DD/MM/YYYY or YYYY-MM-DD",
                "Check for typos in month names",
                "Ensure day and month values are valid"
            ]
        
        return result
    
    def _validate_date_business_rules(self, parsed_date: datetime, field_name: str) -> Dict[str, Any]:
        """Validate business rules for dates"""
        today = datetime.now().date()
        date_obj = parsed_date.date() if hasattr(parsed_date, 'date') else parsed_date
        
        # Field-specific business rules
        if field_name in ['birth_date', 'date_of_birth']:
            # Birth date should not be in future
            if date_obj > today:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'error': 'Birth date cannot be in the future',
                    'suggestions': ['Check birth date accuracy']
                }
            
            # Calculate age
            age = today.year - date_obj.year - ((today.month, today.day) < (date_obj.month, date_obj.day))
            
            if age < 0 or age > 150:
                return {
                    'valid': False,
                    'confidence': 0.2,
                    'error': f'Unrealistic age: {age} years',
                    'suggestions': ['Verify birth date accuracy']
                }
        
        elif field_name in ['expiry_date', 'expiration_date']:
            # Expiry date should be in future (with some tolerance)
            if date_obj < today - timedelta(days=30):
                return {
                    'valid': False,
                    'confidence': 0.3,
                    'error': 'Document appears to be expired',
                    'suggestions': ['Check if document needs renewal']
                }
        
        elif field_name in ['issue_date', 'issued_date']:
            # Issue date should not be in future
            if date_obj > today + timedelta(days=1):
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'error': 'Issue date cannot be in the future',
                    'suggestions': ['Check issue date accuracy']
                }
        
        return {
            'valid': True,
            'confidence': 1.0
        }

class AddressValidator:
    """Validator for addresses"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
        # Malaysian postcode patterns
        self.postcode_pattern = re.compile(r'^\d{5}$')
        
        # Load state-postcode mappings
        self.state_postcodes = config.malaysian_postcodes
    
    def validate(self, address: str, postcode: str = None, state: str = None) -> ValidationResult:
        """Validate address"""
        result = ValidationResult(
            field_name="address",
            field_type=FieldType.ADDRESS,
            is_valid=False,
            confidence_score=0.0
        )
        
        if not address or not isinstance(address, str):
            result.error_message = "Address is required"
            result.validation_level = ValidationLevel.ERROR
            return result
        
        # Basic format validation
        clean_address = address.strip()
        
        if len(clean_address) < 10:
            result.format_valid = False
            result.format_confidence = 0.3
            result.error_message = "Address appears to be too short"
            result.validation_level = ValidationLevel.WARNING
            result.suggestions = ["Provide complete address with street, area, and city"]
        else:
            result.format_valid = True
            result.format_confidence = 0.8
        
        # Validate postcode if provided
        postcode_result = None
        if postcode:
            postcode_result = self._validate_postcode(postcode, state)
        
        # Cross-field validation
        cross_field_result = self._validate_address_consistency(clean_address, postcode, state)
        result.cross_field_valid = cross_field_result['valid']
        result.cross_field_confidence = cross_field_result['confidence']
        
        # Overall validation
        if result.format_valid and result.cross_field_valid:
            result.is_valid = True
            result.validation_level = ValidationLevel.INFO
        elif result.format_valid:
            result.validation_level = ValidationLevel.WARNING
            result.error_message = cross_field_result.get('error', 'Address validation warnings')
        
        # Metadata
        result.metadata = {
            'address_length': len(clean_address),
            'word_count': len(clean_address.split()),
            'postcode_result': postcode_result,
            'cross_field_result': cross_field_result
        }
        
        return result
    
    def _validate_postcode(self, postcode: str, state: str = None) -> Dict[str, Any]:
        """Validate Malaysian postcode"""
        if not self.postcode_pattern.match(postcode):
            return {
                'valid': False,
                'confidence': 0.0,
                'error': 'Invalid postcode format (should be 5 digits)',
                'suggestions': ['Postcode should be exactly 5 digits']
            }
        
        # Check state-postcode consistency
        if state and state in self.state_postcodes:
            postcode_ranges = self.state_postcodes[state]
            postcode_int = int(postcode)
            
            valid_for_state = False
            for range_str in postcode_ranges:
                if '-' in range_str:
                    start, end = map(int, range_str.split('-'))
                    if start <= postcode_int <= end:
                        valid_for_state = True
                        break
                else:
                    if postcode_int == int(range_str):
                        valid_for_state = True
                        break
            
            if not valid_for_state:
                return {
                    'valid': False,
                    'confidence': 0.2,
                    'error': f'Postcode {postcode} does not match state {state}',
                    'suggestions': [f'Check postcode for {state} or verify state information']
                }
        
        return {
            'valid': True,
            'confidence': 1.0
        }
    
    def _validate_address_consistency(self, address: str, postcode: str = None, state: str = None) -> Dict[str, Any]:
        """Validate address consistency"""
        confidence = 1.0
        issues = []
        
        # Check if state mentioned in address matches provided state
        if state:
            address_lower = address.lower()
            state_lower = state.lower()
            
            # Check for state name in address
            if state_lower not in address_lower:
                # Use fuzzy matching for state names
                state_matches = process.extractOne(state_lower, [s.lower() for s in self.config.malaysian_states])
                if state_matches and state_matches[1] > 80:
                    confidence *= 0.9
                    issues.append(f"State '{state}' not explicitly mentioned in address")
                else:
                    confidence *= 0.7
                    issues.append(f"State '{state}' does not match address content")
        
        # Check for common address components
        address_components = ['jalan', 'lorong', 'taman', 'bandar', 'kampung', 'kg', 'no', 'lot']
        found_components = sum(1 for comp in address_components if comp in address.lower())
        
        if found_components == 0:
            confidence *= 0.8
            issues.append("Address lacks common Malaysian address components")
        
        return {
            'valid': confidence >= self.config.address_consistency_threshold,
            'confidence': confidence,
            'error': '; '.join(issues) if issues else None,
            'suggestions': issues
        }

class CrossFieldValidator:
    """Validator for cross-field consistency"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate_document_consistency(self, extracted_fields: Dict[str, Any], 
                                    document_type: DocumentType) -> List[ValidationResult]:
        """Validate consistency across document fields"""
        results = []
        
        # IC-based validations
        if 'ic_number' in extracted_fields:
            ic_results = self._validate_ic_consistency(extracted_fields)
            results.extend(ic_results)
        
        # Address-based validations
        if any(field in extracted_fields for field in ['address', 'postcode', 'state']):
            address_results = self._validate_address_consistency(extracted_fields)
            results.extend(address_results)
        
        # Date-based validations
        date_fields = [k for k in extracted_fields.keys() if 'date' in k.lower()]
        if len(date_fields) > 1:
            date_results = self._validate_date_consistency(extracted_fields, date_fields)
            results.extend(date_results)
        
        # Document-specific validations
        if document_type == DocumentType.IDENTITY_CARD:
            results.extend(self._validate_ic_document_consistency(extracted_fields))
        elif document_type == DocumentType.DRIVING_LICENSE:
            results.extend(self._validate_license_consistency(extracted_fields))
        
        return results
    
    def _validate_ic_consistency(self, fields: Dict[str, Any]) -> List[ValidationResult]:
        """Validate IC number consistency with other fields"""
        results = []
        ic_number = fields.get('ic_number')
        
        if not ic_number:
            return results
        
        # Extract IC metadata
        ic_validator = MalaysianICValidator(self.config)
        ic_result = ic_validator.validate(ic_number)
        
        if not ic_result.is_valid or not ic_result.metadata:
            return results
        
        ic_metadata = ic_result.metadata
        
        # Check gender consistency
        if 'gender' in fields:
            gender_result = ValidationResult(
                field_name="gender_consistency",
                field_type=FieldType.GENDER,
                is_valid=False,
                confidence_score=0.0
            )
            
            extracted_gender = fields['gender'].lower() if fields['gender'] else ''
            ic_gender = ic_metadata.get('gender', '').lower()
            
            if extracted_gender in ic_gender or ic_gender in extracted_gender:
                gender_result.is_valid = True
                gender_result.cross_field_valid = True
                gender_result.cross_field_confidence = 1.0
                gender_result.validation_level = ValidationLevel.INFO
            else:
                gender_result.cross_field_valid = False
                gender_result.cross_field_confidence = 0.0
                gender_result.error_message = f"Gender mismatch: IC indicates {ic_gender}, document shows {extracted_gender}"
                gender_result.validation_level = ValidationLevel.ERROR
                gender_result.suggestions = ["Verify gender information against IC number"]
            
            results.append(gender_result)
        
        # Check birth date consistency
        if 'birth_date' in fields or 'date_of_birth' in fields:
            birth_date_field = fields.get('birth_date') or fields.get('date_of_birth')
            
            birth_date_result = ValidationResult(
                field_name="birth_date_consistency",
                field_type=FieldType.DATE,
                is_valid=False,
                confidence_score=0.0
            )
            
            if birth_date_field:
                # Parse extracted birth date
                date_validator = DateValidator(self.config)
                parsed_result = date_validator.validate(str(birth_date_field), 'birth_date')
                
                if parsed_result.is_valid and parsed_result.metadata:
                    extracted_date = parsed_result.metadata['parsed_date']
                    ic_date = ic_metadata.get('birth_date')
                    
                    if ic_date and extracted_date == ic_date:
                        birth_date_result.is_valid = True
                        birth_date_result.cross_field_valid = True
                        birth_date_result.cross_field_confidence = 1.0
                        birth_date_result.validation_level = ValidationLevel.INFO
                    else:
                        birth_date_result.cross_field_valid = False
                        birth_date_result.cross_field_confidence = 0.0
                        birth_date_result.error_message = f"Birth date mismatch: IC indicates {ic_date}, document shows {extracted_date}"
                        birth_date_result.validation_level = ValidationLevel.ERROR
                        birth_date_result.suggestions = ["Verify birth date against IC number"]
            
            results.append(birth_date_result)
        
        # Check state consistency
        if 'state' in fields:
            state_result = ValidationResult(
                field_name="state_consistency",
                field_type=FieldType.STATE,
                is_valid=False,
                confidence_score=0.0
            )
            
            extracted_state = fields['state']
            ic_state = ic_metadata.get('state')
            
            if extracted_state and ic_state:
                # Use fuzzy matching for state comparison
                similarity = fuzz.ratio(extracted_state.lower(), ic_state.lower())
                
                if similarity >= 80:
                    state_result.is_valid = True
                    state_result.cross_field_valid = True
                    state_result.cross_field_confidence = similarity / 100.0
                    state_result.validation_level = ValidationLevel.INFO
                else:
                    state_result.cross_field_valid = False
                    state_result.cross_field_confidence = similarity / 100.0
                    state_result.error_message = f"State mismatch: IC indicates {ic_state}, document shows {extracted_state}"
                    state_result.validation_level = ValidationLevel.WARNING
                    state_result.suggestions = ["Verify state information"]
            
            results.append(state_result)
        
        return results
    
    def _validate_address_consistency(self, fields: Dict[str, Any]) -> List[ValidationResult]:
        """Validate address field consistency"""
        results = []
        
        address = fields.get('address')
        postcode = fields.get('postcode')
        state = fields.get('state')
        
        if address and (postcode or state):
            address_validator = AddressValidator(self.config)
            address_result = address_validator.validate(address, postcode, state)
            
            # Create cross-field validation result
            cross_field_result = ValidationResult(
                field_name="address_cross_field",
                field_type=FieldType.ADDRESS,
                is_valid=address_result.cross_field_valid,
                confidence_score=address_result.cross_field_confidence,
                cross_field_valid=address_result.cross_field_valid,
                cross_field_confidence=address_result.cross_field_confidence,
                validation_level=address_result.validation_level,
                error_message=address_result.error_message,
                suggestions=address_result.suggestions,
                metadata=address_result.metadata
            )
            
            results.append(cross_field_result)
        
        return results
    
    def _validate_date_consistency(self, fields: Dict[str, Any], date_fields: List[str]) -> List[ValidationResult]:
        """Validate consistency between date fields"""
        results = []
        
        # Parse all dates
        parsed_dates = {}
        date_validator = DateValidator(self.config)
        
        for field_name in date_fields:
            if fields.get(field_name):
                date_result = date_validator.validate(str(fields[field_name]), field_name)
                if date_result.is_valid and date_result.metadata:
                    parsed_dates[field_name] = date_result.metadata['parsed_date']
        
        # Check logical date relationships
        if 'issue_date' in parsed_dates and 'expiry_date' in parsed_dates:
            issue_date = parsed_dates['issue_date']
            expiry_date = parsed_dates['expiry_date']
            
            date_order_result = ValidationResult(
                field_name="date_order_consistency",
                field_type=FieldType.DATE,
                is_valid=False,
                confidence_score=0.0
            )
            
            if issue_date < expiry_date:
                date_order_result.is_valid = True
                date_order_result.cross_field_valid = True
                date_order_result.cross_field_confidence = 1.0
                date_order_result.validation_level = ValidationLevel.INFO
            else:
                date_order_result.cross_field_valid = False
                date_order_result.cross_field_confidence = 0.0
                date_order_result.error_message = "Issue date should be before expiry date"
                date_order_result.validation_level = ValidationLevel.ERROR
                date_order_result.suggestions = ["Check date order: issue date should precede expiry date"]
            
            results.append(date_order_result)
        
        return results
    
    def _validate_ic_document_consistency(self, fields: Dict[str, Any]) -> List[ValidationResult]:
        """Validate IC document specific consistency"""
        results = []
        
        # Check if all required IC fields are present
        required_fields = ['ic_number', 'name', 'address']
        missing_fields = [field for field in required_fields if not fields.get(field)]
        
        if missing_fields:
            completeness_result = ValidationResult(
                field_name="ic_completeness",
                field_type=FieldType.IC_NUMBER,
                is_valid=False,
                confidence_score=0.0,
                cross_field_valid=False,
                cross_field_confidence=0.0,
                error_message=f"Missing required fields: {', '.join(missing_fields)}",
                validation_level=ValidationLevel.WARNING,
                suggestions=[f"Ensure {field} is extracted" for field in missing_fields]
            )
            results.append(completeness_result)
        
        return results
    
    def _validate_license_consistency(self, fields: Dict[str, Any]) -> List[ValidationResult]:
        """Validate driving license specific consistency"""
        results = []
        
        # Check license class consistency with vehicle type
        if 'license_class' in fields and 'vehicle_type' in fields:
            license_class = fields['license_class']
            vehicle_type = fields['vehicle_type']
            
            # Define license class mappings
            license_mappings = {
                'D': ['car', 'automobile', 'vehicle'],
                'DA': ['car', 'automobile', 'automatic'],
                'B2': ['motorcycle', 'motorbike', 'bike'],
                'B': ['motorcycle', 'motorbike', 'heavy bike']
            }
            
            license_result = ValidationResult(
                field_name="license_class_consistency",
                field_type=FieldType.REGISTRATION_NUMBER,
                is_valid=False,
                confidence_score=0.0
            )
            
            if license_class in license_mappings:
                valid_vehicles = license_mappings[license_class]
                vehicle_match = any(vehicle.lower() in vehicle_type.lower() for vehicle in valid_vehicles)
                
                if vehicle_match:
                    license_result.is_valid = True
                    license_result.cross_field_valid = True
                    license_result.cross_field_confidence = 1.0
                    license_result.validation_level = ValidationLevel.INFO
                else:
                    license_result.cross_field_valid = False
                    license_result.cross_field_confidence = 0.0
                    license_result.error_message = f"License class {license_class} may not match vehicle type {vehicle_type}"
                    license_result.validation_level = ValidationLevel.WARNING
                    license_result.suggestions = ["Verify license class matches vehicle type"]
            
            results.append(license_result)
        
        return results

class DataValidationEngine:
    """Main data validation engine"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        
        # Initialize validators
        self.ic_validator = MalaysianICValidator(self.config)
        self.registration_validator = RegistrationNumberValidator(self.config)
        self.phone_validator = PhoneNumberValidator(self.config)
        self.date_validator = DateValidator(self.config)
        self.address_validator = AddressValidator(self.config)
        self.cross_field_validator = CrossFieldValidator(self.config)
        
        # Load custom rules if specified
        self.custom_rules = self._load_custom_rules()
    
    def validate_document(self, extracted_fields: Dict[str, Any], 
                         document_type: DocumentType = DocumentType.IDENTITY_CARD) -> Dict[str, ValidationResult]:
        """Validate all fields in a document"""
        results = {}
        
        # Individual field validation
        for field_name, field_value in extracted_fields.items():
            if field_value is not None and str(field_value).strip():
                field_result = self._validate_field(field_name, field_value)
                if field_result:
                    results[field_name] = field_result
        
        # Cross-field validation
        cross_field_results = self.cross_field_validator.validate_document_consistency(
            extracted_fields, document_type
        )
        
        for cross_result in cross_field_results:
            results[cross_result.field_name] = cross_result
        
        return results
    
    def _validate_field(self, field_name: str, field_value: Any) -> Optional[ValidationResult]:
        """Validate individual field based on field name and type"""
        field_name_lower = field_name.lower()
        field_value_str = str(field_value).strip()
        
        # IC number validation
        if 'ic' in field_name_lower and 'number' in field_name_lower:
            return self.ic_validator.validate(field_value_str)
        
        # Registration number validation
        elif 'registration' in field_name_lower or 'plate' in field_name_lower:
            return self.registration_validator.validate(field_value_str)
        
        # Phone number validation
        elif 'phone' in field_name_lower or 'mobile' in field_name_lower or 'tel' in field_name_lower:
            return self.phone_validator.validate(field_value_str)
        
        # Date validation
        elif 'date' in field_name_lower:
            return self.date_validator.validate(field_value_str, field_name)
        
        # Address validation
        elif 'address' in field_name_lower:
            return self.address_validator.validate(field_value_str)
        
        # Email validation
        elif 'email' in field_name_lower:
            return self._validate_email(field_value_str, field_name)
        
        # Postcode validation
        elif 'postcode' in field_name_lower or 'postal' in field_name_lower:
            return self._validate_postcode(field_value_str, field_name)
        
        # Amount validation
        elif 'amount' in field_name_lower or 'price' in field_name_lower or 'total' in field_name_lower:
            return self._validate_amount(field_value_str, field_name)
        
        return None
    
    def _validate_email(self, email: str, field_name: str) -> ValidationResult:
        """Validate email address"""
        result = ValidationResult(
            field_name=field_name,
            field_type=FieldType.EMAIL,
            is_valid=False,
            confidence_score=0.0
        )
        
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        if email_pattern.match(email):
            result.format_valid = True
            result.format_confidence = 1.0
            result.is_valid = True
            result.validation_level = ValidationLevel.INFO
        else:
            result.format_valid = False
            result.format_confidence = 0.0
            result.error_message = "Invalid email format"
            result.validation_level = ValidationLevel.ERROR
            result.suggestions = ["Email should be in format: user@domain.com"]
        
        return result
    
    def _validate_postcode(self, postcode: str, field_name: str) -> ValidationResult:
        """Validate postcode"""
        result = ValidationResult(
            field_name=field_name,
            field_type=FieldType.POSTCODE,
            is_valid=False,
            confidence_score=0.0
        )
        
        postcode_pattern = re.compile(r'^\d{5}$')
        
        if postcode_pattern.match(postcode):
            result.format_valid = True
            result.format_confidence = 1.0
            result.is_valid = True
            result.validation_level = ValidationLevel.INFO
        else:
            result.format_valid = False
            result.format_confidence = 0.0
            result.error_message = "Invalid postcode format"
            result.validation_level = ValidationLevel.ERROR
            result.suggestions = ["Postcode should be exactly 5 digits"]
        
        return result
    
    def _validate_amount(self, amount: str, field_name: str) -> ValidationResult:
        """Validate monetary amount"""
        result = ValidationResult(
            field_name=field_name,
            field_type=FieldType.AMOUNT,
            is_valid=False,
            confidence_score=0.0
        )
        
        # Clean amount string
        clean_amount = re.sub(r'[^\d.,]', '', amount)
        
        try:
            # Try to parse as float
            amount_value = float(clean_amount.replace(',', ''))
            
            if amount_value >= 0:
                result.format_valid = True
                result.format_confidence = 1.0
                result.is_valid = True
                result.validation_level = ValidationLevel.INFO
                result.metadata = {
                    'parsed_amount': amount_value,
                    'formatted_amount': f"RM {amount_value:,.2f}"
                }
            else:
                result.format_valid = False
                result.format_confidence = 0.0
                result.error_message = "Amount cannot be negative"
                result.validation_level = ValidationLevel.ERROR
        
        except ValueError:
            result.format_valid = False
            result.format_confidence = 0.0
            result.error_message = "Invalid amount format"
            result.validation_level = ValidationLevel.ERROR
            result.suggestions = ["Amount should be a valid number"]
        
        return result
    
    def _load_custom_rules(self) -> Dict[str, Any]:
        """Load custom validation rules from file"""
        custom_rules = {}
        
        if self.config.custom_rules_path and Path(self.config.custom_rules_path).exists():
            try:
                with open(self.config.custom_rules_path, 'r') as f:
                    custom_rules = json.load(f)
                logger.info(f"Loaded custom validation rules from {self.config.custom_rules_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom rules: {e}")
        
        return custom_rules
    
    def calculate_document_confidence(self, validation_results: Dict[str, ValidationResult]) -> float:
        """Calculate overall document confidence score"""
        if not validation_results:
            return 0.0
        
        # Weight different validation types
        weights = {
            ValidationLevel.CRITICAL: 0.0,
            ValidationLevel.ERROR: 0.2,
            ValidationLevel.WARNING: 0.6,
            ValidationLevel.INFO: 1.0
        }
        
        total_weight = 0
        weighted_score = 0
        
        for result in validation_results.values():
            weight = weights.get(result.validation_level, 0.5)
            total_weight += 1
            weighted_score += result.confidence_score * weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def get_validation_summary(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Get validation summary statistics"""
        total_fields = len(validation_results)
        valid_fields = sum(1 for r in validation_results.values() if r.is_valid)
        
        level_counts = {}
        for level in ValidationLevel:
            level_counts[level.value] = sum(1 for r in validation_results.values() if r.validation_level == level)
        
        confidence_scores = [r.confidence_score for r in validation_results.values()]
        
        return {
            'total_fields': total_fields,
            'valid_fields': valid_fields,
            'invalid_fields': total_fields - valid_fields,
            'validation_rate': valid_fields / total_fields if total_fields > 0 else 0.0,
            'level_counts': level_counts,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'min_confidence': min(confidence_scores) if confidence_scores else 0.0,
            'max_confidence': max(confidence_scores) if confidence_scores else 0.0,
            'overall_confidence': self.calculate_document_confidence(validation_results)
        }

def main():
    """Main function for standalone execution"""
    # Example usage
    config = ValidationConfig()
    validator = DataValidationEngine(config)
    
    # Sample document data
    sample_document = {
        'ic_number': '901201-14-5678',
        'name': 'Ahmad bin Abdullah',
        'address': 'No. 123, Jalan Bukit Bintang, 55100 Kuala Lumpur',
        'postcode': '55100',
        'state': 'Kuala Lumpur',
        'phone_number': '+60123456789',
        'birth_date': '01/12/1990',
        'gender': 'Male',
        'email': 'ahmad@example.com'
    }
    
    print("=== Document Validation Demo ===")
    print(f"Sample document: {sample_document}")
    print()
    
    # Validate document
    results = validator.validate_document(sample_document, DocumentType.IDENTITY_CARD)
    
    print("=== Validation Results ===")
    for field_name, result in results.items():
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        confidence = f"({result.confidence_score:.2f})"
        level = result.validation_level.value.upper()
        
        print(f"{field_name}: {status} {confidence} [{level}]")
        
        if result.error_message:
            print(f"  Error: {result.error_message}")
        
        if result.suggestions:
            print(f"  Suggestions: {', '.join(result.suggestions)}")
        
        if result.metadata:
            print(f"  Metadata: {result.metadata}")
        
        print()
    
    # Get summary
    summary = validator.get_validation_summary(results)
    print("=== Validation Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print(f"\nOverall document confidence: {summary['overall_confidence']:.2f}")
    
    # Confidence-based recommendations
    overall_confidence = summary['overall_confidence']
    if overall_confidence >= config.high_confidence_threshold:
        print("✓ RECOMMENDATION: Auto-approve document")
    elif overall_confidence >= config.min_confidence_threshold:
        print("⚠ RECOMMENDATION: Manual review required")
    else:
        print("✗ RECOMMENDATION: Reject or request document resubmission")

if __name__ == "__main__":
    main()