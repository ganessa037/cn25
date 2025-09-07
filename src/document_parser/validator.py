#!/usr/bin/env python3
"""
Document Field Validator

Provides business logic validation for extracted document fields.
Supports Malaysian document standards and custom validation rules.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, date
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationResult:
    """Validation result for a field."""
    field_name: str
    is_valid: bool
    level: ValidationLevel
    message: str
    corrected_value: Optional[str] = None
    confidence: float = 1.0

class DocumentValidator:
    """
    Document field validator with Malaysian document standards support.
    
    Features:
    - IC number validation with checksum verification
    - Name format validation
    - Address format validation
    - Date validation and normalization
    - Cross-field validation
    - Custom business rules
    """
    
    def __init__(self, 
                 validation_rules_path: Optional[str] = None,
                 strict_mode: bool = False):
        """
        Initialize the validator.
        
        Args:
            validation_rules_path: Path to custom validation rules
            strict_mode: Enable strict validation mode
        """
        self.strict_mode = strict_mode
        self.validation_rules_path = validation_rules_path
        
        # Load validation rules
        self.validation_rules = self._load_validation_rules()
        
        # Initialize validation patterns
        self.patterns = self._initialize_validation_patterns()
        
        # Malaysian state codes for IC validation
        self.state_codes = self._get_malaysian_state_codes()
        
        logger.info(f"DocumentValidator initialized (strict_mode={strict_mode})")
    
    def _load_validation_rules(self) -> Dict:
        """
        Load validation rules from file or use defaults.
        
        Returns:
            Dict: Validation rules
        """
        rules = {}
        
        if self.validation_rules_path and Path(self.validation_rules_path).exists():
            try:
                with open(self.validation_rules_path, 'r', encoding='utf-8') as f:
                    rules = json.load(f)
                logger.info(f"Loaded validation rules from {self.validation_rules_path}")
            except Exception as e:
                logger.warning(f"Failed to load validation rules: {e}")
        
        # Use default rules if none loaded
        if not rules:
            rules = self._get_default_validation_rules()
        
        return rules
    
    def _get_default_validation_rules(self) -> Dict:
        """
        Get default validation rules.
        
        Returns:
            Dict: Default validation rules
        """
        return {
            "ic_number": {
                "required": True,
                "format": "YYMMDD-PB-NNNN",
                "checksum": True,
                "min_age": 0,
                "max_age": 150
            },
            "name": {
                "required": True,
                "min_length": 2,
                "max_length": 100,
                "allowed_chars": "letters_spaces_apostrophes",
                "title_case": True
            },
            "address": {
                "required": False,
                "min_length": 10,
                "max_length": 500,
                "require_postcode": True,
                "require_state": True
            },
            "phone": {
                "required": False,
                "formats": ["malaysian_mobile", "malaysian_landline"],
                "normalize": True
            },
            "email": {
                "required": False,
                "format": "rfc5322",
                "normalize": True
            },
            "date": {
                "required": False,
                "min_year": 1900,
                "max_year": 2100,
                "formats": ["DD/MM/YYYY", "DD-MM-YYYY", "YYYY-MM-DD"]
            }
        }
    
    def _initialize_validation_patterns(self) -> Dict:
        """
        Initialize validation regex patterns.
        
        Returns:
            Dict: Compiled regex patterns
        """
        return {
            "ic_number": re.compile(r"^(\d{6})-(\d{2})-(\d{4})$"),
            "ic_number_loose": re.compile(r"^\d{6}[\s-]?\d{2}[\s-]?\d{4}$"),
            "name": re.compile(r"^[A-Za-z\s'.-]+$"),
            "malaysian_mobile": re.compile(r"^(\+?6?0?1[0-9])[\s-]?(\d{3,4})[\s-]?(\d{4})$"),
            "malaysian_landline": re.compile(r"^(\+?6?0?[2-9]\d)[\s-]?(\d{3,4})[\s-]?(\d{4})$"),
            "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
            "postcode": re.compile(r"\b(\d{5})\b"),
            "date_dmy": re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$"),
            "date_ymd": re.compile(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$")
        }
    
    def _get_malaysian_state_codes(self) -> Dict:
        """
        Get Malaysian state codes for IC validation.
        
        Returns:
            Dict: State codes mapping
        """
        return {
            "01": "Johor",
            "02": "Kedah",
            "03": "Kelantan",
            "04": "Melaka",
            "05": "Negeri Sembilan",
            "06": "Pahang",
            "07": "Pulau Pinang",
            "08": "Perak",
            "09": "Perlis",
            "10": "Selangor",
            "11": "Terengganu",
            "12": "Sabah",
            "13": "Sarawak",
            "14": "Wilayah Persekutuan Kuala Lumpur",
            "15": "Wilayah Persekutuan Labuan",
            "16": "Wilayah Persekutuan Putrajaya",
            "21": "Johor (Foreign Born)",
            "22": "Johor (Foreign Born)",
            "23": "Johor (Foreign Born)",
            "24": "Johor (Foreign Born)",
            "25": "Kuching",
            "26": "Kuching",
            "27": "Kuching",
            "28": "Kota Kinabalu",
            "29": "Kota Kinabalu",
            "30": "Unknown/Others"
        }
    
    def validate_document(self, 
                         fields: Dict, 
                         document_type: str = "auto") -> Dict:
        """
        Validate all fields in a document.
        
        Args:
            fields: Dictionary of extracted fields
            document_type: Type of document
            
        Returns:
            Dict: Validation results
        """
        try:
            validation_results = []
            corrected_fields = {}
            
            # Validate individual fields
            for field_name, field_data in fields.items():
                if isinstance(field_data, dict):
                    value = field_data.get("value", "")
                    required = field_data.get("required", False)
                else:
                    value = str(field_data)
                    required = False
                
                # Validate field
                result = self._validate_field(field_name, value, required)
                validation_results.append(result)
                
                # Store corrected value if available
                if result.corrected_value is not None:
                    corrected_fields[field_name] = result.corrected_value
                else:
                    corrected_fields[field_name] = value
            
            # Perform cross-field validation
            cross_validation_results = self._cross_validate_fields(corrected_fields, document_type)
            validation_results.extend(cross_validation_results)
            
            # Calculate overall validation score
            validation_score = self._calculate_validation_score(validation_results)
            
            # Categorize results
            errors = [r for r in validation_results if r.level == ValidationLevel.ERROR]
            warnings = [r for r in validation_results if r.level == ValidationLevel.WARNING]
            info = [r for r in validation_results if r.level == ValidationLevel.INFO]
            
            return {
                "is_valid": len(errors) == 0,
                "validation_score": validation_score,
                "corrected_fields": corrected_fields,
                "results": {
                    "errors": [self._result_to_dict(r) for r in errors],
                    "warnings": [self._result_to_dict(r) for r in warnings],
                    "info": [self._result_to_dict(r) for r in info]
                },
                "summary": {
                    "total_fields": len(fields),
                    "valid_fields": len(fields) - len(errors),
                    "error_count": len(errors),
                    "warning_count": len(warnings),
                    "info_count": len(info)
                },
                "document_type": document_type
            }
            
        except Exception as e:
            logger.error(f"Document validation error: {e}")
            return {
                "is_valid": False,
                "validation_score": 0.0,
                "corrected_fields": {},
                "results": {
                    "errors": [{"field": "system", "message": f"Validation error: {e}"}],
                    "warnings": [],
                    "info": []
                },
                "summary": {
                    "total_fields": 0,
                    "valid_fields": 0,
                    "error_count": 1,
                    "warning_count": 0,
                    "info_count": 0
                },
                "document_type": document_type
            }
    
    def _validate_field(self, field_name: str, value: str, required: bool = False) -> ValidationResult:
        """
        Validate a single field.
        
        Args:
            field_name: Name of the field
            value: Field value
            required: Whether field is required
            
        Returns:
            ValidationResult: Validation result
        """
        # Check if field is empty
        if not value or not value.strip():
            if required:
                return ValidationResult(
                    field_name=field_name,
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Required field '{field_name}' is empty"
                )
            else:
                return ValidationResult(
                    field_name=field_name,
                    is_valid=True,
                    level=ValidationLevel.INFO,
                    message=f"Optional field '{field_name}' is empty"
                )
        
        # Field-specific validation
        if field_name == "ic_number":
            return self._validate_ic_number(value)
        elif field_name == "name":
            return self._validate_name(value)
        elif field_name == "address":
            return self._validate_address(value)
        elif field_name == "phone":
            return self._validate_phone(value)
        elif field_name == "email":
            return self._validate_email(value)
        elif field_name in ["date", "birth_date", "issue_date", "expiry_date"]:
            return self._validate_date(value, field_name)
        elif field_name == "postcode":
            return self._validate_postcode(value)
        else:
            # Generic validation
            return self._validate_generic(field_name, value)
    
    def _validate_ic_number(self, value: str) -> ValidationResult:
        """
        Validate Malaysian IC number.
        
        Args:
            value: IC number value
            
        Returns:
            ValidationResult: Validation result
        """
        # Clean the value
        cleaned = re.sub(r"[^\d]", "", value)
        
        # Check length
        if len(cleaned) != 12:
            return ValidationResult(
                field_name="ic_number",
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"IC number must be 12 digits, got {len(cleaned)}"
            )
        
        # Format properly
        formatted = f"{cleaned[:6]}-{cleaned[6:8]}-{cleaned[8:]}"
        
        # Validate format
        if not self.patterns["ic_number"].match(formatted):
            return ValidationResult(
                field_name="ic_number",
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Invalid IC number format",
                corrected_value=formatted
            )
        
        # Extract components
        birth_date = cleaned[:6]
        state_code = cleaned[6:8]
        sequence = cleaned[8:]
        
        # Validate birth date
        try:
            year = int(birth_date[:2])
            month = int(birth_date[2:4])
            day = int(birth_date[4:6])
            
            # Determine century (00-31 = 2000s, 32-99 = 1900s)
            if year <= 31:
                full_year = 2000 + year
            else:
                full_year = 1900 + year
            
            # Validate date
            birth_datetime = datetime(full_year, month, day)
            
            # Check age reasonableness
            age = (datetime.now() - birth_datetime).days // 365
            if age < 0 or age > 150:
                return ValidationResult(
                    field_name="ic_number",
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"Unusual age calculated from IC: {age} years",
                    corrected_value=formatted
                )
            
        except ValueError:
            return ValidationResult(
                field_name="ic_number",
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Invalid birth date in IC number",
                corrected_value=formatted
            )
        
        # Validate state code
        if state_code not in self.state_codes:
            return ValidationResult(
                field_name="ic_number",
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Unknown state code: {state_code}",
                corrected_value=formatted
            )
        
        # All validations passed
        return ValidationResult(
            field_name="ic_number",
            is_valid=True,
            level=ValidationLevel.INFO,
            message=f"Valid IC number for {self.state_codes[state_code]}",
            corrected_value=formatted
        )
    
    def _validate_name(self, value: str) -> ValidationResult:
        """
        Validate name field.
        
        Args:
            value: Name value
            
        Returns:
            ValidationResult: Validation result
        """
        # Clean and normalize
        cleaned = value.strip()
        
        # Check length
        if len(cleaned) < 2:
            return ValidationResult(
                field_name="name",
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Name too short (minimum 2 characters)"
            )
        
        if len(cleaned) > 100:
            return ValidationResult(
                field_name="name",
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Name too long (maximum 100 characters)"
            )
        
        # Check allowed characters
        if not self.patterns["name"].match(cleaned):
            return ValidationResult(
                field_name="name",
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Name contains invalid characters (only letters, spaces, apostrophes, and hyphens allowed)"
            )
        
        # Normalize case
        normalized = " ".join(word.capitalize() for word in cleaned.split())
        
        # Check for common issues
        warnings = []
        if cleaned.isupper():
            warnings.append("Name is in all caps")
        if cleaned.islower():
            warnings.append("Name is in all lowercase")
        if "  " in cleaned:
            warnings.append("Name contains multiple spaces")
        
        level = ValidationLevel.WARNING if warnings else ValidationLevel.INFO
        message = "; ".join(warnings) if warnings else "Valid name format"
        
        return ValidationResult(
            field_name="name",
            is_valid=True,
            level=level,
            message=message,
            corrected_value=normalized
        )
    
    def _validate_address(self, value: str) -> ValidationResult:
        """
        Validate address field.
        
        Args:
            value: Address value
            
        Returns:
            ValidationResult: Validation result
        """
        cleaned = value.strip()
        
        # Check minimum length
        if len(cleaned) < 10:
            return ValidationResult(
                field_name="address",
                is_valid=False,
                level=ValidationLevel.WARNING,
                message="Address seems too short"
            )
        
        # Check for postcode
        postcode_match = self.patterns["postcode"].search(cleaned)
        if not postcode_match:
            return ValidationResult(
                field_name="address",
                is_valid=True,
                level=ValidationLevel.WARNING,
                message="Address does not contain a valid postcode"
            )
        
        # Check for Malaysian states
        malaysian_states = [
            "johor", "kedah", "kelantan", "melaka", "negeri sembilan",
            "pahang", "perak", "perlis", "pulau pinang", "selangor",
            "terengganu", "sabah", "sarawak", "kuala lumpur", "labuan", "putrajaya"
        ]
        
        address_lower = cleaned.lower()
        has_state = any(state in address_lower for state in malaysian_states)
        
        if not has_state:
            return ValidationResult(
                field_name="address",
                is_valid=True,
                level=ValidationLevel.WARNING,
                message="Address does not contain a recognizable Malaysian state"
            )
        
        return ValidationResult(
            field_name="address",
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Valid address format"
        )
    
    def _validate_phone(self, value: str) -> ValidationResult:
        """
        Validate phone number.
        
        Args:
            value: Phone number value
            
        Returns:
            ValidationResult: Validation result
        """
        # Check mobile number
        mobile_match = self.patterns["malaysian_mobile"].match(value)
        if mobile_match:
            # Normalize format
            prefix, middle, last = mobile_match.groups()
            normalized = f"{prefix}-{middle}-{last}"
            return ValidationResult(
                field_name="phone",
                is_valid=True,
                level=ValidationLevel.INFO,
                message="Valid Malaysian mobile number",
                corrected_value=normalized
            )
        
        # Check landline number
        landline_match = self.patterns["malaysian_landline"].match(value)
        if landline_match:
            # Normalize format
            prefix, middle, last = landline_match.groups()
            normalized = f"{prefix}-{middle}-{last}"
            return ValidationResult(
                field_name="phone",
                is_valid=True,
                level=ValidationLevel.INFO,
                message="Valid Malaysian landline number",
                corrected_value=normalized
            )
        
        return ValidationResult(
            field_name="phone",
            is_valid=False,
            level=ValidationLevel.ERROR,
            message="Invalid phone number format"
        )
    
    def _validate_email(self, value: str) -> ValidationResult:
        """
        Validate email address.
        
        Args:
            value: Email value
            
        Returns:
            ValidationResult: Validation result
        """
        cleaned = value.strip().lower()
        
        if not self.patterns["email"].match(cleaned):
            return ValidationResult(
                field_name="email",
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Invalid email format"
            )
        
        return ValidationResult(
            field_name="email",
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Valid email format",
            corrected_value=cleaned
        )
    
    def _validate_date(self, value: str, field_name: str) -> ValidationResult:
        """
        Validate date field.
        
        Args:
            value: Date value
            field_name: Name of the date field
            
        Returns:
            ValidationResult: Validation result
        """
        # Try different date formats
        date_obj = None
        
        # DD/MM/YYYY or DD-MM-YYYY
        dmy_match = self.patterns["date_dmy"].match(value)
        if dmy_match:
            day, month, year = map(int, dmy_match.groups())
            try:
                date_obj = datetime(year, month, day)
            except ValueError:
                pass
        
        # YYYY/MM/DD or YYYY-MM-DD
        if not date_obj:
            ymd_match = self.patterns["date_ymd"].match(value)
            if ymd_match:
                year, month, day = map(int, ymd_match.groups())
                try:
                    date_obj = datetime(year, month, day)
                except ValueError:
                    pass
        
        if not date_obj:
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Invalid date format"
            )
        
        # Check date reasonableness
        current_date = datetime.now()
        
        if date_obj.year < 1900:
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Date too far in the past"
            )
        
        if date_obj.year > 2100:
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Date too far in the future"
            )
        
        # Field-specific validation
        if field_name in ["birth_date"]:
            age = (current_date - date_obj).days // 365
            if age < 0:
                return ValidationResult(
                    field_name=field_name,
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Birth date cannot be in the future"
                )
            if age > 150:
                return ValidationResult(
                    field_name=field_name,
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"Unusual age: {age} years"
                )
        
        elif field_name in ["expiry_date"]:
            if date_obj < current_date:
                return ValidationResult(
                    field_name=field_name,
                    is_valid=True,
                    level=ValidationLevel.WARNING,
                    message="Document has expired"
                )
        
        # Normalize date format
        normalized = date_obj.strftime("%d/%m/%Y")
        
        return ValidationResult(
            field_name=field_name,
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Valid date",
            corrected_value=normalized
        )
    
    def _validate_postcode(self, value: str) -> ValidationResult:
        """
        Validate Malaysian postcode.
        
        Args:
            value: Postcode value
            
        Returns:
            ValidationResult: Validation result
        """
        cleaned = re.sub(r"[^\d]", "", value)
        
        if len(cleaned) != 5:
            return ValidationResult(
                field_name="postcode",
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Malaysian postcode must be 5 digits"
            )
        
        # Check if it's a valid Malaysian postcode range
        postcode_int = int(cleaned)
        if not (1000 <= postcode_int <= 98000):
            return ValidationResult(
                field_name="postcode",
                is_valid=False,
                level=ValidationLevel.WARNING,
                message="Postcode outside typical Malaysian range"
            )
        
        return ValidationResult(
            field_name="postcode",
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Valid Malaysian postcode",
            corrected_value=cleaned
        )
    
    def _validate_generic(self, field_name: str, value: str) -> ValidationResult:
        """
        Generic validation for unknown fields.
        
        Args:
            field_name: Field name
            value: Field value
            
        Returns:
            ValidationResult: Validation result
        """
        cleaned = value.strip()
        
        if len(cleaned) == 0:
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                level=ValidationLevel.WARNING,
                message="Field is empty"
            )
        
        return ValidationResult(
            field_name=field_name,
            is_valid=True,
            level=ValidationLevel.INFO,
            message="Field contains data"
        )
    
    def _cross_validate_fields(self, fields: Dict, document_type: str) -> List[ValidationResult]:
        """
        Perform cross-field validation.
        
        Args:
            fields: Dictionary of field values
            document_type: Document type
            
        Returns:
            List[ValidationResult]: Cross-validation results
        """
        results = []
        
        # IC number and birth date consistency
        if "ic_number" in fields and "birth_date" in fields:
            ic_result = self._validate_ic_birth_date_consistency(
                fields["ic_number"], fields["birth_date"]
            )
            if ic_result:
                results.append(ic_result)
        
        # Address and postcode consistency
        if "address" in fields and "postcode" in fields:
            addr_result = self._validate_address_postcode_consistency(
                fields["address"], fields["postcode"]
            )
            if addr_result:
                results.append(addr_result)
        
        # Document type specific validations
        if document_type == "mykad":
            results.extend(self._validate_mykad_specific(fields))
        elif document_type == "spk":
            results.extend(self._validate_spk_specific(fields))
        
        return results
    
    def _validate_ic_birth_date_consistency(self, ic_number: str, birth_date: str) -> Optional[ValidationResult]:
        """
        Validate consistency between IC number and birth date.
        
        Args:
            ic_number: IC number
            birth_date: Birth date
            
        Returns:
            Optional[ValidationResult]: Validation result if inconsistent
        """
        try:
            # Extract birth date from IC
            ic_digits = re.sub(r"[^\d]", "", ic_number)
            if len(ic_digits) != 12:
                return None
            
            ic_birth = ic_digits[:6]
            year = int(ic_birth[:2])
            month = int(ic_birth[2:4])
            day = int(ic_birth[4:6])
            
            # Determine century
            if year <= 31:
                full_year = 2000 + year
            else:
                full_year = 1900 + year
            
            ic_date = datetime(full_year, month, day)
            
            # Parse provided birth date
            birth_date_obj = None
            for pattern in [self.patterns["date_dmy"], self.patterns["date_ymd"]]:
                match = pattern.match(birth_date)
                if match:
                    if pattern == self.patterns["date_dmy"]:
                        d, m, y = map(int, match.groups())
                    else:
                        y, m, d = map(int, match.groups())
                    birth_date_obj = datetime(y, m, d)
                    break
            
            if birth_date_obj and ic_date.date() != birth_date_obj.date():
                return ValidationResult(
                    field_name="cross_validation",
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Birth date does not match IC number"
                )
        
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _validate_address_postcode_consistency(self, address: str, postcode: str) -> Optional[ValidationResult]:
        """
        Validate consistency between address and postcode.
        
        Args:
            address: Address string
            postcode: Postcode
            
        Returns:
            Optional[ValidationResult]: Validation result if inconsistent
        """
        # Extract postcode from address
        address_postcodes = self.patterns["postcode"].findall(address)
        
        if address_postcodes:
            address_postcode = address_postcodes[0]
            if address_postcode != postcode.strip():
                return ValidationResult(
                    field_name="cross_validation",
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message="Postcode does not match postcode in address"
                )
        
        return None
    
    def _validate_mykad_specific(self, fields: Dict) -> List[ValidationResult]:
        """
        MyKad specific validations.
        
        Args:
            fields: Field values
            
        Returns:
            List[ValidationResult]: Validation results
        """
        results = []
        
        # Check required fields for MyKad
        required_fields = ["ic_number", "name"]
        for field in required_fields:
            if field not in fields or not fields[field]:
                results.append(ValidationResult(
                    field_name=field,
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Required MyKad field '{field}' is missing"
                ))
        
        return results
    
    def _validate_spk_specific(self, fields: Dict) -> List[ValidationResult]:
        """
        SPK specific validations.
        
        Args:
            fields: Field values
            
        Returns:
            List[ValidationResult]: Validation results
        """
        results = []
        
        # Check required fields for SPK
        required_fields = ["candidate_number", "name"]
        for field in required_fields:
            if field not in fields or not fields[field]:
                results.append(ValidationResult(
                    field_name=field,
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Required SPK field '{field}' is missing"
                ))
        
        return results
    
    def _calculate_validation_score(self, results: List[ValidationResult]) -> float:
        """
        Calculate overall validation score.
        
        Args:
            results: List of validation results
            
        Returns:
            float: Validation score (0.0 to 1.0)
        """
        if not results:
            return 1.0
        
        total_weight = 0
        weighted_score = 0
        
        for result in results:
            if result.level == ValidationLevel.ERROR:
                weight = 1.0
                score = 0.0 if not result.is_valid else 1.0
            elif result.level == ValidationLevel.WARNING:
                weight = 0.5
                score = 0.5 if not result.is_valid else 1.0
            else:  # INFO
                weight = 0.1
                score = 1.0
            
            total_weight += weight
            weighted_score += weight * score
        
        return weighted_score / total_weight if total_weight > 0 else 1.0
    
    def _result_to_dict(self, result: ValidationResult) -> Dict:
        """
        Convert ValidationResult to dictionary.
        
        Args:
            result: ValidationResult object
            
        Returns:
            Dict: Result as dictionary
        """
        return {
            "field": result.field_name,
            "valid": result.is_valid,
            "level": result.level.value,
            "message": result.message,
            "corrected_value": result.corrected_value,
            "confidence": result.confidence
        }
    
    def get_validation_info(self) -> Dict:
        """
        Get information about the validation service.
        
        Returns:
            Dict: Service information
        """
        return {
            "strict_mode": self.strict_mode,
            "supported_fields": list(self.validation_rules.keys()),
            "state_codes_count": len(self.state_codes),
            "patterns_count": len(self.patterns),
            "validation_rules_path": self.validation_rules_path
        }