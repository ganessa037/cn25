#!/usr/bin/env python3
"""
Information Extraction for Document Parser

This module provides Named Entity Recognition (NER) and template matching
capabilities for extracting structured information from Malaysian documents,
following the organizational patterns established by the autocorrect model.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Pattern
from datetime import datetime, date
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from difflib import SequenceMatcher

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

@dataclass
class ExtractionConfig:
    """Configuration for information extraction"""
    extraction_methods: List[str] = None  # ['regex', 'ner', 'template']
    confidence_threshold: float = 0.7
    fuzzy_matching: bool = True
    fuzzy_threshold: float = 0.8
    validation_rules: bool = True
    format_standardization: bool = True
    language_support: List[str] = None
    template_matching_threshold: float = 0.6
    field_proximity_threshold: int = 100  # pixels
    
    def __post_init__(self):
        if self.extraction_methods is None:
            self.extraction_methods = ['regex', 'ner', 'template']
        if self.language_support is None:
            self.language_support = ['en', 'ms', 'chi_sim', 'tam']
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionConfig':
        return cls(**data)

@dataclass
class ExtractedField:
    """Extracted field information"""
    field_name: str
    value: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    extraction_method: str
    validation_status: str  # 'valid', 'invalid', 'warning'
    standardized_value: Optional[str] = None
    language: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class BaseExtractor(ABC):
    """Abstract base class for extractors"""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(f'{self.__class__.__name__}')
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
    
    @abstractmethod
    def extract(self, text: str, bbox_info: Optional[Dict] = None) -> List[ExtractedField]:
        """Extract information from text"""
        pass

class RegexExtractor(BaseExtractor):
    """Regex-based information extractor"""
    
    def __init__(self, config: ExtractionConfig):
        super().__init__(config)
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, Dict[str, Pattern]]:
        """Load regex patterns for Malaysian documents"""
        patterns = {
            # MyKad (Malaysian IC) patterns
            'ic_number': {
                'pattern': re.compile(r'\b\d{6}-\d{2}-\d{4}\b'),
                'description': 'Malaysian IC Number (YYMMDD-PB-NNNN)'
            },
            'ic_number_no_dash': {
                'pattern': re.compile(r'\b\d{12}\b'),
                'description': 'Malaysian IC Number without dashes'
            },
            
            # Names (multilingual support)
            'name': {
                'pattern': re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'),
                'description': 'Name in English/Malay'
            },
            'chinese_name': {
                'pattern': re.compile(r'[\u4e00-\u9fff]+'),
                'description': 'Chinese name'
            },
            'tamil_name': {
                'pattern': re.compile(r'[\u0b80-\u0bff]+'),
                'description': 'Tamil name'
            },
            
            # Addresses
            'address': {
                'pattern': re.compile(r'(?:No\.?\s*\d+[A-Z]?[,-]?\s*)?(?:Jalan|Jln|Lorong|Taman|Kg|Kampung)\s+[A-Za-z0-9\s,.-]+', re.IGNORECASE),
                'description': 'Malaysian address'
            },
            'postcode': {
                'pattern': re.compile(r'\b\d{5}\b'),
                'description': 'Malaysian postcode'
            },
            
            # Vehicle registration
            'vehicle_reg': {
                'pattern': re.compile(r'\b[A-Z]{1,3}\s*\d{1,4}\s*[A-Z]?\b'),
                'description': 'Malaysian vehicle registration'
            },
            
            # Dates
            'date_dmy': {
                'pattern': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
                'description': 'Date in DD/MM/YYYY or DD-MM-YYYY format'
            },
            'date_mdy': {
                'pattern': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
                'description': 'Date in MM/DD/YYYY or MM-DD-YYYY format'
            },
            
            # Phone numbers
            'phone_mobile': {
                'pattern': re.compile(r'\b(?:\+?60)?\s*1[0-9]-?\d{3,4}-?\d{4}\b'),
                'description': 'Malaysian mobile number'
            },
            'phone_landline': {
                'pattern': re.compile(r'\b(?:\+?60)?\s*[3-9]-?\d{3,4}-?\d{4}\b'),
                'description': 'Malaysian landline number'
            },
            
            # Email
            'email': {
                'pattern': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
                'description': 'Email address'
            },
            
            # Gender
            'gender': {
                'pattern': re.compile(r'\b(?:LELAKI|PEREMPUAN|MALE|FEMALE|M|F)\b', re.IGNORECASE),
                'description': 'Gender'
            },
            
            # Race/Ethnicity
            'race': {
                'pattern': re.compile(r'\b(?:MELAYU|CINA|INDIA|LAIN-LAIN|MALAY|CHINESE|INDIAN|OTHERS)\b', re.IGNORECASE),
                'description': 'Race/Ethnicity'
            },
            
            # Religion
            'religion': {
                'pattern': re.compile(r'\b(?:ISLAM|BUDDHA|KRISTIAN|HINDU|LAIN-LAIN|BUDDHIST|CHRISTIAN|OTHERS)\b', re.IGNORECASE),
                'description': 'Religion'
            },
            
            # Citizenship
            'citizenship': {
                'pattern': re.compile(r'\b(?:WARGANEGARA|BUKAN WARGANEGARA|CITIZEN|NON-CITIZEN)\b', re.IGNORECASE),
                'description': 'Citizenship status'
            }
        }
        
        return patterns
    
    def extract(self, text: str, bbox_info: Optional[Dict] = None) -> List[ExtractedField]:
        """Extract information using regex patterns"""
        extracted_fields = []
        
        for field_name, pattern_info in self.patterns.items():
            pattern = pattern_info['pattern']
            matches = pattern.finditer(text)
            
            for match in matches:
                value = match.group().strip()
                
                # Calculate confidence based on pattern specificity
                confidence = self._calculate_regex_confidence(field_name, value)
                
                if confidence >= self.config.confidence_threshold:
                    # Validate the extracted value
                    validation_status = self._validate_field(field_name, value)
                    
                    # Standardize the value
                    standardized_value = self._standardize_value(field_name, value) if self.config.format_standardization else None
                    
                    field = ExtractedField(
                        field_name=field_name,
                        value=value,
                        confidence=confidence,
                        bbox=None,  # Regex doesn't provide bbox info
                        extraction_method='regex',
                        validation_status=validation_status,
                        standardized_value=standardized_value
                    )
                    
                    extracted_fields.append(field)
        
        return extracted_fields
    
    def _calculate_regex_confidence(self, field_name: str, value: str) -> float:
        """Calculate confidence score for regex matches"""
        base_confidence = 0.8
        
        # Adjust confidence based on field type and value characteristics
        if field_name == 'ic_number':
            if len(value.replace('-', '')) == 12:
                return 0.95
            else:
                return 0.7
        elif field_name == 'vehicle_reg':
            if re.match(r'^[A-Z]{1,3}\s*\d{1,4}\s*[A-Z]?$', value):
                return 0.9
            else:
                return 0.6
        elif field_name in ['phone_mobile', 'phone_landline']:
            if '+60' in value or value.startswith('01'):
                return 0.9
            else:
                return 0.7
        elif field_name == 'email':
            if '@' in value and '.' in value:
                return 0.95
            else:
                return 0.6
        
        return base_confidence
    
    def _validate_field(self, field_name: str, value: str) -> str:
        """Validate extracted field values"""
        if not self.config.validation_rules:
            return 'valid'
        
        try:
            if field_name == 'ic_number':
                return self._validate_ic_number(value)
            elif field_name == 'postcode':
                return self._validate_postcode(value)
            elif field_name in ['date_dmy', 'date_mdy']:
                return self._validate_date(value)
            elif field_name == 'email':
                return self._validate_email(value)
            else:
                return 'valid'
        except Exception:
            return 'invalid'
    
    def _validate_ic_number(self, ic_number: str) -> str:
        """Validate Malaysian IC number"""
        # Remove dashes
        ic_clean = ic_number.replace('-', '')
        
        if len(ic_clean) != 12:
            return 'invalid'
        
        # Check if all digits
        if not ic_clean.isdigit():
            return 'invalid'
        
        # Validate birth date part (first 6 digits)
        birth_date = ic_clean[:6]
        year = int(birth_date[:2])
        month = int(birth_date[2:4])
        day = int(birth_date[4:6])
        
        # Assume years 00-30 are 2000-2030, 31-99 are 1931-1999
        if year <= 30:
            full_year = 2000 + year
        else:
            full_year = 1900 + year
        
        # Validate month and day
        if month < 1 or month > 12:
            return 'invalid'
        if day < 1 or day > 31:
            return 'invalid'
        
        # Additional date validation
        try:
            date(full_year, month, day)
        except ValueError:
            return 'invalid'
        
        return 'valid'
    
    def _validate_postcode(self, postcode: str) -> str:
        """Validate Malaysian postcode"""
        if len(postcode) != 5 or not postcode.isdigit():
            return 'invalid'
        
        # Malaysian postcodes range from 01000 to 98859
        code = int(postcode)
        if code < 1000 or code > 98859:
            return 'invalid'
        
        return 'valid'
    
    def _validate_date(self, date_str: str) -> str:
        """Validate date string"""
        try:
            # Try different date formats
            for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%y', '%d-%m-%y']:
                try:
                    datetime.strptime(date_str, fmt)
                    return 'valid'
                except ValueError:
                    continue
            return 'invalid'
        except Exception:
            return 'invalid'
    
    def _validate_email(self, email: str) -> str:
        """Validate email address"""
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if email_pattern.match(email):
            return 'valid'
        else:
            return 'invalid'
    
    def _standardize_value(self, field_name: str, value: str) -> str:
        """Standardize field values"""
        if field_name == 'ic_number':
            # Standardize IC number format
            clean_ic = value.replace('-', '').replace(' ', '')
            if len(clean_ic) == 12:
                return f"{clean_ic[:6]}-{clean_ic[6:8]}-{clean_ic[8:]}"
        elif field_name == 'phone_mobile' or field_name == 'phone_landline':
            # Standardize phone number format
            clean_phone = re.sub(r'[^0-9+]', '', value)
            if clean_phone.startswith('60'):
                clean_phone = '+' + clean_phone
            elif clean_phone.startswith('0'):
                clean_phone = '+60' + clean_phone[1:]
            return clean_phone
        elif field_name == 'vehicle_reg':
            # Standardize vehicle registration
            return value.upper().replace(' ', '')
        elif field_name in ['gender', 'race', 'religion', 'citizenship']:
            # Standardize categorical values
            return value.upper()
        
        return value

class NERExtractor(BaseExtractor):
    """Named Entity Recognition extractor"""
    
    def __init__(self, config: ExtractionConfig):
        super().__init__(config)
        self.nlp_models = self._load_nlp_models()
    
    def _load_nlp_models(self) -> Dict[str, Any]:
        """Load NLP models for different languages"""
        models = {}
        
        try:
            import spacy
            
            # Load English model
            try:
                models['en'] = spacy.load('en_core_web_sm')
            except OSError:
                self.logger.warning("English spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            
            # Load multilingual model if available
            try:
                models['xx'] = spacy.load('xx_core_web_sm')
            except OSError:
                self.logger.warning("Multilingual spaCy model not found. Install with: python -m spacy download xx_core_web_sm")
                
        except ImportError:
            self.logger.error("spaCy not installed. Install with: pip install spacy")
        
        return models
    
    def extract(self, text: str, bbox_info: Optional[Dict] = None) -> List[ExtractedField]:
        """Extract information using NER"""
        extracted_fields = []
        
        if not self.nlp_models:
            return extracted_fields
        
        # Use English model as primary, fallback to multilingual
        nlp = self.nlp_models.get('en') or self.nlp_models.get('xx')
        
        if not nlp:
            return extracted_fields
        
        try:
            doc = nlp(text)
            
            for ent in doc.ents:
                field_name = self._map_entity_label(ent.label_)
                
                if field_name:
                    confidence = self._calculate_ner_confidence(ent)
                    
                    if confidence >= self.config.confidence_threshold:
                        validation_status = self._validate_field(field_name, ent.text)
                        standardized_value = self._standardize_value(field_name, ent.text) if self.config.format_standardization else None
                        
                        field = ExtractedField(
                            field_name=field_name,
                            value=ent.text,
                            confidence=confidence,
                            bbox=None,
                            extraction_method='ner',
                            validation_status=validation_status,
                            standardized_value=standardized_value
                        )
                        
                        extracted_fields.append(field)
                        
        except Exception as e:
            self.logger.error(f"NER extraction failed: {e}")
        
        return extracted_fields
    
    def _map_entity_label(self, label: str) -> Optional[str]:
        """Map spaCy entity labels to our field names"""
        label_mapping = {
            'PERSON': 'name',
            'ORG': 'organization',
            'GPE': 'location',
            'LOC': 'location',
            'DATE': 'date',
            'TIME': 'time',
            'MONEY': 'amount',
            'CARDINAL': 'number',
            'ORDINAL': 'number'
        }
        
        return label_mapping.get(label)
    
    def _calculate_ner_confidence(self, entity) -> float:
        """Calculate confidence for NER entities"""
        # spaCy doesn't provide confidence scores directly
        # Use entity length and type as confidence indicators
        base_confidence = 0.7
        
        # Longer entities are generally more reliable
        length_bonus = min(len(entity.text) * 0.02, 0.2)
        
        # Certain entity types are more reliable
        if entity.label_ in ['PERSON', 'DATE', 'MONEY']:
            type_bonus = 0.1
        else:
            type_bonus = 0.0
        
        return min(base_confidence + length_bonus + type_bonus, 1.0)
    
    def _validate_field(self, field_name: str, value: str) -> str:
        """Validate NER extracted fields"""
        # Reuse validation logic from RegexExtractor
        regex_extractor = RegexExtractor(self.config)
        return regex_extractor._validate_field(field_name, value)
    
    def _standardize_value(self, field_name: str, value: str) -> str:
        """Standardize NER extracted values"""
        # Reuse standardization logic from RegexExtractor
        regex_extractor = RegexExtractor(self.config)
        return regex_extractor._standardize_value(field_name, value)

class TemplateExtractor(BaseExtractor):
    """Template-based information extractor"""
    
    def __init__(self, config: ExtractionConfig):
        super().__init__(config)
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load document templates"""
        templates = {
            'mykad': {
                'fields': {
                    'name': {'keywords': ['NAMA', 'NAME'], 'position': 'after'},
                    'ic_number': {'keywords': ['NO. KAD PENGENALAN', 'IC NO', 'IDENTITY CARD NO'], 'position': 'after'},
                    'gender': {'keywords': ['JANTINA', 'SEX'], 'position': 'after'},
                    'race': {'keywords': ['BANGSA', 'RACE'], 'position': 'after'},
                    'religion': {'keywords': ['AGAMA', 'RELIGION'], 'position': 'after'},
                    'birth_date': {'keywords': ['TARIKH LAHIR', 'DATE OF BIRTH'], 'position': 'after'},
                    'birth_place': {'keywords': ['TEMPAT LAHIR', 'PLACE OF BIRTH'], 'position': 'after'},
                    'address': {'keywords': ['ALAMAT', 'ADDRESS'], 'position': 'after'}
                }
            },
            'vehicle_cert': {
                'fields': {
                    'owner_name': {'keywords': ['NAMA PEMILIK', 'OWNER NAME'], 'position': 'after'},
                    'ic_number': {'keywords': ['NO. KAD PENGENALAN', 'IC NO'], 'position': 'after'},
                    'vehicle_reg': {'keywords': ['NO. PENDAFTARAN', 'REGISTRATION NO'], 'position': 'after'},
                    'vehicle_type': {'keywords': ['JENIS KENDERAAN', 'VEHICLE TYPE'], 'position': 'after'},
                    'engine_no': {'keywords': ['NO. ENJIN', 'ENGINE NO'], 'position': 'after'},
                    'chassis_no': {'keywords': ['NO. CASIS', 'CHASSIS NO'], 'position': 'after'},
                    'make': {'keywords': ['JENAMA', 'MAKE'], 'position': 'after'},
                    'model': {'keywords': ['MODEL'], 'position': 'after'},
                    'year': {'keywords': ['TAHUN', 'YEAR'], 'position': 'after'}
                }
            },
            'spk': {
                'fields': {
                    'name': {'keywords': ['NAMA', 'NAME'], 'position': 'after'},
                    'ic_number': {'keywords': ['NO. KAD PENGENALAN', 'IC NO'], 'position': 'after'},
                    'address': {'keywords': ['ALAMAT', 'ADDRESS'], 'position': 'after'},
                    'postcode': {'keywords': ['POSKOD', 'POSTCODE'], 'position': 'after'},
                    'state': {'keywords': ['NEGERI', 'STATE'], 'position': 'after'},
                    'phone': {'keywords': ['NO. TELEFON', 'PHONE NO'], 'position': 'after'}
                }
            }
        }
        
        return templates
    
    def extract(self, text: str, bbox_info: Optional[Dict] = None) -> List[ExtractedField]:
        """Extract information using template matching"""
        extracted_fields = []
        
        # Detect document type
        doc_type = self._detect_document_type(text)
        
        if doc_type and doc_type in self.templates:
            template = self.templates[doc_type]
            
            for field_name, field_config in template['fields'].items():
                value = self._extract_field_value(text, field_config, bbox_info)
                
                if value:
                    confidence = self._calculate_template_confidence(field_name, value, field_config)
                    
                    if confidence >= self.config.confidence_threshold:
                        validation_status = self._validate_field(field_name, value)
                        standardized_value = self._standardize_value(field_name, value) if self.config.format_standardization else None
                        
                        field = ExtractedField(
                            field_name=field_name,
                            value=value,
                            confidence=confidence,
                            bbox=None,
                            extraction_method='template',
                            validation_status=validation_status,
                            standardized_value=standardized_value
                        )
                        
                        extracted_fields.append(field)
        
        return extracted_fields
    
    def _detect_document_type(self, text: str) -> Optional[str]:
        """Detect document type based on keywords"""
        text_upper = text.upper()
        
        # MyKad indicators
        mykad_keywords = ['MYKAD', 'KAD PENGENALAN', 'IDENTITY CARD', 'MALAYSIA']
        if any(keyword in text_upper for keyword in mykad_keywords):
            return 'mykad'
        
        # Vehicle certificate indicators
        vehicle_keywords = ['GERAN', 'KENDERAAN', 'VEHICLE', 'REGISTRATION', 'PENDAFTARAN']
        if any(keyword in text_upper for keyword in vehicle_keywords):
            return 'vehicle_cert'
        
        # SPK indicators
        spk_keywords = ['SPK', 'SURAT PENGESAHAN KELAHIRAN', 'BIRTH CERTIFICATE']
        if any(keyword in text_upper for keyword in spk_keywords):
            return 'spk'
        
        return None
    
    def _extract_field_value(self, text: str, field_config: Dict[str, Any], bbox_info: Optional[Dict] = None) -> Optional[str]:
        """Extract field value based on template configuration"""
        keywords = field_config['keywords']
        position = field_config['position']
        
        for keyword in keywords:
            # Find keyword in text
            keyword_pattern = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
            match = keyword_pattern.search(text)
            
            if match:
                if position == 'after':
                    # Extract text after the keyword
                    start_pos = match.end()
                    remaining_text = text[start_pos:].strip()
                    
                    # Extract until next line or next keyword
                    lines = remaining_text.split('\n')
                    if lines:
                        value = lines[0].strip()
                        # Clean up common separators
                        value = re.sub(r'^[:\-\s]+', '', value)
                        value = re.sub(r'[\s]+$', '', value)
                        
                        if value:
                            return value
                elif position == 'before':
                    # Extract text before the keyword
                    end_pos = match.start()
                    preceding_text = text[:end_pos].strip()
                    
                    # Extract from previous line or previous keyword
                    lines = preceding_text.split('\n')
                    if lines:
                        value = lines[-1].strip()
                        # Clean up common separators
                        value = re.sub(r'[:\-\s]+$', '', value)
                        value = re.sub(r'^[\s]+', '', value)
                        
                        if value:
                            return value
        
        return None
    
    def _calculate_template_confidence(self, field_name: str, value: str, field_config: Dict[str, Any]) -> float:
        """Calculate confidence for template-based extraction"""
        base_confidence = 0.8
        
        # Adjust confidence based on field characteristics
        if field_name in ['ic_number', 'vehicle_reg']:
            # These have specific formats, so higher confidence if format matches
            if field_name == 'ic_number' and re.match(r'\d{6}-\d{2}-\d{4}', value):
                return 0.95
            elif field_name == 'vehicle_reg' and re.match(r'[A-Z]{1,3}\s*\d{1,4}\s*[A-Z]?', value):
                return 0.9
        
        # Length-based confidence adjustment
        if len(value) < 2:
            return 0.3
        elif len(value) > 100:
            return 0.6
        
        return base_confidence
    
    def _validate_field(self, field_name: str, value: str) -> str:
        """Validate template extracted fields"""
        # Reuse validation logic from RegexExtractor
        regex_extractor = RegexExtractor(self.config)
        return regex_extractor._validate_field(field_name, value)
    
    def _standardize_value(self, field_name: str, value: str) -> str:
        """Standardize template extracted values"""
        # Reuse standardization logic from RegexExtractor
        regex_extractor = RegexExtractor(self.config)
        return regex_extractor._standardize_value(field_name, value)

class InformationExtractor:
    """Main information extraction manager"""
    
    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()
        self.logger = self._setup_logging()
        
        # Initialize extractors
        self.extractors = {}
        self._initialize_extractors()
        
        self.logger.info(f"Information Extractor initialized with methods: {list(self.extractors.keys())}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('InformationExtractor')
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
    
    def _initialize_extractors(self):
        """Initialize extraction methods"""
        if 'regex' in self.config.extraction_methods:
            self.extractors['regex'] = RegexExtractor(self.config)
        
        if 'ner' in self.config.extraction_methods:
            self.extractors['ner'] = NERExtractor(self.config)
        
        if 'template' in self.config.extraction_methods:
            self.extractors['template'] = TemplateExtractor(self.config)
    
    def extract_information(self, text: str, bbox_info: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract information using all configured methods"""
        start_time = datetime.now()
        
        all_extractions = {}
        
        # Run each extractor
        for method_name, extractor in self.extractors.items():
            try:
                extractions = extractor.extract(text, bbox_info)
                all_extractions[method_name] = [field.to_dict() for field in extractions]
            except Exception as e:
                self.logger.error(f"Extraction method {method_name} failed: {e}")
                all_extractions[method_name] = []
        
        # Combine and deduplicate results
        combined_results = self._combine_extractions(all_extractions)
        
        # Post-process results
        final_results = self._post_process_results(combined_results)
        
        return {
            'extracted_fields': final_results,
            'method_results': all_extractions,
            'processing_time': (datetime.now() - start_time).total_seconds(),
            'extraction_date': datetime.now().isoformat(),
            'methods_used': list(self.extractors.keys())
        }
    
    def _combine_extractions(self, all_extractions: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Combine extractions from different methods"""
        field_groups = defaultdict(list)
        
        # Group extractions by field name
        for method_name, extractions in all_extractions.items():
            for extraction in extractions:
                field_name = extraction['field_name']
                extraction['method'] = method_name
                field_groups[field_name].append(extraction)
        
        combined_results = []
        
        # For each field, select the best extraction
        for field_name, extractions in field_groups.items():
            if self.config.fuzzy_matching:
                # Group similar values
                grouped_extractions = self._group_similar_extractions(extractions)
                
                for group in grouped_extractions:
                    best_extraction = max(group, key=lambda x: x['confidence'])
                    combined_results.append(best_extraction)
            else:
                # Just take the highest confidence extraction
                best_extraction = max(extractions, key=lambda x: x['confidence'])
                combined_results.append(best_extraction)
        
        return combined_results
    
    def _group_similar_extractions(self, extractions: List[Dict]) -> List[List[Dict]]:
        """Group similar extractions using fuzzy matching"""
        groups = []
        
        for extraction in extractions:
            added_to_group = False
            
            for group in groups:
                # Check similarity with any item in the group
                for group_item in group:
                    similarity = SequenceMatcher(None, extraction['value'], group_item['value']).ratio()
                    
                    if similarity >= self.config.fuzzy_threshold:
                        group.append(extraction)
                        added_to_group = True
                        break
                
                if added_to_group:
                    break
            
            if not added_to_group:
                groups.append([extraction])
        
        return groups
    
    def _post_process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process extraction results"""
        processed_results = []
        
        for result in results:
            # Apply additional validation rules
            if self.config.validation_rules:
                result = self._apply_validation_rules(result)
            
            # Apply format standardization
            if self.config.format_standardization:
                result = self._apply_format_standardization(result)
            
            processed_results.append(result)
        
        return processed_results
    
    def _apply_validation_rules(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply additional validation rules"""
        field_name = result['field_name']
        value = result['value']
        
        # Cross-field validation
        if field_name == 'ic_number':
            # Additional IC number validation
            if len(value.replace('-', '')) != 12:
                result['validation_status'] = 'invalid'
                result['confidence'] *= 0.5
        
        return result
    
    def _apply_format_standardization(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply format standardization"""
        if not result.get('standardized_value'):
            # Apply standardization if not already done
            regex_extractor = RegexExtractor(self.config)
            standardized = regex_extractor._standardize_value(result['field_name'], result['value'])
            result['standardized_value'] = standardized
        
        return result

def main():
    """Main function for standalone execution"""
    print("🔍 Information Extraction for Malaysian Documents")
    print("=" * 50)
    
    # Initialize extractor
    config = ExtractionConfig(
        extraction_methods=['regex', 'ner', 'template'],
        confidence_threshold=0.7,
        fuzzy_matching=True,
        validation_rules=True
    )
    
    extractor = InformationExtractor(config)
    
    # Display configuration
    print(f"\n⚙️ Configuration:")
    print(f"   Methods: {config.extraction_methods}")
    print(f"   Confidence threshold: {config.confidence_threshold}")
    print(f"   Fuzzy matching: {config.fuzzy_matching}")
    print(f"   Validation rules: {config.validation_rules}")
    print(f"   Available extractors: {list(extractor.extractors.keys())}")
    
    # Example usage
    print("\n📋 Usage Examples:")
    print("1. result = extractor.extract_information(text)")
    print("2. result = extractor.extract_information(text, bbox_info)")
    
    # Example text
    sample_text = """
    NAMA: AHMAD BIN ALI
    NO. KAD PENGENALAN: 901234-12-3456
    JANTINA: LELAKI
    BANGSA: MELAYU
    AGAMA: ISLAM
    ALAMAT: NO. 123, JALAN MERDEKA, TAMAN SETIA
    50000 KUALA LUMPUR
    """
    
    print("\n🧪 Sample Extraction:")
    result = extractor.extract_information(sample_text)
    print(f"   Found {len(result['extracted_fields'])} fields")
    for field in result['extracted_fields'][:3]:  # Show first 3 fields
        print(f"   - {field['field_name']}: {field['value']} (confidence: {field['confidence']:.2f})")
    
    return 0

if __name__ == "__main__":
    exit(main())