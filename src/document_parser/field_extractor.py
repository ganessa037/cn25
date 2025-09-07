#!/usr/bin/env python3
"""
Field Extraction Service

Extracts specific fields from documents using rule-based and ML hybrid approaches.
Supports various Malaysian document types with field-specific validation.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
from pathlib import Path
import spacy
from spacy.matcher import Matcher
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FieldExtractor:
    """
    Field extraction service for Malaysian documents.
    
    Features:
    - Rule-based field extraction using regex patterns
    - Named Entity Recognition (NER) using spaCy
    - Template-based extraction for structured documents
    - Confidence scoring and validation
    """
    
    def __init__(self, 
                 templates_path: Optional[str] = None,
                 spacy_model: str = "en_core_web_sm"):
        """
        Initialize the field extractor.
        
        Args:
            templates_path: Path to document templates directory
            spacy_model: spaCy model name for NER
        """
        self.templates_path = templates_path or self._get_default_templates_path()
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
            self.matcher = Matcher(self.nlp.vocab)
            self._setup_patterns()
            logger.info(f"spaCy model '{spacy_model}' loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}")
            self.nlp = None
            self.matcher = None
        
        # Load document templates
        self.templates = self._load_templates()
        
        # Field extraction patterns
        self.patterns = self._initialize_patterns()
        
        logger.info("FieldExtractor initialized")
    
    def _get_default_templates_path(self) -> str:
        """
        Get default templates directory path.
        
        Returns:
            str: Templates directory path
        """
        return str(Path(__file__).parent.parent.parent / "models" / "document_parser" / "templates")
    
    def _load_templates(self) -> Dict:
        """
        Load document templates from JSON files.
        
        Returns:
            Dict: Document templates
        """
        templates = {}
        templates_dir = Path(self.templates_path)
        
        if templates_dir.exists():
            for template_file in templates_dir.glob("*.json"):
                try:
                    with open(template_file, 'r', encoding='utf-8') as f:
                        template_name = template_file.stem
                        templates[template_name] = json.load(f)
                    logger.info(f"Loaded template: {template_name}")
                except Exception as e:
                    logger.warning(f"Failed to load template {template_file}: {e}")
        
        # Default templates if none found
        if not templates:
            templates = self._get_default_templates()
        
        return templates
    
    def _get_default_templates(self) -> Dict:
        """
        Get default document templates.
        
        Returns:
            Dict: Default templates
        """
        return {
            "mykad": {
                "fields": {
                    "ic_number": {
                        "pattern": r"\b\d{6}-\d{2}-\d{4}\b",
                        "required": True,
                        "validation": "ic_number"
                    },
                    "name": {
                        "pattern": r"(?i)nama[\s:]+([A-Z\s]+)",
                        "required": True,
                        "validation": "name"
                    },
                    "address": {
                        "pattern": r"(?i)alamat[\s:]+([^\n]+(?:\n[^\n]+)*)",
                        "required": False,
                        "validation": "address"
                    }
                }
            },
            "spk": {
                "fields": {
                    "candidate_number": {
                        "pattern": r"(?i)no\.?\s*calon[\s:]+([A-Z0-9]+)",
                        "required": True,
                        "validation": "alphanumeric"
                    },
                    "name": {
                        "pattern": r"(?i)nama[\s:]+([A-Z\s]+)",
                        "required": True,
                        "validation": "name"
                    },
                    "school": {
                        "pattern": r"(?i)sekolah[\s:]+([^\n]+)",
                        "required": False,
                        "validation": "text"
                    }
                }
            }
        }
    
    def _initialize_patterns(self) -> Dict:
        """
        Initialize regex patterns for field extraction.
        
        Returns:
            Dict: Compiled regex patterns
        """
        patterns = {
            # Malaysian IC number patterns
            "ic_number": [
                re.compile(r"\b(\d{6}-\d{2}-\d{4})\b"),
                re.compile(r"\b(\d{6}\s\d{2}\s\d{4})\b"),
                re.compile(r"\b(\d{12})\b")
            ],
            
            # Name patterns
            "name": [
                re.compile(r"(?i)nama[\s:]+([A-Z][A-Z\s]+)"),
                re.compile(r"(?i)name[\s:]+([A-Z][A-Z\s]+)"),
                re.compile(r"\b([A-Z][A-Z\s]{2,30})\b")
            ],
            
            # Date patterns
            "date": [
                re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b"),
                re.compile(r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"),
                re.compile(r"\b(\d{1,2}\s+\w+\s+\d{4})\b")
            ],
            
            # Address patterns
            "address": [
                re.compile(r"(?i)alamat[\s:]+([^\n]+(?:\n[^\n]+)*)"),
                re.compile(r"(?i)address[\s:]+([^\n]+(?:\n[^\n]+)*)"),
                re.compile(r"\b(\d+[^\n]*(?:jalan|road|street)[^\n]*)", re.IGNORECASE)
            ],
            
            # Phone number patterns
            "phone": [
                re.compile(r"\b(\+?6?0?1[0-9]-?\d{7,8})\b"),
                re.compile(r"\b(\+?6?0?[2-9]\d{1}-?\d{7,8})\b"),
                re.compile(r"\b(\d{3}-?\d{7,8})\b")
            ],
            
            # Email patterns
            "email": [
                re.compile(r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b")
            ],
            
            # Postcode patterns
            "postcode": [
                re.compile(r"\b(\d{5})\b")
            ]
        }
        
        return patterns
    
    def _setup_patterns(self):
        """
        Setup spaCy matcher patterns.
        """
        if not self.matcher:
            return
        
        # Malaysian IC pattern
        ic_pattern = [{"TEXT": {"REGEX": r"\d{6}"}}, 
                     {"TEXT": "-"}, 
                     {"TEXT": {"REGEX": r"\d{2}"}}, 
                     {"TEXT": "-"}, 
                     {"TEXT": {"REGEX": r"\d{4}"}}]
        self.matcher.add("IC_NUMBER", [ic_pattern])
        
        # Name patterns
        name_pattern = [{"POS": "PROPN"}, {"POS": "PROPN", "OP": "+"}]
        self.matcher.add("PERSON_NAME", [name_pattern])
    
    def extract_fields(self, 
                      text: str, 
                      document_type: str = "auto",
                      coordinates: Optional[List[Dict]] = None) -> Dict:
        """
        Extract fields from document text.
        
        Args:
            text: Document text content
            document_type: Type of document or 'auto' for detection
            coordinates: Optional word coordinates for spatial extraction
            
        Returns:
            Dict: Extracted fields with confidence scores
        """
        try:
            # Auto-detect document type if needed
            if document_type == "auto":
                document_type = self._detect_document_type(text)
            
            # Get template for document type
            template = self.templates.get(document_type.lower(), {})
            
            # Extract fields using multiple methods
            extracted_fields = {}
            
            # Method 1: Template-based extraction
            if template:
                template_fields = self._extract_with_template(text, template)
                extracted_fields.update(template_fields)
            
            # Method 2: Pattern-based extraction
            pattern_fields = self._extract_with_patterns(text)
            
            # Method 3: NER-based extraction
            ner_fields = self._extract_with_ner(text) if self.nlp else {}
            
            # Method 4: Coordinate-based extraction (if available)
            coord_fields = self._extract_with_coordinates(text, coordinates) if coordinates else {}
            
            # Combine and validate results
            final_fields = self._combine_field_results(
                extracted_fields, pattern_fields, ner_fields, coord_fields
            )
            
            # Validate extracted fields
            validated_fields = self._validate_fields(final_fields, document_type)
            
            return {
                "document_type": document_type,
                "fields": validated_fields,
                "extraction_methods": {
                    "template": bool(template),
                    "patterns": True,
                    "ner": self.nlp is not None,
                    "coordinates": bool(coordinates)
                },
                "field_count": len(validated_fields),
                "confidence": self._calculate_overall_confidence(validated_fields)
            }
            
        except Exception as e:
            logger.error(f"Field extraction error: {e}")
            return {
                "document_type": document_type,
                "fields": {},
                "error": str(e),
                "extraction_methods": {},
                "field_count": 0,
                "confidence": 0.0
            }
    
    def _detect_document_type(self, text: str) -> str:
        """
        Auto-detect document type from text content.
        
        Args:
            text: Document text
            
        Returns:
            str: Detected document type
        """
        text_lower = text.lower()
        
        # Check for MyKad indicators
        mykad_indicators = ["kad pengenalan", "mykad", "warganegara", "malaysia"]
        mykad_score = sum(1 for indicator in mykad_indicators if indicator in text_lower)
        
        # Check for SPK indicators
        spk_indicators = ["sijil pelajaran", "spk", "peperiksaan", "sekolah"]
        spk_score = sum(1 for indicator in spk_indicators if indicator in text_lower)
        
        # Check for passport indicators
        passport_indicators = ["passport", "pasport", "travel document"]
        passport_score = sum(1 for indicator in passport_indicators if indicator in text_lower)
        
        # Return type with highest score
        scores = {
            "mykad": mykad_score,
            "spk": spk_score,
            "passport": passport_score
        }
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "unknown"
    
    def _extract_with_template(self, text: str, template: Dict) -> Dict:
        """
        Extract fields using document template.
        
        Args:
            text: Document text
            template: Document template
            
        Returns:
            Dict: Extracted fields
        """
        fields = {}
        template_fields = template.get("fields", {})
        
        for field_name, field_config in template_fields.items():
            pattern = field_config.get("pattern")
            if pattern:
                matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
                if matches:
                    # Take the first match or the longest one
                    value = max(matches, key=len) if isinstance(matches[0], str) else matches[0]
                    fields[field_name] = {
                        "value": value.strip(),
                        "confidence": 0.8,  # Template-based extraction has high confidence
                        "method": "template",
                        "required": field_config.get("required", False)
                    }
        
        return fields
    
    def _extract_with_patterns(self, text: str) -> Dict:
        """
        Extract fields using regex patterns.
        
        Args:
            text: Document text
            
        Returns:
            Dict: Extracted fields
        """
        fields = {}
        
        for field_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    # Take the first valid match
                    value = matches[0].strip() if matches[0] else ""
                    if value and field_type not in fields:
                        fields[field_type] = {
                            "value": value,
                            "confidence": 0.7,  # Pattern-based has good confidence
                            "method": "pattern",
                            "required": False
                        }
                        break
        
        return fields
    
    def _extract_with_ner(self, text: str) -> Dict:
        """
        Extract fields using Named Entity Recognition.
        
        Args:
            text: Document text
            
        Returns:
            Dict: Extracted fields
        """
        if not self.nlp:
            return {}
        
        fields = {}
        
        try:
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ == "PERSON" and "name" not in fields:
                    fields["name"] = {
                        "value": ent.text.strip(),
                        "confidence": 0.6,  # NER has moderate confidence
                        "method": "ner",
                        "required": False
                    }
                elif ent.label_ == "DATE" and "date" not in fields:
                    fields["date"] = {
                        "value": ent.text.strip(),
                        "confidence": 0.6,
                        "method": "ner",
                        "required": False
                    }
            
            # Use matcher for specific patterns
            if self.matcher:
                matches = self.matcher(doc)
                for match_id, start, end in matches:
                    span = doc[start:end]
                    label = self.nlp.vocab.strings[match_id]
                    
                    if label == "IC_NUMBER" and "ic_number" not in fields:
                        fields["ic_number"] = {
                            "value": span.text.strip(),
                            "confidence": 0.8,
                            "method": "ner_matcher",
                            "required": False
                        }
        
        except Exception as e:
            logger.warning(f"NER extraction error: {e}")
        
        return fields
    
    def _extract_with_coordinates(self, text: str, coordinates: List[Dict]) -> Dict:
        """
        Extract fields using spatial/coordinate information.
        
        Args:
            text: Document text
            coordinates: Word coordinates from OCR
            
        Returns:
            Dict: Extracted fields
        """
        fields = {}
        
        # This is a simplified implementation
        # In practice, would use more sophisticated spatial analysis
        
        try:
            # Group words by lines based on y-coordinates
            lines = self._group_words_by_lines(coordinates)
            
            # Look for field labels and extract adjacent values
            for line in lines:
                line_text = " ".join([word["text"] for word in line])
                
                # Check for field indicators
                if "nama" in line_text.lower() or "name" in line_text.lower():
                    # Extract name from this line or next line
                    name_value = self._extract_field_value_from_line(line, "name")
                    if name_value and "name" not in fields:
                        fields["name"] = {
                            "value": name_value,
                            "confidence": 0.7,
                            "method": "coordinate",
                            "required": False
                        }
        
        except Exception as e:
            logger.warning(f"Coordinate extraction error: {e}")
        
        return fields
    
    def _group_words_by_lines(self, coordinates: List[Dict]) -> List[List[Dict]]:
        """
        Group words into lines based on y-coordinates.
        
        Args:
            coordinates: Word coordinates
            
        Returns:
            List[List[Dict]]: Words grouped by lines
        """
        # Sort by y-coordinate
        sorted_words = sorted(coordinates, key=lambda w: w.get("bbox", {}).get("y", 0))
        
        lines = []
        current_line = []
        current_y = None
        y_threshold = 10  # Pixels
        
        for word in sorted_words:
            bbox = word.get("bbox", {})
            y = bbox.get("y", 0)
            
            if current_y is None or abs(y - current_y) <= y_threshold:
                current_line.append(word)
                current_y = y
            else:
                if current_line:
                    lines.append(sorted(current_line, key=lambda w: w.get("bbox", {}).get("x", 0)))
                current_line = [word]
                current_y = y
        
        if current_line:
            lines.append(sorted(current_line, key=lambda w: w.get("bbox", {}).get("x", 0)))
        
        return lines
    
    def _extract_field_value_from_line(self, line: List[Dict], field_type: str) -> Optional[str]:
        """
        Extract field value from a line of words.
        
        Args:
            line: List of word dictionaries
            field_type: Type of field to extract
            
        Returns:
            Optional[str]: Extracted value
        """
        line_text = " ".join([word["text"] for word in line])
        
        # Use patterns to extract value
        if field_type in self.patterns:
            for pattern in self.patterns[field_type]:
                matches = pattern.findall(line_text)
                if matches:
                    return matches[0].strip()
        
        return None
    
    def _combine_field_results(self, *field_dicts) -> Dict:
        """
        Combine field extraction results from multiple methods.
        
        Args:
            *field_dicts: Variable number of field dictionaries
            
        Returns:
            Dict: Combined fields
        """
        combined = {}
        
        for field_dict in field_dicts:
            for field_name, field_data in field_dict.items():
                if field_name not in combined:
                    combined[field_name] = field_data
                else:
                    # Keep the one with higher confidence
                    if field_data.get("confidence", 0) > combined[field_name].get("confidence", 0):
                        combined[field_name] = field_data
        
        return combined
    
    def _validate_fields(self, fields: Dict, document_type: str) -> Dict:
        """
        Validate extracted fields.
        
        Args:
            fields: Extracted fields
            document_type: Document type
            
        Returns:
            Dict: Validated fields
        """
        validated = {}
        
        for field_name, field_data in fields.items():
            value = field_data.get("value", "")
            
            # Apply field-specific validation
            is_valid, cleaned_value = self._validate_field_value(field_name, value)
            
            if is_valid:
                validated[field_name] = field_data.copy()
                validated[field_name]["value"] = cleaned_value
                validated[field_name]["valid"] = True
            else:
                # Keep invalid fields but mark them
                validated[field_name] = field_data.copy()
                validated[field_name]["valid"] = False
                validated[field_name]["confidence"] *= 0.5  # Reduce confidence
        
        return validated
    
    def _validate_field_value(self, field_name: str, value: str) -> Tuple[bool, str]:
        """
        Validate a specific field value.
        
        Args:
            field_name: Name of the field
            value: Field value
            
        Returns:
            Tuple[bool, str]: (is_valid, cleaned_value)
        """
        if not value or not value.strip():
            return False, value
        
        cleaned_value = value.strip()
        
        if field_name == "ic_number":
            # Validate Malaysian IC format
            ic_pattern = re.compile(r"^\d{6}-?\d{2}-?\d{4}$")
            if ic_pattern.match(cleaned_value.replace(" ", "")):
                # Normalize format
                digits = re.sub(r"[^\d]", "", cleaned_value)
                if len(digits) == 12:
                    cleaned_value = f"{digits[:6]}-{digits[6:8]}-{digits[8:]}"
                    return True, cleaned_value
            return False, cleaned_value
        
        elif field_name == "name":
            # Validate name format
            if len(cleaned_value) >= 2 and cleaned_value.replace(" ", "").isalpha():
                # Capitalize properly
                cleaned_value = " ".join(word.capitalize() for word in cleaned_value.split())
                return True, cleaned_value
            return False, cleaned_value
        
        elif field_name == "phone":
            # Validate Malaysian phone format
            phone_digits = re.sub(r"[^\d]", "", cleaned_value)
            if len(phone_digits) >= 9 and phone_digits.startswith(("01", "03", "04", "05", "06", "07", "08", "09")):
                return True, cleaned_value
            return False, cleaned_value
        
        elif field_name == "email":
            # Basic email validation
            email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
            return bool(email_pattern.match(cleaned_value)), cleaned_value
        
        elif field_name == "postcode":
            # Malaysian postcode validation
            if cleaned_value.isdigit() and len(cleaned_value) == 5:
                return True, cleaned_value
            return False, cleaned_value
        
        # Default validation - just check if not empty
        return len(cleaned_value) > 0, cleaned_value
    
    def _calculate_overall_confidence(self, fields: Dict) -> float:
        """
        Calculate overall confidence score for extracted fields.
        
        Args:
            fields: Extracted fields
            
        Returns:
            float: Overall confidence score
        """
        if not fields:
            return 0.0
        
        confidences = [field.get("confidence", 0) for field in fields.values()]
        valid_fields = [field for field in fields.values() if field.get("valid", True)]
        
        # Weight by validity
        validity_bonus = len(valid_fields) / len(fields) if fields else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return min(avg_confidence * (0.5 + 0.5 * validity_bonus), 1.0)
    
    def get_supported_fields(self, document_type: str = None) -> List[str]:
        """
        Get list of supported fields for extraction.
        
        Args:
            document_type: Specific document type or None for all
            
        Returns:
            List[str]: Supported field names
        """
        if document_type and document_type in self.templates:
            return list(self.templates[document_type].get("fields", {}).keys())
        
        # Return all possible fields
        all_fields = set()
        for template in self.templates.values():
            all_fields.update(template.get("fields", {}).keys())
        
        # Add pattern-based fields
        all_fields.update(self.patterns.keys())
        
        return sorted(list(all_fields))
    
    def get_extraction_info(self) -> Dict:
        """
        Get information about the extraction service.
        
        Returns:
            Dict: Service information
        """
        return {
            "templates_loaded": list(self.templates.keys()),
            "spacy_available": self.nlp is not None,
            "pattern_types": list(self.patterns.keys()),
            "templates_path": self.templates_path,
            "supported_documents": list(self.templates.keys())
        }