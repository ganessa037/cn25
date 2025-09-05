#!/usr/bin/env python3
"""
Year Validation Mechanism
========================

This module implements a validation mechanism for the year of manufacture field
that automatically corrects any input year exceeding 2025 (current year) by
replacing it with 2025. The system maintains a confidence threshold to ensure
consistent correction when users attempt to enter future years (2026 or later).

Features:
- Automatic correction of future years to 2025
- Configurable confidence thresholds
- Integration with existing autocorrect system
- Detailed logging of corrections applied
- Support for various year input formats
"""

import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import json

# Current year constant (can be updated as needed)
CURRENT_YEAR = 2025
MIN_VALID_YEAR = 1980
MAX_VALID_YEAR = CURRENT_YEAR

class YearValidationResult:
    """
    Class to store year validation results
    """
    def __init__(self, original_input: str, corrected_year: str, 
                 correction_applied: bool, confidence_score: float, 
                 correction_reason: str):
        self.original_input = original_input
        self.corrected_year = corrected_year
        self.correction_applied = correction_applied
        self.confidence_score = confidence_score
        self.correction_reason = correction_reason
    
    def to_dict(self) -> Dict:
        return {
            'original_input': self.original_input,
            'corrected_year': self.corrected_year,
            'correction_applied': self.correction_applied,
            'confidence_score': self.confidence_score,
            'correction_reason': self.correction_reason
        }

def normalize_year_input(year_input: Union[str, int, float]) -> str:
    """
    Normalize year input to string format
    """
    if pd.isna(year_input) or year_input == '':
        return ''
    
    # Convert to string and clean
    year_str = str(year_input).strip()
    
    # Remove decimal points for float inputs
    if '.' in year_str:
        year_str = year_str.split('.')[0]
    
    # Remove any non-digit characters except for common OCR mistakes
    year_str = re.sub(r'[^0-9oOlI]', '', year_str)
    
    return year_str

def detect_year_format_issues(year_input: str) -> Tuple[str, float, str]:
    """
    Detect and correct common year format issues
    
    Returns:
        Tuple of (corrected_year, confidence_score, correction_reason)
    """
    if not year_input:
        return '', 0.0, 'empty_input'
    
    original_input = year_input
    corrected = year_input
    confidence = 1.0
    reason = 'no_correction_needed'
    
    # Handle OCR-like mistakes
    ocr_corrections = {
        'o': '0', 'O': '0', 'l': '1', 'I': '1'
    }
    
    for mistake, correction in ocr_corrections.items():
        if mistake in corrected:
            corrected = corrected.replace(mistake, correction)
            confidence = 0.9
            reason = 'ocr_correction'
    
    # Handle truncated years (e.g., '20' -> '2020')
    if len(corrected) == 2 and corrected.isdigit():
        year_int = int(corrected)
        if 80 <= year_int <= 99:
            corrected = '19' + corrected
            confidence = 0.8
            reason = 'century_completion_1900s'
        elif 0 <= year_int <= 30:
            corrected = '20' + corrected
            confidence = 0.8
            reason = 'century_completion_2000s'
    
    # Handle 3-digit years (e.g., '202' -> '2020')
    elif len(corrected) == 3 and corrected.isdigit():
        if corrected.startswith('20'):
            # Try to complete to a reasonable year
            last_digit = corrected[-1]
            if last_digit in ['0', '1', '2']:
                corrected = corrected + '0'
                confidence = 0.7
                reason = 'year_completion'
    
    return corrected, confidence, reason

def validate_year_range(year_str: str, min_year: int = MIN_VALID_YEAR, 
                       max_year: int = MAX_VALID_YEAR) -> Tuple[str, float, str]:
    """
    Validate year is within acceptable range and correct if necessary
    
    Returns:
        Tuple of (corrected_year, confidence_score, correction_reason)
    """
    if not year_str or not year_str.isdigit():
        return year_str, 0.0, 'invalid_format'
    
    year_int = int(year_str)
    
    # Check if year is too far in the future (2026 or later)
    if year_int > max_year:
        return str(max_year), 1.0, f'future_year_corrected_to_{max_year}'
    
    # Check if year is too far in the past
    elif year_int < min_year:
        # For very old years, might be a typo
        if year_int < 1900:
            return str(max_year), 0.8, f'invalid_old_year_corrected_to_{max_year}'
        else:
            return year_str, 0.9, 'old_year_accepted'
    
    # Year is within valid range
    else:
        return year_str, 1.0, 'valid_year'

def apply_year_validation(year_input: Union[str, int, float], 
                         confidence_threshold: float = 0.7) -> YearValidationResult:
    """
    Apply comprehensive year validation with automatic correction
    
    Args:
        year_input: The input year to validate
        confidence_threshold: Minimum confidence score to apply corrections
    
    Returns:
        YearValidationResult object with validation details
    """
    original_input = str(year_input) if year_input else ''
    
    # Step 1: Normalize input
    normalized = normalize_year_input(year_input)
    
    if not normalized:
        return YearValidationResult(
            original_input=original_input,
            corrected_year='',
            correction_applied=False,
            confidence_score=0.0,
            correction_reason='empty_or_invalid_input'
        )
    
    # Step 2: Detect and correct format issues
    format_corrected, format_confidence, format_reason = detect_year_format_issues(normalized)
    
    # Step 3: Validate year range
    final_year, range_confidence, range_reason = validate_year_range(format_corrected)
    
    # Calculate overall confidence
    overall_confidence = min(format_confidence, range_confidence)
    
    # Determine if correction was applied
    correction_applied = (original_input != final_year) and (overall_confidence >= confidence_threshold)
    
    # Combine correction reasons
    if format_reason != 'no_correction_needed' and range_reason != 'valid_year':
        combined_reason = f"{format_reason}+{range_reason}"
    elif format_reason != 'no_correction_needed':
        combined_reason = format_reason
    elif range_reason != 'valid_year':
        combined_reason = range_reason
    else:
        combined_reason = 'no_correction_needed'
    
    return YearValidationResult(
        original_input=original_input,
        corrected_year=final_year if correction_applied else original_input,
        correction_applied=correction_applied,
        confidence_score=overall_confidence,
        correction_reason=combined_reason
    )

def batch_validate_years(year_list: List[Union[str, int, float]], 
                        confidence_threshold: float = 0.7) -> List[YearValidationResult]:
    """
    Apply year validation to a batch of years
    
    Args:
        year_list: List of years to validate
        confidence_threshold: Minimum confidence score to apply corrections
    
    Returns:
        List of YearValidationResult objects
    """
    results = []
    
    for year_input in year_list:
        result = apply_year_validation(year_input, confidence_threshold)
        results.append(result)
    
    return results

def validate_dataframe_years(df: pd.DataFrame, year_column: str, 
                           confidence_threshold: float = 0.7) -> pd.DataFrame:
    """
    Apply year validation to a DataFrame column
    
    Args:
        df: DataFrame containing year data
        year_column: Name of the year column to validate
        confidence_threshold: Minimum confidence score to apply corrections
    
    Returns:
        DataFrame with additional validation columns
    """
    df_copy = df.copy()
    
    # Apply validation to each year
    validation_results = batch_validate_years(
        df_copy[year_column].tolist(), 
        confidence_threshold
    )
    
    # Add validation results to DataFrame
    df_copy[f'{year_column}_original'] = df_copy[year_column]
    df_copy[f'{year_column}_validated'] = [r.corrected_year for r in validation_results]
    df_copy[f'{year_column}_correction_applied'] = [r.correction_applied for r in validation_results]
    df_copy[f'{year_column}_confidence'] = [r.confidence_score for r in validation_results]
    df_copy[f'{year_column}_correction_reason'] = [r.correction_reason for r in validation_results]
    
    return df_copy

def generate_validation_report(validation_results: List[YearValidationResult]) -> Dict:
    """
    Generate a comprehensive validation report
    
    Args:
        validation_results: List of validation results
    
    Returns:
        Dictionary containing validation statistics
    """
    total_records = len(validation_results)
    corrections_applied = sum(1 for r in validation_results if r.correction_applied)
    future_year_corrections = sum(1 for r in validation_results 
                                 if 'future_year_corrected' in r.correction_reason)
    
    # Group by correction reasons
    correction_reasons = {}
    for result in validation_results:
        reason = result.correction_reason
        if reason not in correction_reasons:
            correction_reasons[reason] = 0
        correction_reasons[reason] += 1
    
    # Calculate confidence statistics
    confidence_scores = [r.confidence_score for r in validation_results if r.confidence_score > 0]
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    report = {
        'total_records': total_records,
        'corrections_applied': corrections_applied,
        'correction_rate': corrections_applied / total_records if total_records > 0 else 0,
        'future_year_corrections': future_year_corrections,
        'average_confidence': avg_confidence,
        'correction_reasons': correction_reasons,
        'validation_summary': {
            'valid_years': correction_reasons.get('valid_year', 0) + correction_reasons.get('no_correction_needed', 0),
            'future_years_corrected': future_year_corrections,
            'format_corrections': sum(1 for r in validation_results if 'ocr' in r.correction_reason or 'completion' in r.correction_reason),
            'invalid_inputs': correction_reasons.get('empty_or_invalid_input', 0)
        }
    }
    
    return report

def demonstrate_year_validation():
    """
    Demonstrate the year validation mechanism
    """
    print("=== YEAR VALIDATION MECHANISM DEMONSTRATION ===")
    print()
    
    # Test cases covering various scenarios
    test_cases = [
        # Future years (should be corrected to 2025)
        ('2026', 'Future year - should be corrected to 2025'),
        ('2030', 'Far future year - should be corrected to 2025'),
        ('2027', 'Near future year - should be corrected to 2025'),
        
        # Valid current years
        ('2025', 'Current year - should remain unchanged'),
        ('2024', 'Recent year - should remain unchanged'),
        ('2020', 'Valid year - should remain unchanged'),
        
        # Format issues
        ('2o25', 'OCR error - o instead of 0'),
        ('2O26', 'OCR error with future year'),
        ('202', 'Truncated year'),
        ('26', 'Two-digit year'),
        
        # Edge cases
        ('1979', 'Year below minimum range'),
        ('1850', 'Very old year'),
        ('', 'Empty input'),
        ('abc', 'Invalid text input'),
        (2028.0, 'Float input with future year'),
    ]
    
    print("🔧 TESTING YEAR VALIDATION:")
    print()
    
    validation_results = []
    
    for test_input, description in test_cases:
        result = apply_year_validation(test_input, confidence_threshold=0.7)
        validation_results.append(result)
        
        print(f"Input: '{test_input}' ({description})")
        print(f"  → Corrected: '{result.corrected_year}'")
        print(f"  → Correction Applied: {result.correction_applied}")
        print(f"  → Confidence: {result.confidence_score:.3f}")
        print(f"  → Reason: {result.correction_reason}")
        print()
    
    # Generate and display report
    print("📊 VALIDATION REPORT:")
    report = generate_validation_report(validation_results)
    
    print(f"  Total Records: {report['total_records']}")
    print(f"  Corrections Applied: {report['corrections_applied']} ({report['correction_rate']:.1%})")
    print(f"  Future Year Corrections: {report['future_year_corrections']}")
    print(f"  Average Confidence: {report['average_confidence']:.3f}")
    print()
    
    print("📋 CORRECTION BREAKDOWN:")
    for reason, count in report['correction_reasons'].items():
        print(f"  {reason}: {count}")
    print()
    
    print("🎯 KEY FEATURES:")
    print("  ✅ Automatic correction of future years (2026+) to 2025")
    print("  ✅ Configurable confidence thresholds")
    print("  ✅ OCR error correction (o/O → 0, l/I → 1)")
    print("  ✅ Year format completion (truncated years)")
    print("  ✅ Comprehensive validation reporting")
    print("  ✅ Batch processing capabilities")
    print("  ✅ DataFrame integration")
    
    return validation_results, report

if __name__ == "__main__":
    # Run demonstration
    results, report = demonstrate_year_validation()
    
    print("\n🚀 YEAR VALIDATION MECHANISM READY!")
    print("   The system automatically corrects future years to 2025 with high confidence.")