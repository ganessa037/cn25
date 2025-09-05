"""Autocorrect Module

This module provides comprehensive autocorrect functionality for vehicle data,
including year validation, correction suggestions, and integrated corrections.
"""

from .integrated_year_validation import enhanced_year_correction, process_user_inputs_with_validation, demonstrate_integrated_validation
from .autocorrect_feature_engineering import suggest_year_correction, comprehensive_vehicle_correction
from .year_validation_mechanism import apply_year_validation, YearValidationResult
from .year_validation_summary import demonstrate_year_validation_features
from .process_user_inputs import process_user_inputs
from .clean_vehicle_master import clean_vehicle_master_data

__all__ = [
    'enhanced_year_correction',
    'process_user_inputs_with_validation', 
    'demonstrate_integrated_validation',
    'suggest_year_correction',
    'comprehensive_vehicle_correction',
    'apply_year_validation',
    'YearValidationResult',
    'demonstrate_year_validation_features',
    'process_user_inputs',
    'clean_vehicle_master_data'
]