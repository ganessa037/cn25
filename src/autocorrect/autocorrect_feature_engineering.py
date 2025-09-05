"""Autocorrect Feature Engineering Module

This module provides autocorrect functionality for vehicle data correction,
including year suggestions and comprehensive vehicle data correction.
"""

import pandas as pd
from typing import List, Dict, Any
import difflib
from collections import Counter
from .year_validation_mechanism import apply_year_validation


def suggest_year_correction(year_input: str, vehicle_master_df: pd.DataFrame, 
                          threshold: float = 0.6) -> List[Dict[str, Any]]:
    """
    Suggest year corrections based on vehicle master data.
    
    Args:
        year_input: The input year to correct
        vehicle_master_df: DataFrame containing vehicle master data
        threshold: Minimum similarity threshold for suggestions
        
    Returns:
        List of correction suggestions with confidence scores
    """
    if vehicle_master_df is None or vehicle_master_df.empty:
        return []
    
    suggestions = []
    
    # Get available years from vehicle master data
    if 'year' in vehicle_master_df.columns:
        available_years = vehicle_master_df['year'].astype(str).unique()
        
        # Find similar years using difflib
        matches = difflib.get_close_matches(
            year_input, available_years, n=5, cutoff=threshold
        )
        
        for match in matches:
            similarity = difflib.SequenceMatcher(None, year_input, match).ratio()
            suggestions.append({
                'suggestion': match,
                'confidence': similarity,
                'method': 'similarity_match'
            })
    
    # Sort by confidence score
    suggestions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return suggestions


def comprehensive_vehicle_correction(user_df: pd.DataFrame, 
                                   vehicle_master_df: pd.DataFrame,
                                   year_column: str = 'year',
                                   brand_column: str = 'brand',
                                   model_column: str = 'model',
                                   confidence_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Apply comprehensive vehicle data correction.
    
    Args:
        user_df: DataFrame with user input data
        vehicle_master_df: DataFrame with master vehicle data
        year_column: Name of year column
        brand_column: Name of brand column
        model_column: Name of model column
        confidence_threshold: Minimum confidence for corrections
        
    Returns:
        Dictionary with correction results and statistics
    """
    if vehicle_master_df is None or vehicle_master_df.empty:
        return {
            'corrected_df': user_df.copy(),
            'corrections_applied': 0,
            'correction_rate': 0.0,
            'correction_details': []
        }
    
    corrected_df = user_df.copy()
    corrections_applied = 0
    correction_details = []
    
    for idx, row in user_df.iterrows():
        row_corrections = []
        
        # Correct brand if available
        if brand_column in row and brand_column in vehicle_master_df.columns:
            brand_suggestions = _suggest_brand_correction(
                str(row[brand_column]), vehicle_master_df, confidence_threshold
            )
            if brand_suggestions:
                best_brand = brand_suggestions[0]
                if best_brand['confidence'] >= confidence_threshold:
                    corrected_df.at[idx, brand_column] = best_brand['suggestion']
                    row_corrections.append({
                        'field': brand_column,
                        'original': str(row[brand_column]),
                        'corrected': best_brand['suggestion'],
                        'confidence': best_brand['confidence']
                    })
        
        # Correct model if available
        if model_column in row and model_column in vehicle_master_df.columns:
            model_suggestions = _suggest_model_correction(
                str(row[model_column]), vehicle_master_df, confidence_threshold
            )
            if model_suggestions:
                best_model = model_suggestions[0]
                if best_model['confidence'] >= confidence_threshold:
                    corrected_df.at[idx, model_column] = best_model['suggestion']
                    row_corrections.append({
                        'field': model_column,
                        'original': str(row[model_column]),
                        'corrected': best_model['suggestion'],
                        'confidence': best_model['confidence']
                    })
        
        # Correct year if available
        if year_column in row:
            year_validation_result = apply_year_validation(
                str(row[year_column]), confidence_threshold
            )
            if year_validation_result.correction_applied:
                corrected_df.at[idx, year_column] = year_validation_result.corrected_year
                row_corrections.append({
                    'field': year_column,
                    'original': year_validation_result.original_input,
                    'corrected': year_validation_result.corrected_year,
                    'confidence': year_validation_result.confidence_score
                })
        
        if row_corrections:
            corrections_applied += 1
            correction_details.append({
                'row_index': idx,
                'corrections': row_corrections
            })
    
    correction_rate = corrections_applied / len(user_df) if len(user_df) > 0 else 0.0
    
    return {
        'corrected_df': corrected_df,
        'corrections_applied': corrections_applied,
        'correction_rate': correction_rate,
        'correction_details': correction_details
    }


def _suggest_brand_correction(brand_input: str, vehicle_master_df: pd.DataFrame,
                            threshold: float = 0.6) -> List[Dict[str, Any]]:
    """
    Suggest brand corrections based on vehicle master data.
    """
    if 'brand' not in vehicle_master_df.columns:
        return []
    
    available_brands = vehicle_master_df['brand'].astype(str).unique()
    matches = difflib.get_close_matches(
        brand_input, available_brands, n=3, cutoff=threshold
    )
    
    suggestions = []
    for match in matches:
        similarity = difflib.SequenceMatcher(None, brand_input, match).ratio()
        suggestions.append({
            'suggestion': match,
            'confidence': similarity,
            'method': 'brand_similarity'
        })
    
    return sorted(suggestions, key=lambda x: x['confidence'], reverse=True)


def _suggest_model_correction(model_input: str, vehicle_master_df: pd.DataFrame,
                            threshold: float = 0.6) -> List[Dict[str, Any]]:
    """
    Suggest model corrections based on vehicle master data.
    """
    if 'model' not in vehicle_master_df.columns:
        return []
    
    available_models = vehicle_master_df['model'].astype(str).unique()
    matches = difflib.get_close_matches(
        model_input, available_models, n=3, cutoff=threshold
    )
    
    suggestions = []
    for match in matches:
        similarity = difflib.SequenceMatcher(None, model_input, match).ratio()
        suggestions.append({
            'suggestion': match,
            'confidence': similarity,
            'method': 'model_similarity'
        })
    
    return sorted(suggestions, key=lambda x: x['confidence'], reverse=True)