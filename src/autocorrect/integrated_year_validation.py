#!/usr/bin/env python3
"""
Integrated Year Validation System
================================

This module integrates the year validation mechanism with the existing autocorrect
system to provide a comprehensive solution for vehicle year validation and correction.

Features:
- Automatic correction of future years (2026+) to 2025
- Integration with existing autocorrect algorithms
- Batch processing for CSV files
- Comprehensive reporting and statistics
- Production-ready implementation
"""

import pandas as pd
import numpy as np
from .year_validation_mechanism import (
    apply_year_validation,
    validate_dataframe_years,
    generate_validation_report,
    YearValidationResult
)
from .autocorrect_feature_engineering import (
    suggest_year_correction,
    comprehensive_vehicle_correction
)
import json
from typing import Dict, List, Tuple

def enhanced_year_correction(year_input, vehicle_master_df=None, 
                           confidence_threshold=0.7, use_validation=True):
    """
    Enhanced year correction combining validation and autocorrect
    
    Args:
        year_input: Input year to correct
        vehicle_master_df: Vehicle master DataFrame (optional)
        confidence_threshold: Minimum confidence for corrections
        use_validation: Whether to apply year validation mechanism
    
    Returns:
        Dictionary with correction details
    """
    result = {
        'original_input': str(year_input),
        'final_year': str(year_input),
        'corrections_applied': [],
        'confidence_score': 1.0,
        'correction_methods': []
    }
    
    current_year = str(year_input)
    
    # Step 1: Apply year validation mechanism (future year correction)
    if use_validation:
        validation_result = apply_year_validation(current_year, confidence_threshold)
        
        if validation_result.correction_applied:
            current_year = validation_result.corrected_year
            result['corrections_applied'].append({
                'method': 'year_validation',
                'from': validation_result.original_input,
                'to': validation_result.corrected_year,
                'reason': validation_result.correction_reason,
                'confidence': validation_result.confidence_score
            })
            result['correction_methods'].append('year_validation')
            result['confidence_score'] = min(result['confidence_score'], validation_result.confidence_score)
    
    # Step 2: Apply autocorrect if we have vehicle master data
    if vehicle_master_df is not None and current_year != str(year_input):
        autocorrect_suggestions = suggest_year_correction(current_year, vehicle_master_df, threshold=0.6)
        
        if autocorrect_suggestions:
            best_suggestion = autocorrect_suggestions[0]
            if best_suggestion['combined_score'] > confidence_threshold:
                autocorrected_year = best_suggestion['suggestion']
                
                if autocorrected_year != current_year:
                    result['corrections_applied'].append({
                        'method': 'autocorrect',
                        'from': current_year,
                        'to': autocorrected_year,
                        'reason': 'similarity_match',
                        'confidence': best_suggestion['combined_score']
                    })
                    result['correction_methods'].append('autocorrect')
                    current_year = autocorrected_year
                    result['confidence_score'] = min(result['confidence_score'], best_suggestion['combined_score'])
    
    result['final_year'] = current_year
    
    return result

def process_user_inputs_with_validation(input_file, vehicle_master_file, output_file):
    """
    Process user inputs with integrated year validation
    
    Args:
        input_file: Path to user inputs CSV
        vehicle_master_file: Path to vehicle master CSV
        output_file: Path to output CSV
    
    Returns:
        Processing statistics
    """
    print("🔄 PROCESSING USER INPUTS WITH YEAR VALIDATION")
    print("=" * 55)
    print()
    
    # Load data
    print(f"📂 Loading user inputs from: {input_file}")
    user_df = pd.read_csv(input_file)
    
    print(f"📂 Loading vehicle master from: {vehicle_master_file}")
    vehicle_df = pd.read_csv(vehicle_master_file)
    
    print(f"📊 Processing {len(user_df)} user input records")
    print()
    
    # Apply year validation to user input years
    print("🔧 Applying year validation mechanism...")
    
    year_corrections = []
    validation_stats = {
        'total_records': len(user_df),
        'future_years_corrected': 0,
        'validation_corrections': 0,
        'autocorrect_corrections': 0,
        'no_corrections': 0
    }
    
    for idx, row in user_df.iterrows():
        year_input = row['user_input_year']
        
        # Apply enhanced year correction
        correction_result = enhanced_year_correction(
            year_input, 
            vehicle_df, 
            confidence_threshold=0.7, 
            use_validation=True
        )
        
        year_corrections.append(correction_result)
        
        # Update statistics
        if correction_result['corrections_applied']:
            for correction in correction_result['corrections_applied']:
                if correction['method'] == 'year_validation':
                    validation_stats['validation_corrections'] += 1
                    if 'future_year_corrected' in correction['reason']:
                        validation_stats['future_years_corrected'] += 1
                elif correction['method'] == 'autocorrect':
                    validation_stats['autocorrect_corrections'] += 1
        else:
            validation_stats['no_corrections'] += 1
    
    # Add correction results to DataFrame
    user_df['year_original'] = user_df['user_input_year']
    user_df['year_validated'] = [r['final_year'] for r in year_corrections]
    user_df['year_corrections_applied'] = [len(r['corrections_applied']) > 0 for r in year_corrections]
    user_df['year_correction_methods'] = [','.join(r['correction_methods']) for r in year_corrections]
    user_df['year_confidence_score'] = [r['confidence_score'] for r in year_corrections]
    
    # Apply comprehensive vehicle correction with validated years
    print("🔧 Applying comprehensive vehicle correction...")
    
    comprehensive_results = []
    for idx, row in user_df.iterrows():
        result = comprehensive_vehicle_correction(
            row['user_input_brand'],
            row['user_input_model'], 
            row['year_validated'],  # Use validated year
            vehicle_df
        )
        comprehensive_results.append(result)
    
    # Add comprehensive correction results
    user_df['brand_suggestion'] = [r['brand_suggestions'][0]['suggestion'] if r['brand_suggestions'] else '' for r in comprehensive_results]
    user_df['brand_confidence'] = [r['brand_suggestions'][0]['combined_score'] if r['brand_suggestions'] else 0.0 for r in comprehensive_results]
    
    user_df['model_suggestion'] = [r['model_suggestions'][0]['suggestion'] if r['model_suggestions'] else '' for r in comprehensive_results]
    user_df['model_confidence'] = [r['model_suggestions'][0]['combined_score'] if r['model_suggestions'] else 0.0 for r in comprehensive_results]
    
    # Save results
    print(f"💾 Saving results to: {output_file}")
    user_df.to_csv(output_file, index=False)
    
    # Generate report
    print()
    print("📊 YEAR VALIDATION STATISTICS:")
    print(f"  Total Records: {validation_stats['total_records']}")
    print(f"  Future Years Corrected to 2025: {validation_stats['future_years_corrected']}")
    print(f"  Validation Corrections: {validation_stats['validation_corrections']}")
    print(f"  Autocorrect Corrections: {validation_stats['autocorrect_corrections']}")
    print(f"  No Corrections Needed: {validation_stats['no_corrections']}")
    
    correction_rate = (validation_stats['validation_corrections'] + validation_stats['autocorrect_corrections']) / validation_stats['total_records']
    print(f"  Overall Correction Rate: {correction_rate:.1%}")
    
    future_year_rate = validation_stats['future_years_corrected'] / validation_stats['total_records']
    print(f"  Future Year Correction Rate: {future_year_rate:.1%}")
    
    return validation_stats

def demonstrate_integrated_validation():
    """
    Demonstrate the integrated year validation system
    """
    print("🎯 INTEGRATED YEAR VALIDATION DEMONSTRATION")
    print("=" * 50)
    print()
    
    # Test cases showing integration
    test_cases = [
        {
            'brand': 'Toyota',
            'model': 'Camry', 
            'year': '2026',
            'description': 'Future year with valid brand/model'
        },
        {
            'brand': 'Hond',
            'model': 'Civicy',
            'year': '2030', 
            'description': 'Multiple corrections needed'
        },
        {
            'brand': 'Perodua',
            'model': 'Myvi',
            'year': '2o27',
            'description': 'OCR error in future year'
        },
        {
            'brand': 'Toyota',
            'model': 'Vios',
            'year': '27',
            'description': 'Truncated year that becomes future year'
        },
        {
            'brand': 'Honda',
            'model': 'City',
            'year': '2024',
            'description': 'Valid year - no correction needed'
        }
    ]
    
    print("🔧 TESTING INTEGRATED CORRECTIONS:")
    print()
    
    # Create a simple vehicle master for testing
    vehicle_master = pd.DataFrame({
        'brand': ['toyota', 'honda', 'perodua', 'proton'],
        'model': ['camry', 'civic', 'myvi', 'saga'],
        'year_start': [2000, 2000, 2005, 2000],
        'year_end': [2025, 2025, 2025, 2025]
    })
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['description']}")
        print(f"Input: Brand='{test_case['brand']}', Model='{test_case['model']}', Year='{test_case['year']}'")
        
        # Apply enhanced year correction
        year_result = enhanced_year_correction(
            test_case['year'], 
            vehicle_master, 
            confidence_threshold=0.7
        )
        
        print(f"  Year Correction:")
        print(f"    Original: '{year_result['original_input']}'")
        print(f"    Final: '{year_result['final_year']}'")
        print(f"    Methods: {', '.join(year_result['correction_methods']) if year_result['correction_methods'] else 'None'}")
        print(f"    Confidence: {year_result['confidence_score']:.3f}")
        
        if year_result['corrections_applied']:
            for correction in year_result['corrections_applied']:
                print(f"    Applied: {correction['method']} - {correction['reason']}")
        
        # Apply comprehensive correction with validated year
        comprehensive_result = comprehensive_vehicle_correction(
            test_case['brand'],
            test_case['model'],
            year_result['final_year'],
            vehicle_master
        )
        
        print(f"  Overall Suggestions:")
        if comprehensive_result['brand_suggestions']:
            best_brand = comprehensive_result['brand_suggestions'][0]
            print(f"    Brand: '{best_brand['suggestion']}' (score: {best_brand['combined_score']:.3f})")
        
        if comprehensive_result['model_suggestions']:
            best_model = comprehensive_result['model_suggestions'][0]
            print(f"    Model: '{best_model['suggestion']}' (score: {best_model['combined_score']:.3f})")
        
        print(f"    Year: '{year_result['final_year']}' (validated)")
        print()
    
    print("✅ INTEGRATION FEATURES:")
    print("  🔹 Automatic future year correction to 2025")
    print("  🔹 OCR error correction in years")
    print("  🔹 Year format completion and validation")
    print("  🔹 Integration with existing autocorrect algorithms")
    print("  🔹 Comprehensive vehicle data correction")
    print("  🔹 Configurable confidence thresholds")
    print("  🔹 Detailed correction tracking and reporting")
    print()
    
    print("🚀 PRODUCTION READY:")
    print("   The integrated system ensures all future years are corrected to 2025")
    print("   while maintaining compatibility with existing autocorrect functionality.")

if __name__ == "__main__":
    # Run demonstration
    demonstrate_integrated_validation()
    
    # Test with actual data if available
    try:
        print("\n" + "=" * 60)
        print("🔄 TESTING WITH ACTUAL DATA")
        print("=" * 60)
        
        stats = process_user_inputs_with_validation(
            "../data/autocorrect/user_inputs.csv",
            "../data/autocorrect/vehicle_master_cleaned.csv", 
            "../data/autocorrect/user_inputs_year_validated.csv"
        )
        
        print("\n✅ Processing completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  Data files not found: {e}")
        print("   Demonstration completed with test data only.")
    except Exception as e:
        print(f"\n❌ Error processing data: {e}")