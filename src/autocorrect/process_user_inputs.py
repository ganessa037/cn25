import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re

def normalize_text(text):
    """
    Normalize text using the same rules as vehicle master data:
    - Convert to lowercase
    - Strip whitespace
    - Handle common OCR errors
    """
    if pd.isna(text):
        return ''
    
    text = str(text).lower().strip()
    
    # Handle common OCR errors
    ocr_corrections = {
        '0': 'o',  # Zero to letter O
        '1': 'i',  # One to letter I
        '5': 's',  # Five to letter S
    }
    
    # Apply OCR corrections selectively (only for brand/model, not years)
    for wrong, correct in ocr_corrections.items():
        text = text.replace(wrong, correct)
    
    return text

def normalize_year(year):
    """
    Normalize year format to 4-digit year
    """
    if pd.isna(year):
        return None
    
    try:
        year = int(float(year))
        
        # Handle 2-digit years
        if 0 <= year <= 99:
            if year < 50:
                year += 2000
            else:
                year += 1900
        
        # Validate reasonable year range for vehicles
        if 1900 <= year <= 2030:
            return year
        else:
            return None
    except (ValueError, TypeError):
        return None

def similarity_score(str1, str2):
    """
    Calculate similarity score between two strings
    """
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1, str2).ratio()

def find_best_match(user_input, master_data, threshold=0.8):
    """
    Find the best matching vehicle from master data
    """
    best_match = None
    best_score = 0
    
    user_brand = normalize_text(user_input.get('user_input_brand', ''))
    user_model = normalize_text(user_input.get('user_input_model', ''))
    user_year = normalize_year(user_input.get('user_input_year'))
    
    for _, master_row in master_data.iterrows():
        master_brand = master_row['brand']
        master_model = master_row['model']
        year_start = master_row['year_start']
        year_end = master_row['year_end']
        
        # Calculate brand and model similarity
        brand_score = similarity_score(user_brand, master_brand)
        model_score = similarity_score(user_model, master_model)
        
        # Check if year is within valid range
        year_valid = False
        if user_year and year_start <= user_year <= year_end:
            year_valid = True
        
        # Combined score (weighted)
        combined_score = (brand_score * 0.4 + model_score * 0.4 + (0.2 if year_valid else 0))
        
        if combined_score > best_score and combined_score >= threshold:
            best_score = combined_score
            best_match = {
                'brand': master_brand,
                'model': master_model,
                'year_start': year_start,
                'year_end': year_end,
                'confidence': combined_score
            }
    
    return best_match

def identify_error_types(user_input, master_match, master_data):
    """
    Identify specific types of errors in user input
    """
    errors = []
    
    user_brand = normalize_text(user_input.get('user_input_brand', ''))
    user_model = normalize_text(user_input.get('user_input_model', ''))
    user_year = normalize_year(user_input.get('user_input_year'))
    user_plate = str(user_input.get('user_input_plate', '')).strip()
    
    # Check plate format (should be ABC 1234 or ABC1234)
    plate_pattern = r'^[A-Z]{2,3}\s?\d{3,4}$'
    if not re.match(plate_pattern, user_plate.upper()):
        errors.append('plate_format_error')
    
    if master_match:
        master_brand = master_match['brand']
        master_model = master_match['model']
        year_start = master_match['year_start']
        year_end = master_match['year_end']
        
        # Check brand typos
        if user_brand != master_brand:
            brand_similarity = similarity_score(user_brand, master_brand)
            if brand_similarity > 0.6:  # Likely typo
                errors.append('typo_brand')
            else:
                errors.append('wrong_brand')
        
        # Check model typos
        if user_model != master_model:
            model_similarity = similarity_score(user_model, master_model)
            if model_similarity > 0.6:  # Likely typo
                errors.append('typo_model')
            else:
                errors.append('wrong_model')
        
        # Check year validity
        if user_year:
            if user_year < year_start or user_year > year_end:
                errors.append('invalid_year')
        else:
            errors.append('missing_year')
    else:
        # No match found
        errors.append('no_match_found')
    
    return errors if errors else ['correct']

def process_user_inputs(user_inputs_file, master_data_file, output_file):
    """
    Process user inputs and identify mismatches with master data
    """
    print(f"Loading user inputs from {user_inputs_file}...")
    user_df = pd.read_csv(user_inputs_file)
    
    print(f"Loading master data from {master_data_file}...")
    master_df = pd.read_csv(master_data_file)
    
    print(f"Original user inputs shape: {user_df.shape}")
    print("\nSample user inputs:")
    print(user_df.head())
    
    # Process each user input
    results = []
    
    for idx, row in user_df.iterrows():
        # Find best match in master data
        best_match = find_best_match(row, master_df)
        
        # Identify error types
        error_types = identify_error_types(row, best_match, master_df)
        
        # Prepare result record
        result = {
            'user_input_plate': row.get('user_input_plate', ''),
            'user_input_brand': row.get('user_input_brand', ''),
            'user_input_model': row.get('user_input_model', ''),
            'user_input_year': row.get('user_input_year', ''),
            'normalized_brand': normalize_text(row.get('user_input_brand', '')),
            'normalized_model': normalize_text(row.get('user_input_model', '')),
            'normalized_year': normalize_year(row.get('user_input_year')),
        }
        
        if best_match:
            result.update({
                'matched_brand': best_match['brand'],
                'matched_model': best_match['model'],
                'matched_year_start': best_match['year_start'],
                'matched_year_end': best_match['year_end'],
                'confidence_score': best_match['confidence']
            })
        else:
            result.update({
                'matched_brand': '',
                'matched_model': '',
                'matched_year_start': '',
                'matched_year_end': '',
                'confidence_score': 0.0
            })
        
        result['error_types'] = ', '.join(error_types)
        result['has_errors'] = 'correct' not in error_types
        
        results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print summary statistics
    print("\n=== PROCESSING SUMMARY ===")
    print(f"Total records processed: {len(results_df)}")
    print(f"Records with errors: {results_df['has_errors'].sum()}")
    print(f"Correct records: {(~results_df['has_errors']).sum()}")
    
    print("\nError type distribution:")
    all_errors = []
    for error_list in results_df['error_types']:
        if error_list != 'correct':
            all_errors.extend([e.strip() for e in error_list.split(',')])
    
    error_counts = pd.Series(all_errors).value_counts()
    print(error_counts)
    
    print("\nSample results:")
    print(results_df[['user_input_plate', 'user_input_brand', 'user_input_model', 
                     'matched_brand', 'matched_model', 'error_types', 'confidence_score']].head(10))
    
    return results_df

if __name__ == "__main__":
    # Define file paths
    user_inputs_file = "../data/autocorrect/user_inputs.csv"
    master_data_file = "../data/autocorrect/vehicle_master_cleaned.csv"
    output_file = "../data/autocorrect/user_inputs_processed.csv"
    
    # Process user inputs
    results = process_user_inputs(user_inputs_file, master_data_file, output_file)