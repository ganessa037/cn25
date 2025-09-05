import pandas as pd
import numpy as np

def clean_vehicle_master_data(input_file, output_file):
    """
    Clean and normalize vehicle master data:
    - Convert brand/model names to lowercase
    - Strip whitespace
    - Remove duplicates
    - Standardize year formats
    """
    
    # Load the vehicle master CSV file
    print(f"Loading vehicle master data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print("\nOriginal data sample:")
    print(df.head())
    
    # Clean and normalize the data
    print("\nCleaning and normalizing data...")
    
    # Convert brand and model names to lowercase
    df['brand'] = df['brand'].astype(str).str.lower()
    df['model'] = df['model'].astype(str).str.lower()
    
    # Strip whitespace from all string columns
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        df[col] = df[col].astype(str).str.strip()
    
    # Standardize year formats (ensure they are 4-digit years)
    for year_col in ['year_start', 'year_end']:
        if year_col in df.columns:
            # Convert to numeric, handling any non-numeric values
            df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
            
            # Handle 2-digit years (convert to 4-digit)
            # Assume years < 50 are 20xx, years >= 50 are 19xx
            mask_2digit = (df[year_col] >= 0) & (df[year_col] <= 99)
            df.loc[mask_2digit & (df[year_col] < 50), year_col] = df.loc[mask_2digit & (df[year_col] < 50), year_col] + 2000
            df.loc[mask_2digit & (df[year_col] >= 50), year_col] = df.loc[mask_2digit & (df[year_col] >= 50), year_col] + 1900
            
            # Convert back to integer
            df[year_col] = df[year_col].astype('Int64')  # Use nullable integer type
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    duplicates_removed = initial_count - final_count
    
    print(f"\nData cleaning completed:")
    print(f"- Duplicates removed: {duplicates_removed}")
    print(f"- Final data shape: {df.shape}")
    
    print("\nCleaned data sample:")
    print(df.head())
    
    # Save the cleaned data
    df.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Define input and output file paths
    input_file = "vehicle_master.csv"
    output_file = "vehicle_master_cleaned.csv"
    
    # Clean the vehicle master data
    cleaned_df = clean_vehicle_master_data(input_file, output_file)
    
    # Display summary statistics
    print("\n=== SUMMARY ===")
    print(f"Total records: {len(cleaned_df)}")
    print(f"Unique brands: {cleaned_df['brand'].nunique()}")
    print(f"Unique models: {cleaned_df['model'].nunique()}")
    print("\nBrands in dataset:")
    print(cleaned_df['brand'].value_counts())
    print("\nYear range:")
    if 'year_start' in cleaned_df.columns and 'year_end' in cleaned_df.columns:
        print(f"Start years: {cleaned_df['year_start'].min()} - {cleaned_df['year_start'].max()}")
        print(f"End years: {cleaned_df['year_end'].min()} - {cleaned_df['year_end'].max()}")