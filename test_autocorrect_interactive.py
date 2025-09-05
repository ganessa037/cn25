#!/usr/bin/env python3
"""
Interactive Autocorrect Model Tester
Allows manual testing of the autocorrect functionality through terminal input/output.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from autocorrect import comprehensive_vehicle_correction
import pandas as pd

def main():
    print("🚗 Vehicle Autocorrect Model Tester")
    print("=" * 40)
    print("Enter vehicle brand/model names to test autocorrect functionality.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    print("💡 Example inputs to try:")
    print("   • Brand misspellings: 'toyata', 'hondda', 'mercedez', 'volkswagin'")
    print("   • Model typos: 'civick', 'accrd', 'camrry', 'corola'")
    print("   • Year errors: '202', '2025', '1985', '99'")
    print("   • Case variations: 'TOYOTA', 'honda', 'BMW'")
    print("   • Leave fields empty to skip testing that field\n")
    
    try:
        # Load vehicle master data
        print("Loading vehicle reference data...")
        vehicle_df = pd.read_csv('models/autocorrect/vehicle_reference.csv')
        print("✅ Reference data loaded successfully!\n")
        
        while True:
            print("\n🔍 Enter vehicle information to test (type 'quit' to exit):")
            
            # Get user inputs for all three fields
            brand_input = input("Brand (e.g., 'toyata'): ").strip()
            if brand_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for testing! Goodbye!")
                break
                
            model_input = input("Model (e.g., 'camri'): ").strip()
            if model_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for testing! Goodbye!")
                break
                
            year_input = input("Year (e.g., '202'): ").strip()
            if year_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for testing! Goodbye!")
                break
            
            # Skip if all inputs are empty
            if not brand_input and not model_input and not year_input:
                print("Please enter at least one field to test.\n")
                continue
            
            try:
                # Create a temporary DataFrame with user inputs
                temp_df = pd.DataFrame({
                    'brand': [brand_input if brand_input else 'Unknown'],
                    'model': [model_input if model_input else 'Unknown'],
                    'year': [year_input if year_input else '2020']
                })
                
                # Process the inputs through autocorrect
                result = comprehensive_vehicle_correction(temp_df, vehicle_df)
                
                # Display results
                print(f"\n📝 Original Input:")
                print(f"   Brand: '{brand_input}'")
                print(f"   Model: '{model_input}'")
                print(f"   Year: '{year_input}'")
                
                if result['correction_details']:
                    print("\n✅ Corrections found:")
                    for detail in result['correction_details']:
                        for correction in detail['corrections']:
                            confidence = correction.get('confidence', 0)
                            suggestion = correction.get('corrected', '')
                            field = correction.get('field', 'unknown')
                            print(f"   • {field.title()}: '{suggestion}' (confidence: {confidence:.2f})")
                else:
                    print("\n❓ No corrections found - inputs may already be correct or no close matches available")
                    
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ Error processing inputs: {str(e)}")
                print("-" * 50)
                
    except Exception as e:
        print(f"❌ Failed to initialize autocorrect system: {str(e)}")
        print("\nMake sure you have:")
        print("1. Trained models in models/autocorrect/")
        print("2. Required dependencies installed (pip install -r requirements.txt)")
        print("3. Run this script from the project root directory")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())