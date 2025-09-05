#!/usr/bin/env python3
"""
Visualization Viewer Script

Quick script to display information about generated visualizations
and optionally open them in the default image viewer.
"""

import os
import subprocess
from pathlib import Path

def list_visualizations():
    """List all generated visualization files"""
    current_dir = Path(".")
    png_files = list(current_dir.glob("*.png"))
    
    print("🎨 Generated Autocorrect Model Visualizations")
    print("=" * 50)
    
    if not png_files:
        print("❌ No visualization files found.")
        print("💡 Run 'python3 generate_model_visualizations.py' first.")
        return
    
    descriptions = {
        'model_comparison.png': 'Model accuracy and coverage comparison charts',
        'performance_heatmap.png': 'Performance metrics heatmap across all models',
        'test_analysis.png': 'Comprehensive test results analysis',
        'confusion_matrix.png': 'Vehicle brand prediction confusion matrix',
        'training_progress.png': 'Training accuracy and loss progression',
        'error_analysis.png': 'Error type distribution and correction rates',
        'performance_dashboard.png': 'Comprehensive performance dashboard'
    }
    
    print(f"📁 Found {len(png_files)} visualization files:\n")
    
    for i, png_file in enumerate(sorted(png_files), 1):
        file_size = png_file.stat().st_size / 1024  # KB
        description = descriptions.get(png_file.name, 'Visualization chart')
        
        print(f"{i:2d}. 📊 {png_file.name}")
        print(f"     📝 {description}")
        print(f"     💾 Size: {file_size:.1f} KB")
        print()
    
    return sorted(png_files)

def open_visualization(file_path):
    """Open visualization in default image viewer"""
    try:
        if os.name == 'nt':  # Windows
            os.startfile(file_path)
        elif os.name == 'posix':  # macOS and Linux
            subprocess.run(['open', file_path], check=True)
        else:
            print(f"❌ Cannot open file on this platform: {os.name}")
            return False
        
        print(f"✅ Opened: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error opening file: {e}")
        return False

def interactive_viewer():
    """Interactive visualization viewer"""
    png_files = list_visualizations()
    
    if not png_files:
        return
    
    print("🔍 Interactive Visualization Viewer")
    print("Commands:")
    print("  • Enter number (1-{}) to open a visualization".format(len(png_files)))
    print("  • 'all' to open all visualizations")
    print("  • 'dashboard' to open the main dashboard")
    print("  • 'quit' or 'exit' to stop")
    print()
    
    while True:
        try:
            choice = input("🎯 Enter your choice: ").strip().lower()
            
            if choice in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif choice == 'all':
                print("🚀 Opening all visualizations...")
                for png_file in png_files:
                    open_visualization(png_file)
                break
            
            elif choice == 'dashboard':
                dashboard_file = Path('performance_dashboard.png')
                if dashboard_file.exists():
                    open_visualization(dashboard_file)
                else:
                    print("❌ Dashboard file not found")
            
            elif choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(png_files):
                    open_visualization(png_files[index])
                else:
                    print(f"❌ Invalid number. Please enter 1-{len(png_files)}")
            
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'list':
            list_visualizations()
        elif command == 'dashboard':
            dashboard_file = Path('performance_dashboard.png')
            if dashboard_file.exists():
                open_visualization(dashboard_file)
            else:
                print("❌ Dashboard file not found")
        elif command == 'all':
            png_files = list(Path(".").glob("*.png"))
            for png_file in sorted(png_files):
                open_visualization(png_file)
        else:
            print("❌ Unknown command. Use: list, dashboard, all, or no arguments for interactive mode")
    else:
        interactive_viewer()

if __name__ == "__main__":
    main()