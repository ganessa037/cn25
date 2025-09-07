#!/usr/bin/env python3
"""
License Plate Visualization Viewer

A simple script to view and browse generated visualizations.
"""

import os
import sys
from pathlib import Path
import argparse
from typing import List

try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.widgets import Button
except ImportError:
    print("❌ matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

class VisualizationViewer:
    """Interactive viewer for license plate detection visualizations."""
    
    def __init__(self, viz_dir: str = None):
        """Initialize the viewer.
        
        Args:
            viz_dir: Directory containing visualizations
        """
        self.viz_dir = Path(viz_dir) if viz_dir else Path(__file__).parent
        self.image_files = self._find_visualization_files()
        self.current_index = 0
        
        if not self.image_files:
            print(f"❌ No visualization files found in {self.viz_dir}")
            print("💡 Run generate_visualizations.py first to create visualizations")
            sys.exit(1)
        
        print(f"📊 Found {len(self.image_files)} visualization files")
        
    def _find_visualization_files(self) -> List[Path]:
        """Find all visualization image files."""
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.svg']
        files = []
        
        for ext in extensions:
            files.extend(self.viz_dir.glob(ext))
        
        # Sort by name for consistent ordering
        return sorted(files)
    
    def view_all_grid(self):
        """Display all visualizations in a grid layout."""
        if not self.image_files:
            return
        
        num_images = len(self.image_files)
        cols = min(2, num_images)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 8*rows))
        fig.suptitle('License Plate Detection Visualizations', fontsize=16, fontweight='bold')
        
        if num_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, img_path in enumerate(self.image_files):
            row, col = i // cols, i % cols
            
            try:
                img = mpimg.imread(str(img_path))
                if rows > 1:
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(img_path.stem, fontsize=12, fontweight='bold')
                    axes[row, col].axis('off')
                else:
                    axes[col].imshow(img)
                    axes[col].set_title(img_path.stem, fontsize=12, fontweight='bold')
                    axes[col].axis('off')
            except Exception as e:
                print(f"❌ Error loading {img_path}: {e}")
        
        # Hide empty subplots
        for i in range(num_images, rows * cols):
            row, col = i // cols, i % cols
            if rows > 1:
                axes[row, col].axis('off')
            else:
                axes[col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def view_interactive(self):
        """Interactive viewer with navigation buttons."""
        if not self.image_files:
            return
        
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.suptitle('License Plate Visualization Viewer', fontsize=16, fontweight='bold')
        
        # Create navigation buttons
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.05])
        ax_next = plt.axes([0.8, 0.02, 0.1, 0.05])
        ax_list = plt.axes([0.45, 0.02, 0.1, 0.05])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_list = Button(ax_list, 'List All')
        
        self.btn_prev.on_clicked(self._prev_image)
        self.btn_next.on_clicked(self._next_image)
        self.btn_list.on_clicked(self._list_files)
        
        # Display first image
        self._display_current_image()
        
        plt.show()
    
    def _display_current_image(self):
        """Display the current image."""
        if not self.image_files:
            return
        
        img_path = self.image_files[self.current_index]
        
        try:
            img = mpimg.imread(str(img_path))
            self.ax.clear()
            self.ax.imshow(img)
            self.ax.set_title(f'{img_path.stem} ({self.current_index + 1}/{len(self.image_files)})', 
                            fontsize=14, fontweight='bold')
            self.ax.axis('off')
            
            # Add file info
            file_size = img_path.stat().st_size / 1024  # KB
            info_text = f'File: {img_path.name} | Size: {file_size:.1f} KB'
            self.ax.text(0.02, 0.02, info_text, transform=self.ax.transAxes, 
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            self.fig.canvas.draw()
            
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
    
    def _prev_image(self, event):
        """Navigate to previous image."""
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self._display_current_image()
    
    def _next_image(self, event):
        """Navigate to next image."""
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self._display_current_image()
    
    def _list_files(self, event):
        """Print list of all visualization files."""
        print("\n📊 Available Visualizations:")
        for i, img_path in enumerate(self.image_files):
            marker = "➤" if i == self.current_index else " "
            file_size = img_path.stat().st_size / 1024  # KB
            print(f"{marker} {i+1:2d}. {img_path.name:<30} ({file_size:6.1f} KB)")
        print()
    
    def list_files(self):
        """List all visualization files without opening viewer."""
        if not self.image_files:
            print("❌ No visualization files found")
            return
        
        print(f"\n📊 Found {len(self.image_files)} visualization files in {self.viz_dir}:")
        print("=" * 60)
        
        total_size = 0
        for i, img_path in enumerate(self.image_files):
            file_size = img_path.stat().st_size / 1024  # KB
            total_size += file_size
            
            # Get file modification time
            mod_time = img_path.stat().st_mtime
            mod_date = Path(img_path).stat().st_mtime
            
            print(f"{i+1:2d}. {img_path.name:<35} {file_size:8.1f} KB")
        
        print("=" * 60)
        print(f"Total: {len(self.image_files)} files, {total_size:.1f} KB")
        print()
    
    def open_file(self, filename: str):
        """Open a specific visualization file.
        
        Args:
            filename: Name of the file to open
        """
        target_file = None
        
        for img_path in self.image_files:
            if img_path.name == filename or img_path.stem == filename:
                target_file = img_path
                break
        
        if not target_file:
            print(f"❌ File '{filename}' not found")
            self.list_files()
            return
        
        try:
            img = mpimg.imread(str(target_file))
            
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.imshow(img)
            ax.set_title(target_file.stem, fontsize=16, fontweight='bold')
            ax.axis('off')
            
            # Add file info
            file_size = target_file.stat().st_size / 1024  # KB
            info_text = f'File: {target_file.name} | Size: {file_size:.1f} KB'
            ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error opening {target_file}: {e}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='View license plate detection visualizations')
    parser.add_argument('--dir', help='Directory containing visualizations', 
                       default=str(Path(__file__).parent))
    parser.add_argument('--mode', choices=['interactive', 'grid', 'list'], 
                       default='interactive', help='Viewing mode')
    parser.add_argument('--file', help='Open specific file')
    
    args = parser.parse_args()
    
    # Initialize viewer
    viewer = VisualizationViewer(args.dir)
    
    if args.file:
        viewer.open_file(args.file)
    elif args.mode == 'list':
        viewer.list_files()
    elif args.mode == 'grid':
        viewer.view_all_grid()
    else:  # interactive
        viewer.view_interactive()

if __name__ == '__main__':
    main()