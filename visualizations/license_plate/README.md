# License Plate Detection Visualizations

This directory contains comprehensive visualization tools for license plate detection models, providing insights into training performance, dataset characteristics, and detection results.

## 📊 Available Visualizations

### 1. Training Metrics (`training_metrics.png`)
- **Loss Curves**: Box loss and classification loss over epochs
- **Performance Metrics**: Precision, recall, and F1 score progression
- **mAP Scores**: Mean Average Precision at different IoU thresholds
- **Learning Rate Schedule**: Learning rate changes during training

### 2. Dataset Analysis (`dataset_analysis.png`)
- **Image Dimensions**: Width, height, and aspect ratio distributions
- **Bounding Box Statistics**: Size and area distributions of annotations
- **Dataset Summary**: Training/validation split statistics

### 3. Detection Results (`detection_results.png`)
- **Sample Predictions**: Visual results on test images
- **Confidence Scores**: Detection confidence visualization
- **Bounding Box Overlays**: Predicted license plate locations

### 4. Performance Dashboard (`performance_dashboard.png`)
- **Detection Rate Gauge**: Overall detection success rate
- **Confidence Distribution**: Histogram of prediction confidences
- **Processing Time Analysis**: Speed performance metrics
- **Summary Statistics**: Comprehensive performance table

## 🚀 Quick Start

### Generate All Visualizations
```bash
# From project root
cd visualizations/license_plate
python generate_visualizations.py \
    --data-dir ../../models/license_plate_detection/data \
    --model-path ../../models/license_plate_detection/trained_models/best.pt \
    --results-path ../../models/license_plate_detection/outputs/exp1 \
    --type all
```

### Generate Specific Visualization Types

#### Dataset Analysis Only
```bash
python generate_visualizations.py \
    --data-dir ../../models/license_plate_detection/data \
    --type dataset
```

#### Training Metrics Only
```bash
python generate_visualizations.py \
    --data-dir ../../models/license_plate_detection/data \
    --results-path ../../models/license_plate_detection/outputs/exp1 \
    --type training
```

#### Detection Results Only
```bash
python generate_visualizations.py \
    --data-dir ../../models/license_plate_detection/data \
    --model-path ../../models/license_plate_detection/trained_models/best.pt \
    --type detection
```

#### Performance Dashboard Only
```bash
python generate_visualizations.py \
    --data-dir ../../models/license_plate_detection/data \
    --model-path ../../models/license_plate_detection/trained_models/best.pt \
    --type performance
```

## 📁 Directory Structure

```
visualizations/license_plate/
├── README.md                      # This documentation
├── generate_visualizations.py     # Main visualization script
├── training_metrics.png           # Training performance plots
├── dataset_analysis.png           # Dataset characteristics
├── detection_results.png          # Sample detection results
└── performance_dashboard.png      # Performance summary
```

## 🛠️ Requirements

The visualization script requires the following dependencies:

```python
# Core libraries
numpy
matplotlib
seaborn
pandas
opencv-python
Pillow

# Project modules
license_plate.core.detector
license_plate.core.processor
license_plate.config.settings
```

## 📋 Command Line Options

| Option | Description | Required | Example |
|--------|-------------|----------|----------|
| `--data-dir` | Path to dataset directory | ✅ | `../../models/license_plate_detection/data` |
| `--model-path` | Path to trained model file | ❌ | `../../models/license_plate_detection/trained_models/best.pt` |
| `--results-path` | Path to training results directory | ❌ | `../../models/license_plate_detection/outputs/exp1` |
| `--output-dir` | Output directory for visualizations | ❌ | `./custom_output` |
| `--type` | Type of visualization to generate | ❌ | `all`, `dataset`, `training`, `detection`, `performance` |

## 🎨 Visualization Features

### Color Scheme
- **Primary**: Professional blue (#2E86AB)
- **Secondary**: Elegant purple (#A23B72)
- **Success**: Vibrant orange (#F18F01)
- **Warning**: Bold red (#C73E1D)
- **Info**: Natural green (#6A994E)

### Chart Types
- **Line Plots**: Training metrics over time
- **Histograms**: Distribution analysis
- **Gauge Charts**: Performance indicators
- **Tables**: Summary statistics
- **Image Grids**: Detection result samples

## 📊 Understanding the Visualizations

### Training Metrics
- **Decreasing Loss**: Indicates model is learning
- **Increasing mAP**: Shows improving detection accuracy
- **Stable Learning Rate**: Confirms proper training schedule
- **Converging Metrics**: Suggests training completion

### Dataset Analysis
- **Image Size Distribution**: Helps understand input variety
- **Aspect Ratio Patterns**: Reveals dataset characteristics
- **Bounding Box Sizes**: Shows annotation quality
- **Split Balance**: Confirms proper train/val distribution

### Detection Results
- **Green Boxes**: Successful detections
- **Confidence Scores**: Prediction certainty (0.0-1.0)
- **Multiple Detections**: Multiple license plates per image
- **Visual Quality**: Overall model performance

### Performance Dashboard
- **Detection Rate**: Percentage of images with detections
- **Confidence Distribution**: Quality of predictions
- **Processing Speed**: Real-time performance capability
- **Summary Table**: Key performance indicators

## 🔧 Customization

The visualization script can be customized by modifying:

1. **Color Schemes**: Update the `colors` dictionary in `LicensePlateVisualizer`
2. **Chart Styles**: Modify matplotlib/seaborn settings
3. **Sample Sizes**: Adjust `num_samples` parameter for detection results
4. **Metrics**: Add custom performance calculations
5. **Layout**: Change subplot arrangements and sizes

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure project root is in Python path
   export PYTHONPATH="/path/to/project/root:$PYTHONPATH"
   ```

2. **Missing Data**
   ```bash
   # Check data directory structure
   ls -la ../../models/license_plate_detection/data/
   ```

3. **Model Loading Errors**
   ```bash
   # Verify model file exists and is valid
   ls -la ../../models/license_plate_detection/trained_models/
   ```

4. **Memory Issues**
   - Reduce `num_samples` for detection results
   - Process smaller batches of images
   - Use lower DPI for output images

### Performance Tips

- **Large Datasets**: Use sampling for analysis
- **High-Resolution Images**: Resize for faster processing
- **Multiple Models**: Generate comparisons in batches
- **Automated Generation**: Integrate into training pipeline

## 📈 Integration with Training

To automatically generate visualizations after training:

```python
# Add to training script
from visualizations.license_plate.generate_visualizations import LicensePlateVisualizer

# After training completion
visualizer = LicensePlateVisualizer(output_dir='./outputs/visualizations')
visualizer.generate_all_visualizations(
    data_dir='./data',
    model_path='./trained_models/best.pt',
    results_path='./outputs/exp1'
)
```

## 🤝 Contributing

To add new visualization types:

1. Create new method in `LicensePlateVisualizer` class
2. Follow existing naming conventions
3. Add command-line option in `main()` function
4. Update this README with documentation
5. Test with sample data

## 📝 Notes

- All visualizations are saved as high-resolution PNG files (300 DPI)
- Processing time depends on dataset size and model complexity
- GPU acceleration is used when available for model inference
- Visualizations are optimized for both screen viewing and printing

---

**Generated by**: License Plate Detection Visualization System  
**Last Updated**: January 2025  
**Version**: 1.0.0