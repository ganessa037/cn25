# Environment Setup Guide

## Secure Python Environment

This project uses a Python virtual environment to ensure dependency isolation and security.

### Environment Details
- **Python Version**: 3.11.0
- **Virtual Environment**: `venv/` (created in project root)
- **Dependencies**: Listed in `requirements.txt`

### Setup Instructions

1. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies** (if needed):
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Jupyter notebook**:
   ```bash
   cd models
   jupyter notebook train_autocorrect_model.ipynb
   ```

### Installed Packages

Core packages for the autocorrect model training:
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, transformers, torch
- **String Similarity**: fuzzywuzzy, python-levenshtein, jellyfish
- **Visualization**: matplotlib, seaborn
- **Notebook Environment**: jupyter

### Security Notes

- All dependencies are isolated within the virtual environment
- No system-wide package modifications were made
- All existing code remains unchanged
- Virtual environment can be safely removed if needed

### Deactivation

To deactivate the virtual environment:
```bash
deactivate
```

### Environment Removal

To completely remove the environment:
```bash
rm -rf venv/
rm requirements.txt
```