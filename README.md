# Structure-Performance-prediction
ML models for the prediction of DAC performance parameters (kinetics and equilibrium) from structural measurements (BET measurement, XPS, SEM).

# CO2 Uptake Gaussian Process Regression (GPR) Analysis

A comprehensive Python tool for predicting CO2 uptake and kinetic constants in porous materials using Gaussian Process Regression. This tool analyzes the relationship between material properties (BET surface area and pore volume) and CO2 adsorption performance.

## Features

- **Dual Target Prediction**: Simultaneously models CO2 uptake capacity and kinetic constants
- **Comprehensive Data Exploration**: Correlation analysis, distribution plots, and outlier identification
- **Gaussian Process Regression**: Uses scikit-learn's GPR with RBF kernels and uncertainty quantification
- **Cross-Validation**: Leave-One-Out cross-validation for robust model evaluation
- **Response Surface Visualization**: 3D contour plots showing prediction landscapes
- **Uncertainty Quantification**: Prediction intervals and confidence maps
- **Flexible Length Scale Control**: Option to fix or optimize GP kernel length scales
- **CSV Data Loading**: Easy data import with error handling and validation

## Requirements

```python
numpy
matplotlib
seaborn
scikit-learn
pandas
scipy
```

Install requirements:
```bash
pip install numpy matplotlib seaborn scikit-learn pandas scipy
```

## Data Format

The tool expects a CSV file with the following structure:

| Sample_Name | BET_Area | Pore_Volume | CO2_Uptake | Kinetic_Constant |
|-------------|----------|-------------|------------|------------------|
| Sample_1    | 850.5    | 0.75        | 2.45       | 0.0123          |
| Sample_2    | 1200.3   | 1.20        | 3.12       | 0.0156          |
| ...         | ...      | ...         | ...        | ...             |

**Column Descriptions:**
- `Sample_Name`: Unique identifier for each material sample
- `BET_Area`: BET surface area in m²/g
- `Pore_Volume`: Total pore volume in cm³/g  
- `CO2_Uptake`: CO2 uptake capacity at 400 ppm
- `Kinetic_Constant`: Adsorption kinetic rate constant

## Usage

### Basic Usage

```python
from co2_uptake_gpr import CO2UptakeGPR

# Initialize analyzer
analyzer = CO2UptakeGPR()

# Load data from CSV
sample_names, bet_area, pore_volume, co2_uptake, kinetic_constant = load_csv_data('your_data.csv')
analyzer.load_data(sample_names, bet_area, pore_volume, co2_uptake, kinetic_constant)

# Explore data
analyzer.explore_data()

# Train models
analyzer.train_models()

# Evaluate with cross-validation
analyzer.cross_validate()

# Plot predictions
analyzer.plot_predictions()

# Visualize response surfaces
analyzer.plot_response_surfaces()

# Predict new samples
analyzer.predict_new_sample(bet_area=800, pore_volume=0.8)
```

### Advanced Usage - Fixed Length Scales

```python
# Train models with fixed length scales
analyzer.train_models(
    fixed_length_scale_uptake=5.0,    # Fix uptake model length scale
    fixed_length_scale_kinetic=2.0    # Fix kinetic model length scale
)

# Or fix only one model's length scale
analyzer.train_models(fixed_length_scale_uptake=5.0)  # Only uptake model fixed
```

### Running the Complete Analysis

```python
# Run the main script
python co2_uptake_gpr.py
```

The script will:
1. Attempt to load data from the default CSV path
2. Prompt for a custom file path if default fails
3. Use synthetic example data as fallback
4. Perform complete analysis pipeline automatically

## Output and Visualizations

### 1. Data Exploration
- **Correlation Matrix**: Heatmap showing relationships between variables
- **Distribution Plots**: Histograms of all variables
- **Scatter Plots**: Pairwise relationships with correlation coefficients
- **Top/Bottom Performers**: Identification of best and worst performing samples

### 2. Model Performance
- **Cross-Validation Scores**: Leave-One-Out R² scores for both models
- **Prediction Plots**: Actual vs. predicted with uncertainty bars
- **Performance Metrics**: R², RMSE, and MAE for model evaluation

### 3. Response Surfaces
- **Prediction Landscapes**: 2D contour plots showing how predictions vary with inputs
- **Uncertainty Maps**: Visualization of prediction confidence across input space
- **Training Data Overlay**: Original data points overlaid on prediction surfaces

## Model Details

### Gaussian Process Regression
- **Kernel**: Constant × RBF + White Noise
- **Default Length Scale**: 10.0 (optimizable)
- **Noise Level**: 0.1 (White kernel)
- **Optimization**: 10 random restarts for robust hyperparameter optimization
- **Normalization**: Automatic target normalization (`normalize_y=True`)

### Cross-Validation
- **Method**: Leave-One-Out (LOO-CV)
- **Rationale**: Appropriate for small datasets (typically 20-50 samples)
- **Metrics**: R² score for model comparison

## Example Output

```
Data loaded: 30 samples
BET Area range: 145.2 - 1487.9
Pore Volume range: 0.1234 - 1.4567
CO2 Uptake range: 1.234 - 4.567
Kinetic Constant range: 0.0078 - 0.0234

Cross-Validation Results (Leave-One-Out):
CO2 Uptake - Mean R²: 0.823 ± 0.156
Kinetic Constant - Mean R²: 0.745 ± 0.203

Prediction for BET Area = 800, Pore Volume = 0.8:
CO2 Uptake: 2.456 ± 0.123
Kinetic Constant: 0.0145 ± 0.0023
```

## File Structure

```
project/
├── co2_uptake_gpr.py              # Main analysis script
├── your_data.csv                  # Your experimental data
├── README.md                      # This file
└── outputs/                       # Generated plots and results
    ├── data_exploration.png
    ├── model_predictions.png
    ├── response_surfaces.png
    └── analysis_report.txt
```

## Troubleshooting

### Common Issues

1. **CSV Loading Errors**
   - Check file path and ensure CSV exists
   - Verify column order matches expected format
   - Ensure no missing values in numerical columns

2. **Poor Model Performance**
   - Check for outliers in the data exploration phase
   - Consider data scaling issues
   - Experiment with different length scales

3. **Memory Issues**
   - Reduce grid resolution in response surface plots
   - Use fewer cross-validation folds for large datasets

### Data Quality Checks

The tool automatically:
- Validates CSV format and column structure
- Identifies and reports missing values
- Converts data types and handles non-numeric entries
- Removes samples with incomplete data
- Warns about insufficient sample sizes

## Customization

### Modifying Kernels
```python
# Custom kernel definition in train_models method
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic

# Example: Using Matern kernel instead of RBF
kernel_custom = ConstantKernel(1.0) * Matern(length_scale=10.0, nu=1.5) + WhiteKernel(0.1)
```

### Adding New Features
```python
# Extend input features (modify load_data method)
self.X = np.column_stack([bet_area, pore_volume, additional_feature])
```

### Custom Predictions
```python
# Batch predictions for multiple samples
new_samples = np.array([[800, 0.8], [1000, 1.2], [600, 0.5]])
for i, sample in enumerate(new_samples):
    analyzer.predict_new_sample(sample[0], sample[1])
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{co2_uptake_gpr,
  title={CO2 Uptake Gaussian Process Regression Analysis Tool},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/co2-uptake-gpr}
}
```

## Contact

For questions, issues, or contributions, please contact:
- Email: your.email@institution.edu
- GitHub: [@your-username](https://github.com/your-username)

---

**Note**: This tool is designed for materials science research and CO2 adsorption analysis. Always validate results with experimental data and domain expertise.
