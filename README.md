# Adaptive LASSO Regression for Real Estate Pricing

## Project Overview

This project implements an Adaptive LASSO regression model for feature selection in real estate pricing. The primary goal is to leverage a dynamically adaptive proximal gradient method with a scaled soft-thresholding operator to handle correlated features more effectively than standard LASSO. The project utilizes the "House Prices: Advanced Regression Techniques" dataset to predict house prices, effectively balancing model accuracy with feature selection (sparsity).

## Approach and Modularity

The codebase has been refactored to be modular, reproducible, and executable. 
- **`data_processing.py`**: Handles loading, cleaning, categorical mapping, and standard scaling.
- **`models.py`**: Contains the Ridge Regression baseline, standard LASSO baseline, and our custom **Adaptive LASSO optimizer via Proximal Gradient Descent**.
- **`visualization.py`**: Functions to automatically plot feature coefficients, MSE comparisons, sparsity, and optimization convergence.
- **`main.py`**: Orchestration script tying the modules together.
- **`adaptive_lasso.py`**: Legacy single-file version of the script.

## Setup and Execution

### Prerequisites

Ensure you have Python 3.7+ installed. 
Install the required packages using the generated `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Running the Pipeline

To execute the data processing, model training, and evaluation pipeline, simply run:

```bash
python main.py
```

This will output the performance metrics for the Ridge, LASSO, and Adaptive LASSO models to the console and save several visualizations in the current directory:
- `feature_coefficients.png`
- `sparsity_comparison.png`
- `mse_comparison.png`
- `coefficient_convergence.png`

## Performance Comparison

Using a scaled target and features, the models yield the following benchmark results on the test set:

| Model | Test MSE | Sparsity (Zero Coefficients) |
| --- | --- | --- |
| **Ridge Regression** | 0.5124 | 0 / 12 |
| **Standard LASSO** | 0.8543 | 10 / 12 |
| **Adaptive LASSO** | 0.5113 | 2 / 12 |

### Conclusion

Our results demonstrate the effectiveness of the Adaptive LASSO method:
- **Accuracy**: It achieves the lowest Test Mean Squared Error (MSE: 0.5113), slightly outperforming the robust Ridge regression model.
- **Feature Selection**: Unlike Ridge which yields a fully dense coefficient vector, Adaptive LASSO yields a sparse model (2 features completely pruned to exactly zero). This strikes a superior balance compared to standard LASSO, which aggressively over-regularizes and drops informative features leading to a high Test MSE.
- The Proximal Gradient Descent optimizer effectively converges and selects meaningful real estate features.
