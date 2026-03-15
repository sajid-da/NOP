import sys
from data_processing import load_and_preprocess_data
from models import train_ridge, train_lasso, train_adaptive_lasso
from visualization import create_visualizations

def main():
    print("Starting Adaptive LASSO Project Pipeline...")
    
    # 1. Data Loading and Preprocessing
    try:
        X_train, X_test, y_train, y_test, feature_names, scaler_X, scaler_y = load_and_preprocess_data('dataset/Housing.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
        
    D = len(feature_names)

    # 2. Ridge Regression Baseline
    print("\n--- 1. Ridge Regression ---")
    w_ridge, mse_ridge, sparsity_ridge, ridge_model = train_ridge(X_train, y_train, X_test, y_test)
    print(f"Best Alpha: {ridge_model.alpha_:.4f}")
    print(f"Test MSE: {mse_ridge:.4f}")
    print(f"Sparsity (zero coefficients): {sparsity_ridge} / {D}")

    # 3. Standard LASSO Baseline
    print("\n--- 2. Standard LASSO ---")
    w_lasso, mse_lasso, sparsity_lasso, lasso_model, lasso_alpha = train_lasso(X_train, y_train, X_test, y_test)
    print(f"Alpha (lambda): {lasso_alpha:.4f}")
    print(f"Test MSE: {mse_lasso:.4f}")
    print(f"Sparsity (zero coefficients): {sparsity_lasso} / {D}")

    # 4. Adaptive LASSO with Proximal Gradient Descent
    print("\n--- 3. Adaptive LASSO (Proximal Gradient) ---")
    w_adap, w_hist, mse_adap, sparsity_adap = train_adaptive_lasso(X_train, y_train, X_test, y_test, w_ridge, lasso_alpha)
    print(f"Test MSE: {mse_adap:.4f}")
    print(f"Sparsity (zero coefficients): {sparsity_adap} / {D}")

    # 5. Visualizations
    mses = [mse_ridge, mse_lasso, mse_adap]
    sparsities = [sparsity_ridge, sparsity_lasso, sparsity_adap]
    
    create_visualizations(D, feature_names, w_ridge, w_lasso, w_adap, sparsities, mses, w_hist)
    print("\nPipeline completed successfully! Visualizations saved.")

if __name__ == '__main__':
    main()
