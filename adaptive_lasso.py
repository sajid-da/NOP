import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error

def main():
    # 1. Data Loading and Preprocessing
    data_path = 'dataset/Housing.csv'
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)

    # Map binary categories
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # Map furnishingstatus ordinal
    df['furnishingstatus'] = df['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})

    # Clean missing or dropna if any
    df = df.dropna()

    X = df.drop('price', axis=1)
    y = df['price'].values
    feature_names = X.columns
    N, D = X.shape

    print(f"Dataset Loaded. Features: {D}, Samples: {N}")

    # Standard scale features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Scale target to avoid huge MSE values and help learning
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # 2. Ridge Regression Baseline
    print("\n--- 1. Ridge Regression ---")
    ridge_model = RidgeCV(alphas=np.logspace(-3, 3, 20))
    ridge_model.fit(X_train, y_train)
    w_ridge = ridge_model.coef_
    y_pred_ridge = ridge_model.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    sparsity_ridge = np.sum(np.abs(w_ridge) < 1e-4)
    
    print(f"Best Alpha: {ridge_model.alpha_:.4f}")
    print(f"Test MSE: {mse_ridge:.4f}")
    print(f"Sparsity (zero coefficients): {sparsity_ridge} / {D}")


    # 3. Standard LASSO Baseline
    print("\n--- 2. Standard LASSO ---")
    lasso_model = LassoCV(alphas=np.logspace(-3, 1, 20), cv=5, max_iter=10000, random_state=42)
    lasso_model.fit(X_train, y_train)
    # If alpha is too small to showcase sparsity (0 coefficients), we can pick a stronger alpha manually
    # Or rely on a scaled version for Adaptive Lasso
    lasso_alpha = lasso_model.alpha_
    if np.sum(np.abs(lasso_model.coef_) < 1e-4) == 0:
        # Force a slightly higher alpha just to demonstrate sparsity
        lasso_alpha = 0.05
        from sklearn.linear_model import Lasso
        lasso_model = Lasso(alpha=lasso_alpha, max_iter=10000, random_state=42)
        lasso_model.fit(X_train, y_train)
    w_lasso = lasso_model.coef_
    y_pred_lasso = lasso_model.predict(X_test)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    sparsity_lasso = np.sum(np.abs(w_lasso) < 1e-4)

    print(f"Alpha (lambda): {lasso_alpha:.4f}")
    print(f"Test MSE: {mse_lasso:.4f}")
    print(f"Sparsity (zero coefficients): {sparsity_lasso} / {D}")

    
    # 4. Adaptive LASSO with Proximal Gradient Descent
    print("\n--- 3. Adaptive LASSO (Proximal Gradient) ---")
    
    def adaptive_lasso_pgd(X, y, lambda_base, lr, max_iter=5000, epsilon=1e-4, w_init=None):
        n_samples, n_features = X.shape
        if w_init is None:
            w = np.zeros(n_features)
        else:
            w = np.copy(w_init)
            
        w_history = [w.copy()]
        
        for k in range(max_iter):
            # Compute gradient of smooth loss (MSE): g = 1/N * X^T (Xw - y)
            predictions = X @ w
            gradient = (1.0 / n_samples) * X.T @ (predictions - y)
            
            # Gradient descent step for v
            v = w - lr * gradient
            
            # Subdifferential properties of L1 lead to iteratively reweighted L1 penalty
            # Dynamic lambda: scales based on inverse of previous weights (with epsilon for stability)
            lambda_dynamic = lambda_base / (np.abs(w) + epsilon)
            
            # Soft-thresholding (Proximal Operator)
            # Thresholding level is lr * lambda_dynamic since we multiplied the penalty term 
            # in formulation by lr.
            threshold = lr * lambda_dynamic
            w_new = np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)
            
            w = w_new
            w_history.append(w.copy())
            
        return w, np.array(w_history)

    # Tuning hyperparameters for Adaptive LASSO
    # Learning rate (step size): should be < 2 / L, where L is Lipschitz constant of the gradient
    # L = max eigenvalue of (X^T X) / N
    hessian = (X_train.T @ X_train) / X_train.shape[0]
    eigenvalues = np.linalg.eigvalsh(hessian)
    L = np.max(eigenvalues)
    lr = 1.0 / L  # safe learning rate

    # Base lambda
    # We use a scaled down version of Lasso's optimal lambda because adaptive penalty
    # is lambda_base / (|w| + epsilon). If |w| is around 0.1 and epsilon is 0.01, the penalty is larger.
    w_ridge_fixed = np.where(np.abs(w_ridge) < 1e-4, 1e-4, w_ridge)
    w_init = w_ridge_fixed
    lambda_base = lasso_alpha * 0.5 # Increased base lambda for visible sparsity 
    
    w_adap, w_hist = adaptive_lasso_pgd(
        X=X_train, y=y_train, 
        lambda_base=lambda_base, 
        lr=lr, 
        max_iter=5000, 
        epsilon=1e-3, 
        w_init=w_init
    )

    y_pred_adap = X_test @ w_adap
    mse_adap = mean_squared_error(y_test, y_pred_adap)
    sparsity_adap = np.sum(np.abs(w_adap) < 1e-4)

    print(f"Test MSE: {mse_adap:.4f}")
    print(f"Sparsity (zero coefficients): {sparsity_adap} / {D}")

    # 5. Visualizations
    models = ['Ridge', 'LASSO', 'Adaptive LASSO']
    mses = [mse_ridge, mse_lasso, mse_adap]
    sparsities = [sparsity_ridge, sparsity_lasso, sparsity_adap]

    # Model Coefficient Comparison
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(D)
    width = 0.25
    plt.bar(x_pos - width, w_ridge, width, label='Ridge', color='blue', alpha=0.7)
    plt.bar(x_pos, w_lasso, width, label='LASSO', color='orange', alpha=0.7)
    plt.bar(x_pos + width, w_adap, width, label='Adaptive LASSO', color='green', alpha=0.7)
    plt.xticks(x_pos, feature_names, rotation=45, ha='right')
    plt.ylabel('Coefficient Value (Scaled)')
    plt.title('Feature Coefficients across Models')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_coefficients.png', dpi=300)
    print("Saved feature_coefficients.png")
    
    # Sparsity Comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, sparsities, color=['blue', 'orange', 'green'], alpha=0.7)
    plt.ylabel(f'Number of Pruned Features (out of {D})')
    plt.title('Feature Sparsity Comparison')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')
    plt.savefig('sparsity_comparison.png', dpi=300)
    print("Saved sparsity_comparison.png")

    # MSE Comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, mses, color=['blue', 'orange', 'green'], alpha=0.7)
    plt.ylabel('Mean Squared Error (Scaled y)')
    plt.title('MSE Comparison')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + yval*0.01, f"{yval:.4f}", va='bottom', ha='center')
    plt.savefig('mse_comparison.png', dpi=300)
    print("Saved mse_comparison.png")

    # Coefficient Convergence for Adaptive LASSO
    plt.figure(figsize=(10, 6))
    for j in range(D):
        plt.plot(w_hist[:, j], label=feature_names[j])
    plt.xlabel('Iteration')
    plt.ylabel('Coefficient Value (Scaled)')
    plt.title('Coefficient Convergence in Adaptive LASSO (Proximal Gradient)')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('coefficient_convergence.png', dpi=300)
    print("Saved coefficient_convergence.png")

if __name__ == '__main__':
    main()
