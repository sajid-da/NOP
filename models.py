import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, Lasso
from sklearn.metrics import mean_squared_error

def train_ridge(X_train, y_train, X_test, y_test):
    """
    Trains Ridge Regression baseline
    """
    ridge_model = RidgeCV(alphas=np.logspace(-3, 3, 20))
    ridge_model.fit(X_train, y_train)
    w_ridge = ridge_model.coef_
    y_pred_ridge = ridge_model.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    sparsity_ridge = np.sum(np.abs(w_ridge) < 1e-4)

    return w_ridge, mse_ridge, sparsity_ridge, ridge_model

def train_lasso(X_train, y_train, X_test, y_test):
    """
    Trains standard LASSO Baseline
    """
    lasso_model = LassoCV(alphas=np.logspace(-3, 1, 20), cv=5, max_iter=10000, random_state=42)
    lasso_model.fit(X_train, y_train)
    
    lasso_alpha = lasso_model.alpha_
    if np.sum(np.abs(lasso_model.coef_) < 1e-4) == 0:
        lasso_alpha = 0.05
        lasso_model = Lasso(alpha=lasso_alpha, max_iter=10000, random_state=42)
        lasso_model.fit(X_train, y_train)
        
    w_lasso = lasso_model.coef_
    y_pred_lasso = lasso_model.predict(X_test)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    sparsity_lasso = np.sum(np.abs(w_lasso) < 1e-4)

    return w_lasso, mse_lasso, sparsity_lasso, lasso_model, lasso_alpha

def adaptive_lasso_pgd(X, y, lambda_base, lr, max_iter=5000, epsilon=1e-4, w_init=None):
    """
    Custom Adaptive LASSO optimizer using Proximal Gradient Descent.
    """
    n_samples, n_features = X.shape
    if w_init is None:
        w = np.zeros(n_features)
    else:
        w = np.copy(w_init)
        
    w_history = [w.copy()]
    
    for k in range(max_iter):
        predictions = X @ w
        gradient = (1.0 / n_samples) * X.T @ (predictions - y)
        v = w - lr * gradient
        lambda_dynamic = lambda_base / (np.abs(w) + epsilon)
        threshold = lr * lambda_dynamic
        w_new = np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)
        
        w = w_new
        w_history.append(w.copy())
        
    return w, np.array(w_history)

def train_adaptive_lasso(X_train, y_train, X_test, y_test, w_ridge, lasso_alpha):
    """
    Wrapper to run adaptive LASSO PGD.
    """
    hessian = (X_train.T @ X_train) / X_train.shape[0]
    eigenvalues = np.linalg.eigvalsh(hessian)
    L = np.max(eigenvalues)
    lr = 1.0 / L

    w_ridge_fixed = np.where(np.abs(w_ridge) < 1e-4, 1e-4, w_ridge)
    w_init = w_ridge_fixed
    lambda_base = lasso_alpha * 0.5 
    
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

    return w_adap, w_hist, mse_adap, sparsity_adap

