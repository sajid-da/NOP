import matplotlib.pyplot as plt
import numpy as np
import os

def create_visualizations(D, feature_names, w_ridge, w_lasso, w_adap, sparsities, mses, w_hist, output_dir='.'):
    """
    Creates and saves performance and comparison visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    models = ['Ridge', 'LASSO', 'Adaptive LASSO']

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
    plt.savefig(os.path.join(output_dir, 'feature_coefficients.png'), dpi=300)
    print("Saved feature_coefficients.png")
    
    # Sparsity Comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, sparsities, color=['blue', 'orange', 'green'], alpha=0.7)
    plt.ylabel(f'Number of Pruned Features (out of {D})')
    plt.title('Feature Sparsity Comparison')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')
    plt.savefig(os.path.join(output_dir, 'sparsity_comparison.png'), dpi=300)
    print("Saved sparsity_comparison.png")

    # MSE Comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, mses, color=['blue', 'orange', 'green'], alpha=0.7)
    plt.ylabel('Mean Squared Error (Scaled y)')
    plt.title('MSE Comparison')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + yval*0.01, f"{yval:.4f}", va='bottom', ha='center')
    plt.savefig(os.path.join(output_dir, 'mse_comparison.png'), dpi=300)
    print("Saved mse_comparison.png")

    # Coefficient Convergence for Adaptive LASSO
    plt.figure(figsize=(10, 6))
    for j in range(D):
        plt.plot(w_hist[:, j], label=feature_names[j])
    plt.xlabel('Iteration')
    plt.ylabel('Coefficient Value (Scaled)')
    plt.title('Coefficient Convergence in Adaptive LASSO (Proximal Gradient)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coefficient_convergence.png'), dpi=300)
    print("Saved coefficient_convergence.png")
