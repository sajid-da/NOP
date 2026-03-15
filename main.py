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
    
    # 6. Prediction for a New House
    # Example: A house with the following features (matches D=12 columns in dataset)
    # ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 
    #  'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
    print("\n--- 4. Predicting a New Custom House ---")
    predict_choice = input("Would you like to predict the price of a custom house? (y/n): ")
    if predict_choice.strip().lower() == 'y':
        new_house_features = {}
        print("\nPlease enter the following feature values (press Enter to use default values):")
        print("Note: For yes/no, enter 1 for yes, 0 for no. For furnishing, enter 2 for furnished, 1 for semi, 0 for unfurnished.")
        
        default_features = {
            'area': 6000, 'bedrooms': 3, 'bathrooms': 2, 'stories': 2,
            'mainroad': 1, 'guestroom': 0, 'basement': 1, 'hotwaterheating': 0,
            'airconditioning': 1, 'parking': 2, 'prefarea': 1, 'furnishingstatus': 2
        }
        
        for feature in feature_names:
            val = input(f"{feature} [default: {default_features.get(feature, 0)}]: ")
            if val.strip() == '':
                new_house_features[feature] = default_features.get(feature, 0)
            else:
                try:
                    new_house_features[feature] = float(val)
                except ValueError:
                    print(f"Invalid input, defaulting to {default_features.get(feature, 0)}.")
                    new_house_features[feature] = default_features.get(feature, 0)
                    
        predicted_price = predict_new_house(new_house_features, feature_names, scaler_X, scaler_y, w_adap)
        print(f"\nPredicted Price (Adaptive LASSO): ${predicted_price:,.2f}")
    else:
        print("Skipping custom house prediction.")

    print("\nPipeline completed successfully! Visualizations saved.")

def predict_new_house(features_dict, feature_names, scaler_X, scaler_y, weights):
    """
    Predicts the exact price of a single new house given its raw features.
    Handles scaling explicitly using the training data scalers.
    """
    import numpy as np
    import pandas as pd
    
    # Ensure correct order of features
    feature_values = [features_dict.get(col, 0) for col in feature_names]
    
    # Scale features
    # Passing a DataFrame explicitly prevents the missing valid feature names warning from StandardScaler
    features_df = pd.DataFrame([feature_values], columns=feature_names)
    features_scaled = scaler_X.transform(features_df)
    
    # Predict (results in scaled price)
    price_scaled = features_scaled @ weights
    
    # Inverse scale to get raw dollar value
    price_original = scaler_y.inverse_transform(price_scaled.reshape(1, -1))[0][0]
    
    return price_original

if __name__ == '__main__':
    main()
