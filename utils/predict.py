import joblib
import os
import numpy as np
from utils.transform import boxcox_transform, inverse_boxcox

# Load model and metadata
model_path = os.path.join("model", "linear_model.pkl")
feature_path = os.path.join("model", "feature_names.pkl")
metadata_path = os.path.join("model", "boxcox_metadata.pkl")

model = joblib.load(model_path)
feature_names = joblib.load(feature_path)
boxcox_metadata = joblib.load(metadata_path)

def predict_price(raw_features):
    # Sanity check: all inputs must be finite
    if not np.all(np.isfinite(raw_features)):
        raise ValueError("❌ Input contains NaN or infinite values.")

    # Transform input features using Box-Cox
    transformed_input = boxcox_transform(raw_features, feature_names)

    # Validate transformed input
    if not np.all(np.isfinite(transformed_input)):
        raise ValueError("❌ Transformed input contains NaN or infinite values. Check input ranges.")

    # Predict in transformed space
    y_pred_transformed = model.predict(transformed_input)[0]

    # Log prediction for debugging
    print(f"🔍 y_pred_transformed = {y_pred_transformed}")

    # Inverse-transform prediction to original price
    λ = boxcox_metadata['price']['lambda']
    shift = boxcox_metadata['price']['shift']
    y_pred_actual = inverse_boxcox(y_pred_transformed, λ, shift)

    return y_pred_actual