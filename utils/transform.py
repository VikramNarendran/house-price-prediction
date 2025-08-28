import numpy as np
import joblib
import os

# Load Box-Cox metadata
metadata_path = os.path.join("model", "boxcox_metadata.pkl")
boxcox_metadata = joblib.load(metadata_path)

def boxcox_transform(features, feature_names):
    transformed = []
    for val, name in zip(features, feature_names):
        if name in boxcox_metadata:
            λ = boxcox_metadata[name]['lambda']
            shift = boxcox_metadata[name]['shift']
            val_shifted = val - shift + 1 if shift != 0 else val

            if val_shifted <= 0 or not np.isfinite(val_shifted):
                raise ValueError(f"Feature '{name}' must be positive and finite after shifting. Got: {val_shifted}")

            if λ == 0:
                transformed_val = np.log(val_shifted)
            else:
                transformed_val = (val_shifted ** λ - 1) / λ

            if not np.isfinite(transformed_val):
                raise ValueError(f"Transformed value for '{name}' is not finite. Got: {transformed_val}")

            transformed.append(transformed_val)
        else:
            # No transformation needed (e.g., binary features)
            transformed.append(val)

    transformed_array = np.array(transformed).reshape(1, -1)

    if not np.all(np.isfinite(transformed_array)):
        raise ValueError("Transformed input contains NaN or infinite values.")

    return transformed_array

def inverse_boxcox(y_transformed, λ, shift):
    if not np.isfinite(y_transformed):
        raise ValueError(f"Invalid transformed prediction: {y_transformed}")

    base = y_transformed * λ + 1
    if base <= 0:
        raise ValueError(f"Inverse Box-Cox base is non-positive: {base}")

    if λ == 0:
        y_original = np.exp(y_transformed)
    else:
        y_original = np.power(base, 1 / λ)

    if not np.isfinite(y_original):
        raise ValueError(f"Inverse-transformed prediction is not finite. Got: {y_original}")

    return y_original + shift