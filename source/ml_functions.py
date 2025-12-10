import numpy as np
import matplotlib.pyplot as plt

def grouped_feature_importance(feature_names, categorical_cols, numerical_cols, coefs):
    """
    Groups OHE-expanded coefficients back to original features.
    """
    importance = {}
    idx = 0

    # --- numerical features (1 coefficient each) ---
    for col in numerical_cols:
        importance[col] = abs(coefs[idx])
        idx += 1

    # --- categorical features (multiple coefficients each) ---
    for col in categorical_cols:
        # count how many coefficients belong to this categorical feature
        group_size = sum(1 for name in feature_names if name.startswith(col + "_"))
        group_coefs = coefs[idx:idx + group_size]
        importance[col] = np.sum(np.abs(group_coefs))
        idx += group_size

    return importance




import matplotlib.pyplot as plt
import numpy as np

def plot_grouped_feature_importance(importance_dict, title="Feature Importance Plot"):
    # Convert dict to sorted list of tuples
    items = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    features = [x[0] for x in items]
    values = [x[1] for x in items]

    plt.figure(figsize=(10, 6))

    # Horizontal bar plot centered around zero
    y_positions = np.arange(len(features))
    plt.barh(y_positions, values, color="steelblue")

    # Add feature names on y-axis
    plt.yticks(y_positions, features)

    # Add title + labels
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel("Coefficient Value")

    # Flip y-axis so highest importance is at top
    plt.gca().invert_yaxis()

    # Add vertical line at 0
    plt.axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.show()
