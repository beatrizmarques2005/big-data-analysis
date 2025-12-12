import numpy as np
import matplotlib.pyplot as plt

def grouped_feature_importance(feature_names, categorical_cols, numerical_cols, coefs):
    """
    Compute grouped feature importance by aggregating the importance of OHE-expanded features.

    This function reconstructs feature importances at the original column level.
    Numerical features appear once, while categorical features expanded by one-hot encoding
    are grouped by summing the absolute values of their corresponding coefficients.

    Parameters
    ----------
    feature_names : list of str
        List of expanded feature names output by the VectorAssembler.
    categorical_cols : list of str
        Original categorical columns (pre–one-hot encoding).
    numerical_cols : list of str
        Original numerical columns.
    coefs : array-like
        Model coefficients aligned with the expanded feature list.

    Returns
    -------
    dict
        Dictionary mapping each original feature name to its aggregated importance.
    """
    importance = {}
    idx = 0

    for col in numerical_cols:
        importance[col] = abs(coefs[idx])
        idx += 1

    for col in categorical_cols:
        group_size = sum(1 for name in feature_names if name.startswith(col + "_"))
        group_coefs = coefs[idx:idx + group_size]
        importance[col] = np.sum(np.abs(group_coefs))
        idx += group_size

    return importance


def plot_grouped_feature_importance(importance_dict, title="Feature Importance Plot"):
    """
    Plot grouped feature importances as a horizontal bar chart.

    This visualization helps interpret feature effects after grouping
    one-hot–encoded categories back into their original features.

    Parameters
    ----------
    importance_dict : dict
        Mapping of original feature names to aggregated importance values.
    title : str, optional
        Chart title, by default "Feature Importance Plot".

    Returns
    -------
    None
        Displays the plot.
    """
    items = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    features = [x[0] for x in items]
    values = [x[1] for x in items]

    plt.figure(figsize=(10, 6))

    y_positions = np.arange(len(features))
    plt.barh(y_positions, values, color="steelblue")

    plt.yticks(y_positions, features)

    plt.title(title, fontsize=16, pad=15)
    plt.xlabel("Coefficient Value")

    plt.gca().invert_yaxis()

    plt.axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.show()


def get_expanded_feature_names(pipeline_model, categorical_cols, numerical_cols):
    """
    Retrieve the expanded feature names produced by StringIndexer and OneHotEncoder.

    For each categorical column, this function determines how many encoded categories
    were generated (accounting for Spark's dropLast=True behavior), and reconstructs
    the corresponding expanded feature names. Numerical columns are appended unchanged.

    Parameters
    ----------
    pipeline_model : pyspark.ml.PipelineModel
        Fitted pipeline containing StringIndexerModels and encoders.
    categorical_cols : list of str
        Original categorical column names.
    numerical_cols : list of str
        Original numerical column names.

    Returns
    -------
    list of str
        List of expanded feature names in the same order expected by the model coefficients.
    """
    stages = pipeline_model.stages
    indexer_sizes = {}

    for st in stages:
        if "StringIndexerModel" in type(st).__name__:
            indexer_sizes[st.getOutputCol()] = len(st.labels)

    expanded_cat_features = []
    for col in categorical_cols:
        index_col = f"{col}_index"
        if index_col not in indexer_sizes:
            print(f"Skipping {col} (no StringIndexerModel found)")
            continue
        size = indexer_sizes[index_col]
        for i in range(size - 1):
            expanded_cat_features.append(f"{col}_{i}")

    return expanded_cat_features + numerical_cols
