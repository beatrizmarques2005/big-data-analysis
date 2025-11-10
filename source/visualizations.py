
# Typing
from typing import List, Union
import numpy as np
# PySpark
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def hist_plots_spark(train_df: DataFrame, val_df: DataFrame, cols: List[str]) -> None:
    """
    Generates interactive histogram plots for numerical columns in Spark DataFrames.

    Parameters:
        train_df (DataFrame): Spark DataFrame containing training data.
        val_df (DataFrame): Spark DataFrame containing validation data.
        cols (List[str]): List of numerical column names to plot.
    """
    numerical_columns = cols

    if not numerical_columns:
        print("No numerical columns found in the DataFrames.")
        return

    fig = go.Figure()
    buttons = []

    for idx, column in enumerate(numerical_columns):
        train_col_data = train_df.select(column).dropna().rdd.flatMap(lambda x: x).collect()
        val_col_data = val_df.select(column).dropna().rdd.flatMap(lambda x: x).collect()

        fig.add_trace(go.Histogram(
            x=train_col_data,
            name=f'Train - <i>{column}</i>',
            marker_color='darkgreen',
            visible=(idx == 0),
            nbinsx=30,
            hovertemplate='Interval: %{x}<br>Frequency: %{y}<extra></extra>'
        ))
        fig.add_trace(go.Histogram(
            x=val_col_data,
            name=f'Validation - <i>{column}</i>',
            marker_color='darkmagenta',
            visible=(idx == 0),
            nbinsx=30,
            hovertemplate='Interval: %{x}<br>Frequency: %{y}<extra></extra>'
        ))

        buttons.append({
            'method': 'update',
            'label': column,
            'args': [
                {'visible': [i == idx * 2 or i == idx * 2 + 1 for i in range(len(numerical_columns) * 2)]},
                {'title': f'Histogram Plot for <i>{column}</i>', 'showlegend': True}
            ]
        })

    fig.update_layout(
        updatemenus=[{
            'type': 'dropdown',
            'active': 0,
            'buttons': buttons,
            'x': 1,
            'y': 1.15
        }],
        title=f'Histogram Plot for <i>{numerical_columns[0]}</i>',
        showlegend=True,
        yaxis=dict(showticklabels=False)
    )

    fig.show()

def box_plots_spark(train_df: DataFrame, val_df: DataFrame, cols: List[str]) -> None:
    """
    Generates interactive box plots for numerical columns in Spark DataFrames.

    Parameters:
        train_df (DataFrame): Spark DataFrame containing training data.
        val_df (DataFrame): Spark DataFrame containing validation data.
        cols (List[str]): List of numerical column names to plot.
    """
    numerical_columns = cols

    if not numerical_columns:
        print("No numerical columns found in the DataFrames.")
        return

    fig = go.Figure()
    buttons = []

    for idx, column in enumerate(numerical_columns):
        train_col_data = train_df.select(column).dropna().rdd.flatMap(lambda x: x).collect()
        val_col_data = val_df.select(column).dropna().rdd.flatMap(lambda x: x).collect()

        fig.add_trace(go.Box(
            y=train_col_data,
            name=f'Train - <i>{column}</i>',
            marker_color='darkgreen',
            visible=(idx == 0)
        ))
        fig.add_trace(go.Box(
            y=val_col_data,
            name=f'Validation - <i>{column}</i>',
            marker_color='darkmagenta',
            visible=(idx == 0)
        ))

        buttons.append({
            'method': 'update',
            'label': column,
            'args': [
                {'visible': [i == idx * 2 or i == idx * 2 + 1 for i in range(len(numerical_columns) * 2)]},
                {'title': f'Box Plot for <i>{column}</i>', 'showlegend': True}
            ]
        })

    fig.update_layout(
        updatemenus=[{
            'type': 'dropdown',
            'active': 0,
            'buttons': buttons,
            'x': 1,
            'y': 1.15
        }],
        title=f'Box Plot for <i>{numerical_columns[0]}</i>',
        showlegend=True,
        yaxis=dict(showticklabels=False)
    )

    fig.show()

def bar_plots_spark(train_df: DataFrame, val_df: DataFrame, cols: List[str]) -> None:
    """
    Generates interactive bar plots for categorical columns in Spark DataFrames.

    Parameters:
        train_df (DataFrame): Spark DataFrame containing training data.
        val_df (DataFrame): Spark DataFrame containing validation data.
        cols (List[str]): List of categorical column names to plot.
    """
    categorical_columns = cols

    if not categorical_columns:
        print("No categorical columns found in both DataFrames.")
        return

    fig = go.Figure()
    buttons = []

    for idx, column in enumerate(categorical_columns):
        train_counts = train_df.groupBy(column).agg(F.count("*").alias("count")).dropna().orderBy(column).collect()
        val_counts = val_df.groupBy(column).agg(F.count("*").alias("count")).dropna().orderBy(column).collect()

        train_x = [row[column] for row in train_counts]
        train_y = [row["count"] for row in train_counts]
        val_x = [row[column] for row in val_counts]
        val_y = [row["count"] for row in val_counts]

        fig.add_trace(go.Bar(
            x=train_x,
            y=train_y,
            name=f'Train - <i>{column}</i>',
            marker_color='green',
            visible=(idx == 0)
        ))
        fig.add_trace(go.Bar(
            x=val_x,
            y=val_y,
            name=f'Validation - <i>{column}</i>',
            marker_color='darkmagenta',
            visible=(idx == 0)
        ))

        buttons.append({
            'method': 'update',
            'label': column,
            'args': [
                {'visible': [i == idx * 2 or i == idx * 2 + 1 for i in range(len(categorical_columns) * 2)]},
                {'title': f'Bar Chart for <i>{column}</i>', 'showlegend': True}
            ]
        })

    fig.update_layout(
        updatemenus=[{
            'type': 'dropdown',
            'active': 0,
            'buttons': buttons,
            'x': 1,
            'y': 1.15
        }],
        title=f'Bar Chart for <i>{categorical_columns[0]}</i>',
        showlegend=True
    )

    fig.show()

def plot_feature_distributions_by_target(df: DataFrame, target_col: str, feature_cols: List[str]) -> None:
    """
    Plots grouped bar charts showing the distribution of each feature column by target class.

    Parameters:
        df (DataFrame): Spark DataFrame containing the data.
        target_col (str): Name of the target column.
        feature_cols (List[str]): List of categorical feature columns to visualize.
    """
    if not feature_cols:
        print("No feature columns provided.")
        return

    fig = go.Figure()
    buttons = []

    for idx, feature in enumerate(feature_cols):
        # Aggregate counts by feature and target
        grouped = df.groupBy(feature, target_col).agg(F.count("*").alias("count")).collect()

        # Organize data
        categories = sorted(set(row[feature] for row in grouped if row[feature] is not None))
        targets = sorted(set(row[target_col] for row in grouped if row[target_col] is not None))

        # Build traces for each target class
        for t_idx, target_value in enumerate(targets):
            counts = {row[feature]: row["count"] for row in grouped if row[target_col] == target_value}
            y_vals = [counts.get(cat, 0) for cat in categories]

            fig.add_trace(go.Bar(
                x=categories,
                y=y_vals,
                name=f'{target_col} = {target_value}',
                visible=(idx == 0),
                marker=dict(color=f'rgba({50 + t_idx*50}, {100 + t_idx*30}, {150 - t_idx*40}, 0.8)')
            ))

        # Button for dropdown
        buttons.append({
            'method': 'update',
            'label': feature,
            'args': [
                {'visible': [i // len(targets) == idx for i in range(len(feature_cols) * len(targets))]},
                {'title': f'Distribution of <i>{feature}</i> by <i>{target_col}</i>', 'showlegend': True}
            ]
        })

    fig.update_layout(
        updatemenus=[{
            'type': 'dropdown',
            'active': 0,
            'buttons': buttons,
            'x': 1,
            'y': 1.15
        }],
        title=f'Distribution of <i>{feature_cols[0]}</i> by <i>{target_col}</i>',
        barmode='group',
        showlegend=True
    )

    fig.show()

def plot_numerical_histograms_by_target(df: DataFrame, target_col: str, numeric_cols: List[str]) -> None:
    """
    Creates interactive histograms showing the distribution of numerical features across target classes.

    Parameters:
        df (DataFrame): Spark DataFrame containing the data.
        target_col (str): Name of the target column.
        numeric_cols (List[str]): List of numerical feature columns to visualize.
    """
    if not numeric_cols:
        print("No numerical columns provided.")
        return

    fig = go.Figure()
    buttons = []

    # Get distinct target values
    target_values = [row[target_col] for row in df.select(target_col).distinct().collect()]

    for idx, feature in enumerate(numeric_cols):
        traces = []
        for t_idx, target_value in enumerate(target_values):
            filtered = df.filter(F.col(target_col) == target_value).select(feature).dropna().collect()
            values = [row[feature] for row in filtered]

            traces.append(go.Histogram(
                x=values,
                name=f'{target_col} = {target_value}',
                opacity=0.6,
                nbinsx=30,
                visible=(idx == 0)
            ))

        for trace in traces:
            fig.add_trace(trace)

        buttons.append({
            'method': 'update',
            'label': feature,
            'args': [
                {'visible': [i // len(target_values) == idx for i in range(len(numeric_cols) * len(target_values))]},
                {'title': f'Histogram of <i>{feature}</i> by <i>{target_col}</i>', 'barmode': 'overlay', 'showlegend': True}
            ]
        })

    fig.update_layout(
        updatemenus=[{
            'type': 'dropdown',
            'active': 0,
            'buttons': buttons,
            'x': 1,
            'y': 1.15
        }],
        title=f'Histogram of <i>{numeric_cols[0]}</i> by <i>{target_col}</i>',
        xaxis_title='Feature Value',
        yaxis_title='Count',
        barmode='overlay',
        showlegend=True
    )

    fig.show()

def plot_spark_correlation_heatmap(df: DataFrame, numeric_cols: List[str]) -> None:
    """
    Efficiently computes and plots a correlation heatmap for numerical features in a Spark DataFrame.

    Parameters:
        df (DataFrame): Spark DataFrame containing the data.
        numeric_cols (List[str]): List of numerical column names to include in the correlation matrix.
    """
    # Assemble features into a single vector column
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    vector_df = assembler.transform(df.select(numeric_cols)).select("features")

    # Compute correlation matrix
    corr_matrix = Correlation.corr(vector_df, "features", method="pearson").head()[0].toArray()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=numeric_cols,
        y=numeric_cols,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation')
    ))

    # Add annotations
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            fig.add_annotation(
                x=numeric_cols[j],
                y=numeric_cols[i],
                text=f"{corr_matrix[i][j]:.2f}",
                showarrow=False,
                font=dict(color="black", size=10)
            )

    fig.update_layout(
        title='Correlation Heatmap of Numerical Features (Optimized)',
        xaxis_title='Features',
        yaxis_title='Features',
        width=800,
        height=800
    )

    fig.show()

