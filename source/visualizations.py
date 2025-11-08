from typing import List
from pyspark.sql import DataFrame
import plotly.graph_objects as go
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import List
from pyspark.sql import DataFrame
from typing import Union
import pyspark.sql.functions as F
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

def target_mean_per_category(df: DataFrame, category_cols: List[str], target_col: str = "Rented_Bike_Count") -> None:
    """
    Displays the average target value (e.g., Rented_Bike_Count) per category.

    Parameters:
        df (DataFrame): The input Spark DataFrame.
        category_cols (List[str]): List of categorical columns to group by.
        target_col (str): The target column to average. Defaults to 'Rented_Bike_Count'.
    """
    for col in category_cols:
        print(f"▶ Average {target_col} per {col}:")
        df.groupBy(col).agg(F.mean(target_col).alias("avg_rented_bike_count")) \
          .orderBy('avg_rented_bike_count') \
          .show(truncate=False)

def line_target_distribution(train_df: DataFrame, val_df: DataFrame, title_train: str, title_val: str) -> None:
    """
    Generates side-by-side line charts showing average Rented_Bike_Count over Date
    for training and validation Spark DataFrames.

    Parameters:
        train_df (DataFrame): Spark DataFrame for training data.
        val_df (DataFrame): Spark DataFrame for validation data.
        title_train (str): Title for the training plot.
        title_val (str): Title for the validation plot.
    """

    # Aggregate average Rented_Bike_Count by Date
    train_counts = (
        train_df.groupBy("Date")
        .agg(F.mean("Rented_Bike_Count").alias("avg_rented"))
        .orderBy("Date")
        .collect()
    )
    val_counts = (
        val_df.groupBy("Date")
        .agg(F.mean("Rented_Bike_Count").alias("avg_rented"))
        .orderBy("Date")
        .collect()
    )

    # Extract x (Date) and y (average Rented_Bike_Count) values
    train_x = [row["Date"] for row in train_counts]
    train_y = [row["avg_rented"] for row in train_counts]
    val_x = [row["Date"] for row in val_counts]
    val_y = [row["avg_rented"] for row in val_counts]

    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=[title_train, title_val])

    # Add line plots
    fig.add_trace(
        go.Scatter(
            x=train_x,
            y=train_y,
            mode="lines+markers",
            line=dict(color="green"),
            name=title_train
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=val_x,
            y=val_y,
            mode="lines+markers",
            line=dict(color="darkmagenta"),
            name=title_val
        ),
        row=1,
        col=2
    )

    # Update layout
    fig.update_layout(
        title_text="Average Rented Bike Count Over Time",
        title_x=0.5,
        title_font_size=20,
        height=500,
        showlegend=False,
        xaxis_title="Date",
        yaxis_title="Average Rented_Bike_Count"
    )

    fig.show()




