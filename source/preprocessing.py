
import re
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer
from typing import List


def name_cleaner(name: str, char_list: list) -> str:
    """
    Cleans a given string to make it safe for use in identifiers or filenames.

    This function replaces each character in `char_list` with an underscore,
    except for parentheses '(' and ')' which are removed. After that, any
    remaining non-alphanumeric characters are removed.

    Parameters
    name(str): The input string to be cleaned.
    char_list(list): A list of characters to be removed or replaced with underscores.

    """

    temp_name = name
    for char in char_list:
        if char == '(' or char == ')':
            temp_name = temp_name.replace(char, "")
        else:
            temp_name = temp_name.replace(char, "_")
    #temp_name = name.replace(" ", "_").replace("-", "_").replace("/", "_").replace("&", "and").replace("(", "").replace(")", "")

    return re.sub(r'[^\w]', '', temp_name)

def show_column_types(df: DataFrame):
    """
    Prints the name and data type of each column in a Spark DataFrame.

    Parameters
    df(DataFrame): The Spark DataFrame to inspect.
    """
    print("Column Name - Data Type")
    print("-" * 30)
    for col_name, dtype in df.dtypes:
        print(f"{col_name} - {dtype}")

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def transform_type(df: DataFrame, column_list: list, to_type: str) -> DataFrame:
    """
    Safely converts selected columns to a specific type using try_cast().
    Malformed rows become NULL instead of causing job failures.
    """
    for c in column_list:
        df = df.withColumn(c, F.expr(f"try_cast(`{c}` AS {to_type})"))
    return df


def winsorize_spark(reference_df: DataFrame, apply_to_df: DataFrame, columns: List[str]) -> DataFrame:
    """
    Apply winsorization to specified columns in a Spark DataFrame using IQR from a reference DataFrame.

    Parameters:
        reference_df (DataFrame): Spark DataFrame used to calculate IQR.
        apply_to_df (DataFrame): Spark DataFrame to apply winsorization.
        columns (List[str]): List of column names to winsorize.

    Returns:
        DataFrame: Spark DataFrame with winsorized columns.
    """
    for col in columns:
        # Calculate Q1 and Q3 from reference_df
        q1, q3 = reference_df.approxQuantile(col, [0.25, 0.75], 0.01)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Apply winsorization using when/otherwise
        apply_to_df = apply_to_df.withColumn(
            col,
            F.when(F.col(col) < lower_bound, lower_bound)
             .when(F.col(col) > upper_bound, upper_bound)
             .otherwise(F.col(col))
        )

    return apply_to_df

def index_column(train_df: DataFrame, val_df: DataFrame, col: str) -> tuple:
    """
    Fit a StringIndexer on train_df and transform both train and val DataFrames.

    Returns:
        (train_transformed, val_transformed)
    """
    indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
    indexer_model = indexer.fit(train_df)
    train_df = indexer_model.transform(train_df)
    val_df = indexer_model.transform(val_df)
    return train_df, val_df

def onehot_column(train_df: DataFrame, val_df: DataFrame, col: str) -> tuple:
    """
    Fit a OneHotEncoder on train_df and transform both train and val DataFrames.

    Returns:
        (train_transformed, val_transformed)
    """
    encoder = OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec")
    encoder_model = encoder.fit(train_df)
    train_df = encoder_model.transform(train_df)
    val_df = encoder_model.transform(val_df)
    return train_df, val_df

def assemble_features(train_df: DataFrame, val_df: DataFrame, input_cols: list, output_col: str) -> tuple:
    """
    Assemble a Spark feature vector for given columns.

    Parameters:
        train_df (DataFrame): Training DataFrame
        val_df (DataFrame): Validation DataFrame
        input_cols (list): List of columns to assemble
        output_col (str): Name of the output vector column

    Returns:
        tuple: (train_transformed, val_transformed)
    """
    assembler = VectorAssembler(inputCols=input_cols, outputCol=output_col)
    train_df = assembler.transform(train_df)
    val_df = assembler.transform(val_df)
    return train_df, val_df

def normalize_features(train_df: DataFrame, val_df: DataFrame, input_col: str, output_col: str, p: float = 2.0) -> tuple:
    """
    Normalize a feature vector column in Spark using Lp norm.

    Parameters:
        train_df (DataFrame): Training DataFrame
        val_df (DataFrame): Validation DataFrame
        input_col (str): Name of the input vector column
        output_col (str): Name of the output normalized vector column
        p (float, optional): Lp norm (default 2.0)

    Returns:
        tuple: (train_transformed, val_transformed)
    """
    normalizer = Normalizer(inputCol=input_col, outputCol=output_col, p=p)
    train_df = normalizer.transform(train_df)
    val_df = normalizer.transform(val_df)
    return train_df, val_df



