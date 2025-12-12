import re
import json
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import (col, when)
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import Param, Params


def name_cleaner(name: str, char_list: list) -> str:
    """
    Cleans a string to make it safe for use as an identifier or filename.

    Each character in `char_list` is replaced with an underscore (`_`), 
    except for parentheses '(' and ')' which are removed. 
    Finally, any remaining non-alphanumeric characters are removed.

    Parameters
    ----------
    name : str
        The input string to be cleaned.
    char_list : list of str
        Characters to replace with underscores or remove.

    Returns
    -------
    str
        The cleaned string containing only alphanumeric characters and underscores.
    """

    temp_name = name
    for char in char_list:
        if char == '(' or char == ')':
            temp_name = temp_name.replace(char, "")
        else:
            temp_name = temp_name.replace(char, "_")
    
    return re.sub(r'[^\w]', '', temp_name)

def show_column_types(df: DataFrame) -> None:
    """
    Prints the names and data types of all columns in a Spark DataFrame.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The DataFrame to inspect.

    Returns
    -------
    None
        Prints the column names and types to the console.

    """
    print("Column Name - Data Type")
    print("-" * 30)

    for col_name, dtype in df.dtypes:
        print(f"{col_name} - {dtype}")

def transform_type(df: DataFrame, column_list: list, to_type: str) -> DataFrame:
    """
    Safely converts the data type of multiple columns in a Spark DataFrame using `try_cast`.

    Any value that cannot be converted will result in `NULL` instead of causing a failure,
    making this function safe for dirty or inconsistent data.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The input Spark DataFrame containing the columns to transform.
    column_list : list of str
        A list of column names in `df` that should be converted to the new type.
    to_type : str
        The target data type as a string (e.g., 'INT', 'DOUBLE', 'STRING').

    Returns
    -------
    pyspark.sql.DataFrame
        A new DataFrame with the specified columns converted to the target type.
    """
    for c in column_list:
        df = df.withColumn(c, F.expr(f"try_cast(`{c}` AS {to_type})"))
    return df

# -----------------------------
# OUTLIERS
# -----------------------------
class Winsorizer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    PySpark Transformer that winsorizes numerical columns based on IQR (Interquartile Range).

    This class is fully serializable in PySpark and allows capping values in numerical 
    columns to reduce the effect of extreme outliers.
    """
    bounds = Param(Params._dummy(), "bounds", "JSON dictionary of lower and upper bounds")

    def __init__(self, columns=None, lower_q=0.25, upper_q=0.75, iqr_multiplier=1.5):
        """
        Initialize the Winsorizer.

        Parameters
        ----------
        columns : list of str, optional
            List of numerical columns to winsorize. Defaults to common numeric columns.
        lower_q : float, default=0.25
            Lower quantile for IQR calculation.
        upper_q : float, default=0.75
            Upper quantile for IQR calculation.
        iqr_multiplier : float, default=1.5
            Multiplier applied to the IQR to compute lower and upper bounds.
        """
        super().__init__()
        self.columns = columns or ["age", "campaign", "last_contact_day", "last_contact_duration", "balance_euros"]
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.iqr_multiplier = iqr_multiplier

        self._setDefault(bounds=json.dumps({}))

    def fit(self, df):
        """
        Compute IQR-based lower and upper bounds for each specified column.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to compute column bounds.

        Returns
        -------
        self : Winsorizer
            Returns the fitted Winsorizer with bounds stored as a JSON Param.
        """
        b = {}
        for c in self.columns:
            q1, q3 = df.approxQuantile(c, [self.lower_q, self.upper_q], 0.01)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            b[c] = (lower, upper)
        self._set(bounds=json.dumps(b))
        return self

    def _transform(self, df):
        """
        Apply winsorization to the DataFrame using the stored bounds.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with winsorized columns.
        """
        b = json.loads(self.getOrDefault(self.bounds))
        out_df = df
        for c, (lower, upper) in b.items():
            if c not in df.columns:
                continue 
            out_df = out_df.withColumn(
                c,
                when(col(c) < lower, lower)
                .when(col(c) > upper, upper)
                .otherwise(col(c))
            )
        return out_df

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
class FeatureEngineering(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    PySpark Transformer for custom feature engineering.

    Computes training-based statistics (e.g., median) and creates new features 
    from existing columns. Fully serializable for Spark ML pipelines.
    """
    def __init__(self):
        """Initialize the FeatureEngineering transformer."""
        super().__init__()
        self.campaign_median = None

    def fit(self, df: DataFrame):
        """
        Learn statistics required for feature engineering from the training DataFrame.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The input DataFrame to learn statistics from.

        Returns
        -------
        self : FeatureEngineering
            Returns the fitted transformer with computed statistics stored.
        """
        self.campaign_median = df.approxQuantile("campaign", [0.5], 0.01)[0]
        return self

    def _transform(self, df: DataFrame) -> DataFrame:
        """
        Apply feature engineering transformations to the DataFrame.

        Transformations:
        1. Creates 'had_previous_contact' based on 'pdays' (-1 → 0, else 1).
        2. Drops the original 'pdays' column.
        3. Creates 'high_effort_client' flag where 'campaign' exceeds median value.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pyspark.sql.DataFrame
            Transformed DataFrame with new features.
        """

        df = df.withColumn(
            "had_previous_contact", 
            when(col("pdays") != -1, 1).otherwise(0)
        )
        
        df = df.drop("pdays")

        # df = df.withColumn("balance_per_campaign", col("balance_euros") / (col("campaign") + 1))
        df = df.withColumn(
            "high_effort_client", 
            when(col("campaign") > self.campaign_median, 1).otherwise(0)
        )

        return df
