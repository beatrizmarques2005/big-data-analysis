
# -----------------------------
# IMPORTS
# -----------------------------

import re
from typing import List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col,
    when,
    expr,
    lit,
    concat_ws,
    create_map,
)
from pyspark.sql.types import (
    IntegerType,
    DoubleType,
)

from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    Normalizer,
)

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

def transform_type(df: DataFrame, column_list: list, to_type: str) -> DataFrame:
    """
    Safely converts selected columns to a specific type using try_cast().
    Malformed rows become NULL instead of causing job failures.
    """
    for c in column_list:
        df = df.withColumn(c, F.expr(f"try_cast(`{c}` AS {to_type})"))
    return df

# -----------------------------
# OUTLIERS
# -----------------------------

class Winsorizer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, columns, lower_q=0.25, upper_q=0.75, iqr_multiplier=1.5):
        super().__init__()
        self.columns = columns
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.iqr_multiplier = iqr_multiplier
        self.bounds = {}

    def _fit(self, df: DataFrame):
        """Compute IQR-based winsorization bounds from reference df."""
        for c in self.columns:
            q1, q3 = df.approxQuantile(c, [self.lower_q, self.upper_q], 0.01)
            iqr = q3 - q1

            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            
            self.bounds[c] = (lower, upper)
        return self

    def _transform(self, df: DataFrame):
        """Apply winsorization to DataFrame using learned IQR bounds."""
        out_df = df
        for c in self.columns:
            lower, upper = self.bounds[c]
            out_df = out_df.withColumn(
                c,
                when(col(c) < lower, lower)
                .when(col(c) > upper, upper)
                .otherwise(col(c))
            )
        return out_df

    def fit(self, df):
        return self._fit(df)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

class FeatureEngineering(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self):
        super().__init__()
        self.campaign_median = None

    def fit(self, df: DataFrame):
        """Compute statistics that must be learned from training data."""
        self.campaign_median = df.approxQuantile("campaign", [0.5], 0.01)[0]
        return self

    def _transform(self, df: DataFrame) -> DataFrame:
        """Apply all feature engineering transformations."""
        
        df = df.withColumn("balance_per_campaign", col("balance_euros") / (col("campaign") + 1))
        df = df.withColumn("log_balance", F.log1p(col("balance_euros")))
        df = df.withColumn("had_previous_contact", when(col("pdays") != -1, 1).otherwise(0))

        df = df.withColumn(
            "high_effort_client", 
            when(col("campaign") > self.campaign_median, 1).otherwise(0)
        )
        
        return df
