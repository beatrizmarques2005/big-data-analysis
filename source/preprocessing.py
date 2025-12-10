
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



import json
from pyspark.ml import Transformer, Estimator, Model
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import col, when
# PROFESSORS' CODE!!!
# 1. Define shared parameters
"""class WinsorizerParams(Params):
    lower_q = Param(Params._dummy(), "lower_q", "lower quantile", typeConverter=TypeConverters.toFloat)
    upper_q = Param(Params._dummy(), "upper_q", "upper quantile", typeConverter=TypeConverters.toFloat)
    iqr_multiplier = Param(Params._dummy(), "iqr_multiplier", "iqr multiplier", typeConverter=TypeConverters.toFloat)
    columns_to_winsorize = Param(Params._dummy(), "columns_to_winsorize", "columns to process", typeConverter=TypeConverters.toListString)

    def __init__(self):
        super().__init__()
        self._setDefault(lower_q=0.25, upper_q=0.75, iqr_multiplier=1.5, columns_to_winsorize=[])

# 2. The Estimator: Learns the bounds
class Winsorizer(Estimator, WinsorizerParams, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, lower_q=0.25, upper_q=0.75, iqr_multiplier=1.5, columns_to_winsorize=[]):
        super().__init__()
        self.set(self.lower_q, lower_q)
        self.set(self.upper_q, upper_q)
        self.set(self.iqr_multiplier, iqr_multiplier)
        self.set(self.columns_to_winsorize, columns_to_winsorize)

    def _fit(self, df):
        # Get parameters
        cols = self.getOrDefault(self.columns_to_winsorize)
        lq = self.getOrDefault(self.lower_q)
        uq = self.getOrDefault(self.upper_q)
        mult = self.getOrDefault(self.iqr_multiplier)
        
        # Calculate bounds
        bounds = {}
        for c in cols:
            q1, q3 = df.approxQuantile(c, [lq, uq], 0.01)
            iqr = q3 - q1
            lower = q1 - mult * iqr
            upper = q3 + mult * iqr
            bounds[c] = (lower, upper)
            
        # Return the Model with bounds saved as a JSON string Param
        return WinsorizerModel(bounds=json.dumps(bounds)).setParent(self)

# 3. The Model: Applies the bounds (Transformer)
class WinsorizerModel(Model, WinsorizerParams, DefaultParamsReadable, DefaultParamsWritable):
    # We use a String Param to store the dictionary because PySpark doesn't support Dict Params easily
    bounds = Param(Params._dummy(), "bounds", "json string of bounds", typeConverter=TypeConverters.toString)

    def __init__(self, bounds=None):
        super().__init__()
        if bounds:
            self.set(self.bounds, bounds)

    def _transform(self, df):
        # Load bounds from the Param
        bounds_dict = json.loads(self.getOrDefault(self.bounds))
        
        out_df = df
        for c, (lower, upper) in bounds_dict.items():
            out_df = out_df.withColumn(
                c,
                when(col(c) < lower, lower)
                .when(col(c) > upper, upper)
                .otherwise(col(c))
            )
        return out_df"""


class Winsorizer2(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Winsorizes numerical columns based on IQR. Fully PySpark serializable.
    """
    bounds = Param(Params._dummy(), "bounds", "JSON dictionary of lower and upper bounds")

    def __init__(self, columns=None, lower_q=0.25, upper_q=0.75, iqr_multiplier=1.5):
        super().__init__()
        self.columns = columns or ["age", "campaign", "last_contact_day", "last_contact_duration", "balance_euros"]
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.iqr_multiplier = iqr_multiplier

        # Initialize bounds to empty JSON
        self._setDefault(bounds=json.dumps({}))

    def fit(self, df):
        """
        Compute IQR-based lower and upper bounds for each column.
        """
        b = {}
        for c in self.columns:
            # Approximate quantiles
            q1, q3 = df.approxQuantile(c, [self.lower_q, self.upper_q], 0.01)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            b[c] = (lower, upper)
        # Save bounds as JSON Param (serializable)
        self._set(bounds=json.dumps(b))
        return self

    def _transform(self, df):
        """
        Apply winsorization using the stored bounds.
        """
        b = json.loads(self.getOrDefault(self.bounds))
        out_df = df
        for c, (lower, upper) in b.items():
            if c not in df.columns:
                continue  # skip missing columns
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
    def __init__(self):
        super().__init__()
        self.campaign_median = None

    def fit(self, df: DataFrame):
        """Compute statistics that must be learned from training data."""
        self.campaign_median = df.approxQuantile("campaign", [0.5], 0.01)[0]
        return self

    def _transform(self, df: DataFrame) -> DataFrame:
        """Apply all feature engineering transformations."""
        
        #df = df.withColumn("balance_per_campaign", col("balance_euros") / (col("campaign") + 1))
        #df = df.withColumn("log_balance", F.log1p(when(col("balance_euros") < 0, 0).otherwise(col("balance_euros"))))
        df = df.withColumn("had_previous_contact", when(col("pdays") != -1, 1).otherwise(0))

        df = df.withColumn(
            "high_effort_client", 
            when(col("campaign") > self.campaign_median, 1).otherwise(0)
        )
        
        return df
