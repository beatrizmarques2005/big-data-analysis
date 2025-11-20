
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

# def winsorize_spark(reference_df: DataFrame, apply_to_df: DataFrame, columns: List[str]) -> DataFrame:
#     """
#     Apply winsorization to specified columns in a Spark DataFrame using IQR from a reference DataFrame.

#     Parameters:
#         reference_df (DataFrame): Spark DataFrame used to calculate IQR.
#         apply_to_df (DataFrame): Spark DataFrame to apply winsorization.
#         columns (List[str]): List of column names to winsorize.

#     Returns:
#         DataFrame: Spark DataFrame with winsorized columns.
#     """
#     for col in columns:
#         # Calculate Q1 and Q3 from reference_df
#         q1, q3 = reference_df.approxQuantile(col, [0.25, 0.75], 0.01)
#         iqr = q3 - q1
#         lower_bound = q1 - 1.5 * iqr
#         upper_bound = q3 + 1.5 * iqr

#         # Apply winsorization using when/otherwise
#         apply_to_df = apply_to_df.withColumn(
#             col,
#             F.when(F.col(col) < lower_bound, lower_bound)
#              .when(F.col(col) > upper_bound, upper_bound)
#              .otherwise(F.col(col))
#         )

#     return apply_to_df

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
        
        # ---------- 1. Numeric transformations ----------
        df = df.withColumn("balance_per_campaign", col("balance_euros") / (col("campaign") + 1))
        df = df.withColumn("log_balance", F.log1p(col("balance_euros")))
        df = df.withColumn("had_previous_contact", when(col("pdays") != 999, 1).otherwise(0))
        df = df.withColumn("pdays_since_last", when(col("pdays") != 999, col("pdays")).otherwise(None))
        df = df.withColumn("avg_contact_duration", col("last_contact_duration") / (col("campaign") + 1))
        df = df.withColumn("previous_success_rate", col("previous") / (col("campaign") + 1))

        # ---------- 2. Temporal features ----------
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        df = df.withColumn(
            "month_num",
            create_map([lit(x) for pair in month_map.items() for x in pair])[col("last_contact_month")]
        )
        df = df.withColumn("quarter", ((col("month_num") - 1) / 3 + 1).cast(IntegerType()))
        df = df.withColumn("is_summer_campaign", when(col("month_num").isin([6, 7, 8]), 1).otherwise(0))
        df = df.withColumn("day_category",
                           when(col("last_contact_day") <= 10, "early")
                           .when(col("last_contact_day") <= 20, "mid")
                           .otherwise("late"))
        df = df.withColumn("duration_category",
                           when(col("last_contact_duration") <= 100, "short")
                           .when(col("last_contact_duration") <= 300, "medium")
                           .otherwise("long"))

        # ---------- 3. Credit & financial ----------
        df = df.withColumn("has_credit", when(col("credit") == "yes", 1).otherwise(0))
        df = df.withColumn("has_housing_loan", when(col("housing_loan") == "yes", 1).otherwise(0))
        df = df.withColumn("has_personal_loan", when(col("personal_loan") == "yes", 1).otherwise(0))
        df = df.withColumn("total_loans", col("has_credit") + col("has_housing_loan") + col("has_personal_loan"))
        df = df.withColumn("debt_risk_level",
                           when(col("total_loans") >= 2, "high")
                           .when(col("total_loans") == 1, "medium")
                           .otherwise("low"))
        df = df.withColumn("net_balance", col("balance_euros") - (1000 * col("total_loans")))

        # ---------- 4. Demographics ----------
        df = df.withColumn("is_married", when(col("marital_status") == "married", 1).otherwise(0))
        df = df.withColumn("is_single", when(col("marital_status") == "single", 1).otherwise(0))

        education_map = {"primary": 1, "secondary": 2, "tertiary": 3}
        df = df.withColumn(
            "education_level_num",
            create_map([lit(x) for pair in education_map.items() for x in pair])[col("education")]
        )

        df = df.withColumn("is_blue_collar", when(col("job") == "blue-collar", 1).otherwise(0))
        df = df.withColumn("is_professional",
                           when(col("job").isin(["management", "technician", "admin."]), 1)
                           .otherwise(0))
        df = df.withColumn("age_group",
                           when(col("age") < 30, "<30")
                           .when(col("age") <= 50, "30-50")
                           .otherwise(">50"))
        df = df.withColumn("senior_client", when(col("age") > 60, 1).otherwise(0))

        # ---------- 5. Contact-related ----------
        df = df.withColumn("is_cellphone", when(col("contact") == "cellular", 1).otherwise(0))

        df = df.withColumn(
            "high_effort_client",
            when(col("campaign") > self.campaign_median, 1).otherwise(0)
        )

        df = df.withColumn("contact_intensity",
                           col("campaign") + col("previous") + col("last_contact_duration"))

        # ---------- 6. Combined categorical ----------
        df = df.withColumn("job_education_combo", concat_ws("_", col("job"), col("education")))
        df = df.withColumn("marital_education_combo", concat_ws("_", col("marital_status"), col("education")))
        df = df.withColumn("loan_profile",
                           concat_ws("", col("has_credit"), col("has_housing_loan"), col("has_personal_loan")))

        return df

# def feature_engineering_spark(df: DataFrame) -> DataFrame:
#     """
#     Perform feature engineering on a Spark DataFrame.
#     """
#     # ---------- 1. Numeric transformations ----------
#     df = df.withColumn("balance_per_campaign", F.col("balance_euros") / (F.col("campaign") + 1))
#     df = df.withColumn("log_balance", F.log1p(F.col("balance_euros")))
#     df = df.withColumn("had_previous_contact", F.when(F.col("pdays") != 999, 1).otherwise(0))
#     df = df.withColumn("pdays_since_last", F.when(F.col("pdays") != 999, F.col("pdays")).otherwise(None))
#     df = df.withColumn("avg_contact_duration", F.col("last_contact_duration") / (F.col("campaign") + 1))
#     df = df.withColumn("previous_success_rate", F.col("previous") / (F.col("campaign") + 1))
    
#     # ---------- 2. Temporal features ----------
#     month_map = {
#         'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
#         'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
#     }
#     df = df.withColumn("month_num", F.create_map([F.lit(x) for pair in month_map.items() for x in pair])[F.col("last_contact_month")])
#     df = df.withColumn("quarter", ((F.col("month_num") - 1) / 3 + 1).cast(IntegerType()))
#     df = df.withColumn("is_summer_campaign", F.when(F.col("month_num").isin([6,7,8]), 1).otherwise(0))
#     df = df.withColumn("day_category", F.when(F.col("last_contact_day") <= 10, "early")
#                                         .when(F.col("last_contact_day") <= 20, "mid")
#                                         .otherwise("late"))
#     df = df.withColumn("duration_category", F.when(F.col("last_contact_duration") <= 100, "short")
#                                                .when(F.col("last_contact_duration") <= 300, "medium")
#                                                .otherwise("long"))
    
#     # ---------- 3. Credit and financial interaction ----------
#     df = df.withColumn("has_credit", F.when(F.col("credit") == "yes", 1).otherwise(0))
#     df = df.withColumn("has_housing_loan", F.when(F.col("housing_loan") == "yes", 1).otherwise(0))
#     df = df.withColumn("has_personal_loan", F.when(F.col("personal_loan") == "yes", 1).otherwise(0))
#     df = df.withColumn("total_loans", F.col("has_credit") + F.col("has_housing_loan") + F.col("has_personal_loan"))
#     df = df.withColumn("debt_risk_level", 
#                        F.when(F.col("total_loans") >= 2, "high")
#                         .when(F.col("total_loans") == 1, "medium")
#                         .otherwise("low"))
#     df = df.withColumn("net_balance", F.col("balance_euros") - (1000 * F.col("total_loans")))  # example fixed amount
    
#     # ---------- 4. Demographics ----------
#     df = df.withColumn("is_married", F.when(F.col("marital_status") == "married", 1).otherwise(0))
#     df = df.withColumn("is_single", F.when(F.col("marital_status") == "single", 1).otherwise(0))
#     education_map = {"primary": 1, "secondary": 2, "tertiary": 3}
#     df = df.withColumn("education_level_num", F.create_map([F.lit(x) for pair in education_map.items() for x in pair])[F.col("education")])
#     df = df.withColumn("is_blue_collar", F.when(F.col("job") == "blue-collar", 1).otherwise(0))
#     df = df.withColumn("is_professional", F.when(F.col("job").isin(["management", "technician", "admin."]), 1).otherwise(0))
#     df = df.withColumn("age_group", F.when(F.col("age") < 30, "<30")
#                                     .when(F.col("age") <= 50, "30-50")
#                                     .otherwise(">50"))
#     df = df.withColumn("senior_client", F.when(F.col("age") > 60, 1).otherwise(0))
    
#     # ---------- 5. Contact-related ----------
#     df = df.withColumn("is_cellphone", F.when(F.col("contact") == "cellular", 1).otherwise(0))
#     df = df.withColumn("high_effort_client", F.when(F.col("campaign") > df.agg(F.expr("percentile(campaign, 0.5)")).collect()[0][0], 1).otherwise(0))
#     df = df.withColumn("contact_intensity", F.col("campaign") + F.col("previous") + F.col("last_contact_duration"))
    
#     # ---------- 7. Encoded / grouped categorical features ----------
#     df = df.withColumn("job_education_combo", F.concat_ws("_", F.col("job"), F.col("education")))
#     df = df.withColumn("marital_education_combo", F.concat_ws("_", F.col("marital_status"), F.col("education")))
#     df = df.withColumn("loan_profile", F.concat_ws("", F.col("has_credit"), F.col("has_housing_loan"), F.col("has_personal_loan")))
    
#     return df

# -----------------------------
# ENCODING
# -----------------------------

# def index_column(train_df: DataFrame, val_df: DataFrame, col: str) -> tuple:
#     """
#     Fit a StringIndexer on train_df and transform both train and val DataFrames.

#     Returns:
#         (train_transformed, val_transformed)
#     """
#     indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
#     indexer_model = indexer.fit(train_df)
#     train_df = indexer_model.transform(train_df)
#     val_df = indexer_model.transform(val_df)
#     return train_df, val_df

# def onehot_column(train_df: DataFrame, val_df: DataFrame, col: str) -> tuple:
#     """
#     Fit a OneHotEncoder on train_df and transform both train and val DataFrames.

#     Returns:
#         (train_transformed, val_transformed)
#     """
#     encoder = OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec")
#     encoder_model = encoder.fit(train_df)
#     train_df = encoder_model.transform(train_df)
#     val_df = encoder_model.transform(val_df)
#     return train_df, val_df

# def assemble_features(train_df: DataFrame, val_df: DataFrame, input_cols: list, output_col: str) -> tuple:
#     """
#     Assemble a Spark feature vector for given columns.

#     Parameters:
#         train_df (DataFrame): Training DataFrame
#         val_df (DataFrame): Validation DataFrame
#         input_cols (list): List of columns to assemble
#         output_col (str): Name of the output vector column

#     Returns:
#         tuple: (train_transformed, val_transformed)
#     """
#     assembler = VectorAssembler(inputCols=input_cols, outputCol=output_col)
#     train_df = assembler.transform(train_df)
#     val_df = assembler.transform(val_df)
#     return train_df, val_df

# # -----------------------------
# # SCALLING
# # -----------------------------

# def normalize_features(train_df: DataFrame, val_df: DataFrame, input_col: str, output_col: str, p: float = 2.0) -> tuple:
#     """
#     Normalize a feature vector column in Spark using Lp norm.

#     Parameters:
#         train_df (DataFrame): Training DataFrame
#         val_df (DataFrame): Validation DataFrame
#         input_col (str): Name of the input vector column
#         output_col (str): Name of the output normalized vector column
#         p (float, optional): Lp norm (default 2.0)

#     Returns:
#         tuple: (train_transformed, val_transformed)
#     """
#     normalizer = Normalizer(inputCol=input_col, outputCol=output_col, p=p)
#     train_df = normalizer.transform(train_df)
#     val_df = normalizer.transform(val_df)
#     return train_df, val_df



