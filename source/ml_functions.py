
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType, DoubleType

def feature_engineering_spark(df: DataFrame) -> DataFrame:
    """
    Perform feature engineering on a Spark DataFrame.
    """
    # ---------- 1. Numeric transformations ----------
    df = df.withColumn("balance_per_campaign", F.col("balance_euros") / (F.col("campaign") + 1))
    df = df.withColumn("log_balance", F.log1p(F.col("balance_euros")))
    df = df.withColumn("had_previous_contact", F.when(F.col("pdays") != 999, 1).otherwise(0))
    df = df.withColumn("pdays_since_last", F.when(F.col("pdays") != 999, F.col("pdays")).otherwise(None))
    df = df.withColumn("avg_contact_duration", F.col("last_contact_duration") / (F.col("campaign") + 1))
    df = df.withColumn("previous_success_rate", F.col("previous") / (F.col("campaign") + 1))
    
    # ---------- 2. Temporal features ----------
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df = df.withColumn("month_num", F.create_map([F.lit(x) for pair in month_map.items() for x in pair])[F.col("last_contact_month")])
    df = df.withColumn("quarter", ((F.col("month_num") - 1) / 3 + 1).cast(IntegerType()))
    df = df.withColumn("is_summer_campaign", F.when(F.col("month_num").isin([6,7,8]), 1).otherwise(0))
    df = df.withColumn("day_category", F.when(F.col("last_contact_day") <= 10, "early")
                                        .when(F.col("last_contact_day") <= 20, "mid")
                                        .otherwise("late"))
    df = df.withColumn("duration_category", F.when(F.col("last_contact_duration") <= 100, "short")
                                               .when(F.col("last_contact_duration") <= 300, "medium")
                                               .otherwise("long"))
    
    # ---------- 3. Credit and financial interaction ----------
    df = df.withColumn("has_credit", F.when(F.col("credit") == "yes", 1).otherwise(0))
    df = df.withColumn("has_housing_loan", F.when(F.col("housing_loan") == "yes", 1).otherwise(0))
    df = df.withColumn("has_personal_loan", F.when(F.col("personal_loan") == "yes", 1).otherwise(0))
    df = df.withColumn("total_loans", F.col("has_credit") + F.col("has_housing_loan") + F.col("has_personal_loan"))
    df = df.withColumn("debt_risk_level", 
                       F.when(F.col("total_loans") >= 2, "high")
                        .when(F.col("total_loans") == 1, "medium")
                        .otherwise("low"))
    df = df.withColumn("net_balance", F.col("balance_euros") - (1000 * F.col("total_loans")))  # example fixed amount
    
    # ---------- 4. Demographics ----------
    df = df.withColumn("is_married", F.when(F.col("marital_status") == "married", 1).otherwise(0))
    df = df.withColumn("is_single", F.when(F.col("marital_status") == "single", 1).otherwise(0))
    education_map = {"primary": 1, "secondary": 2, "tertiary": 3}
    df = df.withColumn("education_level_num", F.create_map([F.lit(x) for pair in education_map.items() for x in pair])[F.col("education")])
    df = df.withColumn("is_blue_collar", F.when(F.col("job") == "blue-collar", 1).otherwise(0))
    df = df.withColumn("is_professional", F.when(F.col("job").isin(["management", "technician", "admin."]), 1).otherwise(0))
    df = df.withColumn("age_group", F.when(F.col("age") < 30, "<30")
                                    .when(F.col("age") <= 50, "30-50")
                                    .otherwise(">50"))
    df = df.withColumn("senior_client", F.when(F.col("age") > 60, 1).otherwise(0))
    
    # ---------- 5. Contact-related ----------
    df = df.withColumn("is_cellphone", F.when(F.col("contact") == "cellular", 1).otherwise(0))
    df = df.withColumn("high_effort_client", F.when(F.col("campaign") > df.agg(F.expr("percentile(campaign, 0.5)")).collect()[0][0], 1).otherwise(0))
    df = df.withColumn("contact_intensity", F.col("campaign") + F.col("previous") + F.col("last_contact_duration"))
    
    # ---------- 7. Encoded / grouped categorical features ----------
    df = df.withColumn("job_education_combo", F.concat_ws("_", F.col("job"), F.col("education")))
    df = df.withColumn("marital_education_combo", F.concat_ws("_", F.col("marital_status"), F.col("education")))
    df = df.withColumn("loan_profile", F.concat_ws("", F.col("has_credit"), F.col("has_housing_loan"), F.col("has_personal_loan")))
    
    return df

