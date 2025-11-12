# code
import re

def name_cleaner(name: str, char_list: list) -> str:
    """
    Cleans a given string to make it safe for use in identifiers or filenames.

    This function replaces each character in `char_list` with an underscore,
    except for parentheses '(' and ')' which are removed. After that, any
    remaining non-alphanumeric characters are removed.

    Parameters
    ----------
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


from pyspark.sql import DataFrame

def show_column_types(df: DataFrame):
    """
    Prints the name and data type of each column in a Spark DataFrame.

    Parameters
    df(DataFrame): The Spark DataFrame to inspect.
    """
    print("Column Name\t|\tData Type")
    print("-" * 30)
    for col_name, dtype in df.dtypes:
        print(f"{col_name}\t|\t{dtype}")

def winsorize_spark(reference_df: DataFrame, apply_to_df: DataFrame, columns: list[str]) -> DataFrame:
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