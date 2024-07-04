import pandas as pd
import numpy as np
import re


def is_random_string(column):
    # Adjust pattern based on what you consider random
    random_string_pattern = re.compile(r'[A-Za-z0-9]{5,}')
    return column.apply(lambda x: bool(random_string_pattern.match(str(x)))).any()


def is_sequential(x):
    if x.dtype.kind in 'iufc':  # Check if the column is of a numeric type
        sorted_x = np.sort(x)
        return np.all(np.diff(sorted_x) == 1)
    return False


def is_random_and_unique(x):
    return len(x.unique()) > 50 and is_random_string(x)


def prep_data(df, independent_var):
    # Remove sequential and random unique columns
    sequential_columns = df.apply(is_sequential, axis=0)
    random_and_unique_columns = df.apply(is_random_and_unique, axis=0)
    df = df.loc[:, ~sequential_columns & ~random_and_unique_columns]

    # Ensure independent variable is still in the dataframe before further processing
    if independent_var not in df.columns:
        raise ValueError(
            f"Independent variable '{independent_var}' not found in the dataframe.")

    # Separate the independent variable from the dataset
    independent_data = df[independent_var]
    df = df.drop(columns=[independent_var])

    # Convert boolean columns to integers (0 and 1)
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype(int)

    # Convert categorical variables into dummy variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Add the independent variable back to the dataframe
    df[independent_var] = independent_data

    return df
