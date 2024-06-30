import pandas as pd
import numpy as np
import re


def is_random_string(column):
    # Adjust pattern based on what you consider random
    random_string_pattern = re.compile(r'[A-Za-z0-9]{5,}')
    return column.apply(lambda x: bool(random_string_pattern.match(str(x)))).any()


def remove_id_columns(df):
    def is_sequential(x): return np.all(np.diff(np.sort(x)) == 1)

    def is_random_and_unique(x): return len(
        x.unique()) > 50 and is_random_string(x)
    return df.loc[:, ~df.apply(is_sequential) & ~df.apply(is_random_and_unique)]
