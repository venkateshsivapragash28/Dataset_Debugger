import pandas as pd


def analyze_structure(df):

    structure = {
        "numerical_columns": [],
        "categorical_columns": [],
        "datetime_columns": []
    }

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):
            structure["numerical_columns"].append(column)

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            structure["datetime_columns"].append(column)

        else:
            structure["categorical_columns"].append(column)

    return structure

import pandas as pd


def df_to_column_samples(df, sample_size=10):

    column_samples = {}

    for column in df.columns:

        samples = (
            df[column]
            .dropna()
            .astype(str)
            .sample(min(sample_size, len(df)))
            .tolist()
        )

        column_samples[column] = samples

    return column_samples