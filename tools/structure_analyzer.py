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