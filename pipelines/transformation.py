import pandas as pd
import numpy as np
from scipy import stats

def dummify(df, column, categories, **kwargs):
    if not type(df) == pd.DataFrame:
        raise TypeError("df must be a DataFrame")

    zeroes = np.zeros((df.shape[0], len(categories)))
    dummies = pd.DataFrame(zeroes, columns=categories, index=df.index)
    
    for row_idx, row in df.iterrows():
        indexes = dummies.columns.get_indexer(row)
        dummies.iloc[row_idx, indexes] = 1
    
    return pd.concat([df, dummies], axis=1)

def remove_outliers_iqr(df, column):
    q1, q3 = np.percentile(df[column], [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(lower_bound <= df[column]) & (df[column] <= upper_bound)]
    return df.dropna()

def remove_outliers_zscore(df, column, zscore=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df.loc[z_scores <= zscore]

def replace_outliers_winsor(df, column, limits):
    df[column] = stats.mstats.winsorize(df[column], limits).data
    return df
