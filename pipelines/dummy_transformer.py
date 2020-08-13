import pandas as pd
import numpy as np

def dummify(df, column, **kwargs):
    if not type(df) == pd.DataFrame:
        raise TypeError("df must be a DataFrame")
    dummies = pd.get_dummies(df[column], **kwargs)
    return pd.concat([df, dummies], axis=1)
