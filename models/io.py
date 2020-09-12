import pandas as pd
from joblib import dump, load
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator

def load_table(database, table_name):
    engine = create_engine(f'sqlite:///{database}')
    return pd.read_sql_table(table_name, engine)

def store_model(model, filepath):
    """
    Stores the model in .pkl format.

    Parameters:
        model: Sklearn estimator object
        filepath: relative path for storage. E.g. "models/regressor.pkl"
    """

    if not isinstance(model, BaseEstimator):
        raise TypeError('model should be a sklearn base estimator')
    if not filepath[-4:] == '.pkl':
        raise ValueError('filepath should end with specific extension')
    dump(model, filepath)

def is_valid_model_filepath(filepath):
    if not type(filepath) == str: raise TypeError('filepath should be a string')
    if len(filepath) <= 4:
        raise ValueError('should end with specific extension')
    return filepath.split('.')[-1] == 'pkl'

