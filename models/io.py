from joblib import dump, load
from sklearn.base import BaseEstimator

def store_model(model, filepath):
    if not isinstance(model, BaseEstimator):
        raise TypeError('model should be a sklearn base estimator')
    if not filepath[-7:] == '.joblib':
        raise ValueError('filepath should end with specific extension')
    dump(model, filepath)
