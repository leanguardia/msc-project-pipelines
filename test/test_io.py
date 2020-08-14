import pytest
from unittest.mock import patch

import os

from sklearn.linear_model import LinearRegression
from joblib import dump

from models.io import store_model

model = LinearRegression()

def test_model_datatype_error():
    with pytest.raises(TypeError, match="model should be a sklearn base estimator"):
        store_model(None, 'filepath')

def test_filepath_format():
    with pytest.raises(ValueError, match="filepath should end with specific extension"):
        store_model(model, 'filepath')

@patch('models.io.dump')
def test_new_number_of_columns(dump_mock):
    store_model(model, 'filepath.joblib')
    dump_mock.assert_called_once_with(model, 'filepath.joblib')
