from unittest import TestCase

import pandas as pd
import numpy as np

import joblib
from pipelines.etl_forest_fires import ForestFirePredictor
from unittest.mock import Mock

cols = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain']
vals =    [1, 2, 'dec', 'tue', 20.0,  100.0, 400, 35, 33.2, 60, 5, 4]
# TODO: Transform Raw Args to Features for prediction.

model = Mock()
model.predict.return_value = [23.4324]

def number_of_decimals(num):
    num_str = str(num)
    return len(num_str) - num_str.index('.') - 1

class TestRegressionPredictor(TestCase):
    def test_return_float(self):
        engine = ForestFirePredictor(model)
        inference = engine.predict([vals])[0]
        assert type(inference) == float, 'Inferences should be real numbers'

    def test_returns_prediction(self):
        engine = ForestFirePredictor(model)
        inference = engine.predict([vals])[0]
        assert inference == 23.43, 'Prediction value is incorrect'

    def test_return_two_decimal_points(self):
        engine = ForestFirePredictor(model)
        inference = engine.predict([vals])[0]
        assert number_of_decimals(inference) == 2, 'default number of decimal \
            points should be 2'

    def test_return_specified_decimal_points(self):
        engine = ForestFirePredictor(model)
        inference = engine.predict([vals], decimals=4)[0]
        assert number_of_decimals(inference) == 4, 'number of decimal points \
            should be 4'

    def test_returns_zero_as_lower_bound(self):
        model.predict.return_value = [-23.4324]
        engine = ForestFirePredictor(model)
        inference = engine.predict([vals])[0]
        assert inference == 0, 'Inference lower bound should be zero'

