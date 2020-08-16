import pandas as pd
import numpy as np

import joblib
from pipelines.etl_forest_fires import ForestFirePredictor

cols = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain']
vals =    [1, 2, 'dec', 'tue', 20.0,  100.0, 400, 35, 33.2, 60, 5, 4]

model = joblib.load("models/regrezz.pkl")

def number_of_decimals(num):
    num_str = str(num)
    return len(num_str) - num_str.index('.') - 1

def test_return_a_float():
    engine = ForestFirePredictor(model)
    inference = engine.predict([vals])[0]
    assert type(inference) == float, 'Inferences should be rational numbers'

def test_return_two_decimal_points():
    engine = ForestFirePredictor(model)
    inference = engine.predict([vals])[0]
    assert number_of_decimals(inference) == 2, 'default number of decimal \
        points should be 2'

def test_return_specified_decimal_points():
    engine = ForestFirePredictor(model)
    inference = engine.predict([vals], decimals=4)[0]
    assert number_of_decimals(inference) == 4, 'number of decimal points \
        should be 4'
