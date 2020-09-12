from unittest import TestCase

import pytest
import pandas as pd
import numpy as np

from pipelines.wine_quality_etl import WineQualityProcessor

cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
        'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'quality']
inputs = [7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.0010, 3.00, 0.45, 8.8, 6]

np_inputs = np.array(inputs)
np_inputs_2d = np_inputs.reshape(1, len(np_inputs))

feature_vals = inputs
df_features = pd.DataFrame([feature_vals], columns=cols)


class TestWineQualityProcessor(TestCase):
    def setUp(self):
        self.processor = WineQualityProcessor()

    def test_transform_list(self):
        transformed = self.processor.transform(inputs)
        assert transformed.equals(df_features)

    # def test_transform_wrapped_list(self):
    #     transformed = self.processor.transform([inputs])
    #     assert np.array_equal(transformed.inputs, np_inputs_2d)

    # def test_transform_narray_one_dim(self):
    #     assert np.array_equal(
    #         self.processor.transform(np_inputs).inputs, np_inputs_2d)

    # def test_transform_wrapped_narray_one_dim(self):
    #     assert np.array_equal(
    #         self.processor.transform([np_inputs]).inputs, np_inputs_2d)

    # def test_transform_narray_two_dims(self):
    #     assert np.array_equal(
    #         self.processor.transform(np_inputs_2d).inputs, np_inputs_2d)

    # def test_transform_df(self):
    #     assert np.array_equal(self.processor.transform(df).inputs, np_inputs_2d)

    # def test_transform_requires_minimum_number_of_features(self):
    #     with pytest.raises(ValueError, match='must have 11 or 12 columns'):
    #         self.processor.transform(inputs[:6])

    # def test_transform_requires_maximum_number_of_features(self):
    #     with pytest.raises(ValueError, match='must have 11 or 12 columns.'):
    #         self.processor.transform(inputs * 2)

    # def test_transform_accepts_only_features_as_input(self):
    #     X = np_inputs[:-1]
    #     assert np.array_equal(self.processor.transform([X]).inputs, [X])
