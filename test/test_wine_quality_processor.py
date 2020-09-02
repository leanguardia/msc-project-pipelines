from unittest import TestCase

import pytest
import pandas as pd
import numpy as np

from pipelines.etl_wine_quality import WineQualityProcessor

cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'quality']
values = [7.0, 0.27, 0.36, 20.7, 0.045, 45, 170.0, 1.0010, 3.00, 0.45, 8.8, 6]
np_values = np.array(values)
np_values_2d = np_values.reshape(1, len(np_values))
df = pd.DataFrame([values], columns=cols)

class TestWineQualityProcessor(TestCase):
    def setUp(self):
        self.processor = WineQualityProcessor()

    def test_transform_list(self):
        transformed = self.processor.transform(values)
        assert np.array_equal(transformed.values, np_values_2d)

    def test_transform_wrapped_list(self):
        transformed = self.processor.transform([values])
        assert np.array_equal(transformed.values, np_values_2d)

    def test_transform_narray_one_dim(self):
        assert np.array_equal(
            self.processor.transform(np_values).values, np_values_2d)

    def test_transform_wrapped_narray_one_dim(self):
        assert np.array_equal(
            self.processor.transform([np_values]).values, np_values_2d)

    def test_transform_narray_two_dims(self):
        assert np.array_equal(
            self.processor.transform(np_values_2d).values, np_values_2d)

    def test_transform_df(self):
        assert np.array_equal(self.processor.transform(df).values, np_values_2d)

    def test_transform_requires_minimum_number_of_features(self):
        with pytest.raises(ValueError, match='must have 11 or 12 columns'):
            self.processor.transform(values[:6])

    def test_transform_requires_maximum_number_of_features(self):
        with pytest.raises(ValueError, match='must have 11 or 12 columns.'):
            self.processor.transform(values * 2)

    def test_transform_accepts_only_features_as_input(self):
        X = np_values[:-1]
        assert np.array_equal(self.processor.transform([X]).values, [X])
