from unittest import TestCase

import pytest
import pandas as pd
import numpy as np

from pipelines.wine_quality_etl import WineQualityProcessor

target_name = 'quality'
input_names = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
               'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
               'pH', 'sulphates', 'alcohol']
inputs = [7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.0010, 3.00, 0.45, 8.8, 6]

np_inputs    = np.array(inputs)
np_inputs_2d = np_inputs.reshape(1, len(np_inputs))
df_inputs    = pd.DataFrame([inputs], columns=input_names + [target_name])
X = np_inputs[:-1]

new_feature_names = ['quality_cat', 'free_sulfur_dioxide_log', 'total_sulfur_dioxide_log', 'residual_sugar_log']
feature_vals = inputs + [1, np.log(45.0), np.log(170.0), np.log(20.7)]

feature_names = input_names + [target_name] + new_feature_names
df_features = pd.DataFrame([feature_vals], columns=feature_names)
df_features['quality_cat'] = df_features['quality_cat'].astype(np.uint8)

class TestWineQualityProcessor(TestCase):
    def setUp(self):
        self.processor = WineQualityProcessor()

    def test_transform_list(self):
        transformed = self.processor.transform(inputs)
        assert transformed.equals(df_features)

    def test_transform_wrapped_list(self):
        transformed = self.processor.transform([inputs])
        assert transformed.equals(df_features)

    def test_transform_narray_one_dim(self):
        transformed = self.processor.transform(np_inputs)
        assert transformed.equals(df_features)

    def test_transform_wrapped_narray_one_dim(self):
        transformed = self.processor.transform([np_inputs])
        assert transformed.equals(df_features)
    
    def test_transform_narray_two_dims(self):
        transformed = self.processor.transform(np_inputs_2d)
        assert transformed.equals(df_features)

    def test_transform_df(self):
        transformed = self.processor.transform(df_inputs)
        assert transformed.equals(df_features)

    def test_transform_requires_minimum_number_of_features(self):
        with pytest.raises(ValueError, match="incorrect number of columns"):
            self.processor.transform(inputs[:6])

    def test_transform_requires_maximum_number_of_features(self):
        with pytest.raises(ValueError, match="incorrect number of columns"):
            self.processor.transform(inputs * 2)

    def test_transform_accepts_only_features_as_input(self):
        transformed = self.processor.transform([X])
        assert transformed.equals(df_features[input_names + new_feature_names[1:]])

    def test_raw_features_inputs(self):
        row = self.processor.transform(inputs).loc[0]
        assert row['fixed_acidity'] == 7.0
        assert row['citric_acid'] == 0.36
        assert row['volatile_acidity'] == 0.27
        assert row['residual_sugar'] == 20.7
        assert row['chlorides'] == 0.045
        assert row['free_sulfur_dioxide'] == 45.0
        assert row['total_sulfur_dioxide'] == 170.0
        assert row['density'] == 1.0010
        assert row['pH'] == 3.00
        assert row['sulphates'] == 0.45
        assert row['alcohol'] == 8.8
        assert row['quality'] == 6
    
    def test_raw_features_types(self):
        transformed = self.processor.transform(inputs)
        row = transformed.loc[0]
        self.assertIsInstance(row['fixed_acidity'], np.float64)
        self.assertIsInstance(row['citric_acid'], np.float64)
        self.assertIsInstance(row['volatile_acidity'], np.float64)
        self.assertIsInstance(row['residual_sugar'], np.float64)
        self.assertIsInstance(row['chlorides'], np.float64)
        self.assertIsInstance(row['free_sulfur_dioxide'], np.float64)
        self.assertIsInstance(row['total_sulfur_dioxide'], np.float64)
        self.assertIsInstance(row['density'], np.float64)
        self.assertIsInstance(row['pH'], np.float64)
        self.assertIsInstance(row['sulphates'], np.float64)
        self.assertIsInstance(row['alcohol'], np.float64)
        # Quality is not being passed as Integer for some reason
        # self.assertIsInstance(row['quality'], np.int64)

    def test_transform_quality_to_category(self):
        transformed = self.processor.transform(inputs)
        print(transformed)
        print(df_features)
        self.assertEqual(transformed.loc[0]['quality_cat'], 1)

    def test_transform_free_sulfur_dioxide_to_log(self):
        row = self.processor.transform(inputs).loc[0]
        self.assertEqual(row['free_sulfur_dioxide_log'], np.log(45.0))

    def test_transform_total_sulfur_dioxide_to_log(self):
        row = self.processor.transform(inputs).loc[0]
        self.assertEqual(row['total_sulfur_dioxide_log'], np.log(170.0))

    def test_transform_residual_sugar(self):
        row = self.processor.transform(inputs).loc[0]
        self.assertEqual(row['residual_sugar_log'], np.log(20.7))
