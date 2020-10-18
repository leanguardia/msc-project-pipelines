import numpy as np
import pandas as pd
import pytest
from unittest import TestCase

from pipelines.preparer import Preparer
from fixtures.sample_schema import sample_schema

only_inputs = ['feature1', 'feature2', 'feature3', 'feature4']
target_names = ['target']
input_names  = only_inputs + target_names
input_types = [int, float, str, str, str]
inputs = [1, 2.001, 'feature_3', 'u', 'target_val']

engineered_names = ['feature1_log']
feature_names = input_names + engineered_names

np_inputs = np.array(inputs)
np_inputs2d = np_inputs.reshape(1, len(np_inputs))

class TestPreparer(TestCase):
    def setUp(self):
        self.preparer = Preparer(sample_schema)
        self.df = pd.DataFrame([inputs], columns=sample_schema.inputs())
    
    def test_build_features(self):
        built_df = self.preparer.prepare(inputs)
        assert built_df.columns.to_list() == input_names

    def test_build_shape(self):
        built_df = self.preparer.prepare(inputs)
        assert built_df.shape == (1, 5)

    def test_build_from_list(self):
        built_df  = self.preparer.prepare(inputs)
        assert built_df.equals(self.df)

    def test_build_from_wrapped_list(self):
        built_df = self.preparer.prepare([inputs])
        assert built_df.equals(self.df)

    def test_build_from_narray_one_dim(self):
        built_df = self.preparer.prepare(np_inputs)
        assert built_df.equals(self.df)

    def test_build_from_wrapped_narray_one_dim(self):
        built_df = self.preparer.prepare([np_inputs])
        assert built_df.equals(self.df)

    def test_build_from_narray_two_dims(self):
        built_df = self.preparer.prepare(np_inputs2d)
        assert built_df.equals(self.df)

    def test_transform_requires_minimum_number_of_features(self):
        with pytest.raises(ValueError, match="incorrect number of columns"):
            self.preparer.prepare(inputs[:2])

    def test_transform_requires_maximum_number_of_features(self):
        with pytest.raises(ValueError, match='incorrect number of columns'):
            self.preparer.prepare(inputs * 2)

    def test_transform_features_only(self):
        df_inputs = self.df[input_names[:-1]].copy()
        built_df = self.preparer.prepare(inputs[:-1])
        assert built_df.equals(df_inputs)
