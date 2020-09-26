import pytest
import unittest
import pandas as pd
import numpy as np

from pipelines.schema import Schema, build_df

features_list = [
    dict(
        name='feature1',
        dtype = int,
        type = 'input'
    ),
    dict(
        name='feature2',
        dtype = float,
        type = 'input'
    ),
    dict(
        name='feature3',
        dtype = str,
        type = 'input'
    ),
    dict(
        name='target',
        dtype = str,
        type = 'target'
    )
]

names  = ['feature1', 'feature2', 'feature3', 'target']
inputs = [1, 2.001, 'feature_3', 'target_val']
np_inputs = np.array(inputs)
np_inputs2d = np_inputs.reshape(1, len(np_inputs))
input_types = [int, float, str, str]

class TestSchema(unittest.TestCase):
    def setUp(self):
        self.schema = Schema(features_list)

    def test_schema_columns(self):
        assert self.schema.columns() == names

    def test_num_of_columns(self):
        assert self.schema.n_columns() == 4

    def test_schema_inputs(self):
        assert self.schema.inputs() == names[:-1]

    def test_num_of_inputs(self):
        assert self.schema.n_inputs() == 3

    def test_schema_target(self):
        assert self.schema.target() == 'target'

    def test_schema_target_not_found(self):
        failed_schema = Schema(features_list[:-1])
        with pytest.raises(ValueError, match='Target variable not found.'):
             assert failed_schema.target()

    def test_types(self):
        assert self.schema.dtypes() == input_types


class TestBuildDataFrame(unittest.TestCase):
    def setUp(self):
        self.schema = Schema(features_list)
        self.df = pd.DataFrame([inputs], columns=self.schema.columns())

    def test_build_columns(self):
        built_df = build_df(inputs, self.schema)
        assert built_df.columns.to_list() == names

    def test_build_shape(self):
        built_df = build_df(inputs, self.schema)
        assert built_df.shape == (1, 4)

    def test_build_from_list(self):
        built_df  = build_df(inputs, self.schema)
        assert built_df.equals(self.df)

    def test_build_from_wrapped_list(self):
        built_df = build_df([inputs], self.schema)
        assert built_df.equals(self.df)

    def test_build_from_narray_one_dim(self):
        built_df = build_df(np_inputs, self.schema)
        assert built_df.equals(self.df)
    
    def test_build_from_wrapped_narray_one_dim(self):
        built_df = build_df([np_inputs], self.schema)
        assert built_df.equals(self.df)

    def test_build_from_narray_two_dims(self):
        built_df = build_df(np_inputs2d, self.schema)
        assert built_df.equals(self.df)

    # def test_transform_requires_minimum_number_of_features(self):
    #     with pytest.raises(ValueError, match="incorrect number of columns"):
    #         build_df(inputs[:6], self.schema)

    # def test_transform_requires_maximum_number_of_features(self):
    #     with pytest.raises(ValueError, match='incorrect number of columns'):
    #         build_df(inputs * 2, self.schema)

    # def test_transform_features_only(self):
    #     transformed = build_df([X], self.schema)
    #     assert transformed.equals(df_features[input_names + new_feature_names[1:]])

