import pytest
import unittest
import pandas as pd

from pipelines.schema import Schema, build_df

features_list = [
    dict(
        name='feature0',
        datatype = bool,
        type = 'input'
    ),
    dict(
        name='feature1',
        datatype = int,
        type = 'input'
    ),
    dict(
        name='feature2',
        datatype = float,
        type = 'input'
    ),
    dict(
        name='feature3',
        datatype = str,
        type = 'input'
    ),
    dict(
        name='target',
        datatype = str,
        type = 'target'
    )
]

names  = ['feature0', 'feature1', 'feature2', 'feature3', 'target']
inputs = [False, 1, 2.001, 'feature_3', 'target_vel']

class TestSchema(unittest.TestCase):
    def setUp(self):
        self.schema = Schema(features_list)

    def test_schema_columns(self):
        assert self.schema.columns() == names

    def test_num_of_columns(self):
        assert self.schema.n_columns() == 5

    def test_schema_inputs(self):
        assert self.schema.inputs() == names[:-1]

    def test_num_of_inputs(self):
        assert self.schema.n_inputs() == 4

    def test_schema_target(self):
        assert self.schema.target() == 'target'

    def test_schema_target_not_found(self):
        failed_schema = Schema(features_list[:-1])
        with pytest.raises(ValueError, match='Target variable not found.'):
             assert failed_schema.target()


class TestBuildDataFrame(unittest.TestCase):
    def setUp(self):
        self.schema = Schema(features_list)
        self.df = pd.DataFrame([inputs], columns=self.schema.columns())

    def test_build_columns(self):
        built_df = build_df(inputs, self.schema)
        assert built_df.columns.to_list() == names

    def test_build_shape(self):
        built_df = build_df(inputs, self.schema)
        assert built_df.shape == (1, 5)

    # def test_build_from_list(self):
    #     built_df  = build_df([inputs], self.schema)
    #     assert built_df.equals(self.df)

    # def test_build_from_wrapped_list(self):
    #     built_df = build_df([inputs], self.schema)
    #     assert built_df.equals(self.df)

    # def test_transform_narray_one_dim(self):
    #     df = build_df(np_inputs, self.schema)
    #     assert df.equals(df_features)
    
    # def test_transform_wrapped_narray_one_dim(self):
    #     df = build_df([np_inputs], self.schema)
    #     assert df.equals(df_features)

    # def test_transform_narray_two_dims(self):
    #     df = build_df(np_inputs_2d, self.schema)
    #     assert df.equals(df_features)

    # def test_transform_df(self):
    #     df = build_df(df_inputs, self.schema)
    #     assert df.equals(df_features)

    # def test_transform_requires_minimum_number_of_features(self):
    #     with pytest.raises(ValueError, match="incorrect number of columns"):
    #         build_df(inputs[:6], self.schema)

    # def test_transform_requires_maximum_number_of_features(self):
    #     with pytest.raises(ValueError, match='incorrect number of columns'):
    #         build_df(inputs * 2, self.schema)

    # def test_transform_features_only(self):
    #     transformed = build_df([X], self.schema)
    #     assert transformed.equals(df_features[input_names + new_feature_names[1:]])

