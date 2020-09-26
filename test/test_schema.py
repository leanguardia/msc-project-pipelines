import pytest
import unittest

from pipelines.schema import Schema#, build_df

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


# class TestBuildDf(unittest.TestCase):
#     def setUp(self):
#         self.schema = Schema(features_list)

#     def test_build_df(self):
#         df  = build_df([inputs], self.schema)
#         assert df.columns == names

    # def test_transform_list(self):
    #     transformed = self.processor.transform(inputs)
    #     assert transformed.equals(df_features)

    # def test_transform_wrapped_list(self):
    #     transformed = self.processor.transform([inputs])
    #     assert transformed.equals(df_features)

    # def test_transform_narray_one_dim(self):
    #     transformed = self.processor.transform(np_inputs)
    #     assert transformed.equals(df_features)
    
    # def test_transform_wrapped_narray_one_dim(self):
    #     transformed = self.processor.transform([np_inputs])
    #     assert transformed.equals(df_features)

    # def test_transform_narray_two_dims(self):
    #     transformed = self.processor.transform(np_inputs_2d)
    #     assert transformed.equals(df_features)

    # def test_transform_df(self):
    #     transformed = self.processor.transform(df_inputs)
    #     assert transformed.equals(df_features)

    # def test_transform_requires_minimum_number_of_features(self):
    #     with pytest.raises(ValueError, match="incorrect number of columns"):
    #         self.processor.transform(inputs[:6])

    # def test_transform_requires_maximum_number_of_features(self):
    #     with pytest.raises(ValueError, match='incorrect number of columns'):
    #         self.processor.transform(inputs * 2)

    # def test_transform_features_only(self):
    #     transformed = self.processor.transform([X])
    #     assert transformed.equals(df_features[input_names + new_feature_names[1:]])

