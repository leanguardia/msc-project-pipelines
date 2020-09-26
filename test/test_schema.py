import pytest
import unittest
import pandas as pd
import numpy as np

from pipelines.schema import Schema, build_df
from pipelines.validators import RangeValidator, CategoryValidator, PositiveValidator

features_metadata = [
    dict(
        name = 'feature1',
        dtype = int,
        type = 'input',
        range=(1, 10)
    ),
    dict(
        name = 'feature2',
        dtype = float,
        type = 'input',
        range = (0.0, 99.9)
    ),
    dict(
        name = 'feature3',
        dtype = str,
        type = 'input',
    ),
    dict(
        name = 'feature4',
        dtype = str,
        type = 'input',
        categories = ['a', 'e', 'i', 'o', 'u']
    ),
    dict(
        name = 'target',
        dtype = str,
        type = 'target'
    ),
    dict(
        name = 'feature1_log',
        dtype = str,
        positive = True
    )
]

only_inputs = ['feature1', 'feature2', 'feature3', 'feature4']
target_names = ['target']
input_names  = only_inputs + target_names
input_types = [int, float, str, str, str]
inputs = [1, 2.001, 'feature_3', 'u', 'target_val']

engineered_names = ['feature1_log']
feature_names = input_names + engineered_names

np_inputs = np.array(inputs)
np_inputs2d = np_inputs.reshape(1, len(np_inputs))

class TestSchema(unittest.TestCase):
    def setUp(self):
        self.schema = Schema(features_metadata)

    def test_schema_input_features(self):
        assert self.schema.features(which='input') == input_names
    
    def test_schema_all_features(self):
        assert self.schema.features(which='all') == feature_names

    def test_schema_features_no_target(self):
        assert self.schema.features(which='features') == only_inputs + engineered_names

    def test_schema_all_features_is_default(self):
        assert self.schema.features() == feature_names

    def test_schema_inputs(self):
        assert self.schema.inputs() == input_names

    def test_num_of_inputs(self):
        assert self.schema.n_inputs() == 5

    def test_schema_target(self):
        assert self.schema.target() == 'target'

    def test_schema_target_not_found(self):
        features_with_no_target = features_metadata[:4]
        failed_schema = Schema(features_with_no_target)
        with pytest.raises(ValueError, match='Target variable not found.'):
            assert failed_schema.target()

    def test_types(self):
        assert self.schema.dtypes(which='input') == input_types

    def test_build_validators(self):
        validator_types = [RangeValidator, RangeValidator, CategoryValidator, PositiveValidator]
        validators = self.schema.validators(which='all')
        for validator, vtype in zip(validators, validator_types):
            assert isinstance(validator, vtype)
    
    def test_build_validators_for_inputs(self):
        validator_types = [RangeValidator, RangeValidator, CategoryValidator]
        validators = self.schema.validators(which='input')
        for validator, vtype in zip(validators, validator_types):
            assert isinstance(validator, vtype)

    def test_build_validators_for_engineered(self):
        validators = self.schema.validators(which='engineered')
        assert isinstance(validators[0], PositiveValidator)

class TestBuildDataFrame(unittest.TestCase):
    def setUp(self):
        self.schema = Schema(features_metadata)
        columns = self.schema.inputs()
        self.df = pd.DataFrame([inputs], columns=columns)

    def test_build_features(self):
        built_df = build_df(inputs, self.schema)
        assert built_df.columns.to_list() == input_names

    def test_build_shape(self):
        built_df = build_df(inputs, self.schema)
        assert built_df.shape == (1, 5)

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

    def test_transform_requires_minimum_number_of_features(self):
        with pytest.raises(ValueError, match="incorrect number of columns"):
            build_df(inputs[:2], self.schema)

    def test_transform_requires_maximum_number_of_features(self):
        with pytest.raises(ValueError, match='incorrect number of columns'):
            build_df(inputs * 2, self.schema)

    def test_transform_features_only(self):
        df_inputs = self.df[input_names[:-1]].copy()
        built_df = build_df(inputs[:-1], self.schema)
        assert built_df.equals(df_inputs)
