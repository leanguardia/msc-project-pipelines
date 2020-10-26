import pytest
import unittest
import pandas as pd
import numpy as np

from pipelines.schema import Schema
from pipelines.validators import RangeValidator, CategoryValidator, NonNegativeValidator
from fixtures.sample_schema import sample_features_metadata

sample_schema = Schema(sample_features_metadata)

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
    def test_schema_input_features(self):
        assert sample_schema.features(which='input') == input_names
    
    def test_schema_all_features(self):
        assert sample_schema.features(which='all') == feature_names

    def test_schema_features_no_target(self):
        assert sample_schema.features(which='features') == only_inputs + engineered_names

    def test_schema_all_features_is_default(self):
        assert sample_schema.features() == feature_names

    def test_schema_inputs(self):
        assert sample_schema.inputs() == input_names

    def test_num_of_inputs(self):
        assert sample_schema.n_inputs() == 5

    def test_schema_target(self):
        assert sample_schema.target() == 'target'

    def test_schema_target_not_found(self):
        features_with_no_target = sample_features_metadata[:4]
        failed_schema = Schema(features_with_no_target)
        with pytest.raises(ValueError, match='Target variable not found.'):
            assert failed_schema.target()

    def test_types(self):
        assert sample_schema.dtypes(which='input') == input_types

    def test_build_validators(self):
        validator_types = [RangeValidator, RangeValidator, CategoryValidator, NonNegativeValidator]
        validators = sample_schema.validators(which='all')
        for validator, vtype in zip(validators, validator_types):
            assert isinstance(validator, vtype)
    
    def test_build_validators_for_inputs(self):
        validator_types = [RangeValidator, RangeValidator, CategoryValidator]
        validators = sample_schema.validators(which='input')
        for validator, vtype in zip(validators, validator_types):
            assert isinstance(validator, vtype)

    def test_build_validators_for_engineered(self):
        validators = sample_schema.validators(which='engineered')
        assert isinstance(validators[0], NonNegativeValidator)
    
    # TODO: Implement .get_categories('column')

