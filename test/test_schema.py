import pytest
import unittest

from pipelines.schema import Schema

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

    def test_schema_inputs(self):
        assert self.schema.inputs() == names[:-1]

    def test_schema_target(self):
        assert self.schema.target() == 'target'

    def test_schema_target_not_found(self):
        failed_schema = Schema(features_list[:-1])
        with pytest.raises(ValueError, match='Target variable not found.'):
             assert failed_schema.target()

