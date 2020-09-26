import pytest
import unittest

from pipelines.schema import Schema

schema_dict = dict(
    feature0 = dict(
        datatype = bool,
        type = 'input'
    ),
    feature1 = dict(
        datatype = int,
        type = 'input'
    ),
    feature2 = dict(
        datatype = float,
        type = 'input'
    ),
    feature3 = dict(
        datatype = str,
        type = 'input'
    ),
    target = dict(
        datatype = str,
        type = 'target'
    ),
)

names  = ['feature0', 'feature1', 'feature2', 'feature3', 'target']
inputs = [False, 1, 2.001, 'feature_3', 'target_vel'] 

class TestSchema(unittest.TestCase):
    def setUp(self):
        self.schema = Schema(schema_dict)

    def test_schema_columns(self):
        assert self.schema.columns() == names

    def test_schema_inputs(self):
        assert self.schema.inputs() == names[:-1]

