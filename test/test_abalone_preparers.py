from unittest import TestCase

import pytest
import pandas as pd
import numpy as np

from pipelines.abalone_preparers import AbalonePreparerETL, AbalonePreparer

inputs      = ['M', 0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 0.15, 15]
np_inputs    = np.array(inputs)

class TestAbalonePreparerETL(TestCase):
    def setUp(self):
        self.preparer = AbalonePreparerETL()

    def test_raw_features_inputs(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertEqual(row['sex'], 'M')
        self.assertEqual(row['length'], 0.455)
        self.assertEqual(row['diameter'], 0.365)
        self.assertEqual(row['height'], 0.095)
        self.assertEqual(row['whole_weight'], 0.514)
        self.assertEqual(row['shucked_weight'], 0.2245)
        self.assertEqual(row['viscera_weight'], 0.101)
        self.assertEqual(row['shell_weight'], 0.15)
        self.assertEqual(row['rings'], 15)

    def test_raw_features_types(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertIsInstance(row['sex'], str)
        self.assertIsInstance(row['length'], np.float64)
        self.assertIsInstance(row['diameter'], np.float64)
        self.assertIsInstance(row['height'], np.float64)
        self.assertIsInstance(row['whole_weight'], np.float64)
        self.assertIsInstance(row['shucked_weight'], np.float64)
        self.assertIsInstance(row['viscera_weight'], np.float64)
        self.assertIsInstance(row['shell_weight'], np.float64)
        self.assertIsInstance(row['rings'], np.int64)

    def test_prepare_age(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertEqual(row['age'], 15.0 + 1.5)

    def test_prepare_dummy_sex(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertEqual(row['M'], 1)
        self.assertEqual(row['F'], 0)
        self.assertEqual(row['I'], 0)

class TestAbalonePreparerServing(TestCase):
    def setUp(self):
        self.preparer = AbalonePreparer()

    def test_selected_raw_feature_inputs(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertEqual(row['length'], 0.455)
        self.assertEqual(row['diameter'], 0.365)
        self.assertEqual(row['height'], 0.095)
        self.assertEqual(row['whole_weight'], 0.514)
        self.assertEqual(row['shucked_weight'], 0.2245)
        self.assertEqual(row['viscera_weight'], 0.101)
        self.assertEqual(row['shell_weight'], 0.15)
        self.assertEqual(row['M'], 1)
        self.assertEqual(row['F'], 0)

    def test_rest_of_features_absent(self):
        X = np_inputs[:-1]
        row = self.preparer.prepare(X).loc[0]
        absent_features = ['rings', 'age', 'I']
        with pytest.raises(KeyError):
            for absent_feature in absent_features:
                row[absent_feature]

class TestAbaloneValidations(TestCase):
    def setUp(self):
        self.preparer = AbalonePreparerETL()

    def test_diameter_is_positive(self):
        invalid_inputs = np_inputs.copy()
        invalid_inputs[2] = 0
        with pytest.raises(ValueError, match="'diameter' should be positive"):
            self.preparer.prepare(invalid_inputs)

    def test_rings_invalid_lower_bound(self):
        invalid_inputs = np_inputs.copy()
        invalid_inputs[8] = 0
        with pytest.raises(ValueError, match="'rings' out of range"):
            self.preparer.prepare(invalid_inputs)

    def test_rings_invalid_upper_bound(self):
        invalid_inputs = np_inputs.copy()
        invalid_inputs[8] = 31
        with pytest.raises(ValueError, match="'rings' out of range"):
            self.preparer.prepare(invalid_inputs)

    def test_sex_values(self):
        months = ['M', 'F', 'I']
        for month in months:
            inputs[0] = month
            self.preparer.prepare(inputs)

    def test_sex_invalid(self):
        invalid_inputs = np_inputs.copy()
        invalid_inputs[0] = 'Z'
        with pytest.raises(ValueError, match="Invalid 'sex'"):
            self.preparer.prepare(invalid_inputs)
