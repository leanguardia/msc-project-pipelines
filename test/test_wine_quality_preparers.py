from unittest import TestCase

import pytest
import pandas as pd
import numpy as np

from pipelines.wine_quality_preparers import WinesPreparerETL, WhiteWinesPreparer

inputs = [7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.0010, 3.00, 0.45, 8.8, 6]

np_inputs    = np.array(inputs)

class TestWinesPreparerETL(TestCase):
    def setUp(self):
        self.preparer = WinesPreparerETL()
        
    def test_raw_features_inputs(self):
        row = self.preparer.prepare(inputs).loc[0]
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

    def test_raw_features_type(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertIsInstance(row['fixed_acidity'], float)
        self.assertIsInstance(row['citric_acid'], float)
        self.assertIsInstance(row['volatile_acidity'], float)
        self.assertIsInstance(row['residual_sugar'], float)
        self.assertIsInstance(row['chlorides'], float)
        self.assertIsInstance(row['free_sulfur_dioxide'], float)
        self.assertIsInstance(row['total_sulfur_dioxide'], float)
        self.assertIsInstance(row['density'], float)
        self.assertIsInstance(row['pH'], float)
        self.assertIsInstance(row['sulphates'], float)
        self.assertIsInstance(row['alcohol'], float)
        self.assertIsInstance(row['quality'], float)

    def test_prepare_quality_to_category_1(self):
        transformed = self.preparer.prepare(inputs)
        self.assertEqual(transformed.loc[0]['quality_cat'], 1)

    def test_prepare_quality_to_category_0(self):
        others = np_inputs.copy()
        others[-1] = 1
        transformed = self.preparer.prepare(others)
        self.assertEqual(transformed.loc[0]['quality_cat'], 0)

    def test_prepare_quality_to_category_2(self):
        others = np_inputs.copy()
        others[-1] = 8
        transformed = self.preparer.prepare(others)
        self.assertEqual(transformed.loc[0]['quality_cat'], 2)

    def test_transform_free_sulfur_dioxide_to_log(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertEqual(row['free_sulfur_dioxide_log'], np.log(45.0))

    def test_transform_total_sulfur_dioxide_to_log(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertEqual(row['total_sulfur_dioxide_log'], np.log(170.0))

    def test_transform_residual_sugar(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertEqual(row['residual_sugar_log'], np.log(20.7))

class TestWinesPreparerServing(TestCase):
    def setUp(self):
        self.preparer = WhiteWinesPreparer()

    def test_selected_raw_feature_inputs(self):
        row = self.preparer.prepare(inputs).loc[0]
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

class TestWinesValidations(TestCase):
    def setUp(self):
        self.preparer = WinesPreparerETL()

    def test_valid_quality_lower_bound(self):
        invalid_inputs = np_inputs.copy()
        invalid_inputs[-1] = 0
        row = self.preparer.prepare(invalid_inputs).loc[0]
        self.assertEqual(row['quality'], 0)

    def test_valid_quality_upper_bound(self):
        invalid_inputs = np_inputs.copy()
        invalid_inputs[-1] = 10
        row = self.preparer.prepare(invalid_inputs).loc[0]
        self.assertEqual(row['quality'], 10)

    def test_invalid_quality_lower_bound(self):
        invalid_inputs = np_inputs.copy()
        invalid_inputs[-1] = -1
        with pytest.raises(ValueError, match="'quality' out of range"):
            self.preparer.prepare(invalid_inputs).loc[0]

    def test_invalid_quality_upper_bound(self):
        invalid_inputs = np_inputs.copy()
        invalid_inputs[-1] = 11
        with pytest.raises(ValueError, match="'quality' out of range"):
            self.preparer.prepare(invalid_inputs).loc[0]
