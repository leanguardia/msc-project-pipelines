from unittest import TestCase

import pytest
import pandas as pd
import numpy as np

from pipelines.adult_preparers import AdultPreparerETL#, AbalonePreparer

inputs = [39, 'State-gov', 77516, 'Bachelors', 13, 'Never-married', 
          'Adm-clerical', 'Not-in-family', 'White', 'Male', 2174, 0, 40,
          'United-States', '<=50K', True]

np_inputs    = np.array(inputs)

class TestForestFiresPreparerETL(TestCase):
    def setUp(self):
        self.preparer = AdultPreparerETL()

    def test_raw_features_inputs(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertEqual(row['age'], 39)
        self.assertEqual(row['workclass'], 'State-gov')
        self.assertEqual(row['fnlwgt'], 77516)
        self.assertEqual(row['education'], 'Bachelors'),
        self.assertEqual(row['education_num'], 13),
        self.assertEqual(row['marital_status'], 'Never-married'),
        self.assertEqual(row['occupation'], 'Adm-clerical'),
        self.assertEqual(row['relationship'], 'Not-in-family'),
        self.assertEqual(row['race'], 'White'),
        self.assertEqual(row['sex'], 'Male'),
        self.assertEqual(row['capital_gain'], 2174),
        self.assertEqual(row['capital_loss'], 0),
        self.assertEqual(row['hours_per_week'], 40),
        self.assertEqual(row['native_country'], 'United-States'),
        self.assertEqual(row['>50K<=50K'], '<=50K'),
        self.assertEqual(row['for_training'], True)
        

    def test_raw_features_types(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertIsInstance(row['age'], np.int64)
        self.assertIsInstance(row['workclass'], str)
        self.assertIsInstance(row['fnlwgt'], np.int64)
        self.assertIsInstance(row['education'], str),
        self.assertIsInstance(row['education_num'], np.int64),
        self.assertIsInstance(row['marital_status'], str),
        self.assertIsInstance(row['occupation'], str),
        self.assertIsInstance(row['relationship'], str),
        self.assertIsInstance(row['race'], str),
        self.assertIsInstance(row['sex'], str),
        self.assertIsInstance(row['capital_gain'], np.int64),
        self.assertIsInstance(row['capital_loss'], np.int64),
        self.assertIsInstance(row['hours_per_week'], np.int64),
        self.assertIsInstance(row['native_country'], str),
        self.assertIsInstance(row['>50K<=50K'], str),
        self.assertIsInstance(row['for_training'], np.bool_)

    def test_remove_duplicates(self):
        df = self.preparer.prepare([inputs, inputs])
        self.assertEqual(df.shape[0], 1)

    def test_whitespaces_are_removed(self):
        invalids = inputs.copy()
        invalids[-2] = ' <=50K'
        row = self.preparer.prepare(invalids).loc[0]
        self.assertEqual(row['>50K<=50K'], '<=50K')

    def test_target_does_not_have_dots(self):
        invalids = inputs.copy()
        invalids[-2] = '<=50K.'
        row = self.preparer.prepare(invalids).loc[0]
        self.assertEqual(row['>50K<=50K'], '<=50K')

    def test_replace_question_mark_with_NaN(self):
        invalids = inputs.copy()
        invalids[1] = '?'
        row = self.preparer.prepare(invalids).loc[0]
        self.assertEqual(row['workclass'], None)

#     def test_prepare_dummy_sex(self):
#         row = self.preparer.prepare(inputs).loc[0]
#         self.assertEqual(row['M'], 1)
#         self.assertEqual(row['F'], 0)
#         self.assertEqual(row['I'], 0)

# class TestAbalonePreparerServing(TestCase):
#     def setUp(self):
#         self.preparer = AbalonePreparer()

#     def test_selected_raw_feature_inputs(self):
#         row = self.preparer.prepare(inputs).loc[0]
#         self.assertEqual(row['length'], 0.455)
#         self.assertEqual(row['diameter'], 0.365)
#         self.assertEqual(row['height'], 0.095)
#         self.assertEqual(row['whole_weight'], 0.514)
#         self.assertEqual(row['shucked_weight'], 0.2245)
#         self.assertEqual(row['viscera_weight'], 0.101)
#         self.assertEqual(row['shell_weight'], 0.15)
#         self.assertEqual(row['M'], 1)
#         self.assertEqual(row['F'], 0)

#     def test_rest_of_features_absent(self):
#         X = np_inputs[:-1]
#         row = self.preparer.prepare(X).loc[0]
#         absent_features = ['rings', 'age', 'I']
#         with pytest.raises(KeyError):
#             for absent_feature in absent_features:
#                 row[absent_feature]

# class TestAbaloneValidations(TestCase):
#     def setUp(self):
#         self.preparer = AbalonePreparerETL()

#     def test_diameter_is_positive(self):
#         invalid_inputs = np_inputs.copy()
#         invalid_inputs[2] = 0
#         with pytest.raises(ValueError, match="'diameter' should be positive"):
#             self.preparer.prepare(invalid_inputs)

#     def test_rings_invalid_lower_bound(self):
#         invalid_inputs = np_inputs.copy()
#         invalid_inputs[8] = 0
#         with pytest.raises(ValueError, match="'rings' out of range"):
#             self.preparer.prepare(invalid_inputs)

#     def test_rings_invalid_upper_bound(self):
#         invalid_inputs = np_inputs.copy()
#         invalid_inputs[8] = 31
#         with pytest.raises(ValueError, match="'rings' out of range"):
#             self.preparer.prepare(invalid_inputs)

#     def test_sex_values(self):
#         months = ['M', 'F', 'I']
#         for month in months:
#             inputs[0] = month
#             self.preparer.prepare(inputs)

#     def test_sex_invalid(self):
#         invalid_inputs = np_inputs.copy()
#         invalid_inputs[0] = 'Z'
#         with pytest.raises(ValueError, match="Invalid 'sex'"):
#             self.preparer.prepare(invalid_inputs)
