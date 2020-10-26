# For more information, read [Cortez and Morais, 2007].
# 1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9
# 2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9
# 3. month - month of the year: 'jan' to 'dec'
# 4. day - day of the week: 'mon' to 'sun'

# 5. FFMC - FFMC index from the FWI system: 18.7 to 96.20
# 6. DMC - DMC index from the FWI system: 1.1 to 291.3
# 7. DC - DC index from the FWI system: 7.9 to 860.6
# 8. ISI - ISI index from the FWI system: 0.0 to 56.10

# 9. temp - temperature in Celsius degrees: 2.2 to 33.30
# 10. RH - relative humidity in %: 15.0 to 100
# 11. wind - wind speed in km/h: 0.40 to 9.40
# 12. rain - outside rain in mm/m2 : 0.0 to 6.4

# 13. area - the burned area of the forest (in ha): 0.00 to 1090.84
# (this output variable is very skewed towards 0.0, thus it may make
# sense to model with the logarithm prepare).

from unittest import TestCase

import pytest
import pandas as pd
import numpy as np

from pipelines.abalone_preparers import AbalonePreparerETL, AbalonePreparer

# target_name = 'rings'
# input_names = ['sex', 'length', 'diameter', 'height', 'whole_weight', \
#                'shucked_weight', 'viscera_weight', 'shell_weight']
inputs      = ['M', 0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 0.15, 15]

np_inputs    = np.array(inputs)
# np_inputs_2d = np_inputs.reshape(1, len(np_inputs))
# df_inputs    = pd.DataFrame([inputs], columns=input_names + [target_name])

# new_feature_names  = [f'{target_name}_log', 'FFMC_log', 'ISI_log', 'rain_log', 'rain_cat', 'sep', 'thu']
# feature_vals  = inputs + [np.log1p(1.12), np.log1p(93.7), np.log1p(17.9), np.log1p(0.4), 1, 1, 1]

# feature_names = input_names + [target_name] + new_feature_names
# df_features   = pd.DataFrame([feature_vals], columns=feature_names)
# df_features['sep'] = df_features['sep'].astype(np.uint8)
# df_features['thu'] = df_features['thu'].astype(np.uint8)
# df_features['rain_cat'] = df_features['rain_cat'].astype(np.uint8)


class TestForestFiresPreparerETL(TestCase):
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
