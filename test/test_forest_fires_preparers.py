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

from pipelines.forest_fires_preparers import ForestFiresPreparerETL, ForestFiresPreparer

target_name = 'area'
input_names = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain']
inputs      = [8, 6, 'sep', 'thu', 93.7, 80.9, 685.2, 17.9, 23.7, 25.0, 4.5, 0.4, 1.12]

np_inputs    = np.array(inputs)
np_inputs_2d = np_inputs.reshape(1, len(np_inputs))
df_inputs    = pd.DataFrame([inputs], columns=input_names + [target_name])
X = np_inputs[:-1]

new_feature_names  = [f'{target_name}_log', 'FFMC_log', 'ISI_log', 'rain_log', 'rain_cat', 'sep', 'thu']
feature_vals  = inputs + [np.log1p(1.12), np.log1p(93.7), np.log1p(17.9), np.log1p(0.4), 1, 1, 1]

feature_names = input_names + [target_name] + new_feature_names
df_features   = pd.DataFrame([feature_vals], columns=feature_names)
df_features['sep'] = df_features['sep'].astype(np.uint8)
df_features['thu'] = df_features['thu'].astype(np.uint8)
df_features['rain_cat'] = df_features['rain_cat'].astype(np.uint8)


class TestForestFiresPreparerETL(TestCase):
    def setUp(self):
        self.preparer = ForestFiresPreparerETL()

    def test_raw_features_inputs(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertEqual(row['X'], 8)
        self.assertEqual(row['Y'], 6)
        self.assertEqual(row['month'], 'sep')
        self.assertEqual(row['day'], 'thu')
        self.assertEqual(row['FFMC'], 93.7)
        self.assertEqual(row['DMC'], 80.9)
        self.assertEqual(row['DC'], 685.2)
        self.assertEqual(row['ISI'], 17.9)
        self.assertEqual(row['temp'], 23.7)
        self.assertEqual(row['RH'], 25)
        self.assertEqual(row['wind'], 4.5)
        self.assertEqual(row['rain'], 0.4)
        self.assertEqual(row['area'], 1.12)

    def test_raw_features_types(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertIsInstance(row['X'], np.int64)
        self.assertIsInstance(row['Y'], np.int64)
        self.assertIsInstance(row['month'], str)
        self.assertIsInstance(row['day'], str)
        self.assertIsInstance(row['FFMC'], np.float64)
        self.assertIsInstance(row['DMC'], np.float64)
        self.assertIsInstance(row['DC'], np.float64)
        self.assertIsInstance(row['ISI'], np.float64)
        self.assertIsInstance(row['temp'], np.float64)
        self.assertIsInstance(row['RH'], np.float64)
        self.assertIsInstance(row['wind'], np.float64)
        self.assertIsInstance(row['rain'], np.float64)
        self.assertIsInstance(row['area'], np.float64)

    def test_prepare_area_to_log(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertEqual(row['area_log'], np.log1p(1.12))

    def test_prepare_FFMC_to_log(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertEqual(row['FFMC_log'], np.log1p(93.7))

    def test_prepare_ISI_to_log(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertEqual(row['ISI_log'], np.log1p(17.9))

    def test_prepare_rain_to_log(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertEqual(row['rain_log'], np.log1p(0.4))

    def test_prepare_rain_to_category(self):
        row = self.preparer.prepare(inputs).loc[0]
        self.assertEqual(row['rain_cat'], 1)

    def test_prepare_dummy_month(self):
        feature_cols = self.preparer.prepare(inputs).columns.to_list()
        self.assertIn('sep', feature_cols)

    def test_prepare_dummy_day(self):
        feature_cols = self.preparer.prepare(inputs).columns.to_list()
        self.assertIn('thu', feature_cols)


class TestForestFiresPreparerServing(TestCase):
    def setUp(self):
        self.preparer = ForestFiresPreparer()

    def test_selected_raw_feature_inputs(self):
        row = self.preparer.prepare(X).loc[0]
        self.assertEqual(row['X'], 8)
        self.assertEqual(row['Y'], 6)
        self.assertEqual(row['FFMC'], 93.7)
        self.assertEqual(row['DMC'], 80.9)
        self.assertEqual(row['DC'], 685.2)
        self.assertEqual(round(row['ISI_log'],2), 2.94)
        self.assertEqual(row['temp'], 23.7)
        self.assertEqual(row['RH'], 25)
        self.assertEqual(row['wind'], 4.5)
        self.assertEqual(row['rain_cat'], 1)

    def test_rest_of_features_absent(self):
        row = self.preparer.prepare(X).loc[0]
        absent_features = ['month','day','ISI', 'rain']
        with pytest.raises(KeyError):
            for absent_feature in absent_features:
                row[absent_feature]

class TestForesFiresValidations(TestCase):
    def setUp(self):
        self.preparer = ForestFiresPreparerETL()

    def test_X_invalid_lower_bound(self):
        invalid_inputs = np_inputs.copy()
        invalid_inputs[0] = 0
        with pytest.raises(ValueError, match="'X' out of range"):
            self.preparer.prepare(invalid_inputs)

    def test_X_valid_upper_bound(self):
        invalid_inputs = np_inputs.copy()
        invalid_inputs[0] = 10
        with pytest.raises(ValueError, match="'X' out of range"):
            self.preparer.prepare(invalid_inputs)

    def test_Y_invalid_lower_bound(self):
        invalid_inputs = np_inputs.copy()
        invalid_inputs[1] = 0
        with pytest.raises(ValueError, match="'Y' out of range"):
            self.preparer.prepare(invalid_inputs)

    def test_Y_valid_upper_bound(self):
        invalid_inputs = np_inputs.copy()
        invalid_inputs[1] = 10
        with pytest.raises(ValueError, match="'Y' out of range"):
            self.preparer.prepare(invalid_inputs)

    def test_month_values(self):
        months = ['jan','feb','mar','may','jun','jul','aug','sep','oct','nov','dec']
        for month in months:
            inputs[2] = month
            self.preparer.prepare(inputs)

    def test_month_invalid(self):
        invalid_inputs = np_inputs.copy()
        invalid_inputs[2] = 'Jan'
        with pytest.raises(ValueError, match="Invalid 'month'"):
            self.preparer.prepare(invalid_inputs)

    def test_day_values(self):
        days = ['mon','tue','wed','thu','fri','sat','sun']
        for day in days:
            inputs[3] = day
            self.preparer.prepare(inputs)

    def test_day_invalid(self):
        invalid_inputs = np_inputs.copy()
        invalid_inputs[3] = 'Monday'
        with pytest.raises(ValueError, match="Invalid 'day'"):
            self.preparer.prepare(invalid_inputs)
