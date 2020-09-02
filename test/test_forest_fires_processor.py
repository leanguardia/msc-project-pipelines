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
# sense to model with the logarithm transform).

from unittest import TestCase

import pytest
import pandas as pd
import numpy as np

from pipelines.etl_forest_fires import ForestFiresProcessor

# cols = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']
# values = [8, 6, 'sep', 'thu', 93.7, 80.9, 685.2, 17.9, 23.7, 25, 4.5, 0, 1.12]

cols = ['X','Y','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']
values = [8, 6, 93.7, 80.9, 685.2, 17.9, 23.7, 25, 4.5, 0, 1.12]

np_values    = np.array(values)
np_values_2d = np_values.reshape(1, len(np_values))
df = pd.DataFrame([values], columns=cols)

# features = [8, 6, 1, 1, 93.7, 80.9, 685.2, 17.9, 23.7, 25, 4.5, 0, 1.12]

class TestForestFiresProcessor(TestCase):
    def setUp(self):
        self.processor = ForestFiresProcessor()

    def test_transform_list(self):
        transformed = self.processor.transform(values)
        assert np.array_equal(transformed.values, np_values_2d)

    def test_transform_wrapped_list(self):
        transformed = self.processor.transform([values])
        assert np.array_equal(transformed.values, np_values_2d)

    def test_transform_narray_one_dim(self):
        transformed = self.processor.transform(np_values)
        assert np.array_equal(transformed.values, np_values_2d)

    def test_transform_wrapped_narray_one_dim(self):
        transformed = self.processor.transform([np_values])
        assert np.array_equal(transformed.values, np_values_2d)

    def test_transform_narray_two_dims(self):
        assert np.array_equal(
            self.processor.transform(np_values_2d).values, np_values_2d)

    def test_transform_df(self):
        assert np.array_equal(self.processor.transform(df).values, np_values_2d)

    def test_transform_requires_minimum_number_of_features(self):
        with pytest.raises(ValueError, match="incorrect number of columns"):
            self.processor.transform(values[:6])

    def test_transform_requires_maximum_number_of_features(self):
        with pytest.raises(ValueError, match='incorrect number of columns'):
            self.processor.transform(values * 2)

    def test_transform_accepts_only_features_as_input(self):
        X = np_values[:-1]
        assert np.array_equal(self.processor.transform([X]).values, [X])
    