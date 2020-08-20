import pytest
import pandas as pd
import numpy as np

from pipelines.etl_wine_quality import WineQualityProcessor

cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol']
vals = [7.0, 0.27, 0.36, 20.7, 0.045, 45, 170.0, 1.0010, 3.00, 0.45, 8.8]
nvals = np.array(vals, ndmin=2)
df = pd.DataFrame([vals], columns=cols)

class TestWineQualityProcessor:
    def test_transform_requires_minimum_number_of_features(self):
        processor = WineQualityProcessor()
        with pytest.raises(ValueError, match='must have exactly 11 columns.'):
            processor.transform([7.0, 0.27, 0.36, 20.7, 0.045, 45])

    def test_transform_list(self):
        processor = WineQualityProcessor()
        assert np.array_equal(processor.transform([vals]).values, nvals)

    def test_transform_narray(self):
        processor = WineQualityProcessor()
        assert np.array_equal(
            processor.transform(nvals).values, nvals)


    def test_transform_df(self):
        processor = WineQualityProcessor()
        assert np.array_equal(processor.transform(df).values, nvals)
