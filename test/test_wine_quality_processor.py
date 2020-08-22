import pytest
import pandas as pd
import numpy as np

from pipelines.etl_wine_quality import WineQualityProcessor

cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'quality']
values = [7.0, 0.27, 0.36, 20.7, 0.045, 45, 170.0, 1.0010, 3.00, 0.45, 8.8, 6]
values_np = np.array(values)
values_np2 = values_np.reshape(1, len(values_np))
df = pd.DataFrame([values], columns=cols)

class TestWineQualityProcessor:
    def test_transform_requires_minimum_number_of_features(self):
        processor = WineQualityProcessor()
        with pytest.raises(ValueError, match='must have 11 or 12 columns'):
            processor.transform([7.0, 0.27, 0.36, 20.7, 0.045, 45])

    def test_transform_requires_maximum_number_of_features(self):
        processor = WineQualityProcessor()
        with pytest.raises(ValueError, match='must have 11 or 12 columns.'):
            processor.transform([7.0, 0.27, 0.36, 20.7, 0.045, 45] * 3)

    def test_transform_list(self):
        processor = WineQualityProcessor()
        returned = processor.transform([values_np])
        assert np.array_equal(returned.values, values_np2)

    # def test_transform_narray_one_dim(self):
    #     processor = WineQualityProcessor()
    #     assert np.array_equal(
    #         processor.transform(nvals).values, nvals)

    
    # def test_transform_df(self):
    #     processor = WineQualityProcessor()
    #     assert np.array_equal(processor.transform(df).values, nvals)


    # def test_transform_accepts_only_features_as_input(self):
    #     processor = WineQualityProcessor()
    #     X = nvals[0, :-1]
    #     print(X, X.shape)
    #     assert np.array_equal(processor.transform(X).values, X)


