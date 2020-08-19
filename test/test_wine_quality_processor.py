import pandas as pd
import numpy as np

from pipelines.etl_wine_quality import WineQualityProcessor

cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol']
vals = [7.0, 0.27, 0.36, 20.7, 0.045, 45, 170.0, 1.0010, 3.00, 0.45, 8.8]

def test_transform():
    processor = WineQualityProcessor()
    assert np.array_equal(
        processor.transform(fixed_acidity=7.0, volatile_acidity=0.27,
            citric_acid=0.36, residual_sugar=20.7, chlorides=0.045,
            free_sulfur_dioxide=45, total_sulfur_dioxide=170.0, density=1.0010,
            pH=3.00, sulphates=0.45, alcohol=8.8),
        np.array([7.0, 0.27, 0.36, 20.7, 0.045, 45, \
                  170.0, 1.0010, 3.00, 0.45, 8.8])
    )

# def test_batch_transformation():
#     df = pd.DataFrame([vals, vals], columns=cols)
#     processor = ForestFireProcessor()
#     assert np.array_equal(
#         processor.transform_batch(df).values,
#         [[1,2,20,100,400,35,33.2,60,5,4], [1,2,20,100,400,35,33.2,60,5,4]]
#     )
