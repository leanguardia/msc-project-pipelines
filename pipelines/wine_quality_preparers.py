import numpy as np
import pandas as pd

from pipelines.preparer import Preparer
from pipelines.wine_quality_schema import wines_schema

class WinesPreparerETL(Preparer):
    def __init__(self):
        super(WinesPreparerETL, self).__init__(wines_schema)
        self.input_validator.add_validators(self.schema.validators(which='input'))

    def prepare(self, data):
        df = super(WinesPreparerETL, self).prepare(data)
        self.input_validator.validate(df)

        df['quality_cat'] = pd.cut(df['quality'], bins=[0,5,7,10],
            labels=[0,1,2], include_lowest=True).astype(np.uint8)

        df['free_sulfur_dioxide_log'] = np.log(df['free_sulfur_dioxide'])
        df['total_sulfur_dioxide_log'] = np.log(df['total_sulfur_dioxide'])
        df['residual_sugar_log'] = np.log(df['residual_sugar'])

        return df
    

class WhiteWinesPreparer(Preparer):
    def __init__(self):
        super(WhiteWinesPreparer, self).__init__(wines_schema)

    def prepare(self, data):
        df = super(WhiteWinesPreparer, self).prepare(data)
        
#         df = dummify(df, 'sex', ['M','F','I'])

        selected_features = ['fixed_acidity', 'volatile_acidity', 'citric_acid',
            'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
            'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']

        return df[selected_features].copy()

