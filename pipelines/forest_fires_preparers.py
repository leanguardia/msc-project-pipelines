import numpy as np

from pipelines.transformation import dummify
from pipelines.preparer import Preparer
from pipelines.forest_fires_schema import forest_fires_schema

class ForestFiresPreparerETL(Preparer):
    def __init__(self):
        super(ForestFiresPreparerETL, self).__init__(forest_fires_schema)

    def prepare(self, data):
        df = super(ForestFiresPreparerETL, self).prepare(data)

        for validator in forest_fires_schema.validators(which='input'):
            validator.validate(df)

        _rows, cols = df.shape

        # Target Transformations
        if cols == self.schema.n_inputs():
            df['area_log'] = np.log1p(df['area'])
        
        # Feature Transformations
        df['FFMC_log'] = np.log1p(df['FFMC'])
        df['ISI_log'] = np.log1p(df['ISI'])
        df['rain_log'] = np.log1p(df['rain'])
        df['rain_cat'] = (df['rain'] > 0).astype(np.uint8)

        df = dummify(df, 'month')
        df = dummify(df, 'day')

        for validator in forest_fires_schema.validators(which='engineered'):
            validator.validate(df)

        return df
