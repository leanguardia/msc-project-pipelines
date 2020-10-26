import numpy as np

from pipelines.transformation import dummify
from pipelines.preparer import Preparer
from pipelines.forest_fires_schema import forest_fires_schema
from pipelines.validators import ValidationsRunner

class ForestFiresPreparerETL(Preparer):
    def __init__(self):
        super(ForestFiresPreparerETL, self).__init__(forest_fires_schema)
        self.input_validator.add_validators(self.schema.validators(which='input'))
        self.output_validator = ValidationsRunner()
        self.output_validator.add_validators(self.schema.validators(which='engineered'))

    def prepare(self, data):
        df = super(ForestFiresPreparerETL, self).prepare(data)

        self.input_validator.validate(df)

        _rows, cols = df.shape

        # Target Transformations
        df['area_log'] = np.log1p(df['area'])
        
        # Feature Transformations
        df['FFMC_log'] = np.log1p(df['FFMC'])
        df['ISI_log'] = np.log1p(df['ISI'])
        df['rain_log'] = np.log1p(df['rain'])
        df['rain_cat'] = (df['rain'] > 0).astype(np.uint8)

        df = dummify(df, 'month', ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
        df = dummify(df, 'day', ['mon','tue','wed','thu','fri','sat','sun'])

        self.output_validator.validate(df)

        return df
    

class ForestFiresPreparer(Preparer):
    def __init__(self):
        super(ForestFiresPreparer, self).__init__(forest_fires_schema)
        # self.input_validator = self._build_input_validations()
        # self.output_validator = self._build_output_validations()

    def prepare(self, data):
        df = super(ForestFiresPreparer, self).prepare(data)

        df['rain_cat'] = (df['rain'] > 0).astype(np.uint8)
        df['ISI_log'] = np.log1p(df['ISI'])

        df = dummify(df, 'month', ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
        df = dummify(df, 'day', ['mon','tue','wed','thu','fri','sat','sun'])

        selected_features = ['X', 'Y',
            'FFMC', 'DMC', 'DC', 'ISI_log',
            'temp', 'RH','wind', 'rain_cat', 
            'apr', 'aug', 'dec', 'feb', 'jan',
            'jun', 'mar', 'may', 'nov', 'oct', 'sep', 
            'fri', 'mon', 'sat', 'sun', 'thu',
        ]

        return df[selected_features].copy()

    def _build_input_validations(self):
        input_validator = ValidationsRunner()
        input_validator.add_validators(self.schema.validators(which='input'))
        return input_validator




