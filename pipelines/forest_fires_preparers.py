import numpy as np

from pipelines.transformation import dummify
from pipelines.preparer import Preparer
from pipelines.forest_fires_schema import forest_fires_schema
from pipelines.validators import ValidationsRunner

class ForestFiresPreparerETL(Preparer):
    def __init__(self):
        super(ForestFiresPreparerETL, self).__init__(forest_fires_schema)
        self.input_validator = self._build_input_validations()
        self.output_validator = self._build_output_validations()

    def prepare(self, data):
        df = super(ForestFiresPreparerETL, self).prepare(data)

        self.input_validator.validate(df)

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

        self.output_validator.validate(df)

        return df

    def _build_input_validations(self):
        input_validator = ValidationsRunner()
        input_validator.add_validators(self.schema.validators(which='input'))
        return input_validator

    def _build_output_validations(self):
        output_validator = ValidationsRunner()
        output_validator.add_validators(self.schema.validators(which='engineered'))
        return output_validator
