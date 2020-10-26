import numpy as np

from pipelines.transformation import dummify
from pipelines.preparer import Preparer
from pipelines.abalone_schema import abalone_schema
# from pipelines.validators import ValidationsRunner

class AbalonePreparerETL(Preparer):
    def __init__(self):
        super(AbalonePreparerETL, self).__init__(abalone_schema)
#         self.input_validator.add_validators(self.schema.validators(which='input'))
#         self.output_validator = ValidationsRunner()
#         self.output_validator.add_validators(self.schema.validators(which='engineered'))

    def prepare(self, data):
        df = super(AbalonePreparerETL, self).prepare(data)
        # self.input_validator.validate(df)

        df['age'] = df['rings'] + 1.5
        df = dummify(df, 'sex', ['M','F','I'])

#         self.output_validator.validate(df)

        return df
    

# class ForestFiresPreparer(Preparer):
#     def __init__(self):
#         super(ForestFiresPreparer, self).__init__(forest_fires_schema)
#         # self.input_validator = self._build_input_validations()
#         # self.output_validator = self._build_output_validations()

#     def prepare(self, data):
#         df = super(ForestFiresPreparer, self).prepare(data)

#         df['rain_cat'] = (df['rain'] > 0).astype(np.uint8)
#         df['ISI_log'] = np.log1p(df['ISI'])

#         df = dummify(df, 'month', ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
#         df = dummify(df, 'day', ['mon','tue','wed','thu','fri','sat','sun'])

#         selected_features = ['X', 'Y',
#             'FFMC', 'DMC', 'DC', 'ISI_log',
#             'temp', 'RH','wind', 'rain_cat', 
#             'apr', 'aug', 'dec', 'feb', 'jan',
#             'jun', 'mar', 'may', 'nov', 'oct', 'sep', 
#             'fri', 'mon', 'sat', 'sun', 'thu',
#         ]

#         return df[selected_features].copy()

#     def _build_input_validations(self):
#         input_validator = ValidationsRunner()
#         input_validator.add_validators(self.schema.validators(which='input'))
#         return input_validator




