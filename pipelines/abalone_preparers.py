import numpy as np

from pipelines.transformation import dummify
from pipelines.preparer import Preparer
from pipelines.abalone_schema import abalone_schema

class AbalonePreparerETL(Preparer):
    def __init__(self):
        super(AbalonePreparerETL, self).__init__(abalone_schema)
        self.input_validator.add_validators(self.schema.validators(which='input'))

    def prepare(self, data):
        df = super(AbalonePreparerETL, self).prepare(data)
        self.input_validator.validate(df)

        df['age'] = df['rings'] + 1.5
        df = dummify(df, 'sex', ['M','F','I'])

        return df
    

class AbalonePreparer(Preparer):
    def __init__(self):
        super(AbalonePreparer, self).__init__(abalone_schema)

    def prepare(self, data):
        df = super(AbalonePreparer, self).prepare(data)
        
        df = dummify(df, 'sex', ['M','F','I'])

        selected_features = ['length', 'diameter', 'height', 'whole_weight',
                'shucked_weight', 'viscera_weight', 'shell_weight', 'M', 'F']

        return df[selected_features].copy()
