import numpy as np

from pipelines.transformation import dummify
from pipelines.preparer import Preparer
from pipelines.adult_schema import adult_schema

class AdultPreparerETL(Preparer):
    def __init__(self):
        super(AdultPreparerETL, self).__init__(adult_schema)
        self.input_validator.add_validators(self.schema.validators(which='input'))

    def prepare(self, data):
        df = super(AdultPreparerETL, self).prepare(data)
        self.input_validator.validate(df)

        df.drop_duplicates(keep='first', inplace=True)

        # Select all text columns and strip all values
        text_cols = df.dtypes == np.object
        texts_df = df.loc[:, text_cols].copy()
        df.loc[:, text_cols] = texts_df.applymap(lambda text: text.strip())

        # Remove extra dots from target.
        df['>50K<=50K'] = df['>50K<=50K'].replace({'>50K.': '>50K', '<=50K.': '<=50K'})

        # Replace '?' with None to identify missing values.
        df['workclass'] = df['workclass'].replace({'?': None})
        df['occupation'] = df['occupation'].replace({'?': None})
        df['native_country'] = df['native_country'].replace({'?': None})

        # Feature Engineering
        # workclasses = ['State-gov', 'Self-emp-not-inc', 'Private','Federal-gov',
        #              'Local-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked']
        # df = dummify(df, 'workclass', workclasses, dummy_na=True)
        # df = dummify(df, 'race', ['White', 'Black', 'Asian-Pac-Islander',
        #                             'Amer-Indian-Eskimo', 'Other'])
        # df = dummify(df, 'sex', ['Male', 'Female'])

        return df
    

# class AbalonePreparer(Preparer):
#     def __init__(self):
#         super(AbalonePreparer, self).__init__(adult_schema)

#     def prepare(self, data):
#         df = super(AbalonePreparer, self).prepare(data)
        
#         df = dummify(df, 'sex', ['M','F','I'])

#         selected_features = ['length', 'diameter', 'height', 'whole_weight',
#                 'shucked_weight', 'viscera_weight', 'shell_weight', 'M', 'F']

#         return df[selected_features].copy()
