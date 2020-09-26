import pandas as pd
import numpy as np

from pipelines.schemas_metadata import forest_fires_features_meta
from pipelines.validators import RangeValidator, CategoryValidator

class Schema:
    def __init__(self, features_list):
        self.features_list = features_list

    def columns(self):
        return list(map(lambda column: column['name'], self.features_list))

    def n_columns(self):
        return len(self.columns())

    def inputs(self):
        return self.columns()[:-1]

    def n_inputs(self):
        return self.n_columns()-1

    def target(self):
        for feature_dict in self.features_list:
            if feature_dict['type'] == 'target':
                return feature_dict['name']
        raise ValueError('Target variable not found.')

    def dtypes(self):
        return [dtype['dtype'] for dtype in self.features_list]

    def validators(self):
        validators = []
        for feature in self.features_list:
            if 'range' in feature:
                validators.append(self._build_range_validator(feature))
            if 'categories' in feature:
                validators.append(self._build_category_validator(feature))
        return validators

    def _build_range_validator(self, feature):
        mini, maxi = feature['range']
        return RangeValidator(feature['name'], mini, maxi)

    def _build_category_validator(self, feature):
        return CategoryValidator(feature['name'], feature['categories'])


def build_df(data, schema):
    if not type(data) == pd.DataFrame:
        data = np.array(data, ndmin=2)
        
    _rows, cols = data.shape

    if cols < schema.n_inputs() or cols > schema.n_columns():
        raise ValueError(f"incorrect number of columns")

    columns = schema.columns()
    dtypes = schema.dtypes()
    
    if cols == schema.n_inputs():
        columns = columns[:-1]
        dtypes = dtypes[:-1]

    df = pd.DataFrame(data, columns=columns)
    for col, dtype in zip(schema.columns(), dtypes):
        df[col]= df[col].astype(dtype)

    return df

forest_fires_schema = Schema(forest_fires_features_meta)