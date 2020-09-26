import pandas as pd
import numpy as np

class Schema:
    
    def __init__(self, schema_dict):
        self.schema_dict = schema_dict

    def columns(self):
        return list(map(lambda column: column['name'], self.schema_dict))

    def n_columns(self):
        return len(self.columns())
    
    def inputs(self):
        return self.columns()[:-1]

    def n_inputs(self):
        return self.n_columns()-1

    def target(self):
        for feature_dict in self.schema_dict:
            if feature_dict['type'] == 'target':
                return feature_dict['name']
        raise ValueError('Target variable not found.')
    
            
def build_df(data, schema):
    if not type(data) == pd.DataFrame:
        data = np.array(data, ndmin=2)
        
    # _rows, cols = data.shape
    # if cols < schema.n_columns() or cols > schema.n_columns():
    #     raise ValueError(f"incorrect number of columns")

    # if cols == self.num_of_columns:
    #     columns = self.COLUMNS
    #     dtypes = self.DTYPES
    # else:
    #     columns = self.COLUMNS[:-1]
    #     dtypes = self.DTYPES[:-1]

    df = pd.DataFrame(data, columns=schema.columns())
    # for col, dtype in zip(columns, dtypes):
        # df[col]= df[col].astype(dtype)

    return df