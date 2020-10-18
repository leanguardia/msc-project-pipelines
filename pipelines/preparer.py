import pandas as pd
import numpy as np

class Preparer():
    def __init__(self, schema):
        self.schema = schema

    def prepare(self, data):
        if not type(data) == pd.DataFrame:
            data = np.array(data, ndmin=2)
        
        _rows, cols = data.shape

        if cols < self.schema.n_inputs() - 1 or cols > self.schema.n_inputs():
            raise ValueError(f"incorrect number of columns")

        columns = self.schema.features(which='input')
        dtypes = self.schema.dtypes(which='input')

        if cols + 1 == self.schema.n_inputs():
            columns = columns[:-1]
            dtypes = dtypes[:-1]

        df = pd.DataFrame(data, columns=columns)
        for col, dtype in zip(self.schema.features(), dtypes):
            df[col]= df[col].astype(dtype)

        return df
