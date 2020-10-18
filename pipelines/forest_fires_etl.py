import sys
import argparse

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from pipelines.transformation import dummify
from pipelines.schema import build_df
from pipelines.forest_fires_schema import forest_fires_schema

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

class ForestFiresPreparer():
    def __init__(self, feature_subset=None):
        self.schema = forest_fires_schema
        self.feature_subset = feature_subset

    def prepare(self, data):
        df = build_df(data, forest_fires_schema)

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

        if self.feature_subset: df = df[self.feature_subset]
        return df


def parse_args(args=[]):
    parser = argparse.ArgumentParser(
        description='ETL Preparation Pipeline for Forest Fires data.')

    default_data = 'lake/forest_fires/forestfires.csv'
    parser.add_argument('-i', '--input', default=default_data,
        dest='data', help=f'filepath for data source (default: {default_data})')
    
    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database',
        type=str, dest='database', default=default_db,
        help=f'SQLite database file to store result (default: {default_db})')

    default_db_table = 'forest_fires'
    parser.add_argument('-t', '--table', default=default_db_table,
        dest='table_name',
        help=f'SQLite database table name (default: {default_db_table})')

    default_table_overwrite = False
    parser.add_argument('-o', '--overwrite', default=default_table_overwrite,
        dest='table_overwrite', type=bool,
        help=f'Overwrite database table (default: {default_table_overwrite})')

    return vars(parser.parse_args(args))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    print("≫ Extracting Data")
    data_source = args['data']
    df = load_data(data_source)
    
    print("≫ Transforming Data")
    preparer = ForestFiresPreparer()
    df = preparer.prepare(df)

    print("≫ Loading Data")
    database = args['database']
    engine = create_engine(f'sqlite:///{database}')
    df.to_sql('forest_fires', engine, if_exists='replace', index=False)
    print("≫ ETL - Done")
