import argparse

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# from dummy_transformer import dummify

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

class ForestFireProcessor():
    def transform(self, X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain):
        arr = np.array([X, Y, FFMC, DMC, DC, ISI, temp, RH, wind, rain])
        return list(arr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preparation of Forest Fires dataset.')

    default_data = 'lake/forest_fires/forestfires.csv'
    parser.add_argument('-s', '--source', type=str, default=default_data,
        dest='data', help=f'filepath for data source (default: {default_data})')
    
    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database',
        type=str, dest='database', default=default_db,
        help=f'SQLite database file to store result (default: {default_db})')

    args = vars(parser.parse_args())

    print("≫ Extracting Data")
    data_source = args['data']
    df = load_data(data_source)
    
    print("≫ Transforming Data")
    # df = dummify(df, 'month', prefix='month')
    # df = dummify(df, 'day', prefix='day')

    print("≫ Loading Data")
    database = args['database']
    engine = create_engine(f'sqlite:///{database}')
    df.to_sql("fires", engine, if_exists='replace')
    print("ETL - Done")
