import sys
import argparse

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# from pipelines.dummy_transformer import dummify

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

class ForestFiresProcessor():
    def transform(self, data):
        # if not type(data) == pd.DataFrame:
        data = np.array(data, ndmin=2)
        df = pd.DataFrame(data, columns=['X','Y','FFMC','DMC','DC','ISI','temp','RH','wind','rain'])
        return df

    # def transform_batch(self, df):
    #     """
    #     Transforms a batch of Fores Fire records.
        
    #     Parameters:
    #     - X: DataFrame with the 12 columns of the dataset.

    #     Returns:
    #     - Dataframe containing the transformed records.
    #     """
    #     categorical_features = ['month', 'day']
    #     numerical_features = [feature for feature in df.columns
    #                                   if feature not in categorical_features]
    #     return df[numerical_features]


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
    processor = ForestFireProcessor()
    df = processor.transform_batch(df)
    # df = dummify(df, 'month', prefix='month')
    # df = dummify(df, 'day', prefix='day')

    print("≫ Loading Data")
    database = args['database']
    engine = create_engine(f'sqlite:///{database}')
    df.to_sql('forest_fires', engine, if_exists='replace', index=False)
    print("≫ ETL - Done")
