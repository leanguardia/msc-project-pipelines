import sys
import argparse 

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from pipelines.wine_quality_schema import features

class WineQualityProcessor():
    COLUMNS = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
               'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
               'pH', 'sulphates', 'alcohol', 'type', 'quality']

    def __init__(self):
        self.num_of_columns = len(self.COLUMNS) 
        self.num_of_features = self.num_of_columns - 1
        self.features = features

    def transform(self, data):
        """
        Clean, Transform and Enhance wine input data.

        Parameters
            data: list, ndarray or DataFrame of shape (X, 11 or 12)

        Returns: DataFrame with wine transformed data
        """
        if not type(data) == pd.DataFrame:
            data = np.array(data, ndmin=2)

        _rows, cols = data.shape
        if cols < self.num_of_features or cols > self.num_of_columns:
            raise ValueError(f"incorrect number of columns")

        columns = self.COLUMNS
        features = self.features
        if cols == self.num_of_features:
            columns  = columns[:-1]
            features = features[:-1]

        df = pd.DataFrame(data, columns=columns)

        # Data Types
        for feature in features:
            feature_name = feature['name']
            df[feature_name]= df[feature_name].astype(feature['type'])

        # Target Transformations
        if cols == self.num_of_columns:

            if (df['quality'] < 0).any() or (df['quality'] > 10).any():
                raise ValueError(f"Value out of range 'quality'")

            df['quality_cat'] = pd.cut(df['quality'], bins=[0,5,7,10],
                        labels=[0,1,2], include_lowest=True).astype(np.uint8)

        df['free_sulfur_dioxide_log'] = np.log(df['free_sulfur_dioxide'])
        df['total_sulfur_dioxide_log'] = np.log(df['total_sulfur_dioxide'])
        df['residual_sugar_log'] = np.log(df['residual_sugar'])
        return df


def parse_args(args=[]):
    parser = argparse.ArgumentParser(
        description='ETL Preparation Pipeline for Wine Quality data.')

    default_data = 'lake/wine_quality/winequality.csv'
    parser.add_argument('-s', '--source', default=default_data, dest='data',
        help=f'filepath for data source (default: {default_data})')

    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database', default=default_db, dest='database',
        help=f'SQLite database file to store result (default: {default_db})')

    return vars(parser.parse_args(args))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    print("≫ Extracting Data")
    data_source = args['data']
    df = pd.read_csv(data_source)
    
    print("≫ Transforming Data")
    processor = WineQualityProcessor()
    df = processor.transform(df)

    print("≫ Loading Data")
    database = args['database']
    engine = create_engine(f'sqlite:///{database}')
    df.to_sql('wines', engine, if_exists='replace', index=False)
    print("≫ ETL - Done")

