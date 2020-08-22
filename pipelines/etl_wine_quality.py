import sys
import argparse 

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

class WineQualityProcessor():
    COLUMNS = ['fixed acidity', 'volatile acidity', 'citric acid',
              'residual sugar', 'chlorides', 'free sulfur dioxide',
              'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
              'quality']

    def transform(self, data):
        """
        Clean, Transform and Enhance wine input data.

        Parameters
        data: list, ndarray or DataFrame of shape (X, 11 or 12)

        Returns: DataFrame with wine transformed data
        """
        if not type(data) == pd.DataFrame:
            data = np.array(data, ndmin=2)
            # print(data.shape, "!!!!!!!!!!!")
        rows, cols = data.shape
        if cols < 11 or cols > 12:
            raise ValueError("data must have 11 or 12 columns.")
        columns = self.COLUMNS
        # if cols == 11:/
            # columns = self.COLUMNS[:-1]
        df = pd.DataFrame(data, columns=columns)
        return df


def parse_args(args=[]):
    parser = argparse.ArgumentParser(
        description='ETL Preparation Pipeline for Wine Quality data.')

    default_data = 'lake/wine_quality/winequality-white.csv'
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
    print(data_source)
    df = pd.read_csv(data_source)
    
    print("≫ Transforming Data")
    processor = WineQualityProcessor()
    df = processor.transform(df)

    print("≫ Loading Data")
    database = args['database']
    engine = create_engine(f'sqlite:///{database}')
    df.to_sql("fires", engine, if_exists='replace')
    print("ETL - Done")

