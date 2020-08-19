import sys
import argparse 

import numpy as np
import pandas as pd

class WineQualityProcessor():
    def transform(self, fixed_acidity, volatile_acidity, citric_acid,
                  residual_sugar, chlorides, free_sulfur_dioxide,
                  total_sulfur_dioxide, density, pH, sulphates, alcohol):
        return np.array([7.0, 0.27, 0.36, 20.7, 0.045, 45, \
                          170.0, 1.0010, 3.00, 0.45, 8.8])


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
    # df = processor.transform_batch(df)
    # df = dummify(df, 'month', prefix='month')
    # df = dummify(df, 'day', prefix='day')

    # print("≫ Loading Data")
    # database = args['database']
    # engine = create_engine(f'sqlite:///{database}')
    # df.to_sql("fires", engine, if_exists='replace')
    # print("ETL - Done")

