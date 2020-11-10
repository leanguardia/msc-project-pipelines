import sys
import argparse 

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


#         # Target Transformations
#         if cols == self.num_of_columns:

#             if (df['quality'] < 0).any() or (df['quality'] > 10).any():
#                 raise ValueError(f"Value out of range 'quality'")


#         df['free_sulfur_dioxide_log'] = np.log(df['free_sulfur_dioxide'])
#         df['total_sulfur_dioxide_log'] = np.log(df['total_sulfur_dioxide'])
#         df['residual_sugar_log'] = np.log(df['residual_sugar'])
#         return df


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

