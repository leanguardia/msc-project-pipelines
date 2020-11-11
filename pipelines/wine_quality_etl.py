import sys
import argparse 

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from pipelines.wine_quality_preparers import WinesPreparerETL


def parse_args(args=[]):
    parser = argparse.ArgumentParser(
        description='ETL Preparation Pipeline for Wine Quality data.')

    default_data = 'lake/wine_quality/winequality.csv'
    parser.add_argument('-s', '--source', default=default_data, dest='data',
        help=f'filepath for data source (default: {default_data})')

    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database', default=default_db, dest='database',
        help=f'SQLite database file to store result (default: {default_db})')

    default_db_table = 'wines'
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
    df = pd.read_csv(data_source)
    
    print("≫ Transforming Data")
    preparer = WinesPreparerETL()
    df = preparer.prepare(df)

    print("≫ Loading Data")
    database = args['database']
    db_table = args['table_name']
    engine = create_engine(f'sqlite:///{database}')
    if_table_exists = 'replace' if args['table_overwrite'] else 'fail'
    df.to_sql(db_table, engine, if_exists=if_table_exists, index=False)

