import sys
import argparse
import pandas as pd
from sqlalchemy import create_engine

from pipelines.abalone_preparers import AbalonePreparerETL

def parse_args(args=[]):
    parser = argparse.ArgumentParser(
        description='ETL Preparation Pipeline for Abalone data.')

    default_data = 'lake/abalone/abalone.csv'
    parser.add_argument('-i', '--input', default=default_data, dest='data',
        help=f'Input data filepath (default: {default_data})')

    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database', default=default_db, dest='database',
        help=f'SQLite database file to store result (default: {default_db})')

    default_db_table = 'abalones'
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
    input_data = args['data']
    df = pd.read_csv(input_data)
    
    print("≫ Transforming Data")
    preparer = AbalonePreparerETL()
    df = preparer.prepare(df)

    print("≫ Loading Data")
    database = args['database']
    db_table = args['table_name']
    if_table_exists = 'replace' if args['table_overwrite'] else 'fail'
    engine = create_engine(f'sqlite:///{database}')
    df.to_sql(db_table, engine, if_exists=if_table_exists, index=False)
    print("≫ ETL - Done")
