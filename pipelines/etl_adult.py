import sys
import argparse

import pandas as pd
from sqlalchemy import create_engine

# age: continuous.
# sex: Female, Male.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# hours-per-week: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# fnlwgt: continuous.
# capital-gain: continuous.
# capital-loss: continuous.


def parse_args(args=[]):
    parser = argparse.ArgumentParser(
    description='ETL Preparation Pipeline for Adult Census data.')

    default_data = 'lake/adult/adult_full.csv'
    parser.add_argument('-i', '--input', default=default_data, dest='data',
        help=f'Input data filepath (default: {default_data})')

    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database', default=default_db, dest='database',
        help=f'SQLite database file to store result (default: {default_db})')

    default_db_table = 'adults'
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
    print(args)

    print("≫ Extracting Data")
    input_data = args['data']
    df = pd.read_csv(input_data)

    # print("≫ Transforming Data")
    # processor = Processor()
    # df = processor.transform(df)

    print("≫ Loading Data")
    database = args['database']
    db_table = args['table_name']
    engine = create_engine(f'sqlite:///{database}')
    if_table_exists = 'replace' if args['table_overwrite'] else 'fail'
    df.to_sql(db_table, engine, if_exists=if_table_exists, index=False)
    print("≫ ETL - Done")
