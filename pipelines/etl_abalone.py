import sys
import argparse

# Sex / nominal / -- / M, F, and I (infant)
# Length / continuous / mm / Longest shell measurement
# Diameter / continuous / mm / perpendicular to length
# Height / continuous / mm / with meat in shell
# Whole weight / continuous / grams / whole abalone
# Shucked weight / continuous / grams / weight of meat
# Viscera weight / continuous / grams / gut weight (after bleeding)
# Shell weight / continuous / grams / after being dried
# Rings / integer / -- / +1.5 gives the age in years 

def parse_args(args=[]):
    parser = argparse.ArgumentParser(
        description='ETL Preparation Pipeline for Wine Quality data.')

    default_data = 'lake/abalone/abalone.csv'
    parser.add_argument('-i', '--input', default=default_data, dest='data',
        help=f'Input data filepath (default: {default_data})')

    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database', default=default_db, dest='database',
        help=f'SQLite database file to store result (default: {default_db})')

    default_table_overwrite = False
    parser.add_argument('-o', '--overwrite', default=default_table_overwrite,
        dest='table_overwrite', type=bool,
        help=f'Overwrite database table (default: {default_table_overwrite})')

    return vars(parser.parse_args(args))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    print(args)

    # print("≫ Extracting Data")
    # data_source = args['data']
    # print(data_source)
    # df = pd.read_csv(data_source)
    
    # print("≫ Transforming Data")
    # processor = Processor()
    # df = processor.transform(df)

    # print("≫ Loading Data")
    # database = args['database']
    # engine = create_engine(f'sqlite:///{database}')
    # df.to_sql("wines", engine, if_exists='replace')
    # print("ETL - Done")
