import pandas as pd
import numpy as np

from dummy_transformer import dummify

from sqlalchemy import create_engine

engine = create_engine('sqlite:///lake/warehouse.db')#, echo=True)

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

if __name__ == "__main__":
    
    print("Extracting Data")
    df = load_data('lake/forest_fires/forestfires.csv')
    
    print("Transforming Data")
    df = dummify(df, 'month', prefix='month')
    df = dummify(df, 'day', prefix='day')

    print("Loading Data")
    df.to_sql("fires", engine, if_exists='replace')
    print("ETL - Done")
