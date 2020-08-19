import sys
import argparse

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from pipelines.dummy_transformer import dummify

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

class ForestFireProcessor():
    def transform(self, X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain):
        arr = np.array([X, Y, FFMC, DMC, DC, ISI, temp, RH, wind, rain])
        return arr

    def transform_batch(self, df):
        """
        Transforms a batch of Fores Fire records.
        
        Parameters:
        - X: DataFrame with the 12 columns of the dataset.

        Returns:
        - Dataframe containing the transformed records.
        """
        categorical_features = ['month', 'day']
        numerical_features = [feature for feature in df.columns
                                      if feature not in categorical_features]
        return df[numerical_features]

class ForestFirePredictor():
    def __init__(self, model):
        self.model = model

    def predict(self, X, decimals=2):
        """ Performs one Prediction.
        
        Parameters:
        - X: Input array compatible with model interface for inference.
        - decimals (int): Number of decimals required

        Returns: (int) Prediction result. 
        """

        prediction = self.model.predict(X)[0]
        if prediction < 0: prediction = 0
        return [round(prediction, decimals)]

def parse_args(args=[]):
    parser = argparse.ArgumentParser(
        description='ETL Preparation Pipeline for Forest Fires data.')

    default_data = 'lake/forest_fires/forestfires.csv'
    parser.add_argument('-s', '--source', type=str, default=default_data,
        dest='data', help=f'filepath for data source (default: {default_data})')
    
    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database',
        type=str, dest='database', default=default_db,
        help=f'SQLite database file to store result (default: {default_db})')

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
    df.to_sql("fires", engine, if_exists='replace')
    print("ETL - Done")
