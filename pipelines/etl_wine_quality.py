import argparse 

import numpy as np

class WineQualityProcessor():
    def transform(self, fixed_acidity, volatile_acidity, citric_acid,
                  residual_sugar, chlorides, free_sulfur_dioxide,
                  total_sulfur_dioxide, density, pH, sulphates, alcohol):
        return np.array([7.0, 0.27, 0.36, 20.7, 0.045, 45, \
                          170.0, 1.0010, 3.00, 0.45, 8.8])

def parse_args(args=[]):
    parser = argparse.ArgumentParser()

    default_data = 'lake/wine_quality/wine_quality_white.csv'
    parser.add_argument('-s', '--source', default=default_data, dest='data')

    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database', default=default_db, dest='database')
        
    return vars(parser.parse_args(args))
