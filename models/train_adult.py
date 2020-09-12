import sys
import argparse

# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from models.io import load_table, store_model, is_valid_model_filepath
from models.evaluators import evaluate_classification

def load_data(database, table):
    df = load_table(database, table)
    features = ['age', 'fnlwgt', 'education_num', 'capital_gain',
                'capital_loss', 'hours_per_week']
    X = df[features]
    y = df['>50K']
    return X, y


def parse_args(args):
    if args == None: raise TypeError('An arguments list is required')

    model_filepath = args[0]
    if not is_valid_model_filepath(model_filepath):
        raise ValueError('Invalid model filepath {}'.format(model_filepath))

    parser = argparse.ArgumentParser(
        description='Train Adult income predictive model.')

    parser.add_argument('model',
        help='Output model name (e.g. models/classifier.pkl)')

    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database', dest='database', default=default_db,
        help=f'SQLite database file to query data from (default: {default_db})')

    default_table = 'adults'
    parser.add_argument('-t', '--table', default=default_table,
        help=f'Database table to query (default: {default_table})')

    return vars(parser.parse_args(args))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
 
    print('≫ Loading data')
    X, y = load_data(args['database'], args['table'])

    # print('≫ Feature Engineering')

    # Handle Categorical Data
    # Remove Outliers
    # Polynomial Data

    print('≫ Training Model')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=25)

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    evaluate_classification(y_test, y_pred)

    model_filepath = args['model']
    print(f'≫ Storing Model "{model_filepath}"')
    store_model(logreg, model_filepath)
