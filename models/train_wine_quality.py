import sys
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from models.io import load_table, store_model, is_valid_model_filepath
from models.evaluators import evaluate_regression

def parse_args(args):
    if args == None: raise TypeError("An arguments list is required")

    model_filepath = args[0]
    if not is_valid_model_filepath(model_filepath):
        raise ValueError('Invalid model filepath {}'.format(model_filepath))
    
    parser = argparse.ArgumentParser(
        description='Train Wine Quality predictive model.')

    parser.add_argument('model',
        help='Output model name (e.g. models/regression.pkl)')

    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database', dest='database', default=default_db,
        help=f'SQLite database file to query data from (default: {default_db})')

    default_table = 'wines'
    parser.add_argument('-t', '--table', default=default_table,
        help=f'Database table to query (default: {default_table})')

    return vars(parser.parse_args(args))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
 
    print('≫ Loading data')
    df = load_table(args['database'], args['table'])

    print('≫ Model Specific Transformations')

    # Remove Outliers
    # Polynomial Data

    features = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid',
        'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
        'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
    ]
    X, y = df[features], df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    print('≫ Training Model')
    model_params = {
        "svm": {
            "model": SVC(gamma="auto"),
            "params": {
                'C': [1,10,20],
                'kernel': ["rbf"]
            }
        },
        
        "decision_tree": {
            "model": DecisionTreeClassifier(),
            "params": {
                'criterion': ["entropy","gini"],
                "max_depth": [5,8,9]
            }
        },
        
        "random_forest": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": [1,5,10],
                "max_depth": [5,8,9]
            }
        },

        "naive_bayes": {
            "model": GaussianNB(),
            "params": {}
        },
        
        'logistic_regression': {
            'model': LogisticRegression(solver='liblinear',multi_class = 'auto'),
            'params': {
                "C": [1,5,10]
            }
        }
    }

    scores = []
    models = {}
    for model_name, mp in model_params.items():
        clf = GridSearchCV(mp["model"], mp["params"], cv=6, return_train_score=False)
        clf.fit(X_train, y_train)
        scores.append({
            "model_name" : model_name,
            "best_score": clf.best_score_,
            "best_params": clf.best_params_
        })
        models[model_name] = clf

    scores_df = pd.DataFrame(scores, columns=['model_name',
                                              'best_score',
                                              'best_params'])
    scores_df.sort_values(by='best_score', ascending=False, inplace=True)
    print(scores_df)

    best_model_name = scores_df.iloc[0]['model_name']
    best_clf = models[best_model_name]

    model_filepath = args['model']
    print(f'≫ Storing "{best_model_name}" in "{model_filepath}"')
    store_model(best_clf, model_filepath)
