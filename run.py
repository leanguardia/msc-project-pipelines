from datetime import datetime

import numpy as np
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib

import pipelines
from pipelines.etl_forest_fires import ForestFireProcessor, ForestFirePredictor

app = Flask(__name__, template_folder='app/templates')
model = joblib.load("models/regrezz.pkl")

@app.route('/')
def index():
        return render_template('index.html')

@app.route('/forestfires')
def forest_fires():
    prediction = ''
    if prediction_is_required(request):
        prediction = _process_prediction(request.args)
        print("And the prediction is!", prediction)
    return render_template('forestfires.html', prediction=prediction)

def _process_prediction(args):
    args = _parse_forest_fire_params(args)
    processor = ForestFireProcessor()
    arry = processor.transform(args['X'], args['Y'], args['month'], args['day'],
        args['FFMC'], args['DMC'], args['DC'], args['ISI'],
        args['temp'], args['RH'], args['wind'], args['rain'])
    predictions = ForestFirePredictor(model).predict([arry])
    return str(predictions[0])

def _parse_forest_fire_params(args):
    return {
        'X': float(args['X']), 'Y': float(args['Y']),
        'month': args['month'], 'day': args['day'],
        'FFMC': float(args['FFMC']), 'DMC': float(args['DMC']),
        'DC': float(args['DC']), 'ISI': float(args['ISI']),
        'temp': float(args['temp']), 'RH': float(args['RH']),
        'wind': float(args['wind']), 'rain': float(args['rain']),
    }

def prediction_is_required(request):
    return len(request.args) > 0


if __name__ == "__main__":
    app.run(debug=True)
