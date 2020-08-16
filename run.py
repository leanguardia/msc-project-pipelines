from datetime import datetime

import numpy as np
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib

import pipelines
from pipelines.etl_forest_fires import ForestFireProcessor

app = Flask(__name__, template_folder='app/templates')
model = joblib.load("models/regrezz.pkl")

@app.route('/')
def index():
        return render_template('index.html')

@app.route('/forestfires')
def forest_fires():
    prediction = None
    if get_prediction(request):
        args = _parse_forest_fire_params(request.args)
        processor = ForestFireProcessor()
        array = processor.transform(X=1, Y=2, month= 'jun', day='sat',
            FFMC= 20.0, DMC= 100.0, DC=400, ISI=35,
            temp=33.2, RH=60, wind=5, rain=4); print(array)
        prediction = model.predict([array])
        print("And the prediction is!", prediction)
    return render_template('forestfires.html', prediction=prediction)

def _parse_forest_fire_params(args):
    params = {
        'X': int(args['X']),
        'Y': int(args['Y']),
        'month': args['month'],
        'day': args['day'],
        'temp': int(args['temp']),
        'RH': int(args['RH']),
        'wind': int(args['wind']),
        'FFMC': int(args['FFMC']),
        'DMC': int(args['DMC']),
        'DC': int(args['DC']),
        'ISI': int(args['ISI']),
    }
    return params

def _parse_forest_fire_array(args):
    x = np.array([args['X'], args['Y'],
                  args['FFMC'], args['DMC'], args['DC'], args['ISI'],
                  args['temp'], args['RH'], args['wind'], args['rain']])
    return x

def get_prediction(request):
    return len(request.args) > 0


if __name__ == "__main__":
    app.run(debug=True)
