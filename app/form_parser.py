def parse_forest_fire_params(args):
    return [
        int(args['X']), int(args['Y']),
        args['month'], args['day'],
        float(args['FFMC']), float(args['DMC']),
        float(args['DC']), float(args['ISI']),
        float(args['temp']), float(args['RH']),
        float(args['wind']), float(args['rain']),
    ]

def parse_abalone_params(args):
    return [
        args['sex'],
        float(args['length']),
        float(args['diameter']),
        float(args['height']),
        float(args['whole_weight']),
        float(args['shucked_weight']),
        float(args['viscera_weight']),
        float(args['shell_weight']),
    ]

def parse_wine_quality_params(args):
    return [
        float(args['fixed_acidity']),
        float(args['volatile_acidity']),
        float(args['citric_acid']),
        float(args['residual_sugar']),
        float(args['chlorides']),
        float(args['free_sulfur_dioxide']),
        float(args['total_sulfur_dioxide']),
        float(args['density']),
        float(args['ph']),
        float(args['sulphates']),
        float(args['alcohol'])
    ]

def parse_adult_params(args):
    return [
        int(args['age']),
        args['workclass'],
        int(args['final_weight']),
        args['education'],
        9, # Education_num TODO: Remove this, Add Categorical
        args['marital_status'],
        args['occupation'],
        args['relationship'],
        args['race'],
        args['sex'],
        int(args['capital_gain']),
        int(args['capital_loss']),
        int(args['hours_per_week']),
        args['native_country'],
    ]
