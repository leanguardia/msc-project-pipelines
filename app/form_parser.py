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
        float(args['sulphates']),
        float(args['ph']),
        float(args['alcohol'])
    ]
