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

def parse_abalone_params(args):
    return [
        # float(args['sex']),
        float(args['length']),
        float(args['diameter']),
        float(args['height']),
        float(args['whole_weight']),
        float(args['shucked_weight']),
        float(args['viscera_weight']),
        float(args['shell_weight']),
    ]
