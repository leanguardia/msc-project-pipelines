from pipelines.schema import Schema

wine_features_meta = [ 
    dict(name='fixed_acidity', dtype=float, type='input'),
    dict(name='volatile_acidity', dtype=float, type='input'), 
    dict(name='citric_acid', dtype=float, type='input'), 
    dict(name='residual_sugar', dtype=float, type='input'), 
    dict(name='chlorides', dtype=float, type='input'), 
    dict(name='free_sulfur_dioxide', dtype=float, type='input'), 
    dict(name='total_sulfur_dioxide', dtype=float, type='input'), 
    dict(name='density', dtype=float, type='input'), 
    dict(name='pH', dtype=float, type='input'), 
    dict(name='sulphates', dtype=float, type='input'), 
    dict(name='alcohol', dtype=float, type='input'),
    dict(name='type', dtype=str, type='input'), 
    dict(name='quality', dtype=int, type='target', range=(0,10)),
]

wines_schema = Schema(wine_features_meta)
