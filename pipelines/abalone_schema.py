from pipelines.schema import Schema

abalone_features_meta = [
    dict(name='sex', dtype=str, type='input',
        categories=['M','F','I']),
    dict(name='length', dtype=float, type='input'),
    dict(name='diameter', dtype=float, type='input'),
    dict(name='height', dtype=float, type='input'),
    dict(name='whole_weight', dtype=float, type='input'),
    dict(name='shucked_weight', dtype=float, type='input'),
    dict(name='viscera_weight', dtype=float, type='input'),
    dict(name='shell_weight', dtype=float, type='input'),
    dict(name='rings', dtype=float, type='target'),
]

abalone_schema = Schema(abalone_features_meta)
