from pipelines.schema import Schema

abalone_features_meta = [
    dict(name='sex', dtype='float', type='input',
        categories=['M','F','I']),
    dict(name='length', dtype='float', type='input'),
    dict(name='diameter', dtype='float', type='input'),
    dict(name='height', dtype='float', type='input'),
    dict(name='whole', dtype='float', type='input'),
    dict(name='shucked', dtype='float', type='input'),
    dict(name='viscera', dtype='float', type='input'),
    dict(name='shell', dtype='float', type='input'),
    dict(name='rings', dtype='float', type='target'),
]

abalone_schema = Schema(abalone_features_meta)
