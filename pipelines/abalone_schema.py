from pipelines.schema import Schema

abalone_features_meta = [
    dict(name='sex', dtype=str, type='input',
        categories=['M','F','I']),
    dict(name='length', dtype=float, type='input', positive=True),
    dict(name='diameter', dtype=float, type='input', positive=True),
    dict(name='height', dtype=float, type='input', positive=True),
    dict(name='whole_weight', dtype=float, type='input', positive=True),
    dict(name='shucked_weight', dtype=float, type='input', positive=True),
    dict(name='viscera_weight', dtype=float, type='input', positive=True),
    dict(name='shell_weight', dtype=float, type='input', positive=True),
    dict(name='rings', dtype=int, type='target', positive=True, range=(1,30)),
]

abalone_schema = Schema(abalone_features_meta)
