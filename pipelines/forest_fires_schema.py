from pipelines.schema import Schema

forest_fires_features_meta = [
    dict(name='X', dtype=int, type='input', range=(1,9)),
    dict(name='Y', dtype=int, type='input', range=(1,9)),
    dict(name='month', dtype=str, type='input', 
        categories=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']),
    dict(name='day', dtype=str, type='input',
        categories=['mon','tue','wed','thu','fri','sat','sun']),
    dict(name='FFMC', dtype=float, type='input'),
    dict(name='DMC', dtype=float, type='input'),
    dict(name='DC', dtype=float, type='input'),
    dict(name='ISI', dtype=float, type='input'),
    dict(name='temp', dtype=float, type='input', range=(2.2,33.3)),
    dict(name='RH', dtype=float, type='input', range=(0,100)),
    dict(name='wind', dtype=float, type='input', range=(0.40, 9.40)),
    dict(name='rain', dtype=float, type='input', range=(0.0,6.4)),
    dict(name='area', dtype=float, type='target'),
    # dict(name='area_log', type='target', dtype=float),
    dict(name='FFMC_log', dtype=float, non_negative=True),
    dict(name='ISI_log', dtype=float, non_negative=True),
    dict(name='rain_log', dtype=float, non_negative=True),
    dict(name='rain_cat', dtype=float),
]

forest_fires_schema = Schema(forest_fires_features_meta)
