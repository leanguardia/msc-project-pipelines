from pipelines.schema import Schema

sample_features_metadata = [
    dict(name = 'feature1',
        dtype = int,
        type = 'input',
        range=(1, 10)
    ),
    dict(name = 'feature2',
        dtype = float,
        type = 'input',
        range = (0.0, 99.9)
    ),
    dict(name = 'feature3',
        dtype = str,
        type = 'input',
    ),
    dict(name = 'feature4',
        dtype = str,
        type = 'input',
        categories = ['a', 'e', 'i', 'o', 'u']
    ),
    dict(name = 'target',
        dtype = str,
        type = 'target'
    ),
    dict(name = 'feature1_log',
        dtype = str,
        non_negative = True
    )
]

sample_schema = Schema(sample_features_metadata)
