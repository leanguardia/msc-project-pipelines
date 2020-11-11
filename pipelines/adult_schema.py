from pipelines.schema import Schema

adult_features_meta = [ 
    dict(name='age', type='input', dtype=int, positive=True),
    dict(name='workclass', type='input', dtype=str),
        # ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
        #    'Local-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked']
    dict(name='fnlwgt', type='input', dtype=int),
    dict(name='education', type='input', dtype=str),
    # 'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
    #    'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
    #    '5th-6th', '10th', '1st-4th', 'Preschool', '12th'
    dict(name='education_num', type='input', dtype=int),
    # [13,  9,  7, 14,  5, 10, 12, 11,  4, 16, 15,  3,  6,  2,  1,  8]
    dict(name='marital_status', type='input', dtype=str),
    # 'Never-married', 'Married-civ-spouse', 'Divorced',
    #    'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
    #    'Widowed'
    dict(name='occupation', type='input', dtype=str),
    # ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
    #    'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair',
    #    'Transport-moving', 'Farming-fishing', 'Machine-op-inspct',
    #    'Tech-support', 'Protective-serv', 'Armed-Forces',
    #    'Priv-house-serv']
    dict(name='relationship', type='input', dtype=str),
    # ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried',
    #    'Other-relative']
    dict(name='race', type='input', dtype=str),
    # ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
    #    'Other']
    dict(name='sex', type='input', dtype=str, categories=['Male', 'Female']),
    dict(name='capital_gain', type='input', dtype=int),
    dict(name='capital_loss', type='input', dtype=int),
    dict(name='hours_per_week', type='input', dtype=int, range=(1,99)),
    dict(name='native_country', type='input', dtype=str),
    # ['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
    #    'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany',
    #    'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia',
    #    'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
    #    'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
    #    'China', 'Japan', 'Yugoslavia', 'Peru',
    #    'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
    #    'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
    #    'Holand-Netherlands']
    dict(name='>50K<=50K', type='target', dtype=str),
        # ['<=50K', '>50K']
    dict(name='for_training', type='input', dtype=bool),
    # dict(name='>50K', type='target', dtype=bool),
]

adult_schema = Schema(adult_features_meta)
