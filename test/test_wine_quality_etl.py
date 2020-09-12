from pipelines.wine_quality_etl import parse_args

class TestWineQualityETL:
    def test_argparser_default_data_source(self):
        args = parse_args()
        assert args['data'] == 'lake/wine_quality/winequality-white.csv'

    def test_argparser_valid_data_source(self):
        args = parse_args(['-s', 'specific/path/to/data.csv'])
        assert args['data'] == 'specific/path/to/data.csv'

    def test_argparser_valid_data_source_long(self):
        args = parse_args(['--source', 'specific/path/to/data.csv'])
        assert args['data'] == 'specific/path/to/data.csv'

    def test_argparser_default_output_database(self):
        args = parse_args()
        assert args['database'] == 'lake/warehouse.db'

    def test_argparser_valid_database(self):
        args = parse_args(['-d', 'specific/path/to/database.db'])
        assert args['database'] == 'specific/path/to/database.db'

    def test_argparser_valid_database_long(self):
        args = parse_args(['--database', 'specific/path/to/database.db'])
        assert args['database'] == 'specific/path/to/database.db'
