from unittest import TestCase

from pipelines.adult_etl import parse_args

class TestAdultEtl(TestCase):
    def test_argparser_default_data_source(self):
        args = parse_args()
        self.assertEqual(args['data'], 'lake/adult/adult_full.csv')

    def test_argparser_valid_data_source(self):
        args = parse_args(['-i', 'specific/path/to/data.csv'])
        self.assertEqual(args['data'], 'specific/path/to/data.csv')\

    def test_argparser_valid_data_source_long(self):
        args = parse_args(['--input', 'specific/path/to/data.csv'])
        self.assertEqual(args['data'], 'specific/path/to/data.csv')\

    def test_argparser_default_output_database(self):
        args = parse_args()
        self.assertEqual(args['database'], 'lake/warehouse.db')

    def test_argparser_valid_database(self):
        args = parse_args(['-d', 'specific/path/to/database.db'])
        self.assertEqual(args['database'], 'specific/path/to/database.db')

    def test_argparser_valid_database_long(self):
        args = parse_args(['--database', 'specific/path/to/database.db'])
        self.assertEqual(args['database'], 'specific/path/to/database.db')

    def test_argparser_default_db_table(self):
        args = parse_args()
        self.assertEqual(args['table_name'], 'adults')

    def test_argparser_valid_db_table(self):
        args = parse_args(['-t', 'other_table'])
        self.assertEqual(args['table_name'], 'other_table')

    def test_argparser_valid_db_table_long(self):
        args = parse_args(['--table', 'other_table'])
        self.assertEqual(args['table_name'], 'other_table')
    
    def test_argparser_default_table_overwrite(self):
        args = parse_args()
        self.assertFalse(args['table_overwrite'])

    def test_argparser_default_valid_table_overwrite(self):
        args = parse_args(['-o', 'True'])
        self.assertTrue(args['table_overwrite'])

    def test_argparser_default_valid_table_overwrite_long(self):
        args = parse_args(['--overwrite', 'True'])
        self.assertTrue(args['table_overwrite'])
