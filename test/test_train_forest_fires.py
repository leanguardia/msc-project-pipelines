import pytest
from unittest import TestCase

from models.train_forest_fires import parse_args

class TestTrainForestFires(TestCase):
    def test_argparse_missing_model_name_error(self):
        with pytest.raises(TypeError, match='An arguments list is required'):
            parse_args(None)

    def test_argparse_model_name_without_extension(self):
        with pytest.raises(ValueError, match='Invalid model filepath'):
            parse_args(['model_name'])

    def test_argparse_short_model_name_error(self):
        with pytest.raises(ValueError):
            parse_args(['.pkl'])

    def test_argparse_model_name(self):
        args = parse_args(['model_name.pkl'])
        self.assertEqual(args['model'], 'model_name.pkl')

    def test_argparse_default_database(self):
        args = parse_args(['model_name.pkl'])
        self.assertEqual(args['database'], 'lake/warehouse.db')

    def test_argparse_database(self):
        args = parse_args(['model_name.pkl', '-d', 'other/database.db'])
        self.assertEqual(args['database'], 'other/database.db')
    
    def test_argparse_database_long(self):
        args = parse_args(['model_name.pkl', '--database', 'other/database.db'])
        self.assertEqual(args['database'], 'other/database.db')

    def test_argparse_default_table(self):
        args = parse_args(['model_name.pkl'])
        self.assertEqual(args['table'], 'forest_fires')

    def test_argparse_default_valid_table(self):
        args = parse_args(['model_name.pkl', '-t', 'other_table'])
        self.assertEqual(args['table'], 'other_table')

    def test_argparse_default_valid_table_long(self):
        args = parse_args(['model_name.pkl', '--table', 'other_table'])
        self.assertEqual(args['table'], 'other_table')
