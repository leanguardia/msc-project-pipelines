import pytest
import unittest
from models.train_wine_quality import parse_args

class TestTrainWineQuality(unittest.TestCase):
    def test_argparse_missing_model_name_error(self):
        with pytest.raises(TypeError, match='An arguments list is required'):
            parse_args(None)

    def test_argparse_model_name_without_extension(self):
        with pytest.raises(ValueError):
            parse_args(['model_name'])

# class TestTrainWineQuality(unittest.TestCase):
#     def test_argparse_default_(self):
#         self.assertEqual(parse_args()['database'], 'lake/warehouse.db')