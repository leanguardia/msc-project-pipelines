# load table

    # test presence
    # test 

from unittest import TestCase

import pytest
from models.io import is_valid_model_filepath

class TestModelsIo(TestCase):
    def test_invalid_model_filepath_type(self):
        with pytest.raises(TypeError, match='should be a string'):
            is_valid_model_filepath(2)

    def test_invalid_model_filepath_length(self):
        with pytest.raises(ValueError, match='should end with specific extension'):
            is_valid_model_filepath('.pkl')

    def test_invalid_model_filepath_extension(self):
        self.assertFalse(is_valid_model_filepath('model_filepath.whateva'))

    def test_valid_model_filepath(self):
        self.assertTrue(is_valid_model_filepath('model_filepath.pkl'))
