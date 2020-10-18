import pytest
from unittest import TestCase

from pipelines.validators import ValidationsRunner, RangeValidator, CategoryValidator

range_validator = RangeValidator('col', 0, 10)
category_validator = CategoryValidator('col', ['bad', 'regular', 'good'])

df = 'dataframe_mock'

class TestValidationsRunner(TestCase):
    def setUp(self):
        self.runner = ValidationsRunner()

    def test_append_non_list_raises_an_error(self):
        with pytest.raises(TypeError, match='Parameter should be list of Validators'):
            self.runner.add_validator(3)

    def test_append_non_list_of_validators_raises_an_error(self):
        with pytest.raises(TypeError, match='Parameter should be list of Validators'):
            self.runner.add_validator([3])

    def test_validate(self):
        self.runner.validate(df)

