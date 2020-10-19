import pytest
import pandas as pd
from unittest import TestCase

from pipelines.validators import ValidationsRunner, RangeValidator, CategoryValidator

range_validator = RangeValidator('rank', 0, 10)
category_validator = CategoryValidator('cat', ['bad', 'regular', 'good'])

df = pd.DataFrame({'rank': [1,2,3], 'cat':['bad', 'bad', 'regular']})

class TestValidationsRunner(TestCase):
    def setUp(self):
        self.runner = ValidationsRunner()

    def test_append_non_list_raises_an_error(self):
        with pytest.raises(TypeError, match='Parameter should be list of Validators'):
            self.runner.add_validators(3)

    def test_append_non_list_of_validators_raises_an_error(self):
        with pytest.raises(TypeError, match='Parameter should be list of Validators'):
            self.runner.add_validators([3])

    def test_validate(self):
        self.runner.add_validators([range_validator, category_validator])
        self.runner.validate(df)

