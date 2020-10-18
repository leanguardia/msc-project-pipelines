import pytest
from unittest import TestCase

from pipelines.validators import ValidationsRunner


class TestValidationsRunner(TestCase):
    def setUp(self):
        self.runner = ValidationsRunner()

    def test_append_non_validator_raises_an_error(self):
        with pytest.raises(TypeError, match='Parameter should be a Validator'):
            self.runner.add_validator(3)
