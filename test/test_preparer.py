from unittest import TestCase

from pipelines.preparer import Preparer
from fixtures.sample_schema import sample_schema

class TestPreparer(TestCase):
    def setUp(self):
        self.preparer = Preparer()
    
    def test_anything(self):
        self.assertTrue(True)
