import pytest
import pandas as pd
import numpy as np
from unittest import TestCase

from pipelines.transformation import dummify

df = pd.DataFrame(['a', 'b', 'a'], columns=['category'])

class TestDummifier(TestCase):
    def test_raise_datatype_error(self):
        with pytest.raises(TypeError, match="df must be a DataFrame"):
            dummify(None, 'category', categories=['a', 'b'])

    def test_new_number_of_columns(self):
        dummified = dummify(df, 'category', categories=['a', 'b'])
        assert len(dummified.columns) == 3, "New columns should be concatenated"

    def test_includes_present_columns_explicitly(self):
        dummified = dummify(df, 'category', categories=['a', 'b'])
        assert 'a' in dummified, "New column should be added"
        assert 'b' in dummified, "New column should be added"

    def test_includes_columns_explicitly(self):
        dummified = dummify(df, 'category', categories=['a', 'b', 'c'])
        assert 'a' in dummified, "New column should be added"
        assert 'b' in dummified, "New column should be added"
        assert 'c' in dummified, "New column should be added"
