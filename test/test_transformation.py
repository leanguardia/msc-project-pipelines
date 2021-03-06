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

    def test_dummy_values(self):
        dummified = dummify(df, 'category', categories=['a', 'b'])
        dummy_values = [['a',1,0],
                        ['b',0,1],
                        ['a',1,0]]
        assert np.array_equal(dummified.values.tolist(), dummy_values)

    def test_includes_present_columns_explicitly(self):
        dummified = dummify(df, 'category', categories=['a', 'b'])
        assert 'a' in dummified, "New column should be added"
        assert 'b' in dummified, "New column should be added"

    def test_includes_columns_explicitly(self):
        dummified = dummify(df, 'category', categories=['a', 'b', 'c'])
        assert 'a' in dummified, "New column should be added"
        assert 'b' in dummified, "New column should be added"
        assert 'c' in dummified, "New column should be added"

    def test_includes_dummy_na(self):
        df = pd.DataFrame(['a', 'b', 'a', None], columns=['category'])
        dummified = dummify(df, 'category', categories=['a', 'b', 'c'], dummy_na=True)
        assert 'a' in dummified, "New column should be added"
        assert 'b' in dummified, "New column should be added"
        assert 'c' in dummified, "New column should be added"
        assert None in dummified, "New column should be added"
