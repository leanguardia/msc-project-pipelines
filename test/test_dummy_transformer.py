import pytest

import pandas as pd
import numpy as np

from pipelines.dummy_transformer import dummify

df = pd.DataFrame(['a', 'b', 'a'], columns=['category'])

def test_raise_datatype_error():
    with pytest.raises(TypeError, match="df must be a DataFrame"):
        dummify(None, 'category')

def test_new_number_of_columns():
    dummified = dummify(df, 'category')
    assert len(dummified.columns) == 3, "New columns should be concatenated"

def test_includes_columns():
    dummified_cols = dummify(df, 'category', prefix='category')
    assert 'category_a' in dummified_cols, "New column should be added"
    assert 'category_b' in dummified_cols, "New column should be added"
