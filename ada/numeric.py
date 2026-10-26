"""Text read as numbers, safely.

Every place ADA turns cells into numbers goes through :func:`to_numbers`
instead of calling ``pd.to_numeric`` directly. pandas 2.x crashes the Python
process on text such as "689e8820798347950", which hashed identifiers produce
by chance: digits around an "e" read as an exponent too large for its parser.
Such an exponent is far beyond a float's range, so pandas gives no number for
it in any case; setting it to missing first gives the same result without the
crash.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# An exponent of ten or more significant digits: past 1e1,000,000,000, where the parser's count overflows.
_OVERSIZED_EXPONENT = re.compile(r"[eE][+-]?0*[1-9][0-9]{9,}")


def to_numbers(values):
    """``pd.to_numeric(values, errors="coerce")``, with text the parser cannot take set to missing first.

    Returns what ``pd.to_numeric`` returns for the same input: a Series for a
    Series, an Index for an Index, an array for an array or a list.
    """
    if isinstance(values, list | tuple):
        values = np.asarray(values, dtype=object)
    dtype = getattr(values, "dtype", None)
    if dtype is not None and (pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)):
        text = pd.Series(np.asarray(values, dtype=object), dtype=object).astype(str)
        oversized = text.str.contains(_OVERSIZED_EXPONENT).to_numpy(dtype=bool)
        if oversized.any():
            if isinstance(values, np.ndarray):
                values = np.where(oversized, None, values)
            else:
                values = values.where(~oversized)
    return pd.to_numeric(values, errors="coerce")
