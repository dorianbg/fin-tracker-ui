import pandas as pd

from pipeline.executor import repair_invalid_ohlc


def test_repair_invalid_ohlc_replaces_zero_fields_with_close():
    df = pd.DataFrame(
        {
            "open": [0.0, 10.0],
            "high": [0.0, 11.0],
            "low": [0.0, 9.0],
            "close": [42.0, 10.5],
        }
    )

    repaired = repair_invalid_ohlc(df)

    assert repaired.loc[0, "open"] == 42.0
    assert repaired.loc[0, "high"] == 42.0
    assert repaired.loc[0, "low"] == 42.0
    assert repaired.loc[1, "open"] == 10.0
