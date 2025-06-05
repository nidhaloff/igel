import pandas as pd
import numpy as np
from igel.preprocessing import handle_missing_values

def test_handle_missing_values():
    # Create a sample DataFrame with missing values
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [4, 5, np.nan]
    })

    # Test with default strategy (mean)
    result = handle_missing_values(df)
    assert not result.isna().any().any(), "Missing values should be handled"

    # Test with strategy='drop'
    result_drop = handle_missing_values(df, strategy='drop')
    assert len(result_drop) == 1, "Dropping missing values should leave one row"

    # Test with custom fill_value
    result_custom = handle_missing_values(df, fill_value=0)
    assert result_custom.isna().sum().sum() == 0, "Custom fill value should replace NaNs"
