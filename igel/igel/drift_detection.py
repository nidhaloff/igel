import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

def detect_drift(ref_df, new_df, categorical_features=None, alpha=0.05):
    """
    Detect data drift between reference and new datasets.
    Uses KS test for numerical features and Chi-Squared for categorical features.
    Returns a DataFrame with feature, test type, p-value, and drift flag.
    """
    results = []
    for col in ref_df.columns:
        if categorical_features and col in categorical_features:
            # Categorical: Chi-squared test
            contingency = pd.crosstab(ref_df[col], new_df[col])
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                stat, p, _, _ = chi2_contingency(contingency)
            else:
                stat, p = float('nan'), 1.0  # Not enough data for test
            test_type = "chi2"
        else:
            # Numerical: KS test
            stat, p = ks_2samp(ref_df[col].dropna(), new_df[col].dropna())
            test_type = "ks"
        results.append({
            "feature": col,
            "test": test_type,
            "p_value": p,
            "drift": p < alpha
        })
    return pd.DataFrame(results) 