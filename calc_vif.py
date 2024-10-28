"""
In the case of multicollinearity, this function computes VIF and iteratively drop columns wiht high VIF base on the
threshold set. 5 and 10 are acceptable based on the literature.
"""

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


def vif_func(X):
    # compute VIF for each feature
    vif = pd.DataFrame()
    vif["Feature"] = X.columns

    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Order VIF values from highest to lowest
    vif_sorted = vif.sort_values(by="VIF", ascending=False)

    return vif_sorted


# Iterative process to remove features with high VIF
def iterative_vif_selection(X, threshold=5.0):
    vif = vif_func(X)
    print(f"Initial VIF:\n{vif}\n")

    while vif['VIF'].max() > threshold:
        high_vif_feature = vif.loc[vif['VIF'].idxmax(), 'Feature']
        print(f"Dropping '{high_vif_feature}' with VIF: {vif['VIF'].max()}")
        X = X.drop(columns=high_vif_feature)
        vif = vif_func(X)
        print(f"Updated VIF:\n{vif}\n")

    return X
