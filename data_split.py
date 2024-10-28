"""
Split the dataset.
"""


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def split_data(df_num, df_y):
    xtrain, xtest, ytrain, ytest = train_test_split(df_num, df_y, train_size=0.8, random_state=123,
                                                        stratify=df_y)

    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    print(f"xshape:{xtrain.shape}, yshape: {ytrain.shape}")
    print(f"Count of classes in the data: {np.unique(df_y, return_counts=True)}")
    print(f"Count of classes in ytrain: {np.unique(ytrain, return_counts=True)}")
    print(f"Count of classes in ytest: {np.unique(ytest, return_counts=True)}")

    minority_class_samples = pd.Series(ytrain).value_counts().min()
    k_neighbors = min(5, minority_class_samples - 1)

    return xtrain, xtest, ytrain, ytest, k_neighbors
