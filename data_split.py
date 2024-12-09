import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks


def split_data(df_num, df_y, resampling_technique):
    xtrain, xtest, ytrain, ytest = train_test_split(df_num, df_y, train_size=0.7, random_state=123,
                                                        stratify=df_y)

    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    minority_class_samples = pd.Series(ytrain).value_counts().min()
    k_neighbors = min(5, minority_class_samples - 1)

    if resampling_technique == 'SMOTE':
        oversample = SMOTE(k_neighbors=k_neighbors, random_state=123)
        xtrain, ytrain = oversample.fit_resample(xtrain.copy(), ytrain.copy())
    elif resampling_technique == 'ADASYN':
        oversample = ADASYN(n_neighbors=k_neighbors, sampling_strategy='minority', random_state=123)
        xtrain, ytrain = oversample.fit_resample(xtrain.copy(), ytrain.copy())
    elif resampling_technique == 'Tomek':
        undersample = TomekLinks()
        xtrain, ytrain = undersample.fit_resample(xtrain.copy(), ytrain.copy())

    print(f"Resampling technique is: {resampling_technique}\n")
    print(f"xtrain_shape:{xtrain.shape}, ytrain_shape: {ytrain.shape}\n")
    print(f"Count of classes in the data: {np.unique(df_y, return_counts=True)}\n")
    print(f"Count of classes in ytrain: {np.unique(ytrain, return_counts=True)}\n")
    print(f"Count of classes in ytest: {np.unique(ytest, return_counts=True)}\n")

    return xtrain, xtest, ytrain, ytest, k_neighbors
