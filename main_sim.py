"""
Function to run the bmlr model for simulation experiment.
"""


import warnings
import pandas as pd
import numpy as np

from data_split import split_data
from bmlr_fn_sim import bmlr
from baseline_fn import evaluate_model
from common import initialize_files, close_files, append_files
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


warnings.filterwarnings('ignore')


def main(file1, file2):
    datapath = "./simulated_data"
    n_class = ['3', '5', '10']
    n_samples = ['2000', '5000', '10000']
    scenarios = ['1', '2', '3']
    i = '0'

    models = {
        'SVC': SVC(),
        'RF': RandomForestClassifier(),
        'AdaB': AdaBoostClassifier(algorithm='SAMME')
    }

    for classes in n_class:
        for sample in n_samples:
            for sce in scenarios:
                read_data = pd.read_csv(f"{datapath}/sim_class{classes}_{sample}samples_{sce}.csv")
                df_num = read_data.iloc[:, :-1]
                df_y = read_data.iloc[:, -1]
                use_features = df_num.columns.tolist()
                class_names = np.unique(df_y).tolist()

                for resampling_technique in ['Imbalanced', 'SMOTE', 'ADASYN', 'Tomek']:
                    xtrain, xtest, ytrain, ytest, k_neighbors = split_data(df_num, df_y, resampling_technique)
                    bmlr(file1, file2, use_features, xtrain, ytrain, xtest, ytest, class_names,
                         resampling_technique, classes, sample, sce, i)
                    for name, model in models.items():
                        for j in tqdm(range(5)):
                            evaluate_model(file1, file2, resampling_technique, xtrain, ytrain, xtest, ytest, model,
                                           name, j, classes, sample, sce)


if __name__ == "__main__":
    train_f = "./results/sim_train_result3.csv"
    test_f = "./results/sim_test_result3.csv"

    file1, file2 = initialize_files(train_f, test_f)

    try:
        main(file1, file2)
    finally:
        close_files(file1, file2)



