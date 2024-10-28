"""
Function to run baseline models for the simulation experiment.
"""

import pandas as pd

from tqdm import tqdm
from common import append_files, close_files
from data_split import split_data
from baseline_fn import evaluate_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def main(file1, file2):
    datapath = "./simulated_data"
    models = {
        'SVM': SVC(),
        'RF': RandomForestClassifier(),
        'AdaB': AdaBoostClassifier(algorithm='SAMME')
    }

    n_class = ['3', '5', '10']
    n_samples = ['2000', '5000', '10000']
    scenarios = ['1', '2', '3']
    for classes in n_class:
        for sample in n_samples:
            for sce in scenarios:
                read_data = pd.read_csv(f"{datapath}/sim_class{classes}_{sample}samples_{sce}.csv")
                df_num = read_data.iloc[:, :-1]
                df_y = read_data.iloc[:, -1]
                xtrain, xtest, ytrain, ytest, class_names, k_neighbors = split_data(df_num, df_y)
                for name, model in models.items():
                    for i in tqdm(range(5)):
                        for resampling_technique in ['Imbalanced', 'SMOTE', 'ADASYN', 'Tomek']:
                            evaluate_model(file1, file2, resampling_technique, xtrain, ytrain, xtest, ytest, model,
                                           name, i, classes, sample, sce, k_neighbors)


if __name__ == "__main__":
    train_f = "./results/sim_train_result.csv"
    test_f = "./results/sim_test_result.csv"

    file1, file2 = append_files(train_f, test_f)

    try:
        main(file1, file2)
    finally:
        close_files(file1, file2)
