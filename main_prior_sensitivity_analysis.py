import warnings
import pandas as pd
import numpy as np

from data_split import split_data
from bmlr_fn_prior_sensitivity import bmlr
from prior_sensitivity_result_header import initialize_files, close_files


warnings.filterwarnings('ignore')


def main(file1, file2):
    datapath = "./simulated_data"
    n_class = ['3', '5']
    n_samples = ['100', '200', '500', '1000', '2000', '5000', '10000']
    scenarios = ['Moderate', 'Extreme', 'EqualMinorities']
    sigmas = [0.01, 0.5, 1, 5, 10, 100, 1000]

    for classes in n_class:
        for sample in n_samples:
            for sce in scenarios:
                read_data = pd.read_csv(f"{datapath}/sim_{classes}classes_{sample}samples_{sce}.csv")
                df_num = read_data.iloc[:, :-1]
                df_y = read_data.iloc[:, -1]
                use_features = df_num.columns.tolist()
                class_names = np.unique(df_y).tolist()[1:]

                for resampling_technique in ['Imbalanced']:
                    xtrain, xtest, ytrain, ytest, k_neighbors = split_data(df_num, df_y, resampling_technique)
                    for sigma in sigmas:
                        bmlr(file1, file2, use_features, xtrain, ytrain, xtest, ytest, class_names,
                             classes, sample, sce, sigma)


if __name__ == "__main__":
    train_f = "./prior_sensitivity_results/sim_train_results.csv"
    test_f = "./prior_sensitivity_results/sim_test_results.csv"

    file1, file2 = initialize_files(train_f, test_f)

    try:
        main(file1, file2)
    finally:
        close_files(file1, file2)

