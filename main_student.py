"""
Function to run the bmlr model for real data experiment.
"""

import warnings

from student_preprocess import *
from data_split import split_data
from bmlr_fn_student import bmlr
from common import initialize_files, close_files
from tqdm import tqdm

warnings.filterwarnings('ignore')


def main(file1, file2):
    data = 'Student'
    n_samples = '4424'
    scenarios = 'NA'
    df_num, df_y, class_names = preprocess_student()
    exploratory_analysis(df_num, df_y)

    xtrain, xtest, ytrain, ytest, k_neighbors = split_data(df_num, df_y)
    use_features = df_num.columns.values

    for i in tqdm(range(1)):
        for resampling_technique in ['Imbalanced', 'SMOTE', 'ADASYN', 'Tomek']:
            bmlr(file1, file2, use_features, xtrain, ytrain, xtest, ytest, i, class_names,
                 resampling_technique, data, n_samples, scenarios, k_neighbors)


if __name__ == "__main__":
    train_f = "./results/student_train_result.csv"
    test_f = "./results/student_test_result.csv"

    file1, file2 = initialize_files(train_f, test_f)

    try:
        main(file1, file2)
    finally:
        close_files(file1, file2)

