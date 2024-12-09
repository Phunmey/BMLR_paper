"""
Function to run the bmlr model for real student data.
"""

import warnings

from student_preprocess import *
from data_split import split_data
from bmlr_fn_student import bmlr
from student_header import initialize_files, close_files
from baseline_fn import evaluate_model
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


warnings.filterwarnings('ignore')


def main(file1, file2):
    data = 'Student'
    n_samples = '4424'
    scenarios = 'NA'
    i = '0'

    df_num, df_y, class_names = read_data()
    print(f"class_names are: {class_names}")
    exploratory_analysis(df_num, df_y)
    # perform VIF
    df_new = obtain_vif_data(df_num)
    vif_exploratory_analysis(df_new, df_y)

    use_features = df_new.columns.values

    models = {
        'SVC': SVC(),
        'RF': RandomForestClassifier(),
        'AdaB': AdaBoostClassifier(algorithm='SAMME')
    }

    for resampling_technique in ['Imbalanced', 'SMOTE', 'ADASYN', 'Tomek']:
        xtrain, xtest, ytrain, ytest, k_neighbors = split_data(df_new, df_y, resampling_technique)
        bmlr(file1, file2, use_features, xtrain, ytrain, xtest, ytest, class_names, resampling_technique,
             data, n_samples, scenarios, i)
        for name, model in models.items():
            for j in tqdm(range(5)):
                evaluate_model(file1, file2, resampling_technique, xtrain, ytrain, xtest, ytest, model, name, j,
                               data, n_samples, scenarios)


if __name__ == "__main__":
    train_f = "./results/student_train_result.csv"
    test_f = "./results/student_test_result.csv"

    file1, file2 = initialize_files(train_f, test_f)

    try:
        main(file1, file2)
    finally:
        close_files(file1, file2)

