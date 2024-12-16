"""
Function to run the BMLR model for real student data.
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


def run_exploratory_analysis(df_x, df_y):
    """
    Perform exploratory analysis and VIF-based reduction.
    """
    exploratory_analysis(df_x, df_y)
    df_new = obtain_vif_data(df_x)
    vif_exploratory_analysis(df_new, df_y)
    return df_new


def train_and_evaluate(file1, file2, df_x, df_y, use_features, class_names, models):
    """
    Train and evaluate models with different resampling techniques.
    """
    data, n_samples, scenarios, i = "Student", "4424", "NA", "0"

    for resampling_technique in ['Imbalanced', 'SMOTE', 'ADASYN', 'Tomek']:
        xtrain, xtest, ytrain, ytest, k_neighbors = split_data(df_x, df_y, resampling_technique)

        # Train BMLR
        bmlr(file1, file2, use_features, xtrain, ytrain, xtest, ytest, class_names,
             resampling_technique, data, n_samples, scenarios, i)

        # Train and evaluate other models
        for name, model in models.items():
            for j in tqdm(range(5), desc=f"Evaluating {name}"):
                evaluate_model(file1, file2, resampling_technique, xtrain, ytrain, xtest, ytest,
                               model, name, j, data, n_samples, scenarios)


def main(file1, file2):
    """
    Main function to preprocess data, run exploratory analysis, and train models.
    """
    df_x, df_y, class_names = read_data()
    print(f"Class names: {class_names}")

    df_new = run_exploratory_analysis(df_x, df_y)
    use_features = df_new.columns.values

    models = {
        'SVC': SVC(),
        'RF': RandomForestClassifier(),
        'AdaB': AdaBoostClassifier(algorithm='SAMME')
    }

    train_and_evaluate(file1, file2, df_new, df_y, use_features, class_names, models)


if __name__ == "__main__":
    train_f, test_f = "./student_train_result.csv", "./student_test_result.csv"
    file1, file2 = initialize_files(train_f, test_f)

    try:
        main(file1, file2)
    finally:
        close_files(file1, file2)
