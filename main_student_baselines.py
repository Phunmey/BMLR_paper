"""
Function to run bmlr model for the real data experiment.
"""

from student_preprocess import preprocess_student
from tqdm import tqdm
from common import append_files, close_files
from data_split import split_data
from baseline_fn import evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def main(file1, file2):
    data = 'Student'
    n_samples = '4424'
    scenarios = 'Student'
    df_num, df_y, class_names = preprocess_student()
    use_features = df_num.columns.values

    xtrain, xtest, ytrain, ytest, k_neighbors = split_data(df_num, df_y)
    models = {
        'LR': LogisticRegression(multi_class='multinomial', solver='lbfgs',),
        'SVM': SVC(),
        'RF': RandomForestClassifier(),
        'AdaB': AdaBoostClassifier(algorithm='SAMME')
    }

    for name, model in models.items():
        for i in tqdm(range(5)):
            for resampling_technique in ['Imbalanced', 'SMOTE', 'ADASYN', 'Tomek']:
                evaluate_model(file1, file2, resampling_technique, xtrain, ytrain, xtest, ytest, model, name, i,
                               data, n_samples, scenarios, k_neighbors)

if __name__ == "__main__":
    train_f = "./results/student_train_result.csv"
    test_f = "./results/student_test_result.csv"

    file1, file2 = append_files(train_f, test_f)

    try:
        main(file1, file2)
    finally:
        close_files(file1, file2)
