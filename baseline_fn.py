"""
Code to fit the baseline models
"""


from time import time
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def resample_data(method, xtrain, ytrain, k_neighbors):
    if method == 'SMOTE':
        sampler = SMOTE(k_neighbors=k_neighbors, random_state=123)
    elif method == 'ADASYN':
        sampler = ADASYN(n_neighbors=k_neighbors, sampling_strategy='minority', random_state=123)
    elif method == 'Tomek':
        sampler = TomekLinks()
    else:
        return xtrain, ytrain
    x_resampled, y_resampled = sampler.fit_resample(xtrain.copy(), ytrain.copy())

    return x_resampled, y_resampled


def evaluate_model(file1, file2, method, xtrain, ytrain, xtest, ytest, model, name, i, classes, sample, sce,
                   k_neighbors=None):
    x_resampled, y_resampled = resample_data(method, xtrain, ytrain, k_neighbors)
    print(f"xshape:{x_resampled.shape}, yshape: {y_resampled.shape}")

    start1 = time()

    model.fit(x_resampled.copy(), y_resampled.copy())  # Train model
    y_pred_train = model.predict(x_resampled.copy())  # Predict on train data
    train_acc = accuracy_score(y_resampled.copy(), y_pred_train)  # Calculate training accuracy
    train_prec = precision_score(y_resampled.copy(), y_pred_train, average='macro')
    train_rec = recall_score(y_resampled.copy(), y_pred_train, average='macro')
    train_f1 = f1_score(y_resampled.copy(), y_pred_train, average='macro')
    train_conf = (str(confusion_matrix(y_resampled.copy(), y_pred_train).flatten(order='C')))[1:-1]

    end1 = time()
    train_time = end1 - start1

    file1.write(f"{classes} \t {sample} \t {sce} \t {method} \t {name} \t {i} \t {train_acc} \t {train_prec} \t "
                f"{train_rec} \t {train_f1} \t {train_conf} \t {train_time}\n")
    file1.flush()

    start2 = time()

    y_pred_test = model.predict(xtest.copy())  # Predict on test data
    test_acc = accuracy_score(ytest.copy(), y_pred_test)  # Calculate testing accuracy
    test_prec = precision_score(ytest.copy(), y_pred_test, average='macro')
    test_rec = recall_score(ytest.copy(), y_pred_test, average='macro')
    test_f1 = f1_score(ytest.copy(), y_pred_test, average='macro')
    test_conf = (str(confusion_matrix(ytest.copy(), y_pred_test).flatten(order='C')))[1:-1]

    end2 = time()
    test_time = end2 - start2

    file2.write(f"{classes} \t {sample} \t {sce} \t {method} \t {name} \t {i} \t {test_acc} \t {test_prec} \t "
                f"{test_rec} \t {test_f1}\t{test_conf}\t{test_time}\n")
    file2.flush()
