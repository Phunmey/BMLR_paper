"""
Code to fit the baseline models
"""


from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(file1, file2, resampling_technique, xtrain, ytrain, xtest, ytest, model, name, i, classes, sample,
                   sce):
    start1 = time()

    model.fit(xtrain.copy(), ytrain.copy())
    y_pred_train = model.predict(xtrain.copy())
    train_acc = accuracy_score(ytrain.copy(), y_pred_train)
    train_prec = precision_score(ytrain.copy(), y_pred_train, average='weighted')
    train_rec = recall_score(ytrain.copy(), y_pred_train, average='weighted')
    train_f1 = f1_score(ytrain.copy(), y_pred_train, average='weighted')

    end1 = time()
    train_time = end1 - start1

    file1.write(
        f"{classes}\t{sample}\t{sce}\t{resampling_technique}\t{name}\t{i}\t{train_acc}\t{train_prec}\t"
        f"{train_rec}\t{train_f1}\t{train_time}\n")
    file1.flush()

    start2 = time()

    y_pred_test = model.predict(xtest.copy())
    test_acc = accuracy_score(ytest.copy(), y_pred_test)
    test_prec = precision_score(ytest.copy(), y_pred_test, average='weighted')
    test_rec = recall_score(ytest.copy(), y_pred_test, average='weighted')
    test_f1 = f1_score(ytest.copy(), y_pred_test, average='weighted')

    end2 = time()
    test_time = end2 - start2

    file2.write(
        f"{classes}\t{sample}\t{sce}\t{resampling_technique}\t{name}\t{i}\t{test_acc}\t{test_prec}\t"
        f"{test_rec}\t{test_f1}\t{test_time}\n")
    file2.flush()
