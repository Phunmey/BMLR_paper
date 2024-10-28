"""
Bayesian Multinomial Logistic Regression funtion for the simulated data.
"""


import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt

from time import time
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from bayesian_plotting import *

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-white")

plt.rcParams["figure.figsize"] = [7, 6]
plt.rcParams["figure.dpi"] = 100


def bmlr(file1, file2, use_features, xtrain, ytrain, xtest, ytest, i, class_names, resampling_technique, classes,
         sample, sce, k_neighbors=None):
    if resampling_technique == 'SMOTE':
        oversample = SMOTE(k_neighbors=k_neighbors, random_state=123)
        xtrain, ytrain = oversample.fit_resample(xtrain.copy(), ytrain.copy())
    elif resampling_technique == 'ADASYN':
        oversample = ADASYN(n_neighbors=k_neighbors, sampling_strategy='minority', random_state=123)
        xtrain, ytrain = oversample.fit_resample(xtrain.copy(), ytrain.copy())
    elif resampling_technique == 'Tomek':
        undersample = TomekLinks()
        xtrain, ytrain = undersample.fit_resample(xtrain.copy(), ytrain.copy())

    start1 = time()
    coords = {"xvars": use_features, "classes": class_names}
    with pm.Model(coords=coords) as model:
        xNormal = pm.Data("xNormal", xtrain.copy(), mutable=True)
        yNormal = pm.ConstantData('yNormal', ytrain.copy())

        betaI = pm.Normal('betaI', mu=0, sigma=10, dims='classes')
        betaP = pm.Normal('betaP', mu=0, sigma=1, dims=('xvars', 'classes'))

        betaIR = pm.Deterministic('betaIR', pt.concatenate([[0], betaI]))
        betaPR = pm.Deterministic('betaPR', pt.concatenate([pt.zeros((xNormal.shape[1], 1)), betaP],
                                                           axis=1))

        muNormal = betaIR + pm.math.dot(xNormal, betaPR)
        thetaNormal = pm.Deterministic(f"thetaNormal_{resampling_technique}", pt.special.softmax(muNormal,
                                                                                                 axis=1))

        pm.Categorical('observed', p=thetaNormal, observed=yNormal, shape=thetaNormal.shape[0])
        trace = pm.sample(5000, chains=4, target_accept=0.95, idata_kwargs={'log_likelihood': True},
                          random_seed=rng)
        trace.extend(pm.sample_posterior_predictive(trace, random_seed=rng))

    summary = az.summary(trace, var_names=['betaI', 'betaP'], hdi_prob=0.95, round_to=3)
    summary.to_csv(f"./summary/{classes}_{sample}_{sce}_{resampling_technique}_summary_{i}.csv")

    theta_train_pred = trace.posterior[f'thetaNormal_{resampling_technique}'].mean(dim=['chain', 'draw'])
    row_max = theta_train_pred.argmax(axis=1)

    train_acc = accuracy_score(ytrain, row_max)
    train_prec = precision_score(ytrain, row_max, average='macro')
    train_rec = recall_score(ytrain, row_max, average='macro')
    train_f1 = f1_score(ytrain, row_max, average='macro')
    train_conf = (str(confusion_matrix(ytrain, row_max).flatten(order='C')))[1:-1]

    end1 = time()
    train_time = end1 - start1

    file1.write(f"{classes} \t {sample} \t {sce} \t {resampling_technique} \t BMLR \t {i} \t {train_acc} \t {train_prec} "
                f"\t {train_rec} \t {train_f1} \t {train_conf} \t {train_time}\n")
    file1.flush()

    start2 = time()
    with model:
        pm.set_data({"xNormal": xtest.copy()})
        post_predictive = pm.sample_posterior_predictive(trace, var_names=[f'thetaNormal_{resampling_technique}', 'observed'],
                                             predictions=True)

    theta_test_pred = post_predictive.predictions[f'thetaNormal_{resampling_technique}'].mean(dim=['chain', 'draw'])
    row_max_test = theta_test_pred.argmax(axis=1)

    test_acc = accuracy_score(ytest, row_max_test)
    test_prec = precision_score(ytest, row_max_test, average='macro')
    test_rec = recall_score(ytest, row_max_test, average='macro')
    test_f1 = f1_score(ytest, row_max_test, average='macro')
    test_conf = (str(confusion_matrix(ytest, row_max_test).flatten(order='C')))[1:-1]

    end2 = time()
    test_time = end2 - start2

    file2.write(f"{classes} \t {sample} \t {sce} \t {resampling_technique} \t BMLR \t {i} \t {test_acc} \t {test_prec} "
                f"\t {test_rec} \t {test_f1} \t {test_conf} \t {test_time}\n")
    file2.flush()

