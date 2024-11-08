"""
Bayesian Multinomial Logistic Regression funtion for the student data. I have used {classes}_{sample}_{sce} to occupy
the space in the result file. It is not exactly necessary.
"""

import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt

from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from bayesian_plotting import *

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-white")

plt.rcParams["figure.figsize"] = [7, 6]
plt.rcParams["figure.dpi"] = 100


def bmlr(file1, file2, use_features, xtrain, ytrain, xtest, ytest, class_names, resampling_technique, classes,
         sample, sce, i):
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

    # trace.to_netcdf(f"./traces/{classes}_{sample}_{sce}_{resampling_technique}_trace.nc")

    traceplotting(trace)
    plt.savefig(f"./plots/{classes}_{sample}_{sce}_{resampling_technique}_trace.png")
    plt.clf()

    posterior_predictiveplotting(trace)
    plt.savefig(f"./plots/{classes}_{sample}_{sce}_{resampling_technique}_ppc.png")
    plt.clf()

    summary = az.summary(trace, var_names=['betaI', 'betaP'], hdi_prob=0.95, round_to=3)
    summary.to_csv(f"./summary/{classes}_{sample}_{sce}_{resampling_technique}_summary_{i}.csv")

    theta_train_pred = trace.posterior[f'thetaNormal_{resampling_technique}'].mean(dim=['chain', 'draw'])
    row_max = theta_train_pred.argmax(axis=1)

    train_acc = accuracy_score(ytrain, row_max)
    train_prec = precision_score(ytrain, row_max, average='macro')
    train_rec = recall_score(ytrain, row_max, average='macro')
    train_f1 = f1_score(ytrain, row_max, average='macro')
    # train_conf = (str(confusion_matrix(ytrain, row_max).flatten(order='C')))[1:-1]

    end1 = time()
    train_time = end1 - start1

    file1.write(f"{classes}\t{sample}\t{sce}\t{resampling_technique}\tBMLR\t{i}\t{train_acc}\t{train_prec}\t"
                f"{train_rec}\t{train_f1}\t{train_time}\n")
    file1.flush()

    start2 = time()
    with model:
        pm.set_data({"xNormal": xtest.copy()})
        post_predictive = pm.sample_posterior_predictive(trace,
                                                         var_names=[f'thetaNormal_{resampling_technique}', 'observed'],
                                                         predictions=True)

    theta_test_pred = post_predictive.predictions[f'thetaNormal_{resampling_technique}'].mean(dim=['chain', 'draw'])
    row_max_test = theta_test_pred.argmax(axis=1)

    test_acc = accuracy_score(ytest, row_max_test)
    test_prec = precision_score(ytest, row_max_test, average='macro')
    test_rec = recall_score(ytest, row_max_test, average='macro')
    test_f1 = f1_score(ytest, row_max_test, average='macro')
    # test_conf = (str(confusion_matrix(ytest, row_max_test).flatten(order='C')))[1:-1]

    end2 = time()
    test_time = end2 - start2

    file2.write(f"{classes}\t{sample}\t{sce}\t{resampling_technique}\tBMLR\t{i}\t{test_acc}\t{test_prec}\t"
                f"{test_rec}\t{test_f1}\t{test_time}\n")
    file2.flush()



