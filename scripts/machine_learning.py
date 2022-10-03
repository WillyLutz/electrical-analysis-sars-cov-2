import os
import pickle
import time

import forestci
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm, linear_model
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import data_processing as dpr
import data_analysis as dan
from firelib.firelib import firefiles as ff
import PATHS as P


def train_model_from_dataset(dataset, model_save_path="", save=False):
    """
    Train an RFC model from a dataset and save the model.

    :param dataset: The dataset to train the model.
    :param model_save_path: Path to save the model
    :return: the trained RFC model
    """

    clf = RandomForestClassifier(n_estimators=1000)
    X = dataset[dataset.columns[:-1]]
    y = dataset["status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    clf.fit(X_train, y_train)
    clf.feature_names = [x for x in X.columns]
    if save:
        pickle.dump(clf, open(model_save_path, "wb"))
    return clf


def get_feature_of_interest(timepoint, path, detection_factor=2.0, plot=True, by_percentage=False, percentage=0.05):
    dataset = pd.read_csv(path)
    # training
    print("learning")
    X = dataset[dataset.columns[:-1]]
    y = dataset["status"]

    model_directory = P.MODELS
    ff.verify_dir(model_directory)
    importances_over_iterations = []
    std_over_iterations = []
    for i in range(10):
        clf = RandomForestClassifier(n_estimators=1000)
        clf.fit(X, y)
        mean = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)

        importances_over_iterations.append(mean)

    arrays = [np.array(x) for x in importances_over_iterations]
    mean_importances_over_iterations = [np.mean(k) for k in zip(*arrays)]
    std_arrays = [np.array(x) for x in importances_over_iterations]
    std_importances_over_iterations = [np.std(k) for k in zip(*std_arrays)]

    low_std = []
    for i in range(len(mean_importances_over_iterations)):
        low_std.append(mean_importances_over_iterations[i] - std_importances_over_iterations[i])
    high_std = []
    for i in range(len(mean_importances_over_iterations)):
        high_std.append(mean_importances_over_iterations[i] + std_importances_over_iterations[i])

    hertz = []
    factor = 5000 / 300
    for i in range(300):
        hertz.append(int(i * factor))

    whole_mean = np.mean(mean_importances_over_iterations)
    whole_std = np.std(mean_importances_over_iterations)

    high_mean_thresh = whole_mean + whole_std * detection_factor
    low_mean_thresh = whole_mean - whole_mean
    factor_mean_thresh = 1
    if plot:
        fig, ax = plt.subplots()
        ax.plot(hertz, mean_importances_over_iterations, color="red", linewidth=0.5)
        ax.fill_between(hertz, low_std, high_std, facecolor="blue", alpha=0.5)

        ax.axhline(y=whole_mean, xmin=0, xmax=300, color="black", linewidth=0.5)
        ax.fill_between(hertz, low_mean_thresh * factor_mean_thresh, high_mean_thresh * factor_mean_thresh,
                        facecolor="black", alpha=0.3)

        idx = []
        for i in range(len(mean_importances_over_iterations) - 1):
            value1 = mean_importances_over_iterations[i]
            value2 = mean_importances_over_iterations[i + 1]

            if value1 >= high_mean_thresh * factor_mean_thresh >= value2 or value1 <= high_mean_thresh * factor_mean_thresh <= value2:
                idx.append(hertz[i])

        for x in idx:
            ax.axvline(x=x, color="green", linewidth=0.5)

        title = f"features of interest at {timepoint} with {detection_factor} factor detection"
        plt.show()

    if by_percentage:
        n = int(percentage * len(mean_importances_over_iterations))
        idx_foi = sorted(range(len(mean_importances_over_iterations)),
                         key=lambda i: mean_importances_over_iterations[i], reverse=True)[:n]
        return idx_foi
    else:
        idx_foi = []
        for i in range(len(mean_importances_over_iterations) - 1):
            if mean_importances_over_iterations[i] >= high_mean_thresh * factor_mean_thresh:
                idx_foi.append(i)

        return idx_foi


def get_features_of_interest_from_trained_model(clf, percentage=0.05):
    """
    Only for model not trained on restricted features.

    :param clf:
    :param percentage:
    :return:
    """
    importances_over_iterations = []
    std_over_iterations = []
    for i in range(10):
        mean = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)

        importances_over_iterations.append(mean)

    arrays = [np.array(x) for x in importances_over_iterations]
    mean_importances_over_iterations = [np.mean(k) for k in zip(*arrays)]
    std_arrays = [np.array(x) for x in importances_over_iterations]
    std_importances_over_iterations = [np.std(k) for k in zip(*std_arrays)]

    low_std = []
    for i in range(len(mean_importances_over_iterations)):
        low_std.append(mean_importances_over_iterations[i] - std_importances_over_iterations[i])
    high_std = []
    for i in range(len(mean_importances_over_iterations)):
        high_std.append(mean_importances_over_iterations[i] + std_importances_over_iterations[i])

    hertz = []
    factor = 5000 / 300
    for i in range(300):
        hertz.append(int(i * factor))
    n = int(percentage * len(mean_importances_over_iterations))
    idx_foi = sorted(range(len(mean_importances_over_iterations)),
                     key=lambda i: mean_importances_over_iterations[i], reverse=True)[:n]

    return idx_foi


def test_model(clf, dataset, iterations=1):
    """
    Test a model on a dataset.

    :param clf: The model
    :param dataset: dataset used for the testing
    :param iterations: number of iteration for testing
    :return: scores
    """
    X = dataset[dataset.columns[:-1]]
    y = dataset["status"]

    scores = []
    overall_tp = []
    overall_tn = []
    overall_fp = []
    overall_fn = []

    for iter in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # get predictions and probabilities
        predictions = []
        targets = []
        for i in X_test.index:
            row = X_test.loc[i, :]
            #y_true = y_test[i]
            y_pred = clf.predict([row])[0]
            proba_class = clf.predict_proba([row])[0]
            predictions.append((y_pred, proba_class[0], proba_class[1]))
        for i in y_test.index:
            targets.append(y_test.loc[i])

        # get metrics out of predictions
        proba_tp = []
        proba_tn = []
        proba_fp = []
        proba_fn = []
        for i in range(len(targets)):
            y_true = targets[i]
            y_pred = predictions[i][0]
            proba_ni = predictions[i][1]
            proba_inf = predictions[i][2]

            if y_true == y_pred:  # true...
                if y_true == 0:  # ...negative
                    proba_tn.append(proba_ni)
                else: # ...positive
                    proba_tp.append(proba_inf)
            else:  # false...
                if y_true == 0:  # ...negative
                    proba_fn.append(proba_ni)
                else:  # ...positive
                    proba_fp.append(proba_inf)

        print("number of true positives: ", len(proba_tp), " with a mean probability of: ", np.mean(proba_tp))
        print("number of true negatives: ", len(proba_tn), " with a mean probability of: ", np.mean(proba_tn))
        print("number of false positives: ", len(proba_fp), " with a mean probability of: ", np.mean(proba_fp))
        print("number of false negatives: ", len(proba_fn), " with a mean probability of: ", np.mean(proba_fn))
        print("accuracy: ", (len(proba_tn)+len(proba_tp))/(len(proba_fn)+len(proba_fp)))
        # todo: differences between vote probabilities and confidence in the prediction ?
        forestci.random_forest_error()
        # todo: calculer le score manuellement et avoir les probabilités pour chaque prédiction
        #scores.append(clf.score(X_test, y_test))

    return scores
