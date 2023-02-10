import os
import pickle
import time
from random import randint

import forestci
import numpy as np
import sklearn
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm, linear_model
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import data_processing as dpr
import fiiireflyyy.firelearn as fl
import fiiireflyyy.firefiles as ff
import PATHS as P
import sys


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


def test_model(clf, dataset, iterations=1, verbose=False, show_metrics=False):
    """
    Test a model on a dataset.

    :param clf: The model
    :param dataset: dataset used for the testing
    :param iterations: number of iteration for testing
    :return: scores
    """
    X = dataset[dataset.columns[:-1]]
    y = dataset["status"]

    accuracies_over_iterations = []
    tp_confidence_over_iterations = []
    tn_confidence_over_iterations = []
    fp_confidence_over_iterations = []
    fn_confidence_over_iterations = []
    tp_count_over_iterations = []
    tn_count_over_iterations = []
    fp_count_over_iterations = []
    fn_count_over_iterations = []
    entries_count_over_iterations = []

    a, b, c, d = train_test_split(X, y, test_size=0.3)  # Do not do that at home, kids
    number_of_operations = iterations * (len(b.index) + 2*len(d.index))
    ongoing_operation = 0

    if verbose:
        progress = 0
        sys.stdout.write(f"\rTesting model: {progress}%")
        sys.stdout.flush()

    for iter in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        entries_count_over_iterations.append(len(y_test))

        # get predictions and probabilities
        predictions = []
        targets = []
        for i in X_test.index:
            row = X_test.loc[i, :]
            # y_true = y_test[i]
            y_pred = clf.predict([row])[0]
            proba_class = clf.predict_proba([row])[0]
            predictions.append((y_pred, proba_class[0], proba_class[1]))

            ongoing_operation += 1

        if verbose:
            progress = int(np.ceil(ongoing_operation / number_of_operations * 100))
            sys.stdout.write(f"\rTesting model: {progress}%")
            sys.stdout.flush()

        for i in y_test.index:
            targets.append(y_test.loc[i])
            ongoing_operation += 1

        if verbose:
            progress = int(np.ceil(ongoing_operation / number_of_operations * 100))
            sys.stdout.write(f"\rTesting model: {progress}%")
            sys.stdout.flush()

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

            if y_pred == 1:  # predicted infected
                if y_true == 0:  # but is not infected => FP
                    proba_fp.append(proba_inf)
                if y_true == 1:  # and is infected => TP
                    proba_tp.append(proba_inf)
            if y_pred == 0:  # predicted not infected
                if y_true == 0:  # and is not infected => TN
                    proba_tn.append(proba_ni)
                if y_true == 1:  # but is infected => FN
                    proba_fn.append(proba_ni)

            if verbose:
                progress = int(np.ceil(ongoing_operation / number_of_operations * 100))
                sys.stdout.write(f"\rTesting model: {progress}%")
                sys.stdout.flush()
            ongoing_operation += 1

        accuracy = (len(proba_tp) + len(proba_tn)) / (len(proba_tp) + len(proba_tn) + len(proba_fp) + len(proba_fn))
        accuracies_over_iterations.append(accuracy)

        tp_count_over_iterations.append(len(proba_tp))
        if tp_count_over_iterations:
            tp_confidence_over_iterations.append(np.mean(proba_tp))

        tn_count_over_iterations.append(len(proba_tn))
        if tn_count_over_iterations:
            tn_confidence_over_iterations.append(np.mean(proba_tn))

        fp_count_over_iterations.append(len(proba_fp))
        if fp_count_over_iterations:
            fp_confidence_over_iterations.append(np.mean(proba_fp))

        fn_count_over_iterations.append(len(proba_fn))
        if fn_count_over_iterations:
            fn_confidence_over_iterations.append(np.mean(proba_fn))

        progress = int(np.ceil(ongoing_operation / number_of_operations * 100))
        sys.stdout.write(f"\rTesting model: {progress}%")
        sys.stdout.flush()
        ongoing_operation += 1

    tp_confidence_mean = np.mean(tp_confidence_over_iterations)
    tn_confidence_mean = np.mean(tn_confidence_over_iterations)
    fp_confidence_mean = np.mean(fp_confidence_over_iterations)
    fn_confidence_mean = np.mean(fn_confidence_over_iterations)
    accuracy_mean = np.mean(accuracies_over_iterations)

    tp_confidence_std = np.std(tp_confidence_over_iterations)
    tn_confidence_std = np.std(tn_confidence_over_iterations)
    fp_confidence_std = np.std(fp_confidence_over_iterations)
    fn_confidence_std = np.std(fn_confidence_over_iterations)
    accuracy_std = np.std(accuracies_over_iterations)
    print("\n")
    if show_metrics:
        print(f"Mean number of tested entries: {int(np.mean(entries_count_over_iterations))}\n"
              f"True positives count: {int(np.mean(tp_count_over_iterations))}\n"
              f"True negatives count: {int(np.mean(tn_count_over_iterations))}\n"
              f"False positives count:{int(np.mean(fp_count_over_iterations))}\n"
              f"False negatives count:{int(np.mean(fn_count_over_iterations))}")
    if iterations > 1:
        return (accuracy_mean, accuracy_std), (tp_confidence_mean, tp_confidence_std), (tn_confidence_mean, tn_confidence_std), (fp_confidence_mean, fp_confidence_std), (fn_confidence_mean, fn_confidence_std)
    else:
        return (accuracy_mean, 0), (tp_confidence_mean, 0), (tn_confidence_mean, 0), (fp_confidence_mean, 0), (fn_confidence_mean, 0)
