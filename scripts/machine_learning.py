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
import data_analysis as dan
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

def plot_pca(dataframe: pd.DataFrame, **kwargs): # todo: to fiiireflyyy
    """
    plot the result of PCA.

    Parameters
    ----------
    dataframe: DataFrame
        The data to plot. Must contain a 'label' column.
    n_components: int, optional, default: 2
        Number of principal components. Also, teh dimension
        of the graph. Must be equal to 2 or 3.
    show: bool, optional, default: True
        Whether to show the plot or not.
    save: bool, optional, default: False
        Whether to save the plot or not.
    commentary: str, optional, default: "T=48H"
        Any specification to include in the file name while saving.
    points: bool, optional, default: True
        whether to plot the points or not.
    metrics: bool, optional, default: False
        Whether to plot the metrics or not
    savedir: str, optional, default: ""
        Directory where to save the resulting plot, if not empty.
    title: str, optional, defualt: ""
        The filename of the resulting plot. If empty,
        an automatic name will be generated.
    """

    options = {
        'n_components': 2,
        'show': True,
        'commentary': "",
        'points': True,
        'metrics': False,
        'savedir': "",
        'pc_ratios': [],
        'title': ""

    }
    options.update(kwargs)
    targets = (sorted(list(set(dataframe["label"]))))
    colors = ['g', 'b','r', 'k', 'sandybrown', 'deeppink', 'gray']
    if len(targets) > len(colors):
        n = len(targets) - len(colors) + 1
        for i in range(n):
            colors.append('#%06X' % randint(0, 0xFFFFFF))

    label_params = {'fontsize': 30, "labelpad": 8}
    ticks_params = {'fontsize': 30, }
    if options['n_components'] == 2:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        plt.xticks(**ticks_params)
        plt.yticks(**ticks_params)
        xlabel = 'Principal Component-1'
        ylabel = 'Principal Component-2'
        if len(options['pc_ratios']):
            xlabel += f" ({round(options['pc_ratios'][0] * 100, 2)}%)"
            ylabel += f" ({round(options['pc_ratios'][1] * 100, 2)}%)"

        plt.xlabel(xlabel, **label_params)
        plt.ylabel(ylabel, **label_params)

        for target, color in zip(targets, colors):
            indicesToKeep = dataframe['label'] == target
            x = dataframe.loc[indicesToKeep, 'principal component 1']
            y = dataframe.loc[indicesToKeep, 'principal component 2']
            if options['points']:
                alpha = 1
                if options['metrics']:
                    alpha = .2
                plt.scatter(x, y, c=color, s=10, alpha=alpha, label=target)
            if options['metrics']:
                plt.scatter(np.mean(x), np.mean(y), marker="+", color=color, linewidth=2, s=160)
                fl.confidence_ellipse(x, y, ax, n_std=1.0, color=color, fill=False, linewidth=2)

        def update(handle, orig):
            handle.update_from(orig)
            handle.set_alpha(1)

        plt.legend(prop={'size': 25}, handler_map={PathCollection: HandlerPathCollection(update_func=update),
                                                   plt.Line2D: HandlerLine2D(update_func=update)})
    elif options['n_components'] == 3:
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')

        xlabel = 'Principal Component-1'
        ylabel = 'Principal Component-2'
        zlabel = 'Principal Component-3'
        if len(options['pc_ratios']):
            xlabel += f" ({round(options['pc_ratios'][0] * 100, 2)}%)"
            ylabel += f" ({round(options['pc_ratios'][1] * 100, 2)}%)"
            zlabel += f" ({round(options['pc_ratios'][2] * 100, 2)}%)"

        ax.set_xlabel(xlabel, **label_params)
        ax.set_ylabel(ylabel, **label_params)
        ax.set_zlabel(zlabel, **label_params)
        for target, color in zip(targets, colors):
            indicesToKeep = dataframe['label'] == target
            x = dataframe.loc[indicesToKeep, 'principal component 1']
            y = dataframe.loc[indicesToKeep, 'principal component 2']
            z = dataframe.loc[indicesToKeep, 'principal component 3']
            ax.scatter3D(x, y, z, c=color, s=10)
        plt.legend(targets, prop={'size': 18})

    if options['savedir']:
        if options["title"] == "":
            if options['commentary']:
                options["title"] += options["commentary"]

        plt.savefig(os.path.join(options['savedir'], options["title"] + ".png"), dpi=1200)

    if options['show']:
        plt.show()
    plt.close()

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
