import pickle
import time

import statistics
import pandas as pd
import os
import shutil
import fileinput
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import scipy
import sklearn
from numpy import trapz
from scipy.integrate import simps
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelBinarizer
from os import listdir
import re
import signal_processing as spr
import data_processing as dpr
import data_analysis as dan
import machine_learning as ml
import seaborn as sns
from sklearn import preprocessing
import complete_procedures as cp
import PATHS as P
import data_processing as dp
import FireFiles as ff
import get_plots as gp
import firefilespip

pd.set_option('display.max_columns', None)


def main():
    #gp.inf_ni_frequency_pattern("T=24H", process=False, ds=True, zoom_in=True)
    #gp.inf_ni_frequency_pattern("T=24H", process=False, ds=False, zoom_in=True)
    #gp.inf_ni_frequency_pattern_ROI_bar("T=24H", process=False, ds=True)
    #gp.inf_ni_frequency_pattern_ROI_bar("T=24H", process=False, ds=False)

    figure_2_13()

def figure_2_13():
    trained_time = "T=24H"
    times = ("T=0MIN", "T=30MIN", "T=24H")

    all_scores = []
    all_mean_scores = []
    all_std = []
    for test_time in times:
        print(test_time)
        path_trained = fr"Four organoids/datasets/filtered_50_highest_5.0%features_frequency_top35_nfeatures_300_{trained_time}.csv"

        dataset = pd.read_csv(path_trained)

        X = dataset[dataset.columns[1:-1]]
        y = dataset["status"]
        folder = "four organoids" + "\\models\\"
        ff.verify_dir(folder)
        modelpath = f"{folder}filtered_50_highest_5.0%features_frequency_top35_nfeatures_300_trained_{trained_time}_test_{test_time}"
        modelpathname = modelpath + "\\rfc1000.sav"

        clf = ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname="rfc1000",
                                          modelpath=modelpath, )

        path_tested = fr"Four organoids/datasets/filtered_50_highest_5.0%features_frequency_top35_nfeatures_300_{test_time}.csv"
        scores_orga = ml.test_model_across_organoids(modelpathname, "rfc", path_tested, "status", 0.7,
                                                     f"test_{test_time}",
                                                     commentary=f"testing on {test_time} a model trained on {trained_time}")
        all_mean_scores.append(np.mean(scores_orga))
        all_std.append(np.std(scores_orga))
        all_scores.append(scores_orga)

    plotting_df = pd.DataFrame(columns=["T=0MIN", "T=30MIN", "T=24H"])
    for x in range(len(all_scores)):
        plotting_df[plotting_df.columns[x]] = all_scores[x]

    sns.barplot(data=plotting_df, capsize=.1, ci="sd", color="gray")
    sns.swarmplot(data=plotting_df, color="0", alpha=.5)

    plotting_df.to_csv(r"Four organoids/datasets/scores model trained on 24H tested on T0 T30 T24.csv", index=False)

    plt.xticks((0, 1, 2), ("T=0MIN", "T=30MIN", "T=24H"))
    plt.ylabel("Testing scores")
    plt.xlabel("Testing time point")
    plt.title(f"Scores obtained with a model trained at {trained_time} and tested at other time points.")
    plt.show()

def ttest():
    # cp.make_complete_filtered_frequencies_files(ds=True)
    # gp.inf_ni_frequency_pattern(process=False, ds=True)
    trained_time = "T=24H"
    times = ("T=24H",)

    all_scores = []
    all_mean_scores = []
    all_std = []
    for test_time in times:
        print(test_time)
        path_trained = fr"Four organoids/datasets/filtered_50_highest_5.0%features_frequency_top35_nfeatures_300_{trained_time}.csv"

        dataset = pd.read_csv(path_trained)

        X = dataset[dataset.columns[1:-1]]
        y = dataset["status"]
        folder = "four organoids" + "\\models\\"
        ff.verify_dir(folder)
        modelpath = f"{folder}filtered_50_highest_5.0%features_frequency_top35_nfeatures_300_trained_{trained_time}_test_{test_time}"
        modelpathname = modelpath + "\\rfc1000.sav"

        clf = ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname="rfc1000",
                                          modelpath=modelpath, )

        path_tested = fr"Four organoids/datasets/filtered_50_highest_5.0%features_frequency_top35_nfeatures_300_{test_time}.csv"
        scores_orga = ml.test_model_across_organoids(modelpathname, "rfc", path_tested, "status", 0.7,
                                                     f"test_{test_time}",
                                                     commentary=f"testing on {test_time} a model trained on {trained_time}")
        all_mean_scores.append(np.mean(scores_orga))
        all_std.append(np.std(scores_orga))
        all_scores.append(scores_orga)

    print(all_scores)


main()
