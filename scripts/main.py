import pandas as pd
import os
import fileinput
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import signal_processing as spr
import data_processing as dpr
import machine_learning as ml
import seaborn as sns
import PATHS as P
import data_processing as dp
from firelib.firelib import firefiles as ff, firelearn as fl
import pickle
import forestci as fci
from pathlib import Path

pd.set_option('display.max_columns', None)
import complete_procedures as cp
import get_plots as gp


def main():
    fig, axes = plt.subplots(3, 1, figsize=(6,9), sharex='col')
    fn_t0 = [78, 57]
    tp_t0 = [71, None]
    fp_t0 = [62, 84]
    tn_t0 = [56, 79]
    axes[0].plot(fn_t0, label="FN", marker=".")
    axes[0].plot(tp_t0, label="TP", marker=".")
    axes[0].plot(fp_t0, label="FP", marker=".")
    axes[0].plot(tn_t0, label="TN", marker=".")
    axes[0].set_title("Recording: T=0 hpi")


    fn_t30 = [77, 56]
    tp_t30 = [77, 63]
    fp_t30 = [56, 78]
    tn_t30 = [None, 72]
    axes[1].plot(fn_t30, label="FN", marker=".")
    axes[1].plot(tp_t30, label="TP", marker=".")
    axes[1].plot(fp_t30, label="FP", marker=".")
    axes[1].plot(tn_t30, label="TN", marker=".")
    axes[1].set_title("Recording: T=0.5 hpi")

    fn_t24 = [55, 63, 91]
    tp_t24 = [88, 72, 67]
    fp_t24 = [66, 75, 60]
    tn_t24 = [85, 68, 66]
    axes[2].plot(fn_t24, label="FN", marker=".")
    axes[2].plot(tp_t24, label="TP", marker=".")
    axes[2].plot(fp_t24, label="FP", marker=".")
    axes[2].plot(tn_t24, label="TN", marker=".")
    axes[2].set_title("Recording: T=24hpi")

    for i in [0, 1, 2]:
        axes[i].set_xticks([0, 1, 2], ["None", "T=0.5 hpi", "T=24 hpi"])
        axes[i].set_ylim(50, 100)
        axes[i].set_ylabel("Confidence upon prediction")
        axes[i].legend()#(loc='upper right', bbox_to_anchor=(1, -0.2), ncol=4)
    axes[2].set_xlabel("Stachel addition time")
    axes[2].legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=4)
    fig.suptitle("impact of stachel on prediction confidence")
    plt.show()
    #cp.impact_of_stachel_on_classification_performance()

def test_from_scratch():
    df24 = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                            to_include=("freq_50hz_sample", "T=24H"),
                                            to_exclude=("TTX", "STACHEL"),
                                            verbose=True,
                                            save=False, )
    clf = ml.train_model_from_dataset(df24)
    foi = ml.get_features_of_interest_from_trained_model(clf)
    del clf
    hdf24 = dpr.make_highest_features_dataset_from_complete_dataset(foi, df24)
    clf = ml.train_model_from_dataset(hdf24)

    scores = ml.test_model(clf, hdf24, iterations=15, verbose=True)
    print(scores)


def test_with_loaded_model():
    clf = pickle.load(open(os.path.join(P.MODELS, "T=24H mixed organoids - base foi - no stachel.sav"), "rb"))
    df24 = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                            to_include=("freq_50hz_sample", "T=24H", "STACHEL"),
                                            to_exclude=("TTX",),
                                            verbose=True,
                                            save=False, )

    hdf24 = dpr.make_highest_features_dataset_from_complete_dataset(clf.feature_names, df24)

    scores = ml.test_model(clf, hdf24, iterations=15, verbose=True, show_metrics=True)
    print(scores)


main()
