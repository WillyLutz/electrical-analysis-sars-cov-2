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
    # make dataset from scratch :                       dpr.make_filtered_sampled_freq_files
    #                                                   dpr.make_dataset_from_freq_files
    # train first model
    # get the highest features :                        ml.get_features_of_interest_from_trained_model
    # make new dataset based on highest features :      dpr.make_highest_features_dataset_from_complete_dataset
    # train new model :                                 ml.train_model_on_features_of_interest
    # test                                              ml.test_model

    # procedure()

    # clf = pickle.load(open(os.path.join(P.MODELS, "T=24H mixed organoids - base foi - no stachel.sav"), "rb"))
    # df24 = dpr.make_dataset_from_freq_files(timepoint="T=30MIN",
    #                                         parent_dir=P.NOSTACHEL,
    #                                         to_include=("freq_50hz_sample",),
    #                                         to_exclude=("TTX", "STACHEL"),
    #                                         verbose=True,
    #                                         save=True,
    #                                         title="test df24.csv"
    #                                         )
    df = pd.read_csv(os.path.join(P.FOUR_ORGANOIDS, "T=24H/INF/3/",
                                  "freq_50hz_sample3_2022-06-10T10-57.csv"))

    plt.plot(df[df.columns[1]], df[df.columns[0]], linewidth=.5, color="black")
    title = "organoid signal in frequencies domain"
    plt.title(title)
    plt.xlabel("Frequencies [Hz]")
    plt.ylabel("Amplitude [pV]")
    plt.legend()
    plt.show()

    dfds = dpr.down_sample(df["mean"], 300, 'mean')
    plt.plot(dfds, linewidth=1, color="black")
    title = "Smoothened signal in frequencies domain"
    plt.title(title)
    plt.xlabel("Frequencies [Hz]")
    plt.ylabel("Amplitude [pV]")
    plt.legend()
    plt.show()
    # df24 = dpr.make_raw_frequency_plots_from_pr_files(parent_dir=P.NOSTACHEL,
    #                                                   to_include=("pr_", "T=24H"),
    #                                                   to_exclude=("TTX", "STACHEL"),
    #                                                   verbose=True,
    #                                                   save=True,
    #                                                   )

    # df24 = pd.read_csv(os.path.join(P.DATASETS, "test df24.csv"))
    # hdf24 = dpr.make_highest_features_dataset_from_complete_dataset(clf.feature_names, df24)

    # scores = ml.test_model(clf, hdf24)
    # print(np.mean(scores))


def procedure():
    parent_dir = P.FOUR_ORGANOIDS
    df24 = dpr.make_dataset_from_freq_files(parent_dir=parent_dir,
                                            to_include=("freq_50hz_sample", "T=24H"),
                                            to_exclude=("TTX", "STACHEL")
                                            )
    clf = ml.train_model_from_dataset(df24)
    foi = ml.get_features_of_interest_from_trained_model(clf)
    print(foi)
    hdf24 = dpr.make_highest_features_dataset_from_complete_dataset(foi, df24)
    clf2 = ml.train_model_from_dataset(hdf24)

    scores = ml.test_model(clf2, hdf24)
    print(np.mean(scores))


main()
