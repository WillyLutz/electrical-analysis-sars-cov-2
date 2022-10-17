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
    print()


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
