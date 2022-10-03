import pickle
import os
from pathlib import Path
import PATHS as P
import pandas as pd
from sklearn.model_selection import train_test_split

import signal_processing as spr
import data_processing as dpr
import numpy as np
import machine_learning as ml
import matplotlib.pyplot as plt
import statistics
from firelib.firelib import firefiles as ff
import data_analysis as dan
from random import randint
import get_plots as gp
from sklearn.model_selection import train_test_split, KFold, cross_val_score

def generate_stachel_dataset():
    timepoint = "T=24H"
    spec = "manip stachel"

    files = ff.get_all_files(os.path.join(P.NOSTACHEL, "T=24H"))
    freq_files = []
    for f in files:
        if "freq_50hz" in f and "TTX" not in f and "STACHEL" not in f:
            freq_files.append(f)

    dataset = pd.DataFrame(columns=[x for x in range(0, 300)])
    target = pd.DataFrame(columns=["status", ])

    for f in freq_files:
        print(f)
        df = pd.read_csv(f)
        df_top = dpr.top_N_electrodes(df, 35, "Frequency [Hz]")
        samples = dpr.equal_samples(df_top, 30)

        for df_s in samples:

            df_mean = dpr.merge_all_columns_to_mean(df_s, "Frequency [Hz]").round(3)

            downsampled_df = dpr.down_sample(df_mean["mean"], 300, 'mean')

            # construct the dataset with n features
            dataset.loc[len(dataset)] = downsampled_df

            path = Path(f)
            if os.path.basename(path.parent.parent) == "NI":
                target.loc[len(target)] = 0
            elif os.path.basename(path.parent.parent) == "INF":
                target.loc[len(target)] = 1

    dataset["status"] = target["status"]
    ff.verify_dir(P.DATASETS)
    dataset.to_csv(os.path.join(P.DATASETS, f"training dataset {timepoint} {spec}.csv"), index=False)


def generate_basic_dataset():
    files = ff.get_all_files(os.path.join(P.FOUR_ORGANOIDS, "T=24H"))
    pr_paths = []
    # mean between the topped channels
    for f in files:
        if "pr_" in f:
            pr_paths.append(f)

    dataset = pd.DataFrame(columns=[x for x in range(0, 300)])
    target = pd.DataFrame(columns=["status", ])

    for f in pr_paths:
        print(f)
        df = pd.read_csv(f)
        df_top = dpr.top_N_electrodes(df, 35, "TimeStamp")
        samples = dpr.equal_samples(df_top, 30)
        channels = df_top.columns

        for df_s in samples:
            fft_all_channels = pd.DataFrame()

            # fft of the signal
            for ch in channels[1:]:
                filtered = spr.butter_filter(df_s[ch], order=3, lowcut=50)
                clean_fft, clean_freqs = spr.fast_fourier(filtered, 10000)
                fft_all_channels[ch] = clean_fft
                fft_all_channels["Frequency [Hz]"] = clean_freqs
            # mean between the topped channels
            df_mean = dpr.merge_all_columns_to_mean(fft_all_channels, "Frequency [Hz]").round(3)

            # Down sampling by n
            downsampled_df = dpr.down_sample(df_mean["mean"], 300, 'mean')

            # construct the dataset with n features
            dataset.loc[len(dataset)] = downsampled_df

            path = Path(f)
            if os.path.basename(path.parent.parent) == "NI":
                target.loc[len(target)] = 0
            elif os.path.basename(path.parent.parent) == "INF":
                target.loc[len(target)] = 1

    dataset["status"] = target["status"]
    ff.verify_dir(P.DATASETS)
    dataset.to_csv(os.path.join(P.DATASETS, "training dataset T=24H basic batch.csv"))


def classify_frequency_top_monoTime_bothTypes(paths_pr, mono_time, top_n, truncate,
                                              n_features, location, learning=True, ):
    """
    Classify, for the same cell type, between two time points

    :param location: location to save the resulting dataset
    :param paths_pr: paths of the processed files
    :param tn: first time point
    :param tm: second time point
    :param top_n: number of top channels to keep by stp
    :param mono_type: type of cell to considerate
    :param truncate: number of samples to divide the original signal to.
    :param n_features: number of feature to consider for down sampling
    :param learning: boolean. Activate machine learning algorithm
    """
    columns = list(range(0, n_features))
    dataset = pd.DataFrame(columns=columns)

    target = pd.DataFrame(columns=["status", ])
    for p in paths_pr:
        if p.split("\\")[3] == mono_time:
            print("path = ", p)
            df = pd.read_csv(p)
            # selecting top channels by their std

            df_top = dpr.top_N_electrodes(df, top_n, "TimeStamp")

            samples = dpr.equal_samples(df_top, truncate)
            channels = df_top.columns
            for df_s in samples:
                fft_all_channels = pd.DataFrame()

                # fft of the signal
                for ch in channels[1:]:
                    clean_fft, clean_freqs = spr.fast_fourier(df_s[ch], 10000)
                    fft_all_channels[ch] = clean_fft
                    fft_all_channels["frequency"] = clean_freqs
                # mean between the topped channels
                df_mean = dpr.merge_all_columns_to_mean(fft_all_channels, "frequency").round(3)

                # Downsampling by n
                downsampled_df = dpr.down_sample(df_mean["mean"], n_features, 'mean')

                # construct the dataset with n features
                dataset.loc[len(dataset)] = downsampled_df

                if p.split("\\")[4] == "NI":
                    target.loc[len(target)] = 0
                elif p.split("\\")[4] == "COV":
                    target.loc[len(target)] = 1
        else:
            continue

    dataset["status"] = target["status"]
    folder = location + "\\datasets\\"
    ff.verify_dir(folder)
    title = f"{folder}frequency_top{str(top_n)}_nfeatures_{n_features}_{mono_time}.csv"
    dataset.to_csv(title, index=False)

    if learning:
        # training
        print("learning")
        dataset = pd.read_csv(title)
        X = dataset[dataset.columns[:-1]]
        y = dataset["status"]
        folder = location + "\\models\\"
        ff.verify_dir(folder)
        modelpath = f"{folder}Importance_frequency_top{str(top_n)}_nfeatures_{n_features}_{mono_time}"

        for i in range(0, 10):
            modelname = "rfc1000"
            model_perf = modelpath + "\\" + modelname + ".sav"
            ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname=modelname, modelpath=modelpath, )
            ml.get_feature_importance_rfc(model_perf, 300,
                                          f"{folder}frequency_top{str(top_n)}_nfeatures_{n_features}_{mono_time}_IT{i}.png")
            ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)
            print(modelname)



def T24H_mixed_organoids_base_foi_no_stachel():
    df24orga = dpr.make_dataset_from_freq_files(timepoint="T=24H",
                                                parent_dir=P.FOUR_ORGANOIDS,
                                                to_include=("freq_50hz_sample",),
                                                to_exclude=("TTX", "STACHEL"))

    df30orga = dpr.make_dataset_from_freq_files(timepoint="T=30MIN",
                                                parent_dir=P.FOUR_ORGANOIDS,
                                                to_include=("freq_50hz_sample",),
                                                to_exclude=("TTX", "STACHEL")
                                                )
    df0orga = dpr.make_dataset_from_freq_files(timepoint="T=0MIN",
                                               parent_dir=P.FOUR_ORGANOIDS,
                                               to_include=("freq_50hz_sample",),
                                               to_exclude=("TTX", "STACHEL")
                                               )

    df24nostachel = dpr.make_dataset_from_freq_files(timepoint="T=24H",
                                                     parent_dir=P.NOSTACHEL,
                                                     to_include=("freq_50hz_sample",),
                                                     to_exclude=("TTX", "STACHEL"))

    df30nostachel = dpr.make_dataset_from_freq_files(timepoint="T=30MIN",
                                                     parent_dir=P.NOSTACHEL,
                                                     to_include=("freq_50hz_sample",),
                                                     to_exclude=("TTX", "STACHEL")
                                                     )
    df0nostachel = dpr.make_dataset_from_freq_files(timepoint="T=0MIN",
                                                    parent_dir=P.NOSTACHEL,
                                                    to_include=("freq_50hz_sample",),
                                                    to_exclude=("TTX", "STACHEL")
                                                    )

    df24 = pd.concat([df24orga, df24nostachel], ignore_index=True)
    df30 = pd.concat([df30orga, df30nostachel], ignore_index=True)
    df0 = pd.concat([df0orga, df0nostachel], ignore_index=True)

    print(df24orga.info())
    print(df24nostachel.info())
    print(df24.info())
    clf_orga = ml.train_model_from_dataset(df24orga)
    foi_orga = ml.get_features_of_interest_from_trained_model(clf_orga)
    print(foi_orga)
    clf = ml.train_model_from_dataset(df24)
    # foi = ml.get_features_of_interest_from_trained_model(clf)
    # foi = clf.feature_names
    print(foi_orga)
    hdf24 = dpr.make_highest_features_dataset_from_complete_dataset(foi_orga, df24)
    hdf30 = dpr.make_highest_features_dataset_from_complete_dataset(foi_orga, df30)
    hdf0 = dpr.make_highest_features_dataset_from_complete_dataset(foi_orga, df0)
    #
    clf = ml.train_model_from_dataset(hdf24)
    #

    hdf24orga = dpr.make_highest_features_dataset_from_complete_dataset(foi_orga, df24orga)
    hdf30orga = dpr.make_highest_features_dataset_from_complete_dataset(foi_orga, df30orga)
    hdf0orga = dpr.make_highest_features_dataset_from_complete_dataset(foi_orga, df0orga)
    hdf24nostachel = dpr.make_highest_features_dataset_from_complete_dataset(foi_orga, df24nostachel)
    hdf30nostachel = dpr.make_highest_features_dataset_from_complete_dataset(foi_orga, df30nostachel)
    hdf0nostachel = dpr.make_highest_features_dataset_from_complete_dataset(foi_orga, df0nostachel)

    scores = ml.test_model(clf, hdf24orga)
    print("24 orga", np.mean(scores))
    scores = ml.test_model(clf, hdf30orga)
    print("30 orga", np.mean(scores))
    scores = ml.test_model(clf, hdf0orga)
    print("0 orga", np.mean(scores))

    scores = ml.test_model(clf, hdf24nostachel)
    print("24 nostachel", np.mean(scores))
    scores = ml.test_model(clf, hdf30nostachel)
    print("30 nostachel", np.mean(scores))
    scores = ml.test_model(clf, hdf0nostachel)
    print("0 nostachel", np.mean(scores))

    scores = ml.test_model(clf, hdf24)
    print("24 mix", np.mean(scores))
    scores = ml.test_model(clf, hdf30)
    print("30 mix", np.mean(scores))
    scores = ml.test_model(clf, hdf0)
    print("0 mix", np.mean(scores))

    pickle.dump(clf, open(os.path.join(P.MODELS, "T=24H mixed organoids - base foi - no stachel.sav"), "wb"))