import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

import signal_processing as spr
import data_processing as dpr
import numpy as np
import machine_learning as ml
import matplotlib.pyplot as plt
import statistics
import FireFiles as ff
import data_analysis as dan
from random import randint
import get_plots as gp
from sklearn.model_selection import train_test_split, KFold, cross_val_score


def test_all_top(PATHS_FREQ_HDD):
    n_features = 600
    truncate = 30
    top_n = 10
    fft_freq = 10000
    low_window = []
    high_window = []
    for i in range(200, 2600, 200):
        low_window.append(i)
    for i in range(600, 3000, 200):
        high_window.append(i)
    learning = True

    for low in low_window:
        for high in high_window:
            if high > low + 400:
                print(low, high)
                classify_top_tn_tm_typeCell(PATHS_FREQ_HDD, "T=0", "T=4H", top_n, "COV", truncate, fft_freq, low, high,
                                            n_features, learning)
                classify_top_tn_tm_typeCell(PATHS_FREQ_HDD, "T=0", "T=4H", top_n, "NI", truncate, fft_freq, low, high,
                                            n_features, learning)


def classify_top_tn_tm_typeCell(PATHS_PR_HDD: list, tn: str, tm: str, top_n: int, mono_type: str, truncate: int,
                                fft_freq: int, low_window: int, high_window: int, n_features: int, learning=True,
                                key=""):
    """
    Features: frequencies. Target: Infected (1), non infected (0).
    Classification nature: On one cell type, (COV or NI), classify between 2 times

    :param PATHS_PR_HDD: paths of the temporal csv.
    :param tn: first time point of classification. Takes the class value 0.
    :param tm: second time point of classification. Takes the class value 1.
    :param top_n: number of most relevant channels kept by their std.
    :param mono_type: 'COV' or 'NI'. Type of cell to classify on.
    :param truncate: Number of sample to divide the original signal to.
    :param fft_freq: frequency of the signal.
    :param low_window: low frequency of the window.
    :param high_window: high frequency of the window.
    :param n_features: number of features to consider.
    :param learning: Boolean. if True, launches a machine learning script.
    :param key: str to add to paths and model names. Optional.
    :return:
    """
    columns = list(range(0, n_features))
    dataset = pd.DataFrame(columns=columns)

    target = pd.DataFrame(columns=["status", ])
    for p in PATHS_PR_HDD:
        if p.split("/")[3] == mono_type:
            if p.split("/")[2] == tn or p.split("/")[2] == tm:
                print("path = ", p)
                df = pd.read_csv(p)
                df_top = dpr.top_N_electrodes(df, top_n, "TimeStamp [µs]")

                samples = dpr.equal_samples(df_top, truncate)
                channels = df_top.columns
                # fft for each samples
                for df_s in samples:
                    fft_all_channels = pd.DataFrame()
                    for ch in channels[1:]:
                        clean_fft, clean_freqs = spr.fast_fourier(df_s[ch], fft_freq)
                        fft_all_channels[ch] = clean_fft
                        fft_all_channels["frequency"] = clean_freqs

                    # mean between the cleaned channels
                    df_mean = dpr.merge_all_columns_to_mean(fft_all_channels, "frequency")

                    # windowing
                    windowed_df = pd.DataFrame(columns=["frequency", "mean", ])
                    windowed_df["frequency"] = df_mean["frequency"][
                        (df_mean['frequency'] >= low_window) & (df_mean['frequency'] <= high_window)]
                    windowed_df["mean"] = df_mean["mean"][
                        (df_mean['frequency'] >= low_window) & (df_mean['frequency'] <= high_window)]

                    # Downsampling by n_features
                    downsampled_df = dpr.down_sample(windowed_df["mean"], n_features, 'mean')

                    # construct the dataset with n_features
                    dataset.loc[len(dataset)] = downsampled_df

                    if p.split("/")[2] == tn:
                        target.loc[len(target)] = 0
                    elif p.split("/")[2] == tm:
                        target.loc[len(target)] = 1
            else:
                continue
        else:
            continue

    dataset["status"] = target["status"]
    title = "ml_datasets/" + key + "top" + str(top_n) + "_" + tn + "_" + tm + "_" + mono_type + "_" + str(n_features) + \
            "features_" + str(low_window) + "_" + str(high_window) + "_Hz.csv"
    dataset.to_csv(title, index=False)

    if learning:
        # training
        print("learning")
        dataset = pd.read_csv(title)
        X = dataset[dataset.columns[:-1]]
        y = dataset["status"]

        modelpath = "ml_models/" + key + "top" + str(top_n) + "_" + tn + "_" + tm + "_" + mono_type + "_" + \
                    str(n_features) + "features_" + str(low_window) + "_" + str(high_window) + "_Hz/"

        modelname = "svm_linear"
        model_perf = modelpath + modelname + ".sav"
        ml.support_vector_machine(X, y, kernel='linear', save=True, modelname=modelname, modelpath=modelpath,
                                  decision_function_shape="ovr")
        ml.model_performance_analysis(model_perf, "svm", X, y, train_size=0.7)
        print(modelname)

        modelname = "rfc1000"
        model_perf = modelpath + modelname + ".sav"
        ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname=modelname, modelpath=modelpath, )
        ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)
        print(modelname)


def classify_tn_tm_typeCell(PATHS_PR_HDD, CHANNELS, tn, tm, typeCell, truncate, fft_freq, std_threshold,
                            low_window, high_window, n_features, learning=True):
    columns = list(range(0, n_features))
    dataset = pd.DataFrame(columns=columns)

    target = pd.DataFrame(columns=["status", ])
    # truncate into 30 samples TEMPORAL
    for p in PATHS_PR_HDD:
        if p.split("/")[3] == typeCell:
            if p.split("/")[2] == tn or p.split("/")[2] == tm:
                print("path = ", p)
                df = pd.read_csv(p)
                samples = dpr.equal_samples(df, truncate)

                # fft for each samples
                for df_s in samples:

                    fft_all_channels = pd.DataFrame()
                    for ch in CHANNELS[1:]:
                        clean_fft, clean_freqs = spr.fast_fourier(df_s[ch], fft_freq)
                        fft_all_channels[ch] = clean_fft
                        fft_all_channels["frequency"] = clean_freqs

                    # clean the channels in each samples
                    df_clean = dpr.clean_std_threshold(fft_all_channels, std_threshold)

                    # mean between the cleaned channels
                    df_mean = dpr.merge_all_columns_to_mean(df_clean, "frequency")

                    # windowing
                    windowed_df = pd.DataFrame(columns=["frequency", "mean", ])
                    windowed_df["frequency"] = df_mean["frequency"][
                        (df_mean['frequency'] >= low_window) & (df_mean['frequency'] <= high_window)]
                    windowed_df["mean"] = df_mean["mean"][
                        (df_mean['frequency'] >= low_window) & (df_mean['frequency'] <= high_window)]

                    # Downsampling by 300
                    downsampled_df = dpr.down_sample(windowed_df["mean"], n_features, 'mean')

                    # construct the dataset with 300 features
                    dataset.loc[len(dataset)] = downsampled_df

                    if p.split("/")[2] == tn:
                        target.loc[len(target)] = 0
                    elif p.split("/")[2] == tm:
                        target.loc[len(target)] = 1
            else:
                continue
        else:
            continue

    dataset["status"] = target["status"]
    title = "ml_datasets/" + tn + "_" + tm + "_" + typeCell + "_" + str(n_features) + "features_" + str(low_window) \
            + "_" + str(high_window) + "_Hz.csv"
    dataset.to_csv(title, index=False)

    if learning:
        # training
        print("learning")
        dataset = pd.read_csv(title)
        X = dataset[dataset.columns[:-1]]
        y = dataset["status"]

        modelpath = "ml_models/" + tn + "_" + tm + "_" + typeCell + "_" + str(n_features) + "features_" + str(
            low_window) \
                    + "_" + str(high_window) + "_Hz/"

        modelname = "svm_linear"
        model_perf = modelpath + modelname + ".sav"
        ml.support_vector_machine(X, y, kernel='linear', save=True, modelname=modelname, modelpath=modelpath,
                                  decision_function_shape="ovr")
        ml.model_performance_analysis(model_perf, "svm", X, y, train_size=0.7)
        print(modelname)

        modelname = "rfc1000"
        model_perf = modelpath + modelname + ".sav"
        ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname=modelname, modelpath=modelpath, )
        ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)
        print(modelname)

        modelname = "rfc10000"
        model_perf = modelpath + modelname + ".sav"
        ml.random_forest_classifier(X, y, n_estimators=10000, save=True, modelname=modelname, modelpath=modelpath, )
        ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)
        print(modelname)


def make_std_cleaned_mean_merged_dataset(paths_freq_std_clean_mean_merged):
    nb_features = 30
    columns = list(range(0, nb_features))
    dataset = pd.DataFrame(columns=columns)
    target = pd.DataFrame(columns=["status", ])

    for p in paths_freq_std_clean_mean_merged:
        df = pd.read_csv(p)

        df_mean = pd.DataFrame(columns=["mean", ])
        df_mean["mean"] = df["mean"]

        samples = dpr.equal_samples(df_mean, 30)
        for s in samples:
            clean_fft_df, clean_freqs = spr.fast_fourier(s["mean"], 10000)
            ds_fft = dpr.down_sample(clean_fft_df[400:1200], nb_features, 'mean')

            dataset.loc[len(dataset)] = pd.Series(ds_fft)

            if p.split("/")[1][-2:] == "IN":
                target.loc[len(target)] = 1
            elif p.split("/")[1][-2:] == "NI":
                target.loc[len(target)] = 0

    dataset["status"] = target["status"]
    dataset.to_csv("ml_datasets/std_clean_mean_merged.csv", index=False)


def IN_NI_mean_channel_dataset_freq(paths):
    """
    Create a dataset for binary classification meaning all channels in temporal domain into one signal
    then transforming it in frequencies domain.
    :param paths: list of paths of temporal signals
    :return:
    """
    nb_features = 30
    columns = list(range(0, nb_features))
    dataset = pd.DataFrame(columns=columns)

    target = pd.DataFrame(columns=["status", ])

    for p in paths:
        df = pd.read_csv(p)

        df_no_x = df.loc[:, df.columns != "TimeStamp [µs]"]
        df_mean = pd.DataFrame(columns=["mean", ])
        df_mean['mean'] = df_no_x.mean(axis=1)

        samples = dpr.equal_samples(df_mean, 30)
        for s in samples:
            clean_fft_df, clean_freqs = spr.fast_fourier(s["mean"], 10000)
            ds_fft = dpr.down_sample(clean_fft_df[400:1200], nb_features, 'mean')

            dataset.loc[len(dataset)] = pd.Series(ds_fft)

            if p.split("/")[1][-2:] == "IN":
                target.loc[len(target)] = 1
            elif p.split("/")[1][-2:] == "NI":
                target.loc[len(target)] = 0

    dataset["status"] = target["status"]
    dataset.to_csv("ml_datasets/down_sampled_features_ds.csv", index=False)


def downsampled_n_features_low_high_Hz(paths, channels, n_features, low_ds, high_ds):
    columns = list(range(0, n_features))
    dataset = pd.DataFrame(columns=columns)

    target = pd.DataFrame(columns=["status", ])

    for p in paths:
        df = pd.read_csv(p)

        samples = dpr.equal_samples(df, 30)
        for s in samples:
            values = pd.DataFrame()
            for ch in channels[1:]:
                clean_fft_df, clean_freqs = spr.fast_fourier(s[ch], 10000)
                ds_fft = dpr.down_sample(clean_fft_df[low_ds * 2:high_ds * 2], n_features, 'mean')
                values[ch] = ds_fft

            values_mean = dpr.merge_all_columns_to_mean(values)
            dataset.loc[len(dataset)] = pd.Series(values_mean["mean"])

            if p.split("/")[1][-2:] == "IN":
                target.loc[len(target)] = 1
            elif p.split("/")[1][-2:] == "NI":
                target.loc[len(target)] = 0

    dataset["status"] = target["status"]
    title = "ml_datasets/downsampled" + str(n_features) + "features_" + str(low_ds) + "_" + str(high_ds) + "_Hz.csv"
    dataset.to_csv(title, index=False)


def IN_NI_all_channels_dataset(paths, channels):
    """
    create a dataset for binary classification considering each electrode as a feature.

    :param paths: list of all the csv paths
    :param channels: list of all the channels (columns names)
    :return:
    """
    dataset = pd.DataFrame(columns=channels)
    target = pd.DataFrame(columns=["status", ])

    for p in paths:
        df = pd.read_csv(p)

        samples = dpr.equal_samples(df, 30)
        for s in samples:
            values = []
            for ch in channels[1:]:
                clean_fft_df, clean_freqs = spr.fast_fourier(s[ch], 10000)
                ds_fft = dpr.down_sample(clean_fft_df[0:1200], 30, 'mean')
                values.append(ds_fft)

            dataset.loc[len(dataset)] = np.mean(values)
            if p.split("/")[1][-2:] == "IN":
                day = p.split("/")[1][1:3]
                target.loc[len(target)] = int(str("2" + day))
            elif p.split("/")[1][-2:] == "NI":
                day = p.split("/")[1][1:3]
                target.loc[len(target)] = int(str("1" + day))

    dataset["status"] = target["status"]
    dataset.to_csv("ml_datasets/day_stage_ds.csv", index=False)


def IN_NI_temp_all_channels_dataset(paths, channels, n_features):
    """
    create a dataset for binary classification considering each electrode as a feature.

    :param n_features: numer of features (number for the down sampling)
    :param paths: list of all the csv paths
    :param channels: list of all the channels (columns names)
    :return:
    """

    columns = list(range(0, n_features))
    dataset = pd.DataFrame(columns=columns)
    target = pd.DataFrame(columns=["status", ])

    for p in paths:
        df = pd.read_csv(p)

        samples = dpr.equal_samples(df, 30)
        for s in samples:
            values = []
            for ch in channels[1:]:
                clean_fft_df, clean_freqs = spr.fast_fourier(s[ch], 10000)
                ds_fft = dpr.down_sample(clean_fft_df[0:1200], 30, 'mean')
                values.append(ds_fft)

            dataset.loc[len(dataset)] = np.mean(values)
            if p.split("/")[1][-2:] == "IN":
                day = p.split("/")[1][1:3]
                target.loc[len(target)] = int(str("2" + day))
            elif p.split("/")[1][-2:] == "NI":
                day = p.split("/")[1][1:3]
                target.loc[len(target)] = int(str("1" + day))

    dataset["status"] = target["status"]
    dataset.to_csv("ml_datasets/day_stage_ds.csv", index=False)


def plot_spikes_count_per_day(paths_pr_hdd: list, threshold: float):
    x = ("T=0", "T=30", "T=4H", "T=24H", "T=48H", "T=6J", "T=7J")

    rep1_spike_cov_0 = []
    rep1_spike_cov_30 = []
    rep1_spike_cov_4 = []
    rep1_spike_cov_24 = []
    rep1_spike_cov_48 = []
    rep1_spike_cov_6 = []
    rep1_spike_cov_7 = []
    rep1_spike_ni_0 = []
    rep1_spike_ni_30 = []
    rep1_spike_ni_4 = []
    rep1_spike_ni_24 = []
    rep1_spike_ni_48 = []
    rep1_spike_ni_6 = []
    rep1_spike_ni_7 = []

    rep2_spike_cov_0 = []
    rep2_spike_cov_30 = []
    rep2_spike_cov_4 = []
    rep2_spike_cov_24 = []
    rep2_spike_cov_48 = []
    rep2_spike_cov_6 = []
    rep2_spike_cov_7 = []
    rep2_spike_ni_0 = []
    rep2_spike_ni_30 = []
    rep2_spike_ni_4 = []
    rep2_spike_ni_24 = []
    rep2_spike_ni_48 = []
    rep2_spike_ni_6 = []
    rep2_spike_ni_7 = []

    rep3_spike_cov_0 = []
    rep3_spike_cov_30 = []
    rep3_spike_cov_4 = []
    rep3_spike_cov_24 = []
    rep3_spike_cov_48 = []
    rep3_spike_cov_6 = []
    rep3_spike_cov_7 = []
    rep3_spike_ni_0 = []
    rep3_spike_ni_30 = []
    rep3_spike_ni_4 = []
    rep3_spike_ni_24 = []
    rep3_spike_ni_48 = []
    rep3_spike_ni_6 = []
    rep3_spike_ni_7 = []

    top_n = 15
    for p in paths_pr_hdd:
        print(p)
        df_full = pd.read_csv(p)
        df = dpr.top_N_electrodes(df_full, top_n, "TimeStamp [µs]")
        channels = df.columns

        all_channels_spikes = []
        for ch in channels[1:]:
            channel_spikes_count = []
            std = np.std(df[ch])
            for amp in df[ch]:
                if amp > threshold * std:
                    index = df[df[ch] == amp].index.tolist()
                    channel_spikes_count.append(index)
            all_channels_spikes.append(len(channel_spikes_count))

        if p.split("/")[2] == "T=0" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_1.csv":
            rep1_spike_cov_0 = all_channels_spikes
        elif p.split("/")[2] == "T=0" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_1.csv":
            rep1_spike_ni_0 = all_channels_spikes
        elif p.split("/")[2] == "T=30" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_1.csv":
            rep1_spike_cov_30 = all_channels_spikes
        elif p.split("/")[2] == "T=30" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_1.csv":
            rep1_spike_ni_30 = all_channels_spikes
        elif p.split("/")[2] == "T=4H" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_1.csv":
            rep1_spike_cov_4 = all_channels_spikes
        elif p.split("/")[2] == "T=4H" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_1.csv":
            rep1_spike_ni_4 = all_channels_spikes
        elif p.split("/")[2] == "T=24H" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_1.csv":
            rep1_spike_cov_24 = all_channels_spikes
        elif p.split("/")[2] == "T=24H" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_1.csv":
            rep1_spike_ni_24 = all_channels_spikes
        elif p.split("/")[2] == "T=48H" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_1.csv":
            rep1_spike_cov_48 = all_channels_spikes
        elif p.split("/")[2] == "T=48H" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_1.csv":
            rep1_spike_ni_48 = all_channels_spikes
        elif p.split("/")[2] == "T=6J" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_1.csv":
            rep1_spike_cov_6 = all_channels_spikes
        elif p.split("/")[2] == "T=6J" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_1.csv":
            rep1_spike_ni_6 = all_channels_spikes
        elif p.split("/")[2] == "T=7J" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_1.csv":
            rep1_spike_cov_7 = all_channels_spikes
        elif p.split("/")[2] == "T=7J" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_1.csv":
            rep1_spike_ni_7 = all_channels_spikes

        elif p.split("/")[2] == "T=0" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_2.csv":
            rep2_spike_cov_0 = all_channels_spikes
        elif p.split("/")[2] == "T=0" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_2.csv":
            rep2_spike_ni_0 = all_channels_spikes
        elif p.split("/")[2] == "T=30" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_2.csv":
            rep2_spike_cov_30 = all_channels_spikes
        elif p.split("/")[2] == "T=30" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_2.csv":
            rep2_spike_ni_30 = all_channels_spikes
        elif p.split("/")[2] == "T=4H" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_2.csv":
            rep2_spike_cov_4 = all_channels_spikes
        elif p.split("/")[2] == "T=4H" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_2.csv":
            rep2_spike_ni_4 = all_channels_spikes
        elif p.split("/")[2] == "T=24H" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_2.csv":
            rep2_spike_cov_24 = all_channels_spikes
        elif p.split("/")[2] == "T=24H" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_2.csv":
            rep2_spike_ni_24 = all_channels_spikes
        elif p.split("/")[2] == "T=48H" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_2.csv":
            rep2_spike_cov_48 = all_channels_spikes
        elif p.split("/")[2] == "T=48H" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_2.csv":
            rep2_spike_ni_48 = all_channels_spikes
        elif p.split("/")[2] == "T=6J" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_2.csv":
            rep2_spike_cov_6 = all_channels_spikes
        elif p.split("/")[2] == "T=6J" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_2.csv":
            rep2_spike_ni_6 = all_channels_spikes
        elif p.split("/")[2] == "T=7J" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_2.csv":
            rep2_spike_cov_7 = all_channels_spikes
        elif p.split("/")[2] == "T=7J" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_2.csv":
            rep2_spike_ni_7 = all_channels_spikes

        elif p.split("/")[2] == "T=0" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_3.csv":
            rep3_spike_cov_0 = all_channels_spikes
        elif p.split("/")[2] == "T=0" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_3.csv":
            rep3_spike_ni_0 = all_channels_spikes
        elif p.split("/")[2] == "T=30" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_3.csv":
            rep3_spike_cov_30 = all_channels_spikes
        elif p.split("/")[2] == "T=30" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_3.csv":
            rep3_spike_ni_30 = all_channels_spikes
        elif p.split("/")[2] == "T=4H" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_3.csv":
            rep3_spike_cov_4 = all_channels_spikes
        elif p.split("/")[2] == "T=4H" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_3.csv":
            rep3_spike_ni_4 = all_channels_spikes
        elif p.split("/")[2] == "T=24H" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_3.csv":
            rep3_spike_cov_24 = all_channels_spikes
        elif p.split("/")[2] == "T=24H" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_3.csv":
            rep3_spike_ni_24 = all_channels_spikes
        elif p.split("/")[2] == "T=48H" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_3.csv":
            rep3_spike_cov_48 = all_channels_spikes
        elif p.split("/")[2] == "T=48H" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_3.csv":
            rep3_spike_ni_48 = all_channels_spikes
        elif p.split("/")[2] == "T=6J" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_3.csv":
            rep3_spike_cov_6 = all_channels_spikes
        elif p.split("/")[2] == "T=6J" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_3.csv":
            rep3_spike_ni_6 = all_channels_spikes
        elif p.split("/")[2] == "T=7J" and p.split("/")[3] == "COV" and p.split("/")[4] == "pr_data_3.csv":
            rep3_spike_cov_7 = all_channels_spikes
        elif p.split("/")[2] == "T=7J" and p.split("/")[3] == "NI" and p.split("/")[4] == "pr_data_3.csv":
            rep3_spike_ni_7 = all_channels_spikes

    spike_cov_0 = [statistics.mean(k) for k in zip(rep1_spike_cov_0, rep2_spike_cov_0, rep3_spike_cov_0)]
    spike_cov_30 = [statistics.mean(k) for k in zip(rep1_spike_cov_30, rep2_spike_cov_30, rep3_spike_cov_30)]
    spike_cov_4 = [statistics.mean(k) for k in zip(rep1_spike_cov_4, rep2_spike_cov_4, rep3_spike_cov_4)]
    spike_cov_24 = [statistics.mean(k) for k in zip(rep1_spike_cov_24, rep2_spike_cov_24, rep3_spike_cov_24)]
    spike_cov_48 = [statistics.mean(k) for k in zip(rep1_spike_cov_48, rep2_spike_cov_48, rep3_spike_cov_48)]
    spike_cov_6 = [statistics.mean(k) for k in zip(rep1_spike_cov_6, rep2_spike_cov_6, rep3_spike_cov_6)]
    spike_cov_7 = [statistics.mean(k) for k in zip(rep1_spike_cov_7, rep2_spike_cov_7, rep3_spike_cov_7)]
    spike_ni_0 = [statistics.mean(k) for k in zip(rep1_spike_ni_0, rep2_spike_ni_0, rep3_spike_ni_0)]
    spike_ni_30 = [statistics.mean(k) for k in zip(rep1_spike_ni_30, rep2_spike_ni_30, rep3_spike_ni_30)]
    spike_ni_4 = [statistics.mean(k) for k in zip(rep1_spike_ni_4, rep2_spike_ni_4, rep3_spike_ni_4)]
    spike_ni_24 = [statistics.mean(k) for k in zip(rep1_spike_ni_24, rep2_spike_ni_24, rep3_spike_ni_24)]
    spike_ni_48 = [statistics.mean(k) for k in zip(rep1_spike_ni_48, rep2_spike_ni_48, rep3_spike_ni_48)]
    spike_ni_6 = [statistics.mean(k) for k in zip(rep1_spike_ni_6, rep2_spike_ni_6, rep3_spike_ni_6)]
    spike_ni_7 = [statistics.mean(k) for k in zip(rep1_spike_ni_7, rep2_spike_ni_7, rep3_spike_ni_7)]

    N = 7
    covMeans = (
        np.mean(spike_cov_0), np.mean(spike_cov_30), np.mean(spike_cov_4), np.mean(spike_cov_24), np.mean(spike_cov_48),
        np.mean(spike_cov_6), np.mean(spike_cov_7))
    covStd = (
        np.std(spike_cov_0), np.std(spike_cov_30), np.std(spike_cov_4), np.std(spike_cov_24), np.std(spike_cov_48),
        np.std(spike_cov_6), np.std(spike_cov_7))
    print(covStd)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, covMeans, width, color='royalblue', yerr=covStd)

    niMeans = (
        np.mean(spike_ni_0), np.mean(spike_ni_30), np.mean(spike_ni_4), np.mean(spike_ni_24), np.mean(spike_ni_48),
        np.mean(spike_ni_6), np.mean(spike_ni_7))
    niStd = (np.std(spike_ni_0), np.std(spike_ni_30), np.std(spike_ni_4), np.std(spike_ni_24), np.std(spike_ni_48),
             np.std(spike_ni_6), np.std(spike_ni_7))

    rects2 = ax.bar(ind + width, niMeans, width, color='seagreen', yerr=niStd)

    # add some
    ax.set_ylabel('number of spikes')
    ax.set_title('signal spikes count for top ' + str(top_n) + 'electrodes with std_threshold=' + str(threshold))
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(x)

    ax.legend((rects1[0], rects2[0]), ('COV', 'NI'))

    plt.show()


def classify_temporal_spike_top_tn_tm_typeCell(paths_pr, tn, tm, top_n, spike_threshold, mono_type, truncate,
                                               n_features, location, learning=True, ):
    """
    Classify, for the same cell type, between two time points

    :param location: location to save the resulting dataset
    :param paths_pr: paths of the processed files
    :param tn: first time point
    :param tm: second time point
    :param top_n: number of top channels to keep by stp
    :param spike_threshold: threshold to consider a spike
    :param mono_type: type of cell to considerate
    :param truncate: number of samples to divide the original signal to.
    :param n_features: number of feature to consider for down sampling
    :param learning: boolean. Activate machine learning algorithm
    """
    columns = list(range(0, n_features))
    columns.append("spikes")
    dataset = pd.DataFrame(columns=columns)

    target = pd.DataFrame(columns=["status", ])
    for p in paths_pr:
        if p.split("\\")[2] == mono_type:
            if p.split("\\")[1] == tn or p.split("\\")[1] == tm:
                print("path = ", p)
                df = pd.read_csv(p)
                # selecting top channels by their std
                df_top = dpr.top_N_electrodes(df, top_n, "TimeStamp [µs]")

                # counting mean spikes of topped channels
                spikes = dan.count_spikes_by_std_all_channels(df_top, spike_threshold).round(3)

                samples = dpr.equal_samples(df_top, truncate)
                for df_s in samples:
                    # mean between the topped channels
                    df_mean = dpr.merge_all_columns_to_mean(df_s, "TimeStamp [µs]").round(3)

                    # Downsampling by n
                    downsampled_df = dpr.down_sample(df_mean["mean"], n_features, 'mean')
                    downsampled_df.append(spikes)
                    # construct the dataset with n features
                    dataset.loc[len(dataset)] = downsampled_df

                    if p.split("\\")[1] == tn:
                        target.loc[len(target)] = 0
                    elif p.split("\\")[1] == tm:
                        target.loc[len(target)] = 1
            else:
                continue
        else:
            continue

    dataset["status"] = target["status"]
    folder = location + "/ml_datasets/"
    ff.verify_dir(folder)
    title = folder + "temp_spikes" + str(spike_threshold) + "_top" + str(
        top_n) + "_" + tn + "_" + tm + "_" + mono_type + "_" + str(
        n_features) + "features.csv"
    dataset.to_csv(title, index=False)

    if learning:
        # training
        print("learning")
        dataset = pd.read_csv(title)
        X = dataset[dataset.columns[:-1]]
        y = dataset["status"]

        folder = location + "/ml_models/"
        ff.verify_dir(folder)
        modelpath = folder + "temp_spikes" + str(spike_threshold) + "top" + str(
            top_n) + "_" + tn + "_" + tm + "_" + mono_type + "_" + str(
            n_features) + "features"

        modelname = "svm_linear"
        model_perf = modelpath + "/" + modelname + ".sav"
        ml.support_vector_machine(X, y, kernel='linear', save=True, modelname=modelname, modelpath=modelpath,
                                  decision_function_shape="ovr")
        ml.model_performance_analysis(model_perf, "svm", X, y, train_size=0.7)
        print(modelname)

        modelname = "rfc1000"
        model_perf = modelpath + "/" + modelname + ".sav"
        ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname=modelname, modelpath=modelpath, )
        ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)
        print(modelname)


def classify_temporal_top_tn_tm_typeCell(paths_pr, tn, tm, top_n, mono_type, truncate,
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
        if p.split("\\")[2] == mono_type:
            if p.split("\\")[1] == tn or p.split("\\")[1] == tm:
                print("path = ", p)
                df = pd.read_csv(p)
                # selecting top channels by their std
                df_top = dpr.top_N_electrodes(df, top_n, "TimeStamp [µs]")

                samples = dpr.equal_samples(df_top, truncate)
                for df_s in samples:
                    # mean between the topped channels
                    df_mean = dpr.merge_all_columns_to_mean(df_s, "TimeStamp [µs]").round(3)

                    # Downsampling by n
                    downsampled_df = dpr.down_sample(df_mean["mean"], n_features, 'mean')

                    # construct the dataset with n features
                    dataset.loc[len(dataset)] = downsampled_df

                    if p.split("\\")[1] == tn:
                        target.loc[len(target)] = 0
                    elif p.split("\\")[1] == tm:
                        target.loc[len(target)] = 1
            else:
                continue
        else:
            continue

    dataset["status"] = target["status"]
    folder = location + "/ml_datasets/"
    ff.verify_dir(folder)
    title = folder + "temp_top" + str(top_n) + "_" + tn + "_" + tm + "_" + mono_type + "_" + str(
        n_features) + "features.csv"
    dataset.to_csv(title, index=False)

    if learning:
        # training
        print("learning")
        dataset = pd.read_csv(title)
        X = dataset[dataset.columns[:-1]]
        y = dataset["status"]

        folder = location + "/ml_models/"
        ff.verify_dir(folder)
        modelpath = folder + "temp_top" + str(top_n) + "_" + tn + "_" + tm + "_" + mono_type + "_" + str(
            n_features) + "features"

        modelname = "svm_linear"
        model_perf = modelpath + "/" + modelname + ".sav"
        ml.support_vector_machine(X, y, kernel='linear', save=True, modelname=modelname, modelpath=modelpath,
                                  decision_function_shape="ovr")
        ml.model_performance_analysis(model_perf, "svm", X, y, train_size=0.7)
        print(modelname)

        modelname = "rfc1000"
        model_perf = modelpath + "/" + modelname + ".sav"
        ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname=modelname, modelpath=modelpath, )
        ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)
        print(modelname)


def classify_temporal_spike_top_monoTime_bothTypes(paths_pr, mono_time, top_n, spike_threshold, truncate,
                                                   n_features, location, learning=True, ):
    """
    Classify, for the two cell types, between the same time point respectively

    :param location: location to save the resulting dataset
    :param paths_pr: paths of the processed files
    :param mono_time: first time point
    :param tm: second time point
    :param top_n: number of top channels to keep by stp
    :param spike_threshold: threshold to consider a spike
    :param mono_type: type of cell to considerate
    :param truncate: number of samples to divide the original signal to.
    :param n_features: number of feature to consider for down sampling
    :param learning: boolean. Activate machine learning algorithm
    """
    columns = list(range(0, n_features))
    columns.append("spikes")
    dataset = pd.DataFrame(columns=columns)

    target = pd.DataFrame(columns=["status", ])
    for p in paths_pr:
        if p.split("\\")[1] == mono_time:
            print("path = ", p)
            df = pd.read_csv(p)
            # selecting top channels by their std
            df_top = dpr.top_N_electrodes(df, top_n, "TimeStamp [µs]")

            # counting mean spikes of topped channels
            spikes = dan.count_spikes_by_std_all_channels(df_top, spike_threshold).round(3)

            samples = dpr.equal_samples(df_top, truncate)
            for df_s in samples:
                # mean between the topped channels
                df_mean = dpr.merge_all_columns_to_mean(df_s, "TimeStamp [µs]").round(3)

                # Downsampling by n
                downsampled_df = dpr.down_sample(df_mean["mean"], n_features, 'mean')
                downsampled_df.append(spikes)
                # construct the dataset with n features
                dataset.loc[len(dataset)] = downsampled_df

                if p.split("\\")[2] == "NI":
                    target.loc[len(target)] = 0
                elif p.split("\\")[2] == "COV":
                    target.loc[len(target)] = 1
            else:
                continue
        else:
            continue

    dataset["status"] = target["status"]
    folder = location + "/ml_datasets/"
    ff.verify_dir(folder)
    title = folder + "temp_spikes" + str(spike_threshold) + "_top" + str(top_n) + "_" + mono_time + "_NI0_COV1_" + str(
        n_features) + "features.csv"
    dataset.to_csv(title, index=False)

    if learning:
        # training
        print("learning")
        dataset = pd.read_csv(title)
        X = dataset[dataset.columns[:-1]]
        y = dataset["status"]

        folder = location + "/ml_models/"
        ff.verify_dir(folder)
        modelpath = folder + "temp_spikes" + str(spike_threshold) + "_top" + str(
            top_n) + "_" + mono_time + "_NI0_COV1_" + str(
            n_features) + "features"

        modelname = "svm_linear"
        model_perf = modelpath + "/" + modelname + ".sav"
        ml.support_vector_machine(X, y, kernel='linear', save=True, modelname=modelname, modelpath=modelpath,
                                  decision_function_shape="ovr")
        ml.model_performance_analysis(model_perf, "svm", X, y, train_size=0.7)
        print(modelname)

        modelname = "rfc1000"
        model_perf = modelpath + "/" + modelname + ".sav"
        ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname=modelname, modelpath=modelpath, )
        ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)
        print(modelname)


def classify_temporal_top_monoTime_bothTypes(paths_pr, mono_time, top_n, truncate,
                                             n_features, location, learning=True, ):
    """
    Classify, for the two cell types, between the same time point respectively

    :param location: location to save the resulting dataset
    :param paths_pr: paths of the processed files
    :param mono_time: first time point
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
        if p.split("\\")[1] == mono_time:
            print("path = ", p)
            df = pd.read_csv(p)
            # selecting top channels by their std
            df_top = dpr.top_N_electrodes(df, top_n, "TimeStamp [µs]")

            samples = dpr.equal_samples(df_top, truncate)
            for df_s in samples:
                # mean between the topped channels
                df_mean = dpr.merge_all_columns_to_mean(df_s, "TimeStamp [µs]").round(3)

                # Downsampling by n
                downsampled_df = dpr.down_sample(df_mean["mean"], n_features, 'mean')
                # construct the dataset with n features
                dataset.loc[len(dataset)] = downsampled_df

                if p.split("\\")[2] == "NI":
                    target.loc[len(target)] = 0
                elif p.split("\\")[2] == "COV":
                    target.loc[len(target)] = 1
            else:
                continue
        else:
            continue

    dataset["status"] = target["status"]
    folder = location + "/ml_datasets/"
    ff.verify_dir(folder)
    title = folder + "temp_top" + str(top_n) + "_" + mono_time + "_NI0_COV1_" + str(
        n_features) + "features.csv"
    dataset.to_csv(title, index=False)

    if learning:
        # training
        print("learning")
        dataset = pd.read_csv(title)
        X = dataset[dataset.columns[:-1]]
        y = dataset["status"]

        folder = location + "/ml_models/"
        ff.verify_dir(folder)
        modelpath = folder + "temp_top" + str(top_n) + "_" + mono_time + "_NI0_COV1_" + str(
            n_features) + "features"

        modelname = "svm_linear"
        model_perf = modelpath + "/" + modelname + ".sav"
        ml.support_vector_machine(X, y, kernel='linear', save=True, modelname=modelname, modelpath=modelpath,
                                  decision_function_shape="ovr")
        ml.model_performance_analysis(model_perf, "svm", X, y, train_size=0.7)
        print(modelname)

        modelname = "rfc1000"
        model_perf = modelpath + "/" + modelname + ".sav"
        ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname=modelname, modelpath=modelpath, )
        ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)
        print(modelname)


def spike_classify_by_T0(paths_pr_hdd, it, truncate, n_features, spike_threshold, learning=True, key=""):
    columns = list(range(0, n_features))
    dataset = pd.DataFrame(columns=columns)

    target = pd.DataFrame(columns=["status", ])
    for p in paths_pr_hdd:
        if p.split("/")[2] == "T=0" or (p.split("/")[2] == it and p.split("/")[3] == "NI"):
            print("0 path = ", p)
            df = pd.read_csv(p)
            # df_top = dpr.top_N_electrodes(df, top_n, "TimeStamp [µs]")
            samples = dpr.equal_samples(df, truncate)
            channels = df.columns
            # detect spikes for each sample
            for df_s in samples:
                spikes_all_channels = pd.DataFrame()
                for ch in channels[1:]:
                    channel_spikes_count = []
                    std = np.std(df_s[ch])
                    for amp in df_s[ch]:
                        if amp > spike_threshold * std:
                            index = df_s[df_s[ch] == amp].index.tolist()
                            channel_spikes_count.append(index)

                    spikes_per_second = len(channel_spikes_count) / (60 / truncate)  # 1 min recording
                    spikes_all_channels[ch] = [spikes_per_second, ]
                # mean between the cleaned channels
                # construct the dataset with 300 features
                dataset = dataset.append(spikes_all_channels.values.tolist(), ignore_index=True)

                target.loc[len(target)] = 0

        elif p.split("/")[2] == it:
            if p.split("/")[3] == "COV":
                print("1 path = ", p)
                df = pd.read_csv(p)
                # df_top = dpr.top_N_electrodes(df, top_n, "TimeStamp [µs]")
                samples = dpr.equal_samples(df, truncate)
                channels = df.columns
                # detect spikes for each sample
                for df_s in samples:
                    spikes_all_channels = pd.DataFrame()
                    for ch in channels[1:]:
                        channel_spikes_count = []
                        std = np.std(df_s[ch])
                        for amp in df_s[ch]:
                            if amp > spike_threshold * std:
                                index = df_s[df_s[ch] == amp].index.tolist()
                                channel_spikes_count.append(index)

                        spikes_per_second = len(channel_spikes_count) / (60 / truncate)  # 1 min recording
                        spikes_all_channels[ch] = [spikes_per_second, ]
                    # mean between the cleaned channels

                    # construct the dataset with 300 features
                    dataset = dataset.append(spikes_all_channels.values.tolist(), ignore_index=True)

                    target.loc[len(target)] = 1

    dataset["status"] = target["status"]
    title = "ml_datasets/" + key + "unbalanced_spikes_" + str(spike_threshold) + "_" + it + ".csv"
    dataset.to_csv(title, index=False)

    if learning:
        # training
        print("learning")
        dataset = pd.read_csv(title)
        X = dataset[dataset.columns[:-1]]
        y = dataset["status"]

        modelpath = "ml_models/" + key + "unbalanced_spikes_" + str(spike_threshold) + "_" + it + "/"

        modelname = "svm_linear"
        model_perf = modelpath + modelname + ".sav"
        ml.support_vector_machine(X, y, kernel='linear', save=True, modelname=modelname, modelpath=modelpath,
                                  decision_function_shape="ovr")
        ml.model_performance_analysis(model_perf, "svm", X, y, train_size=0.7)
        print(modelname)

        modelname = "rfc1000"
        model_perf = modelpath + modelname + ".sav"
        ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname=modelname, modelpath=modelpath, )
        ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)
        print(modelname)

    print()


def spike_classify_monoType(paths_pr_hdd, tn, tm, mono_type, truncate, n_features, spike_threshold, learning=True,
                            key=""):
    columns = list(range(0, n_features))
    dataset = pd.DataFrame(columns=columns)

    target = pd.DataFrame(columns=["status", ])
    for p in paths_pr_hdd:
        if p.split("/")[3] == mono_type:
            if p.split("/")[2] == tn or p.split("/")[2] == tm:
                print("path = ", p)
                df = pd.read_csv(p)
                # df_top = dpr.top_N_electrodes(df, top_n, "TimeStamp [µs]")
                samples = dpr.equal_samples(df, truncate)
                channels = df.columns
                # detect spikes for each sample
                for df_s in samples:
                    spikes_all_channels = pd.DataFrame()
                    for ch in channels[1:]:
                        channel_spikes_count = []
                        std = np.std(df_s[ch])
                        for amp in df_s[ch]:
                            if amp > spike_threshold * std:
                                index = df_s[df_s[ch] == amp].index.tolist()
                                channel_spikes_count.append(index)

                        spikes_per_second = len(channel_spikes_count) / (60 / truncate)  # 1 min recording
                        spikes_all_channels[ch] = [spikes_per_second, ]
                    # mean between the cleaned channels

                    # construct the dataset with 300 features
                    dataset = dataset.append(spikes_all_channels.values.tolist(), ignore_index=True)

                    if p.split("/")[2] == tn:
                        target.loc[len(target)] = 0
                    elif p.split("/")[2] == tm:
                        target.loc[len(target)] = 1
            else:
                continue
        else:
            continue

    dataset["status"] = target["status"]
    title = "ml_datasets/" + key + "spikes_" + str(spike_threshold) + "_" + tn + "_" + tm + "_" + mono_type + ".csv"
    dataset.to_csv(title, index=False)

    if learning:
        # training
        print("learning")
        dataset = pd.read_csv(title)
        X = dataset[dataset.columns[:-1]]
        y = dataset["status"]

        modelpath = "ml_models/" + key + "spikes_" + str(spike_threshold) + "_" + tn + "_" + tm + "_" + mono_type + "/"

        modelname = "svm_linear"
        model_perf = modelpath + modelname + ".sav"
        ml.support_vector_machine(X, y, kernel='linear', save=True, modelname=modelname, modelpath=modelpath,
                                  decision_function_shape="ovr", class_weight={1: 3})
        ml.model_performance_analysis(model_perf, "svm", X, y, train_size=0.7)
        print(modelname)

        modelname = "rfc1000"
        model_perf = modelpath + modelname + ".sav"
        ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname=modelname, modelpath=modelpath, )
        ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)
        print(modelname)


def classify_std_range_spikes_top_monoTime_bothTypes(paths_pr, min_spike_thresh, max_spike_thresh, step_spike_thresh,
                                                     top_n, mono_time, truncate,
                                                     location, learning=True, add_random_label=False,
                                                     add_constant_label=False):
    """
    Classify, for the same cell type, between two time points.
    Features : std, (number of spikes per time, mean amplitude of spikes)*(different spikes std threshold)

    :param step_spike_thresh: step between min_spike_thresh and max_spike_thresh
    :param max_spike_thresh: minimum spike threshold based on std to consider
    :param min_spike_thresh: minimum spike threshold based on std to consider
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
    spikes_thresholds = []
    columns = ["std", ]

    for threshold in range(min_spike_thresh, max_spike_thresh + step_spike_thresh, step_spike_thresh):
        spikes_thresholds.append(threshold)
        columns.append("n_spikes_t" + str(threshold))
    if add_random_label:
        columns.append("rand_label")
    if add_constant_label:
        columns.append("const_label")

    dataset = pd.DataFrame(columns=columns)

    target = pd.DataFrame(columns=["status", ])
    for p in paths_pr:
        if p.split("\\")[3] == mono_time:
            print("path = ", p)
            df = pd.read_csv(p)
            # selecting top channels by their std
            df_top = dpr.top_N_electrodes(df, top_n, "TimeStamp [µs]")

            # divide by sample
            samples = dpr.equal_samples(df_top, truncate)
            for df_s in samples:
                # number of spikes and mean spikes amplitude per sample
                dataset_line = []
                sample_std = 0
                for threshold in range(min_spike_thresh, max_spike_thresh + step_spike_thresh, step_spike_thresh):
                    std, spikes, = dan.count_spikes_and_channel_std_by_std_all_channels(df_s, threshold)
                    dataset_line.append(spikes.round(3))
                    sample_std = std.round(3)
                dataset_line.insert(0, sample_std)
                if add_random_label:
                    dataset_line.append(randint(0, 100))
                if add_constant_label:
                    dataset_line.append(0)

                # construct the dataset with n features
                dataset.loc[len(dataset)] = dataset_line
                if p.split("\\")[4] == "NI":
                    target.loc[len(target)] = 0
                elif p.split("\\")[4] == "INF":
                    target.loc[len(target)] = 1
            else:
                continue
        else:
            continue

    dataset["status"] = target["status"]
    folder = location + "/datasets/"
    ff.verify_dir(folder)
    title = folder + f"std_range_spikes_min{str(min_spike_thresh)}_max{str(max_spike_thresh)}_step{str(step_spike_thresh)}" \
                     f"_top{str(top_n)}_{mono_time}_randLabel_{str(add_random_label)}_constLabel_{str(add_constant_label)}.csv"
    dataset.to_csv(title, index=False)

    if learning:
        # training
        print("learning")
        dataset = pd.read_csv(title)
        X = dataset[dataset.columns[:-1]]
        y = dataset["status"]

        folder = location + "/models/"
        ff.verify_dir(folder)
        modelpath = folder + f"std_range_spikes_min{str(min_spike_thresh)}_max{str(max_spike_thresh)}_step{str(step_spike_thresh)}" \
                             f"_top{str(top_n)}_{mono_time}_randLabel_{str(add_random_label)}_constLabel_{str(add_constant_label)}"

        modelname = "rfc1000"
        model_perf = modelpath + "/" + modelname + ".sav"
        ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname=modelname, modelpath=modelpath, )
        ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)


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


def classify_range_spikes_top_monoTime_bothTypes(paths_pr, min_spike_thresh, max_spike_thresh, step_spike_thresh,
                                                 top_n, mono_time, truncate,
                                                 location, learning=True, add_random_label=False):
    """
    Classify, for the same cell type, between two time points.
    Features : std, (number of spikes per time, mean amplitude of spikes)*(different spikes std threshold)

    :param step_spike_thresh: step between min_spike_thresh and max_spike_thresh
    :param max_spike_thresh: minimum spike threshold based on std to consider
    :param min_spike_thresh: minimum spike threshold based on std to consider
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
    spikes_thresholds = []
    columns = []

    for threshold in range(min_spike_thresh, max_spike_thresh + step_spike_thresh, step_spike_thresh):
        spikes_thresholds.append(threshold)
        columns.append("n_spikes_t" + str(threshold))
    if add_random_label:
        columns.append("rand_label")
    n_features = len(columns)

    dataset = pd.DataFrame(columns=columns)

    target = pd.DataFrame(columns=["status", ])
    for p in paths_pr:
        if p.split("\\")[3] == mono_time:
            print("path = ", p)
            df = pd.read_csv(p)
            # selecting top channels by their std
            df_top = dpr.top_N_electrodes(df, top_n, "TimeStamp [µs]")

            # divide by sample
            samples = dpr.equal_samples(df_top, truncate)
            for df_s in samples:
                # number of spikes and mean spikes amplitude per sample
                dataset_line = []
                for threshold in range(min_spike_thresh, max_spike_thresh + step_spike_thresh, step_spike_thresh):
                    std, spikes, = dan.count_spikes_and_channel_std_by_std_all_channels(df_s, threshold)
                    dataset_line.append(spikes.round(3))
                if add_random_label:
                    dataset_line.append(randint(0, 100))
                # construct the dataset with n features
                dataset.loc[len(dataset)] = dataset_line
                if p.split("\\")[4] == "NI":
                    target.loc[len(target)] = 0
                elif p.split("\\")[4] == "COV":
                    target.loc[len(target)] = 1
            else:
                continue
        else:
            continue

    dataset["status"] = target["status"]
    folder = location + "/ml_datasets/"
    ff.verify_dir(folder)
    title = folder + f"range_spikes_min{str(min_spike_thresh)}_max{str(max_spike_thresh)}_step{str(step_spike_thresh)}" \
                     f"_top{str(top_n)}_{mono_time}_randLabel_{str(add_random_label)}.csv"
    dataset.to_csv(title, index=False)

    if learning:
        # training
        print("learning")
        dataset = pd.read_csv(title)
        X = dataset[dataset.columns[:-1]]
        y = dataset["status"]

        folder = location + "/ml_models/"
        ff.verify_dir(folder)
        modelpath = folder + f"range_spikes_min{str(min_spike_thresh)}_max{str(max_spike_thresh)}_step{str(step_spike_thresh)}" \
                             f"_top{str(top_n)}_{mono_time}_randLabel_{str(add_random_label)}"

        modelname = "rfc1000"
        model_perf = modelpath + "/" + modelname + ".sav"
        ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname=modelname, modelpath=modelpath, )
        ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)
        print(modelname)


def make_filtered_numbered_freq_files(mono_time, top_n=35, truncate=30, n_features=300, lowcut=10):
    files = ff.get_all_files("E:\\Organoids\\four organoids per label\\")
    paths_pr = []
    columns = list(range(0, n_features))

    dataset = pd.DataFrame(columns=columns)
    identities = pd.DataFrame(columns=["organoid number", ])
    target = pd.DataFrame(columns=["status", ])
    for f in files:
        if "pr_" in f:
            paths_pr.append(f)
    print(paths_pr)
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
                    filtered = spr.filter_peak(df_s[ch], order=3, lowcut=lowcut)
                    clean_fft, clean_freqs = spr.fast_fourier(filtered, 10000)
                    fft_all_channels[ch] = clean_fft
                    fft_all_channels["frequency"] = clean_freqs
                # mean between the topped channels
                df_mean = dpr.merge_all_columns_to_mean(fft_all_channels, "frequency").round(3)

                # Downsampling by n
                downsampled_df = dpr.down_sample(df_mean["mean"], n_features, 'mean')

                # construct the dataset with n features
                dataset.loc[len(dataset)] = downsampled_df
                identities.loc[len(identities)] = p.split("\\")[5]
                if p.split("\\")[4] == "NI":
                    target.loc[len(target)] = 0
                elif p.split("\\")[4] == "INF":
                    target.loc[len(target)] = 1

    dataset.insert(loc=0, column="organoid number", value=identities["organoid number"])
    dataset["status"] = target["status"]
    folder = "Four organoids\\datasets\\"
    ff.verify_dir(folder)
    title = f"{folder}filtered_{lowcut}_numbered_frequency_top{str(top_n)}_nfeatures_{n_features}_{mono_time}.csv"
    dataset.to_csv(title, index=False)




def make_numbered_freq_files(mono_time, top_n=35, truncate=30, n_features=300):
    files = ff.get_all_files("E:\\Organoids\\four organoids per label\\")
    paths_pr = []
    columns = list(range(0, n_features))

    dataset = pd.DataFrame(columns=columns)
    identities = pd.DataFrame(columns=["organoid number", ])
    target = pd.DataFrame(columns=["status", ])
    for f in files:
        if "pr_" in f:
            paths_pr.append(f)
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
                identities.loc[len(identities)] = p.split("\\")[5]
                if p.split("\\")[4] == "NI":
                    target.loc[len(target)] = 0
                elif p.split("\\")[4] == "INF":
                    target.loc[len(target)] = 1

    dataset.insert(loc=0, column="organoid number", value=identities["organoid number"])
    dataset["status"] = target["status"]
    folder = "Four organoids\\datasets\\"
    ff.verify_dir(folder)
    title = f"{folder}numbered_frequency_top{str(top_n)}_nfeatures_{n_features}_{mono_time}.csv"
    dataset.to_csv(title, index=False)


def create_no_DS_freq_files_numbered():
    mono_time = "T=24H"
    top_n = 35
    n_features = 300
    files = ff.get_all_files("E:\\Organoids\\four organoids per label\\")
    paths_pr = []
    columns = list(range(0, n_features))
    dataset = pd.DataFrame(columns=columns)
    identities = pd.DataFrame(columns=["organoid number", ])
    target = pd.DataFrame(columns=["status", ])
    for f in files:
        if "pr_" in f:
            paths_pr.append(f)
    for p in paths_pr:
        if p.split("\\")[3] == mono_time:
            print("path = ", p)
            df = pd.read_csv(p)
            # selecting top channels by their std

            df_top = dpr.top_N_electrodes(df, top_n, "TimeStamp")

            channels = df_top.columns
            fft_all_channels = pd.DataFrame()

            # fft of the signal
            for ch in channels[1:]:
                clean_fft, clean_freqs = spr.fast_fourier(df_top[ch], 10000)
                fft_all_channels[ch] = clean_fft
                fft_all_channels["frequency"] = clean_freqs
            # mean between the topped channels
            df_mean = dpr.merge_all_columns_to_mean(fft_all_channels, "frequency").round(3)

            # Downsampling by n
            downsampled_df = dpr.down_sample(df_mean["mean"], n_features, 'mean')

            # construct the dataset with n features
            dataset.loc[len(dataset)] = downsampled_df
            identities.loc[len(identities)] = p.split("\\")[5]
            if p.split("\\")[4] == "NI":
                target.loc[len(target)] = 0
            elif p.split("\\")[4] == "INF":
                target.loc[len(target)] = 1

    dataset.insert(loc=0, column="organoid number", value=identities["organoid number"])
    dataset["status"] = target["status"]
    folder = "Four organoids\\datasets\\"
    ff.verify_dir(folder)
    title = f"{folder}noDS_numbered_frequency_top{str(top_n)}_nfeatures_{n_features}_{mono_time}.csv"
    dataset.to_csv(title, index=False)


def classify_filtered_frequency_top_monoTime_bothTypes(mono_time, top_n, truncate,
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
    files = ff.get_all_files("Four organoids\\datasets\\")
    filtered_paths = []
    for f in files:
        if "filtered_" in f:
            filtered_paths.append(f)
    for p in filtered_paths:
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


def get_barplot_by_foi_5percents_highest(mono_time, percentage=0.05, is_filtered=False, show=True, save=True,
                                         lowcut=10):
    filtered = ""
    if is_filtered:
        filtered = f"filtered_{lowcut}_"
    dataset_path = f"Four organoids\\datasets\\{filtered}highest_{percentage * 100}%features_frequency_top35_nfeatures_300_{mono_time}.csv"

    dataset = pd.read_csv(dataset_path)
    idx_foi = sorted(list(map(int, dataset.columns[1:-1].tolist())))

    roi_separator = int(len(idx_foi) / 2)
    print(roi_separator)
    first_roi = idx_foi[:roi_separator]
    second_roi = idx_foi[roi_separator:]
    print(first_roi, second_roi)
    numbered_dataset = pd.read_csv(
        f"Four organoids\\datasets\\{filtered}numbered_frequency_top35_nfeatures_300_{mono_time}.csv")

    roi_idx = 1
    for roi in (first_roi, second_roi):
        orga_inf1 = numbered_dataset[numbered_dataset["organoid number"] == "INF1"] \
            .drop("organoid number", axis=1).drop("status", axis=1).mean(axis=0)
        orga_inf2 = numbered_dataset[numbered_dataset["organoid number"] == "INF2"] \
            .drop("o'rganoid number", axis=1).drop("status", axis=1).mean(axis=0)
        orga_inf3 = numbered_dataset[numbered_dataset["organoid number"] == "INF3"] \
            .drop("organoid number", axis=1).drop("status", axis=1).mean(axis=0)
        orga_inf4 = numbered_dataset[numbered_dataset["organoid number"] == "INF4"] \
            .drop("organoid number", axis=1).drop("status", axis=1).mean(axis=0)
        orga_ni1 = numbered_dataset[numbered_dataset["organoid number"] == "NI1"] \
            .drop("organoid number", axis=1).drop("status", axis=1).mean(axis=0)
        orga_ni2 = numbered_dataset[numbered_dataset["organoid number"] == "NI2"] \
            .drop("organoid number", axis=1).drop("status", axis=1).mean(axis=0)
        orga_ni3 = numbered_dataset[numbered_dataset["organoid number"] == "NI3"] \
            .drop("organoid number", axis=1).drop("status", axis=1).mean(axis=0)
        orga_ni4 = numbered_dataset[numbered_dataset["organoid number"] == "NI4"] \
            .drop("organoid number", axis=1).drop("status", axis=1).mean(axis=0)

        mean_amplitude_orga_inf1 = np.mean([orga_inf1[i] for i in roi])
        mean_amplitude_orga_inf2 = np.mean([orga_inf2[i] for i in roi])
        mean_amplitude_orga_inf3 = np.mean([orga_inf3[i] for i in roi])
        mean_amplitude_orga_inf4 = np.mean([orga_inf4[i] for i in roi])
        std_across_organoids_inf = np.std([mean_amplitude_orga_inf1, mean_amplitude_orga_inf2,
                                           mean_amplitude_orga_inf3, mean_amplitude_orga_inf4])
        mean_across_organoids_inf = np.mean([mean_amplitude_orga_inf1, mean_amplitude_orga_inf2,
                                             mean_amplitude_orga_inf3, mean_amplitude_orga_inf4])

        mean_amplitude_orga_ni1 = np.mean([orga_ni1[i] for i in roi])
        mean_amplitude_orga_ni2 = np.mean([orga_ni2[i] for i in roi])
        mean_amplitude_orga_ni3 = np.mean([orga_ni3[i] for i in roi])
        mean_amplitude_orga_ni4 = np.mean([orga_ni4[i] for i in roi])
        std_across_organoids_ni = np.std([mean_amplitude_orga_ni1, mean_amplitude_orga_ni2,
                                          mean_amplitude_orga_ni3, mean_amplitude_orga_ni4])
        mean_across_organoids_ni = np.mean([mean_amplitude_orga_ni1, mean_amplitude_orga_ni2,
                                            mean_amplitude_orga_ni3, mean_amplitude_orga_ni4])

        print(f"infected roi{roi_idx}",
              [mean_amplitude_orga_inf1, mean_amplitude_orga_inf2, mean_amplitude_orga_inf3, mean_amplitude_orga_inf4])
        print(f"non infected roi{roi_idx}",
              [mean_amplitude_orga_ni1, mean_amplitude_orga_ni2, mean_amplitude_orga_ni3, mean_amplitude_orga_ni4])

        # plt.plot(orga_inf1, label="inf1")
        # plt.plot(orga_inf2, label="inf2")
        # plt.plot(orga_inf3, label="inf3")
        # plt.plot(orga_inf4, label="inf4")
        #
        # plt.plot(orga_ni1, label="ni1")
        # plt.plot(orga_ni2, label="ni2")
        # plt.plot(orga_ni3, label="ni3")
        # plt.plot(orga_ni4, label="ni4")

        plt.bar(0, mean_across_organoids_inf, yerr=std_across_organoids_inf)
        plt.scatter(0, mean_amplitude_orga_inf1, label="inf1")
        plt.scatter(0, mean_amplitude_orga_inf2, label="inf2")
        plt.scatter(0, mean_amplitude_orga_inf3, label="inf3")
        plt.scatter(0, mean_amplitude_orga_inf4, label="inf4")

        plt.bar(1, mean_across_organoids_ni, yerr=std_across_organoids_ni)
        plt.scatter(1, mean_amplitude_orga_ni1, label="ni1")
        plt.scatter(1, mean_amplitude_orga_ni2, label="ni2")
        plt.scatter(1, mean_amplitude_orga_ni3, label="ni3")
        plt.scatter(1, mean_amplitude_orga_ni4, label="ni4")

        plt.xticks((0, 1), ("INF", "NI"))
        plt.legend()
        title = f"mean of frequencies of interest (based on the {percentage * 100}% top features)\nfor region of interest {roi_idx} at {mono_time}" \
                f""
        plt.title(title)
        plt.legend()
        if save:
            plt.savefig(
                f"Four organoids\\figures\\{filtered}mean frequencies of interest ({percentage * 100}% foi) for ROI {roi_idx} {mono_time}.png")

        if show:
            plt.show()
        plt.close()

        roi_idx += 1


def calculget_freq_plot_by_foi_5percent_highest(mono_time, percentage=0.05, is_filtered=False, show=True, save=True,
                                                lowcut=10):
    filtered = ""
    if is_filtered:
        filtered = f"filtered_{lowcut}_"
    dataset_path = f"Four organoids\\datasets\\{filtered}highest_{percentage * 100}%features_frequency_top35_nfeatures_300_{mono_time}.csv"

    numbered_dataset = pd.read_csv(
        f"Four organoids\\datasets\\{filtered}numbered_frequency_top35_nfeatures_300_{mono_time}.csv")

    orga_inf1 = numbered_dataset[numbered_dataset["organoid number"] == "INF1"] \
        .drop("organoid number", axis=1).drop("status", axis=1).mean(axis=0)
    orga_inf2 = numbered_dataset[numbered_dataset["organoid number"] == "INF2"] \
        .drop("organoid number", axis=1).drop("status", axis=1).mean(axis=0)
    orga_inf3 = numbered_dataset[numbered_dataset["organoid number"] == "INF3"] \
        .drop("organoid number", axis=1).drop("status", axis=1).mean(axis=0)
    orga_inf4 = numbered_dataset[numbered_dataset["organoid number"] == "INF4"] \
        .drop("organoid number", axis=1).drop("status", axis=1).mean(axis=0)
    orga_ni1 = numbered_dataset[numbered_dataset["organoid number"] == "NI1"] \
        .drop("organoid number", axis=1).drop("status", axis=1).mean(axis=0)
    orga_ni2 = numbered_dataset[numbered_dataset["organoid number"] == "NI2"] \
        .drop("organoid number", axis=1).drop("status", axis=1).mean(axis=0)
    orga_ni3 = numbered_dataset[numbered_dataset["organoid number"] == "NI3"] \
        .drop("organoid number", axis=1).drop("status", axis=1).mean(axis=0)
    orga_ni4 = numbered_dataset[numbered_dataset["organoid number"] == "NI4"] \
        .drop("organoid number", axis=1).drop("status", axis=1).mean(axis=0)

    plt.plot(orga_inf1, label="inf1")
    plt.plot(orga_inf2, label="inf2")
    plt.plot(orga_inf3, label="inf3")
    plt.plot(orga_inf4, label="inf4")

    plt.plot(orga_ni1, label="ni1")
    plt.plot(orga_ni2, label="ni2")
    plt.plot(orga_ni3, label="ni3")
    plt.plot(orga_ni4, label="ni4")

    mean_inf = [np.mean(k) for k in zip(orga_inf1, orga_inf2, orga_inf3, orga_inf4)]
    mean_ni = [np.mean(k) for k in zip(orga_ni1, orga_ni2, orga_ni3, orga_ni4)]

    minus = [k - j for k, j in zip(mean_inf, mean_ni)]
    plt.plot(minus, label="minus")

    # plt.xticks(list(range(300)), list(range(0, 5000, 500)))
    plt.legend()
    title = f"processed frequencies top {percentage * 100}% as features for classification at {mono_time}"
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig(
            f"Four organoids\\figures\\{filtered}{title}.png")

    if show:
        plt.show()
    plt.close()


def get_scores_for_top_features(mono_time, percentage, learning=True, is_filtered=False, lowcut=10):
    filtered = ""
    if is_filtered:
        filtered = f"filtered_{lowcut}_"
    print("getting top features")
    idx_foi = gp.get_feature_of_interest(mono_time, detection_factor=2.0, plot=False, by_percentage=True,
                                         percentage=percentage, is_filtered=is_filtered, lowcut=lowcut)
    weights_title = f"Four organoids\\objects\\{filtered}numbered_frequency_top35_nfeatures_300_{mono_time}.fti"
    importance = pickle.load(open(weights_title, 'rb'))
    n_features = len(idx_foi)
    print("creating new dataset")
    plt.bar(idx_foi, [importance[i] for i in idx_foi])
    title = f"highest {percentage * 100}% important features"
    plt.title(title)
    plt.savefig(f"Four organoids\\figures\\{filtered}{title}.png")
    plt.close()
    dataset = pd.DataFrame()

    for p in (f"Four organoids\\datasets\\{filtered}numbered_frequency_top35_nfeatures_300_{mono_time}.csv",):
        df = pd.read_csv(p)
        dataset["organoid number"] = df["organoid number"]
        for col in idx_foi:
            dataset[str(col)] = df[str(col)]
        dataset["status"] = df["status"]

    folder = "Four organoids\\datasets\\"
    ff.verify_dir(folder)
    title = f"{folder}{filtered}highest_{percentage * 100}%features_frequency_top35_nfeatures_300_{mono_time}.csv"
    dataset.to_csv(title, index=False)

    if learning:
        # training
        print("learning")
        dataset = pd.read_csv(title)
        X = dataset[dataset.columns[1:-1]]
        y = dataset["status"]
        folder = "Four organoids\\models\\"
        modelpath = f"{folder}{filtered}highest_{percentage * 100}_importance_frequency_top35_nfeatures_300_{mono_time}"
        ff.verify_dir(modelpath)

        tp_mean = []
        tn_mean = []
        fp_mean = []
        fn_mean = []
        f1_mean = []
        precision_mean = []
        recall_mean = []
        kfold_mean = []
        for i in range(0, 10):
            print(i)
            modelname = "rfc1000"
            model_perf = modelpath + "\\" + modelname + ".sav"
            clf = ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname=modelname,
                                              modelpath=modelpath, )
            ml.get_feature_importance_rfc(model_perf, n_features,
                                          f"{folder}{filtered}highest_{percentage * 100}_importance_frequency_top35_nfeatures_300_{mono_time}_IT{i}.png")
            ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)

            n_splits = 10
            random_state = 1
            shuffle = True
            cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
            scores_kf = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
            tp, tn, fp, fn, precision, recall, f1score = ml.f1_score_calculation(y, X, clf, pos=1, neg=0)
            tp_mean.append(tp)
            tn_mean.append(tn)
            fp_mean.append(fp)
            fn_mean.append(fn)
            f1_mean.append(f1score)
            precision_mean.append(precision)
            recall_mean.append(recall)
            kfold_mean.append(np.mean(scores_kf))
        text = f"{mono_time}\nmean f1-score {round(np.mean(f1_mean), 3)}\n" \
               f"std f1-score {round(np.std(f1_mean), 3)}\n" \
               f"mean TP={round(np.mean(tp_mean), 3)}\n" \
               f"std TP={round(np.std(tp_mean), 3)}\n" \
               f"mean TN={round(np.mean(tn_mean), 3)}\n" \
               f"std TN={round(np.std(tn_mean), 3)}\n" \
               f"mean FP={round(np.mean(fp_mean), 3)}\n" \
               f"std FP={round(np.std(fp_mean), 3)}\n" \
               f"mean FN={round(np.mean(fn_mean), 3)}\n" \
               f"std FN={round(np.std(fn_mean), 3)}\n" \
               f"mean precision={round(np.mean(precision_mean), 3)}\n" \
               f"std precision={round(np.std(precision_mean), 3)}\n" \
               f"mean recall={round(np.mean(recall_mean), 3)}\n" \
               f"std recall={round(np.std(recall_mean), 3)}\n" \
               f"mean score K-fold={round(np.mean(kfold_mean), 3)}\n" \
               f"std score K-fold={round(np.std(kfold_mean), 3)}"
        with open(
                f"Four organoids\\models\\{filtered}highest_{percentage * 100}_importance_frequency_top35_nfeatures_300_{mono_time}\\scores_report.txt",
                "w") as f:
            f.write(text)


def make_complete_filtered_frequencies_files(mono_time="T=24H", lowcut=50, top_n=35, truncate=30, ds=True):
    files = ff.get_all_files("E:\\Organoids\\four organoids per label\\")
    paths_pr = []
    columns = ["INF1", "INF2", "INF3", "INF4", "NI1", "NI2", "NI3", "NI4"]

    dataset = pd.DataFrame()
    for f in files:
        if "pr_" in f:
            paths_pr.append(f)

    for p in paths_pr:
        if p.split("\\")[3] == mono_time:
            print("path = ", p)
            df = pd.read_csv(p)
            # selecting top channels by their std

            df_top = dpr.top_N_electrodes(df, top_n, "TimeStamp")

            channels = df_top.columns
            fft_all_channels = pd.DataFrame()

            # fft of the signal
            for ch in channels[1:]:
                filtered = spr.filter_peak(df_top[ch], order=3, lowcut=lowcut)
                clean_fft, clean_freqs = spr.fast_fourier(filtered, 10000)
                fft_all_channels[ch] = clean_fft
                fft_all_channels["frequency"] = clean_freqs
            # mean between the topped channels
            df_mean = dpr.merge_all_columns_to_mean(fft_all_channels, "frequency").round(3)

            identity = p.split("\\")[5] +"_"+ p.split("\\")[6][:-4]

            # construct the dataset with n features
            if ds:
                dataset[identity] = dpr.down_sample(df_mean["mean"], 300, "mean")
            else:
                dataset[identity] = df_mean["mean"]

    folder = "Four organoids\\datasets\\"
    ff.verify_dir(folder)
    title = f"{folder}DS_{ds}_filtered_{lowcut}_frequencies_per_organoid_repeats_{mono_time}.csv"
    dataset.to_csv(title, index=False)
