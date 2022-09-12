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
from scipy.integrate import trapz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from os import listdir
import re
import signal_processing as spr
import data_processing as dpr
import data_analysis as dan
import machine_learning as ml
from sklearn import preprocessing
import complete_procedures as cp
import PATHS as P
import data_processing as dp
import FireFiles as ff


def scores_heatmap(report_paths, title):
    print(report_paths)
    arr = np.zeros((3, 3))
    relative_index = {"T=0MIN": 0, "T=30MIN": 1, "T=24H": 2}
    for report_path in report_paths:
        timer = report_path.split("\\")[2].split("_")[4]
        target = report_path.split("\\")[3].split("_")[2]
        testing = ""
        with open(report_path, "r") as report:
            lines = report.readlines()
            for line in lines:
                if "Testing:" in line:
                    testing = line.split(" ")[2][:-1]
            arr[relative_index[target]][relative_index[timer]] = float(testing)

    ax = sns.heatmap(arr, annot=True, cmap="coolwarm", cbar_kws={'label': 'Testing scores'})
    ax.set_xticklabels(["T=0MIN", "T=30MIN", "T=24H"])
    ax.set_yticklabels(["T=0MIN", "T=30MIN", "T=24H"])
    ax.set_xlabel("Model training time point")
    ax.set_ylabel("Testing time point")
    # ax.xaxis.set_label_position('top')
    # ax.xaxis.tick_top()

    plt.title("Testing scores based on training time point \n"
              f"{title}")
    plt.tight_layout()
    plt.savefig(f"Four organoids\\figures\\testing scores {title}.png")


def features_importance_std_spikes3_randlabel_n_iterations(model_paths, save=False, title="", stacked=True):
    stds = []
    spikes = []
    randoms = []
    times = ["T=0", "T=30MIN", "T=24H"]
    std = []
    for model_path in model_paths:
        record_time = model_path.split("\\")[2].split("_")[4]
        model = ml.load_model(model_path)
        importances = model.feature_importances_
        std.append(np.std([tree.feature_importances_ for tree in model.estimators_], axis=0))

        forest_importances = pd.Series(importances, index=range(0, 3))
        stds.append(forest_importances[0])
        spikes.append(forest_importances[1])
        randoms.append(forest_importances[2])
    print(std)

    stds_std = []
    for s in std:
        stds_std.append(s[0])
    spikes_std = []
    for s in std:
        spikes_std.append(s[1])
    randoms_std = []
    for s in std:
        randoms_std.append(s[2])
    bot = [x + y for x, y in zip(spikes, randoms)]

    if stacked:
        plt.bar(times, stds, bottom=bot, color="lightgray", label="std", yerr=np.mean(stds_std))
        plt.bar(times, spikes, color="darkgray", bottom=randoms, label="spikes count t=3", yerr=np.mean(spikes_std))
        plt.bar(times, randoms, color="dimgray", label="random feature", yerr=np.mean(randoms_std))
    else:
        plt.bar(times, stds, color="lightgray", label="std", yerr=np.mean(stds_std))
        plt.bar(times, spikes, color="darkgray", label="spikes count t=3", yerr=np.mean(spikes_std))
        plt.bar(times, randoms, color="dimgray", label="random feature", yerr=np.mean(randoms_std))

    if stacked:
        title = title + " stacked"
    plt.title(title)
    plt.legend(bbox_to_anchor=(0.5, 1.0), loc='upper left')
    if save:
        plt.savefig("Four organoids\\figures\\" + title)
    else:
        plt.show()

    # forest_importances.plot.bar(ax=ax)
    # ax.set_title("Feature importances using MDI - " + str(path.split("\\")[-2]))
    # ax.set_ylabel("Mean decrease in impurity")
    # plt.tight_layout()
    # plt.savefig("figures/features_importance_std_spikes3_randlabel")


def features_importance_std_spikes3_randlabel(save=False, title=""):
    model_paths = ["Two organoids/ml_models/std_range_spikes_min3_max3_step1_top35_t=0_randLabel_True/rfc1000.sav",
                   "Two organoids/ml_models/std_range_spikes_min3_max3_step1_top35_t=30 min_randLabel_True/rfc1000.sav",
                   "Two organoids/ml_models/std_range_spikes_min3_max3_step1_top35_t=24 h_randLabel_True/rfc1000.sav", ]

    stds = []
    spikes = []
    randoms = []
    times = ["t=0", "t=30 min", "t=24 h"]
    for model_path in model_paths:
        record_time = model_path.split("/")[2].split("_")[7]
        model = ml.load_model(model_path)
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

        forest_importances = pd.Series(importances, index=range(0, 3))
        stds.append(forest_importances[0])
        spikes.append(forest_importances[1])
        randoms.append(forest_importances[2])

    bot = [x + y for x, y in zip(spikes, randoms)]
    plt.bar(times, stds, bottom=bot, color="silver", label="std")
    plt.bar(times, spikes, color="gray", bottom=randoms, label="spikes count t=3")
    plt.bar(times, randoms, color="black", label="random feature")

    plt.title(title)
    plt.legend(bbox_to_anchor=(0.5, 1.0), loc='upper left')
    if save:
        plt.savefig("figures/" + title)
    else:
        plt.show()

    # forest_importances.plot.bar(ax=ax)
    # ax.set_title("Feature importances using MDI - " + str(path.split("\\")[-2]))
    # ax.set_ylabel("Mean decrease in impurity")
    # plt.tight_layout()
    # plt.savefig("figures/features_importance_std_spikes3_randlabel")


def frequencies_cov_ni_separated_organoids(mono_time):
    fig, ax = plt.subplots()

    top_n = 35
    truncate = 30

    files = ff.get_all_files(f"E:\\Organoids\\four organoids per label\\{mono_time}")
    paths_pr = []
    for f in files:
        if "pr_" in f:
            paths_pr.append(f)
    identities = []
    for p in paths_pr:
        identity = p.split("\\")[5] + "_" + p.split("\\")[-1].split("_")[1].split(".")[0][:-3]
        if identity not in identities:
            if p.split("\\")[4] == "NI":
                identities.append(identity)
    organoids = {"frequency": []}
    for i in identities:
        organoids[i] = []

    for status in ["NI", ]:
        for p in paths_pr:

            identity = p.split("\\")[5] + "_" + p.split("\\")[-1].split("_")[1].split(".")[0][:-3]
            if p.split("\\")[3] == mono_time and p.split("\\")[4] == status:
                print(p)
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
                    organoids[identity].append(df_mean["mean"])
                    organoids["frequency"] = df_mean["frequency"]

    for identity in identities:
        arrays = [np.array(x) for x in organoids[identity]]
        mean_amplitudes = [np.mean(k) for k in zip(*arrays)]
        ax.errorbar(organoids["frequency"], mean_amplitudes, label=identity)

    ax.set_title(f"separated organoids NI frequencies at {mono_time}")
    ax.legend()
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("Spectral power")
    fig.savefig(f"Four organoids\\figures\\freq_NI_per_organoids_repeats_{mono_time}.png")
    plt.show()


def frequencies_cov_ni(mono_time):
    top_n = 35
    truncate = 30

    files = ff.get_all_files(f"E:\\Organoids\\four organoids per label\\{mono_time}")
    paths_pr = []
    for f in files:
        if "pr_" in f:
            paths_pr.append(f)

    for status in ["INF", "NI"]:
        amplitudes = []
        frequency = []
        for p in paths_pr:
            if p.split("\\")[3] == mono_time and p.split("\\")[4] == status:
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
                    amplitudes.append(df_mean["mean"])
                    frequency = df_mean["frequency"]
                    # Downsampling by n
                    # downsampled_df = dpr.down_sample(df_mean["mean"], n_features, 'mean')
        arrays = [np.array(x) for x in amplitudes]
        mean_amplitudes = [np.mean(k) for k in zip(*arrays)]
        plt.errorbar(frequency, mean_amplitudes, label=status)

    plt.title(f"COV and NI frequencies at {mono_time}")
    plt.legend()
    plt.xlabel("frequency (Hz)")
    plt.ylabel("Spectral power")
    plt.savefig(f"Four organoids\\figures\\freq_COV_NI_{mono_time}.png")
    plt.show()


def get_feature_of_interest_with_AUC(mono_time, AUC_limit=2.0):
    dataset = pd.read_csv(f"Four organoids\\datasets\\numbered_frequency_top35_nfeatures_300_{mono_time}.csv")
    # training
    print("learning")
    X_numbered = dataset[dataset.columns[:-1]]
    X = X_numbered.drop("organoid number", axis=1)
    y = dataset["status"]
    folder = "Four organoids\\models\\"

    model_directory = f"{folder}numbered_frequency_top_35_nfeatures_300_{mono_time}"
    ff.verify_dir(model_directory)
    importances_over_iterations = []
    std_over_iterations = []
    for i in range(1):
        model_name = "rfc1000"
        model_path = model_directory + "\\" + model_name + ".sav"
        clf = ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname=model_name,
                                          modelpath=model_directory, )
        importances = clf.feature_importances_
        mean = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)

        importances_over_iterations.append(mean)
        title = f"Four organoids\\figures\\numbered_frequency_top35_nfeatures_300_{mono_time}.png"
        indices = [x for x in range(0, 300)]
        plt.bar(indices, importances)
        plt.savefig(title, dpi=500)
        print(model_name)

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

    fig, ax = plt.subplots()
    ax.plot(hertz, mean_importances_over_iterations, color="red", linewidth=0.5)
    ax.fill_between(hertz, low_std, high_std, facecolor="blue", alpha=0.5)
    ax.fill_between(hertz, mean_importances_over_iterations, facecolor="green", alpha=0.5)

    total_AUC = trapz(mean_importances_over_iterations, dx=3)
    print(total_AUC)
    partial_AUC = total_AUC
    limit_AUC = 0
    while partial_AUC > 0.1 * total_AUC:
        partial_AUC = trapz([x - limit_AUC for x in mean_importances_over_iterations], dx=3)
        limit_AUC += 0.0001
        print(partial_AUC)
    title = f"features of interest at {mono_time} with {AUC_limit} factor detection"
    ax.axhline(y=limit_AUC, color="red")
    plt.show()
    # pickle.dump(fig, open(f"Four organoids\\figures\\{title}.plt", 'wb'))
    # plt.savefig(f"Four organoids\\figures\\{title}.png", dpi=500)

    idx_foi = []
    # for i in range(len(mean_importances_over_iterations)-1):
    #     if mean_importances_over_iterations[i] >= high_mean_thresh*factor_mean_thresh :
    #         idx_foi.append(i)
    #
    # return idx_foi

    # ax2 = pickle.load(open("Four organoids\\figures\\test.plt", 'rb'))
    # plt.show()


def get_feature_of_interest(mono_time, detection_factor=2.0, plot=True, by_percentage=False, percentage=0.0,
                            is_filtered=False, lowcut=10):
    filtered = ""
    if is_filtered:
        filtered = f"filtered_{lowcut}_"
    dataset = pd.read_csv(f"Four organoids\\datasets\\{filtered}numbered_frequency_top35_nfeatures_300_{mono_time}.csv")
    # training
    print("learning")
    X_numbered = dataset[dataset.columns[:-1]]
    X = X_numbered.drop("organoid number", axis=1)
    y = dataset["status"]
    folder = "Four organoids\\models\\"

    model_directory = f"{folder}{filtered}numbered_frequency_top_35_nfeatures_300_{mono_time}"
    ff.verify_dir(model_directory)
    importances_over_iterations = []
    std_over_iterations = []
    for i in range(10):
        model_name = "rfc1000"
        model_path = model_directory + "\\" + model_name + ".sav"
        clf = ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname=model_name,
                                          modelpath=model_directory, )
        importances = clf.feature_importances_
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

        title = f"features of interest at {mono_time} with {detection_factor} factor detection"
        pickle.dump(fig, open(f"Four organoids\\figures\\{title}.plt", 'wb'))
        plt.savefig(f"Four organoids\\figures\\{filtered}{title}.png", dpi=500)

    weights_title = f"{filtered}numbered_frequency_top35_nfeatures_300_{mono_time}.fti"
    pickle.dump(mean_importances_over_iterations, open(f"Four organoids\\objects\\{weights_title}", 'wb'))
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


def inf_ni_frequency_pattern_ROI_bar(mono_time="T=24H", process=False, ds=False):
    roi1 = (0, 330)
    roi2 = (3800, 4200)

    dsd = ""
    if ds:
        dsd = "DS"
    if process:
        path = f"Four organoids/datasets/{dsd}filtered_50_frequencies_per_organoid_repeats_{mono_time}.csv"
        df = pd.read_csv(path)
        repeated_organoids = {"INF1": [], "INF2": [], "INF3": [], "INF4": [],
                              "NI1": [], "NI2": [], "NI3": [], "NI4": []}
        for col in df.columns[1:]:
            status = col.split("_")[0]
            repeated_organoids[status].append(df[col])
        organoids = {"INF1": [], "INF2": [], "INF3": [], "INF4": [],
                     "NI1": [], "NI2": [], "NI3": [], "NI4": []}
        for key in repeated_organoids:
            arrays = [np.array(x) for x in repeated_organoids[key]]
            organoids[key] = [np.mean(k) for k in zip(*arrays)]

        title = f"Four organoids/datasets/{dsd}filtered_50_frequencies_per_organoid_{mono_time}.csv"
        frequencies_df = pd.DataFrame()

        for key in organoids:
            frequencies_df[key] = organoids[key]
        frequencies_df.to_csv(title, index=False)

    freq_df = pd.read_csv(f"Four organoids/datasets/{dsd}filtered_50_frequencies_per_organoid_{mono_time}.csv")

    roi1_signal_inf = []
    roi2_signal_inf = []
    roi1_signal_ni = []
    roi2_signal_ni = []
    if ds:
        ratio = 5000 / 300
        for ind in freq_df.index:
            frequency = ind*ratio
            if roi1[0] <= frequency <= roi1[1]:
                roi1_signal_inf.append(np.mean([freq_df["INF1"].iloc[[ind]], freq_df["INF2"].iloc[[ind]],
                                   freq_df["INF3"].iloc[[ind]], freq_df["INF4"].iloc[[ind]]]))
            if roi2[0] <= frequency <= roi2[1]:
                roi2_signal_inf.append(np.mean([freq_df["INF1"].iloc[[ind]], freq_df["INF2"].iloc[[ind]],
                                            freq_df["INF3"].iloc[[ind]], freq_df["INF4"].iloc[[ind]]]))
        for ind in freq_df.index:
            frequency = ind*ratio
            if roi1[0] <= frequency <= roi1[1]:
                roi1_signal_ni.append(np.mean([freq_df["NI1"].iloc[[ind]], freq_df["NI2"].iloc[[ind]],
                                   freq_df["NI3"].iloc[[ind]], freq_df["NI4"].iloc[[ind]]]))
            if roi2[0] <= frequency <= roi2[1]:
                roi2_signal_ni.append(np.mean([freq_df["NI1"].iloc[[ind]], freq_df["NI2"].iloc[[ind]],
                                            freq_df["NI3"].iloc[[ind]], freq_df["NI4"].iloc[[ind]]]))

        plt.bar(0, np.mean(roi1_signal_inf), yerr=np.std(roi1_signal_inf), label=f"Infected organoids", color="dimgrey")
        plt.bar(1, np.mean(roi1_signal_ni), yerr=np.std(roi1_signal_ni), label=f"Not infected organoids", color="silver")
        plt.bar(3, np.mean(roi2_signal_inf), yerr=np.std(roi2_signal_inf), color="dimgrey")
        plt.bar(4, np.mean(roi2_signal_ni), yerr=np.std(roi2_signal_ni), color="silver")

        plt.xticks((0.5, 3.5), (f"ROI 1 ({roi1[0]}-{roi1[1]} Hz)", f"ROI 2 ({roi2[0]}-{roi2[1]} Hz)"))
        plt.xlabel("Regions of Interest")
        plt.ylabel("mean spectral power")
        data1 = pd.DataFrame()
        data2 = pd.DataFrame()
        data1["roi1 signal inf"] = roi1_signal_inf
        data1["roi1 signal ni"] = roi1_signal_ni
        data2["roi2 signal inf"] = roi2_signal_inf
        data2["roi2 signal ni"] = roi2_signal_ni
        data1.to_csv(r"Four organoids/datasets/inf_ni_down_sampled_frequency_ROI1_bar_data.csv", index=False)
        data2.to_csv(r"Four organoids/datasets/inf_ni_down_sampled_frequency_ROI2_bar_data.csv", index=False)

    else:
        ratio = 5000 / 300000
        for ind in freq_df.index:
            frequency = ind * ratio
            if roi1[0] <= frequency <= roi1[1]:
                roi1_signal_inf.append(np.mean([freq_df["INF1"].iloc[[ind]], freq_df["INF2"].iloc[[ind]],
                                                freq_df["INF3"].iloc[[ind]], freq_df["INF4"].iloc[[ind]]]))
            if roi2[0] <= frequency <= roi2[1]:
                roi2_signal_inf.append(np.mean([freq_df["INF1"].iloc[[ind]], freq_df["INF2"].iloc[[ind]],
                                                freq_df["INF3"].iloc[[ind]], freq_df["INF4"].iloc[[ind]]]))
        for ind in freq_df.index:
            frequency = ind * ratio
            if roi1[0] <= frequency <= roi1[1]:
                roi1_signal_ni.append(np.mean([freq_df["NI1"].iloc[[ind]], freq_df["NI2"].iloc[[ind]],
                                               freq_df["NI3"].iloc[[ind]], freq_df["NI4"].iloc[[ind]]]))
            if roi2[0] <= frequency <= roi2[1]:
                roi2_signal_ni.append(np.mean([freq_df["NI1"].iloc[[ind]], freq_df["NI2"].iloc[[ind]],
                                               freq_df["NI3"].iloc[[ind]], freq_df["NI4"].iloc[[ind]]]))

        plt.bar(0, np.mean(roi1_signal_inf), yerr=np.std(roi1_signal_inf), label=f"Infected organoids", color="dimgrey")
        plt.bar(1, np.mean(roi1_signal_ni), yerr=np.std(roi1_signal_ni), label=f"Not infected organoids",
                color="silver")
        plt.bar(3, np.mean(roi2_signal_inf), yerr=np.std(roi2_signal_inf), color="dimgrey")
        plt.bar(4, np.mean(roi2_signal_ni), yerr=np.std(roi2_signal_ni), color="silver")

        plt.xticks((0.5, 3.5), (f"ROI 1 ({roi1[0]}-{roi1[1]} Hz)", f"ROI 2 ({roi2[0]}-{roi2[1]} Hz)"))
        plt.xlabel("Regions of Interest")
        plt.ylabel("mean spectral power")

        data1 = pd.DataFrame()
        data2 = pd.DataFrame()
        data1["roi1 signal inf"] = roi1_signal_inf
        data1["roi1 signal ni"] = roi1_signal_ni
        data2["roi2 signal inf"] = roi2_signal_inf
        data2["roi2 signal ni"] = roi2_signal_ni
        data1.to_csv(r"Four organoids/datasets/inf_ni_original_frequency_ROI1_bar_data.csv", index=False)
        data2.to_csv(r"Four organoids/datasets/inf_ni_original_frequency_ROI2_bar_data.csv", index=False)
    plt.legend()
    if ds:
        plt.savefig(f"Four organoids\\figures\\{dsd} filtered frequencies for {mono_time}")
    else:
        plt.savefig(f"Four organoids\\figures\\filtered frequencies for {mono_time}")
    plt.show()


def inf_ni_frequency_pattern(mono_time="T=24H", process=False, ds=False, zoom_in=False):
    dsd = ""
    if ds:
        dsd = "DS"
    if process:
        path = f"Four organoids/datasets/{dsd}filtered_50_frequencies_per_organoid_repeats_{mono_time}.csv"
        df = pd.read_csv(path)
        repeated_organoids = {"INF1": [], "INF2": [], "INF3": [], "INF4": [],
                              "NI1": [], "NI2": [], "NI3": [], "NI4": []}
        for col in df.columns[1:]:
            status = col.split("_")[0]
            repeated_organoids[status].append(df[col])
        organoids = {"INF1": [], "INF2": [], "INF3": [], "INF4": [],
                     "NI1": [], "NI2": [], "NI3": [], "NI4": []}
        for key in repeated_organoids:
            arrays = [np.array(x) for x in repeated_organoids[key]]
            organoids[key] = [np.mean(k) for k in zip(*arrays)]

        title = f"Four organoids/datasets/{dsd}filtered_50_frequencies_per_organoid_{mono_time}.csv"
        frequencies_df = pd.DataFrame()

        for key in organoids:
            frequencies_df[key] = organoids[key]
        frequencies_df.to_csv(title, index=False)

    freq_df = pd.read_csv(f"Four organoids/datasets/{dsd}filtered_50_frequencies_per_organoid_{mono_time}.csv")
    mean_inf = dpr.merge_all_columns_to_mean(freq_df[["INF1", "INF2", "INF3", "INF4"]])
    mean_ni = dpr.merge_all_columns_to_mean(freq_df[["NI1", "NI2", "NI3", "NI4"]])
    std_inf = []
    std_ni = []
    for ind in freq_df.index:
        std_inf.append(np.std([freq_df["INF1"].iloc[[ind]], freq_df["INF2"].iloc[[ind]],
                               freq_df["INF3"].iloc[[ind]], freq_df["INF4"].iloc[[ind]]]))
        std_ni.append(np.std([freq_df["NI1"].iloc[[ind]], freq_df["NI2"].iloc[[ind]],
                              freq_df["NI3"].iloc[[ind]], freq_df["NI4"].iloc[[ind]]]))

    length = [x for x in range(len(mean_inf))]
    std_inf_low = [mean_inf["mean"][x] - std_inf[x] for x in length]
    std_inf_high = [mean_inf["mean"][x] + std_inf[x] for x in length]
    std_ni_low = [mean_ni["mean"][x] - std_ni[x] for x in length]
    std_ni_high = [mean_ni["mean"][x] + std_ni[x] for x in length]

    plt.plot(mean_inf, label="mean inf", color="red")
    plt.fill_between(length, y1=std_inf_high, y2=std_inf_low, alpha=.4, color="red",
                     label="std inf across organoids")

    plt.plot(mean_ni, label="mean ni", color="blue")
    plt.fill_between(length, y1=std_ni_high, y2=std_ni_low, alpha=0.4, color="blue",
                     label="std ni across organoids")
    if zoom_in:
        if ds:
            ratio = 5000 / 300
            x_ds = [x for x in range(0, 301, 15)]
            x_freq = [int(x * ratio) for x in x_ds]
            plt.xticks(x_ds, x_freq, rotation=45)
            plt.xlabel("frequencies (50-5000 Hz) as features (0-300)")
            plt.ylabel("spectral power")
            data = pd.DataFrame()
            data["mean inf spectral power"] = mean_inf
            data["std inf spectral power"] = std_inf
            data["mean ni spectral power"] = mean_ni
            data["std ni spectral power"] = std_ni
            data.to_csv(r"Four organoids/datasets/down_sampled_frequencies_across_inf_ni_organoids.csv", index=False)
        else:
            ratio = 5000 / 300000
            x_ds = [x for x in range(0, 300001, 10000)]
            x_freq = [int(x * ratio) for x in x_ds]
            plt.xticks(x_ds, x_freq, rotation=45)
            plt.xlabel("frequencies (50-5000 Hz)")
            plt.ylabel("spectral power")
            data = pd.DataFrame()
            data["mean inf spectral power"] = mean_inf
            data["std inf spectral power"] = std_inf
            data["mean ni spectral power"] = mean_ni
            data["std ni spectral power"] = std_ni
            data.to_csv(r"Four organoids/datasets/original_frequencies_across_inf_ni_organoids.csv", index=False)

    plt.legend()
    if ds:
        plt.savefig(f"Four organoids\\figures\\{dsd} filtered frequencies for {mono_time}")
    else:
        plt.savefig(f"Four organoids\\figures\\filtered frequencies for {mono_time}")
    plt.show()
