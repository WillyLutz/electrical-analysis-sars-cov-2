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
import fiiireflyyy.firefiles as ff
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


def impact_of_stachel_on_classification_performance():
    path = os.path.join(P.RESULTS, "Stachel results.xlsx")
    df = pd.read_excel(path, index_col=False)

    recording = ["T=0MIN", "T=30MIN", "T=24H", ]
    recording_idx = [x for x in range(len(recording))]
    stachel = ["None", "T=0MIN", "T=24H", "T=24H+TTX"]
    stachel_idx = [x for x in range(len(stachel))]
    fig, axes = plt.subplots(len(recording), len(stachel), figsize=(4 * len(recording), 5 * len(stachel)))

    # --------- GETTING DATA ----------------------

    for r in recording_idx:
        for s in stachel_idx:
            sub_df = df[(df['Stachel addition'] == stachel[s]) & (df["Recording time"] == recording[r])].reset_index(
                drop=True)
            if not sub_df.empty:
                number_of_positive_entries = int(sub_df["TP cnt"]) + int(sub_df["FN cnt"])
                number_of_negative_entries = int(sub_df["TN cnt"]) + int(sub_df["FP cnt"])

                # -----------PLOTTING DATA --------------------

                acc = int(sub_df["Accuracy"][0] * 100)
                axes[r, s].bar(0, acc, edgecolor='black', color="black")
                axes[r, s].text(-0.1, acc / 2, str(acc) + "%", color="white")

                tp_ratio = int(sub_df["TP cnt"][0] / number_of_positive_entries * 100)
                fn_ratio = int(sub_df["FN cnt"][0] / number_of_positive_entries * 100)
                tp_cup = (int(sub_df["TP CUP"][0] * 100), sub_df["TP CUP std"][0])
                fn_cup = (int(sub_df["FN CUP"][0] * 100), sub_df["FN CUP std"][0])
                axes[r, s].bar(1, tp_ratio, edgecolor='black', color='darkgray')
                axes[r, s].bar(1, fn_ratio, bottom=tp_ratio, edgecolor='black', color='whitesmoke')
                axes[r, s].text(0.6, tp_ratio / 2, "CUP TP\n=" + str(tp_cup[0]) + "%")
                axes[r, s].text(0.6, fn_ratio / 2 + tp_ratio, "CUP FN\n=" + str(fn_cup[0]) + "%")

                tn_ratio = int(sub_df["TN cnt"][0] / number_of_negative_entries * 100)
                fp_ratio = int(sub_df["FP cnt"][0] / number_of_negative_entries * 100)
                tn_cup = (int(sub_df["TN CUP"][0] * 100), sub_df["TN CUP std"][0])
                fp_cup = (int(sub_df["FP CUP"][0] * 100), sub_df["FP CUP std"][0])
                axes[r, s].bar(2, tn_ratio, edgecolor='black', color='darkgray', label="Correctly predicted")
                axes[r, s].bar(2, fp_ratio, bottom=tn_ratio, edgecolor='black', color='whitesmoke',
                               label="Misclassified INF/NI")
                axes[r, s].text(1.6, tn_ratio / 2, "CUP TN\n=" + str(tn_cup[0]) + "%")
                axes[r, s].text(1.6, fp_ratio / 2 + tn_ratio, "CUP FP\n=" + str(fp_cup[0]) + "%")

                # -----------------------------------------------
                axes[r, s].set_axisbelow(True)
                axes[r, s].yaxis.grid(color='black', linestyle='dotted', alpha=0.7)
                axes[r, s].set_xticks([0, 1, 2], ["Model acc.", "INF", "NI"])
                axes[r, s].set_aspect("auto")
                axes[r, s].plot([], [], ' ', label="CUP: Confidence upon prediction")
                if s == 0:
                    axes[r, s].set_ylabel("Prediction ratio")

    cols = ["Stachel: None", "Stachel: T=0MIN", "Stachel: T=24H", "Stachel: T=24H+TTX"]
    rows = ["Recording:\nT=0MIN", "Recording:\nT=30MIN", "Recording:\nT=24H"]
    pad = 5
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline', )

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90.0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center')
    fig.suptitle("impact of Stachel on predictions done by the model trained at T=24H", fontsize=15)

    plt.show()
