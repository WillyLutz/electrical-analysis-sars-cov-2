import os
import re
import sys

import pandas as pd
import signal_processing as spr
import numpy as np
import machine_learning as ml
import fiiireflyyy.firefiles as ff
import PATHS as P
from pathlib import Path
import matplotlib.pyplot as plt


def concatenate_datasets():
    print()


def discard_outliers_by_iqr(df: pd.DataFrame, **kwargs):
    """
    remove outliers from a dataframe using the interquartile range method.

        Parameters
        ----------
        df: pd.Dataframe
            the dataframe containing the data.
            Must contain a column 'label'.

        **kwargs: keyword arguments
            low_percentile : float, default: 0.25
                the low percentile for IQR outliers removal.
            high_percentile : float, default: 0.75
                the high percentile for IQR outliers removal.
            iqr_limit_factor : float, default: 1.5
                the factor used to determine when the point is
                an outlier compared to the percentiles.
            show: bool, default: False
                Whether to show the resulting plot or not.
                WARNING: may cause randomly 'Key error: O'.
            save: str, default: ""
                Where to save the resulting plot.
                If empty, the plot will not be saved.
                WARNING : may cause randomly 'Key error: 0'.
            mode: {'capping', 'trimming'}, default: capping
                The method used to discard the outliers.

        Returns
        -------
        out: pd.Dataframe
            the dataframe without the outliers
    """
    options = {"low_percentile": 0.25,
               "high_percentile": 0.75,
               "iqr_limit_factor": 1.5,
               "show": False,
               "save": "",
               "mode": "capping"}
    options.update(kwargs)
    labels = list(set(list(df["label"])))
    labels_values = {}
    for n in range(len(labels)):
        labels_values[labels[n]] = n

    features = list(df.columns.values)
    features.remove('label')
    metrics = np.empty(shape=(len(labels), len(features)), dtype=object)

    # obtaining the metrics [lower_limit, upper_limit]
    for i_label in range(len(labels)):
        label = labels[i_label]
        sub_df = df.loc[df['label'] == label]
        for i_feat in range(len(features)):
            feat = features[i_feat]
            # finding the iqr
            low_percentile = sub_df[feat].quantile(options["low_percentile"])
            high_percentile = sub_df[feat].quantile(options["high_percentile"])
            iqr = high_percentile - low_percentile
            # finding upper and lower limit
            lower_limit = low_percentile - options["iqr_limit_factor"] * iqr
            upper_limit = high_percentile + options["iqr_limit_factor"] * iqr
            metrics[i_label][i_feat] = [lower_limit, upper_limit]

    discarded_df = pd.DataFrame(columns=features)
    discarded_labels = pd.DataFrame(columns=["label", ])

    for i in range(len(df.values)):
        discarded_row = []
        discarded_label = df['label'].iloc[i]
        for j in range(len(df.values[i]) - 1):
            lower_limit, upper_limit = metrics[labels_values[discarded_label], j]
            if options["mode"] == "capping":
                if df.iloc[i, j] > upper_limit:
                    discarded_row.append(upper_limit)
                elif df.iloc[i, j] < lower_limit:
                    discarded_row.append(lower_limit)
                else:
                    discarded_row.append(df.iloc[i, j])

            elif options["mode"] == "trimming":
                if lower_limit < df.iloc[i, j] < upper_limit:
                    discarded_row.append(df.iloc[i, j])
                else:
                    discarded_row.append(np.nan)  # todo: how to manage ? (not the same amount of values)
        discarded_labels.loc[len(discarded_labels)] = discarded_label
        discarded_df.loc[len(discarded_df)] = discarded_row
        #     if options["save"] or options["show"]:
        #         fig, axes = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
        #         sns.distplot(sub_df[feat], ax=axes[0, 0])
        #         axes[0, 0].set_ylabel("Density")
        #         axes[0, 0].set_xlabel("Amplitude [pV]")
        #         axes[0, 0].set_title("Before outliers removal")
        #         sns.distplot(modified_df[feat], ax=axes[0, 1])
        #         axes[0, 1].set_ylabel("Density")
        #         axes[0, 1].set_xlabel("Amplitude [pV]")
        #         axes[0, 1].set_title("After outliers removal")
        #         sns.boxplot(data=sub_df[feat], orient="h", ax=axes[1, 0])
        #
        #         sns.boxplot(data=modified_df[feat], orient="h", ax=axes[1, 1])
        #
        #         title = f"Feature {feat} distribution, label {label}"
        #         plt.suptitle(title)
        #         plt.tight_layout()
        #         if options["save"]:
        #             plt.savefig(os.path.join(options["save"], title + ".png"), dpi=500)
        #         if options["show"]:
        #             plt.show()
        #         plt.close()
        #
        #     sub_df[feat] = modified_df[feat]
        #     sub_df["label"] = modified_df["label"]
        # label_specific_dfs.append(sub_df)
    discarded_df["label"] = discarded_labels["label"]
    return discarded_df


def make_highest_features_dataset_from_complete_dataset(foi, complete_dataset, percentage=0.05, save=False):
    """
    Extract columns corresponding to features of interest from a complete dataset and saves/returns it.

    :param foi: the columns names of the features of interests
    :param complete_dataset: the complete dataset to extract the features from.
    :param percentage: for the title. Corresponding percentage for the highest features of interest.
    :return: dataframe of interest
    """
    df_foi = complete_dataset[[f for f in foi]]
    df_foi["label"] = complete_dataset["label"]
    if save:
        df_foi.to_csv(os.path.join(os.path.dirname(complete_dataset), f"highest {percentage * 100}% features - "
                                                                      f"{os.path.basename(complete_dataset)}"),
                      index=False)
    return df_foi


def make_raw_frequency_plots_from_pr_files(parent_dir, to_include=(), to_exclude=(), save=False, verbose=False):
    all_files = ff.get_all_files(os.path.join(parent_dir))
    files = []
    organoids = []
    for f in all_files:
        if all(i in f for i in to_include) and (not any(e in f for e in to_exclude)):
            files.append(f)

            organoid_key = os.path.basename(Path(f).parent.parent.parent.parent) + "_" + \
                           os.path.basename(Path(f).parent.parent) + "_" + os.path.basename(Path(f).parent)
            if organoid_key not in organoids:
                organoids.append(organoid_key)  # for parent: P.NOSTACHEL ==> - StachelINF2

            if verbose:
                print("added: ", f)
    number_of_organoids = len(organoids)

    print(number_of_organoids, organoids)
    columns = list(range(0, 300))
    dataset = pd.DataFrame(columns=columns)
    target = pd.DataFrame(columns=["label", ])

    n_processed_files = 0
    infected_organoids = []
    non_infected_organoids = []
    for f in files:
        print(f)
        organoid_key = os.path.basename(Path(f).parent.parent.parent.parent) + "_" + \
                       os.path.basename(Path(f).parent.parent) + "_" + os.path.basename(Path(f).parent)

        df = pd.read_csv(f)

        df_top = top_N_electrodes(df, 35, "TimeStamp [µs]")

        channels = df_top.columns

        fft_all_channels = pd.DataFrame()

        # fft of the signal
        for ch in channels[1:]:
            filtered = spr.butter_filter(df_top[ch], order=3, lowcut=50)
            clean_fft, clean_freqs = spr.fast_fourier(filtered, 10000)
            fft_all_channels[ch] = clean_fft
            fft_all_channels["Frequency [Hz]"] = clean_freqs
        # mean between the topped channels
        df_mean = merge_all_columns_to_mean(fft_all_channels, "Frequency [Hz]").round(3)
        downsampled_df = down_sample(df_mean["mean"], 300, 'mean')
        if "INF" in organoid_key:
            infected_organoids.append(downsampled_df)
            print("added infected: ", organoid_key, len(downsampled_df))
        if "NI" in organoid_key:
            non_infected_organoids.append(downsampled_df)
            print("added not infected: ", organoid_key, len(downsampled_df))

    non_infected_arrays = [np.array(x) for x in non_infected_organoids]
    mean_non_infected = [np.mean(k) for k in zip(*non_infected_arrays)]
    std_non_infected = [np.std(k) for k in zip(*non_infected_arrays)]
    low_std_non_infected = [mean_non_infected[x] - std_non_infected[x] for x in range(len(mean_non_infected))]
    high_std_non_infected = [mean_non_infected[x] + std_non_infected[x] for x in range(len(mean_non_infected))]
    plt.plot([int(x * 5000 / 300) for x in range(0, 300)], mean_non_infected, color='blue', label='not infected')
    plt.fill_between(x=[int(x * 5000 / 300) for x in range(0, 300)], y1=low_std_non_infected, y2=high_std_non_infected,
                     color='blue', alpha=.5)

    infected_arrays = [np.array(x) for x in infected_organoids]
    mean_infected = [np.mean(k) for k in zip(*infected_arrays)]
    std_infected = [np.std(k) for k in zip(*infected_arrays)]
    low_std_infected = [mean_infected[x] - std_infected[x] for x in range(len(mean_infected))]
    high_std_infected = [mean_infected[x] + std_infected[x] for x in range(len(mean_infected))]
    plt.plot([int(x * 5000 / 300) for x in range(0, 300)], mean_infected, color='red', label='infected')
    plt.fill_between(x=[int(x * 5000 / 300) for x in range(0, 300)], y1=low_std_infected, y2=high_std_infected,
                     color='red', alpha=.5)

    title = "Smoothened frequencies for all - Stachel organoids at T=24H"
    plt.title(title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [pV]")
    plt.legend()
    if save:
        plt.savefig(os.path.join(P.RESULTS, title + ".png"), dpi=1000)
    plt.show()

    #
    #
    #
    #     # construct the dataset with n features
    #     dataset.loc[len(dataset)] = downsampled_df
    #
    #     path = Path(f)
    #     if "NI" in os.path.basename(path.parent.parent):
    #         target.loc[len(target)] = 0
    #     elif "INF" in os.path.basename(path.parent.parent):
    #         target.loc[len(target)] = 1

    # if verbose:
    #     progress = int(np.ceil(n_processed_files / len(files) * 100))
    #     sys.stdout.write(f"\rProgression of processing all_files: {progress}%")
    #     sys.stdout.flush()
    #     n_processed_files += 1


def make_dataset_from_freq_files(parent_dir, title="", to_include=(), to_exclude=(), save=False, verbose=False,
                                 separate_organoids=False, select_organoids=False, target_keys=None, label_comment="",
                                 freq_range=()):
    """
    Use frequency files of format two columns (one column 'Frequencies [Hz]' and one column 'mean') to generate a
    dataset used for classification.

    :param separate_organoids:
    :param to_exclude:
    :param to_include:
    :param timepoint: The time point to study.
    :param title: name of the resulting dataset.
    :param parent_dir: name of the parent directory that contains all files to make the dataset from.
    :return:
    """  # todo : update on other projects
    if select_organoids is False:
        select_organoids = [1, 2, 3, 4, 5, 6, 7]
    if target_keys is None:
        target_keys = {'NI': 0, 'INF': 1}
    files = ff.get_all_files(os.path.join(parent_dir))
    freq_files = []
    for f in files:
        if all(i in f for i in to_include) and (not any(e in f for e in to_exclude)) and int(
                os.path.basename(Path(f).parent)) in select_organoids:
            freq_files.append(f)
    if verbose:
        print("added: ", freq_files)
    columns = list(range(0, 300))
    dataset = pd.DataFrame(columns=columns)
    target = pd.DataFrame(columns=["label", ])

    n_processed_files = 0
    for f in freq_files:
        df = pd.read_csv(f)
        if freq_range:
            # selecting the frequencies range
            df = df.loc[(df["Frequency [Hz]"] >= freq_range[0]) & (df["Frequency [Hz]"] <= freq_range[1])]
        # Downsampling by n
        downsampled_df = down_sample(df["mean"], 300, 'mean')

        # construct the dataset with n features
        dataset.loc[len(dataset)] = downsampled_df

        path = Path(f)

        if "NI" in os.path.basename(path.parent.parent):
            if separate_organoids:
                target.loc[len(target)] = "NI" + str(os.path.basename(path.parent)) + label_comment
            else:
                target.loc[len(target)] = "NI" + label_comment
        elif "INF" in os.path.basename(path.parent.parent):
            if separate_organoids:
                target.loc[len(target)] = "INF" + str(os.path.basename(path.parent)) + label_comment
            else:
                target.loc[len(target)] = "INF" + label_comment

        if verbose:
            progress = int(np.ceil(n_processed_files / len(freq_files) * 100))
            sys.stdout.write(f"\rProgression of processing files: {progress}%")
            sys.stdout.flush()
            n_processed_files += 1
    dataset["label"] = target["label"]
    if verbose:
        print("\n")
    if save:
        dataset.to_csv(os.path.join(P.DATASETS, title), index=False)
    return dataset


def make_filtered_sampled_freq_files():
    """
    make frequency files of format two columns (one column 'Frequencies [Hz]' and one column 'mean') from raw files.

    :return:
    """
    for timepoint in ("T=0MIN", "T=30MIN", "T=24H"):
        for stach in (P.NOSTACHEL, P.STACHEL, P.FOUR_ORGANOIDS):
            files = ff.get_all_files(os.path.join(stach, timepoint))
            raw_files = []
            for f in files:
                if "pr_" in f:
                    raw_files.append(f)

            for f in raw_files:
                print(f)
                df = pd.read_csv(f)
                df_top = top_N_electrodes(df, 35, "TimeStamp")
                samples = equal_samples(df_top, 30)
                channels = df_top.columns
                n_sample = 0
                for df_s in samples:
                    fft_all_channels = pd.DataFrame(columns=["Frequency [Hz]", "mean"])

                    # fft of the signal
                    for ch in channels[1:]:
                        filtered = spr.butter_filter(df_s[ch], order=3, lowcut=50)
                        clean_fft, clean_freqs = spr.fast_fourier(filtered, 10000)
                        fft_all_channels["Frequency [Hz]"] = clean_freqs
                        fft_all_channels[ch] = clean_fft

                    # mean between the topped channels
                    df_mean = merge_all_columns_to_mean(fft_all_channels, "Frequency [Hz]").round(3)

                    id = os.path.basename(f).split("_")[1]
                    df_mean.to_csv(os.path.join(os.path.dirname(f), f"freq_50hz_sample{n_sample}_{id}"), index=False)
                    n_sample += 1


def make_filtered_numbered_freq_files(mono_time, top_n=35, truncate=30, n_features=300, lowcut=10):
    files = ff.get_all_files("E:\\Organoids\\four organoids per label\\")
    paths_pr = []
    columns = list(range(0, n_features))

    dataset = pd.DataFrame(columns=columns)
    identities = pd.DataFrame(columns=["organoid number", ])
    target = pd.DataFrame(columns=["label", ])
    for f in files:
        if "pr_" in f:
            paths_pr.append(f)
    print(paths_pr)
    for p in paths_pr:
        if p.split("\\")[3] == mono_time:
            print("path = ", p)
            df = pd.read_csv(p)
            # selecting top channels by their std

            df_top = top_N_electrodes(df, top_n, "TimeStamp")

            samples = equal_samples(df_top, truncate)
            channels = df_top.columns
            for df_s in samples:
                fft_all_channels = pd.DataFrame()

                # fft of the signal
                for ch in channels[1:]:
                    filtered = spr.butter_filter(df_s[ch], order=3, lowcut=lowcut)
                    clean_fft, clean_freqs = spr.fast_fourier(filtered, 10000)
                    fft_all_channels[ch] = clean_fft
                    fft_all_channels["frequency"] = clean_freqs
                # mean between the topped channels
                df_mean = merge_all_columns_to_mean(fft_all_channels, "frequency").round(3)

                # Downsampling by n
                downsampled_df = down_sample(df_mean["mean"], n_features, 'mean')

                # construct the dataset with n features
                dataset.loc[len(dataset)] = downsampled_df
                identities.loc[len(identities)] = p.split("\\")[5]
                if p.split("\\")[4] == "NI":
                    target.loc[len(target)] = 0
                elif p.split("\\")[4] == "INF":
                    target.loc[len(target)] = 1

    dataset.insert(loc=0, column="organoid number", value=identities["organoid number"])
    dataset["label"] = target["label"]
    folder = "Four organoids\\datasets\\"
    ff.verify_dir(folder)
    title = f"{folder}filtered_{lowcut}_numbered_frequency_top{str(top_n)}_nfeatures_{n_features}_{mono_time}.csv"
    dataset.to_csv(title, index=False)


def down_sample(data, n: int, mode: str):
    if len(data.index) > n:
        step = int(len(data.index) / n)
        lower_limit = 0
        upper_limit = step
        ds_data = []
        if mode == 'mean':
            while upper_limit <= len(data):
                ds_data.append(np.mean(data[lower_limit:upper_limit]).round(3))
                lower_limit = upper_limit
                upper_limit += step
        excedent = len(ds_data) - n
        ds_data = ds_data[:-excedent or None]
        return ds_data
    else:
        raise Exception("downsampling: length of data " + str(len(data.index)) + "< n " + str(n))


def equal_samples(df, n):
    step = int(len(df) / n)
    lower_limit = 0
    upper_limit = step
    samples = []
    while upper_limit <= len(df):
        samples.append(df[lower_limit:upper_limit])
        lower_limit = upper_limit
        upper_limit += step
    return samples


def make_freq_file(path, channels, file_path):
    """
    Make the fft of a temporal signal file and save it
    :param path: path of the temporal signal file
    :param channels: all the column to apply the fft to.
    :param file_path: path of the new file
    :return:
    """
    df = pd.read_csv(path)
    freq_df = pd.DataFrame()
    for channel in channels[1:]:
        clean_fft, clean_freq = spr.fast_fourier(df[channel], 10000)
        freq_df["frequency"] = clean_freq
        freq_df[channel] = clean_fft

    folder_path = os.path.dirname(file_path)
    isExist = os.path.exists(folder_path)
    if not isExist:
        os.makedirs(folder_path)
        freq_df.to_csv(file_path, index=False)
    else:
        freq_df.to_csv(file_path, index=False)


def clean_std_threshold(df, threshold):
    """
    generate a dataframe where some channels are omitted because of their too low standard deviation based on the
    threshold.

    :param df: dataframe in the frequencies domain.
    :param threshold: acceptable standard deviation compared to max and min std of all channels. Between 0 and 1.
    :return:
    """
    chans = []  # all the headers
    for col in df.columns:
        chans.append(col)

    # getting the std
    standards = {}
    for ch in chans:
        standards[ch] = np.std(df[ch])

    # keeping the channel or not
    min_key = min(standards, key=standards.get)
    min_value = standards.get(min_key)
    max_key = max(standards, key=standards.get)
    max_value = standards.get(max_key)
    limit = min_value + threshold * (max_value - min_value)
    clean_channels = []
    for key in standards:
        if standards.get(key) > limit:
            clean_channels.append(key)

    dfc = pd.DataFrame()  # cleaned dataframe
    dfc["frequency"] = df["frequency"]
    dfc[clean_channels] = df[clean_channels]

    return dfc
    # write new file
    # folder_path = "exp_may2021freq_std_clean_" + str(threshold) + "/" + path.split("/")[1]
    # isExist = os.path.exists(folder_path)
    # if not isExist:
    #     os.makedirs(folder_path)
    #     dfc.to_csv(folder_path + "/std_cleaned_freq.csv", index=False)
    # else:
    #     dfc.to_csv(folder_path + "/std_cleaned_freq.csv", index=False)


def merge_all_columns_to_mean(df: pd.DataFrame, except_column=""):
    excepted_column = pd.DataFrame()
    if except_column != "":
        for col in df.columns:
            if except_column in col:
                except_column = col
        excepted_column = df[except_column]
        df.drop(except_column, axis=1, inplace=True)

    df_mean = pd.DataFrame(columns=["mean", ])
    df_mean['mean'] = df.mean(axis=1)

    if except_column != "":
        for col in df.columns:
            if except_column in col:
                except_column = col
        df_mean[except_column] = excepted_column

    return df_mean


def top_N_electrodes(df, n, except_column):
    """
    keep the n electrodes with the highest std
    :param dfc: dataframe to filter
    :param n: number of electrodes to keep
    :return: filtered dataframe
    """
    for col in df.columns:
        if except_column in col:
            except_column = col
    dfc = df.drop(except_column, axis=1)
    df_filtered = pd.DataFrame()
    df_filtered[except_column] = df[except_column]

    all_std = []
    for c in dfc.columns:
        all_std.append(np.std(dfc[c]))
    top_indices = sorted(range(len(all_std)), key=lambda i: all_std[i], reverse=True)[:n]

    for c in dfc.columns:
        id = c.split(")")[0].split("=")[1]
        if int(id) in top_indices:
            df_filtered[c] = dfc[c]

    return df_filtered
