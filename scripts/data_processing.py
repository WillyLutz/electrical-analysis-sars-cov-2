import os
import re

import pandas as pd
import signal_processing as spr
import numpy as np
import machine_learning as ml
import FireFiles as ff

def get_hdd_raw_paths():
    path_per_day = ["E:/Organoids/T=0", "E:/Organoids/T=4H", "E:/Organoids/T=6J", "E:/Organoids/T=7J",
                    "E:/Organoids/T=24H", "E:/Organoids/T=30", "E:/Organoids/T=48H"]
    path_in_ni = ["COV", "NI"]
    files_paths = []
    for day in path_per_day:
        for in_ni in path_in_ni:
            for file in os.listdir(day + "/" + in_ni):
                if re.search("Analog", file):
                    files_paths.append(day + "/" + in_ni + "/" + file)
    return files_paths

def process_raw_to_csv(path):
    files = ff.get_all_files(path)
    for f in files:
        sign = os.path.basename(f).split("-")
        title = "pr_" + sign[0] + "-" + sign[1] + "-" + sign[2] + "-" + sign[3] + ".csv"

        raw_to_csv(f, title, 7, 600007)
        print(f)

def raw_to_csv(initial_path: str, csv_name: str, head_line=0, end_line=0):
    """
    Delete lines before a limit line in a text file.

    :param csv_name: saving name of the final csv
    :param initial_path: path of the text file.
    :param head_line: index number of the beginning limit line.
    :param end_line: index number of the end limit line
    :return: beheaded file.
    """
    csv_path = os.path.dirname(initial_path) + "\\" + csv_name
    with open(initial_path, "r") as f:
        if end_line == 0:
            end_line = len(f.readlines())
        rows = f.readlines()[head_line:end_line]
        separated_columns = []
        for r in rows:
            r = r[:-2]
            separated_columns.append(r.split(","))
        df = pd.DataFrame(separated_columns, columns=["TimeStamp [µs]", "47 (ID=0) [pV]", "48 (ID=1) [pV]", "46 (ID=2) [pV]", "45 (ID=3) [pV]", "38 (ID=4) [pV]",
            "37 (ID=5) [pV]", "28 (ID=6) [pV]", "36 (ID=7) [pV]", "27 (ID=8) [pV]", "17 (ID=9) [pV]", "26 (ID=10) [pV]",
            "16 (ID=11) [pV]", "35 (ID=12) [pV]", "25 (ID=13) [pV]", "15 (ID=14) [pV]", "14 (ID=15) [pV]",
            "24 (ID=16) [pV]", "34 (ID=17) [pV]", "13 (ID=18) [pV]", "23 (ID=19) [pV]", "12 (ID=20) [pV]",
            "22 (ID=21) [pV]", "33 (ID=22) [pV]", "21 (ID=23) [pV]", "32 (ID=24) [pV]", "31 (ID=25) [pV]",
            "44 (ID=26) [pV]", "43 (ID=27) [pV]", "41 (ID=28) [pV]", "42 (ID=29) [pV]", "52 (ID=30) [pV]",
            "51 (ID=31) [pV]", "53 (ID=32) [pV]", "54 (ID=33) [pV]", "61 (ID=34) [pV]", "62 (ID=35) [pV]",
            "71 (ID=36) [pV]", "63 (ID=37) [pV]", "72 (ID=38) [pV]", "82 (ID=39) [pV]", "73 (ID=40) [pV]",
            "83 (ID=41) [pV]", "64 (ID=42) [pV]", "74 (ID=43) [pV]", "84 (ID=44) [pV]", "85 (ID=45) [pV]",
            "75 (ID=46) [pV]", "65 (ID=47) [pV]", "86 (ID=48) [pV]", "76 (ID=49) [pV]", "87 (ID=50) [pV]",
            "77 (ID=51) [pV]", "66 (ID=52) [pV]", "78 (ID=53) [pV]", "67 (ID=54) [pV]", "68 (ID=55) [pV]",
            "55 (ID=56) [pV]", "56 (ID=57) [pV]", "58 (ID=58) [pV]", "57 (ID=59) [pV]"])
        df.to_csv(csv_path, index=False)


def extract_sample_from_dataset(dataset_path: str, percentage: int):
    """
    Cut and keep a percentage of a dataset. The part that will be kept is starting from the beginning of the dataset.

    :param dataset_path: path to the dataset
    :param percentage: the percentage of the original dataset that will be kept
    :return: Pandas DataFrame object. Size reduced dataset
    """
    csv_path = os.path.dirname(os.path.abspath(dataset_path)) + "\\sample_data.csv"

    df = pd.read_csv(dataset_path)
    length = len(df.index)
    sample_limit = int(percentage * length)

    dfc = df[:sample_limit]
    dfc.to_csv(csv_path, index=False)
    return dfc


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
        raise Exception("downsampling: length of data "+str(len(data.index)) + "< n "+str(n))


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


def organoid_cartography_std_based(standards, threshold):
    min_key = min(standards, key=standards.get)
    min_value = standards.get(min_key)
    max_key = max(standards, key=standards.get)
    max_value = standards.get(max_key)
    ratio = max_value / min_value

    electrodes_by_row = 8
    electrodes_number = len(standards) - 1
    current_drawn = 0
    nrows = int(np.ceil(electrodes_number / electrodes_by_row))
    organoid = np.zeros((nrows, electrodes_by_row))

    row = 0
    column = 0
    keylist = list(standards)
    while current_drawn <= electrodes_number:
        detected = False
        if standards.get(keylist[current_drawn]) > (min_value + threshold * (max_value - min_value)):
            detected = True
        if detected:
            organoid[row, column] = 1
        else:
            organoid[row, column] = 0

        # increments
        if column < electrodes_by_row - 1:
            column += 1
            current_drawn += 1
        elif column == electrodes_by_row - 1:
            row += 1
            column = 0
            current_drawn += 1

    # print section
    printable_organoid = organoid.tolist()
    for ir in range(len(printable_organoid)):
        for ic in range(len(printable_organoid[ir])):
            if printable_organoid[ir][ic] == 0.0:
                printable_organoid[ir][ic] = " |"
            else:
                printable_organoid[ir][ic] = "0|"

    for row in printable_organoid:
        print(''.join(row))
    print()


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


def truncate_fft_cleanSTD_window_downsampling_mergeChannels(paths_full, channels, truncate, fft_freq, std_threshold,
                                                            low_window, high_window, n_features):
    columns = list(range(0, n_features))
    dataset = pd.DataFrame(columns=columns)

    target = pd.DataFrame(columns=["status", ])
    # truncate into 30 samples TEMPORAL
    for p in paths_full:
        print("path = ", p)
        df = pd.read_csv(p)
        samples = equal_samples(df, truncate)

        # fft for each samples
        for df_s in samples:

            fft_all_channels = pd.DataFrame()
            for ch in channels[1:]:
                clean_fft, clean_freqs = spr.fast_fourier(df_s[ch], fft_freq)
                fft_all_channels[ch] = clean_fft
                fft_all_channels["frequency"] = clean_freqs

            # clean the channels in each samples
            df_clean = clean_std_threshold(fft_all_channels, std_threshold)

            # mean between the cleaned channels
            df_mean = merge_all_columns_to_mean(df_clean, "frequency")

            # windowing 200-3000 Hz
            windowed_df = pd.DataFrame(columns=["frequency", "mean", ])
            windowed_df["frequency"] = df_mean["frequency"][low_window * 2:high_window * 2]
            windowed_df["mean"] = df_mean["mean"][low_window * 2:high_window * 2]

            # Downsampling by 300
            downsampled_df = down_sample(windowed_df["mean"], n_features, 'mean')

            # construct the dataset with 300 features
            dataset.loc[len(dataset)] = downsampled_df

            if p.split("/")[1][-2:] == "IN":
                target.loc[len(target)] = 1
            elif p.split("/")[1][-2:] == "NI":
                target.loc[len(target)] = 0

    dataset["status"] = target["status"]
    title = "ml_datasets/downsampled" + str(n_features) + "features_" + str(low_window) + "_" + str(
        high_window) + "_Hz.csv"
    dataset.to_csv(title, index=False)

    # training
    dataset = pd.read_csv(title)
    X = dataset[dataset.columns[:-1]]
    y = dataset["status"]

    modelpath = "ml_models_" + str(n_features) + "features_" + str(low_window) + "_" + str(high_window) + "_Hz/"

    modelname = "svm_linear"
    model_perf = modelpath + modelname + ".sav"
    ml.support_vector_machine(X, y, kernel='linear', save=True, modelname=modelname, modelpath=modelpath,
                              decision_function_shape="ovr")
    ml.model_performance_analysis(model_perf, "svm", X, y, train_size=0.7)
    print(0)

    modelname = "rfc1000"
    model_perf = modelpath + modelname + ".sav"
    ml.random_forest_classifier(X, y, n_estimators=1000, save=True, modelname=modelname, modelpath=modelpath, )
    ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)
    print(0)

    modelname = "rfc10000"
    model_perf = modelpath + modelname + ".sav"
    ml.random_forest_classifier(X, y, n_estimators=10000, save=True, modelname=modelname, modelpath=modelpath, )
    ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)
    print(0)

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