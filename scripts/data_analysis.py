import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

import signal_processing as spr
import data_processing as dpr


def solo_analysis(path, y, full=True, upper=False, lower=False, show=False, threshold=1e7, distance=500,
                  title_signature=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    df = pd.read_csv(path)
    if full:
        axes[0].plot(df["TimeStamp [µs]"], df[y], color="blue")

    if upper:
        upeaks = spr.make_envelope(df[y], threshold=threshold, distance=distance)[0]
        axes[0].plot(df["TimeStamp [µs]"][upeaks], df[y][upeaks], color="green", label="upper envelope")
        axes[0].legend()
    if lower:
        lpeaks = spr.make_envelope(-df[y], threshold=threshold, distance=distance)[0]
        axes[0].plot(df["TimeStamp [µs]"][lpeaks], df[y][lpeaks], color="red", label="lower envelope")
        axes[0].legend()

    axes[0].set_title("raw temp " + path.split("/")[1])
    axes[0].set_xlabel('Timestamp [µs]')
    axes[0].set_ylabel('Intensity [pV]')

    duration = df["TimeStamp [µs]"][len(df) - 1]
    freq = len(df) / duration
    fft_df = np.fft.fft(df[y])
    freqs = np.fft.fftfreq(len(df), d=1 / freq)
    clean_fft_df = abs(fft_df)
    clean_freqs = abs(freqs)
    if full:
        axes[1].plot(clean_freqs, clean_fft_df, color="orange")
    if upper:
        upeaks_fft = spr.make_envelope(clean_fft_df, threshold=threshold, distance=distance)[0]
        axes[1].plot(clean_freqs[upeaks_fft], clean_fft_df[upeaks_fft], color="green", label="upper envelope")
        axes[1].legend()
    if lower:
        lpeaks_fft = spr.make_envelope(-clean_fft_df, threshold=threshold, distance=distance)[0]
        axes[1].plot(clean_freqs[lpeaks_fft], clean_fft_df[lpeaks_fft], color="red", label="lower envelope")
        axes[1].legend()

    axes[1].set_title("raw freq " + path.split("/")[1])
    axes[1].set_xlabel('freq (Hz)')
    axes[1].set_ylabel('spectral power')

    plt.savefig("RES\\solo\\" + path.split("/")[1] + "_" + y.split(" ")[0] + "_" + title_signature + ".png")
    if show:
        plt.show()

    plt.close(fig)


def electrode_temporal_comparison(paths, channels, threshold=1e7, distance=500, upper=True, title_signature="",
                                  max_plot_per_row=7):
    if len(channels) >= max_plot_per_row:
        subplots_in_width = max_plot_per_row
    else:
        subplots_in_width = len(channels)
    subplots_in_height = math.ceil(len(channels) / subplots_in_width)

    fig, axes = plt.subplots(subplots_in_height, subplots_in_width, sharey='all',
                             figsize=(4.5 * subplots_in_width, 4 * subplots_in_height), constrained_layout=True)

    for p in paths:
        r = 0
        c = 0
        df = pd.read_csv(p)
        for ch in channels:
            if upper:
                upeaks = spr.make_envelope(df[ch], threshold=threshold, distance=distance)[0]
                axes[r, c].plot(df["TimeStamp [µs]"][upeaks], df[ch][upeaks], label=p.split("/")[1])
                axes[r, c].legend()

            axes[r, c].set_title("raw temp " + ch.split(" ")[1][1:-1])
            axes[r, c].set_xlabel('Timestamp [µs]')
            axes[r, c].set_ylabel('Intensity [pV]')

            if c < subplots_in_width - 1:
                c += 1
            elif c == subplots_in_width - 1:
                c = 0
                r += 1
    plt.savefig("RES\\comparison\\" + paths[0].split("/")[1] + "_temp_" + title_signature + ".png")


def electrode_frequential_comparison(paths, channels, threshold=1e7, distance=500, upper=True, title_signature="",
                                     max_plot_per_row=7):
    if len(channels) >= max_plot_per_row:
        subplots_in_width = max_plot_per_row
    else:
        subplots_in_width = len(channels)
    subplots_in_height = math.ceil(len(channels) / subplots_in_width)

    fig, axes = plt.subplots(subplots_in_height, subplots_in_width, sharey='all',
                             figsize=(4.5 * subplots_in_width, 4 * subplots_in_height), constrained_layout=True)

    for p in paths:
        r = 0
        c = 0
        df = pd.read_csv(p)
        for ch in channels:
            freq = 10000
            fft_df = np.fft.fft(df[ch])
            freqs = np.fft.fftfreq(len(df), d=1 / freq)
            clean_fft_df = abs(fft_df)
            cleans = abs(freqs)
            if upper:
                upeaks_fft = spr.make_envelope(clean_fft_df, threshold=threshold, distance=distance)[0]
                axes[r, c].plot(cleans[upeaks_fft], clean_fft_df[upeaks_fft], label=p.split("/")[1])
                axes[r, c].legend()

            axes[r, c].set_title("raw freq " + ch.split(" ")[1][1:-1])
            axes[r, c].set_xlabel('freq (Hz)')
            axes[r, c].set_ylabel('spectral power')

            if c < subplots_in_width - 1:
                c += 1
            elif c == subplots_in_width - 1:
                c = 0
                r += 1
    plt.savefig("RES\\comparison\\" + paths[0].split("/")[1] + "_freq_" + title_signature + ".png")


def adaptable_subplot(elements_to_plot: list, max_plot_per_row=5, group_n_plots=1, sharey='none',
                      sharex='none', envelope_threshold=float(1), envelope_distance=1, envelope=False, save_plot=False,
                      plot_path='', show_plot=True, ratio_width=4, ratio_height=4.5, plot_std=False, plot_std_threshold=0):
    """
    Allow to plot different graphs in a single plot using different parameters.

    :param plot_std_threshold: If the value is different than 0, plot a horizontal line at std * plot_threshold
    :param plot_std: Boolean. If true, show the std as a horizontal line
    :param ratio_height: multiplication for height for the fig size.
    :param ratio_width: multiplication for width of the fig size.
    :param envelope: Boolean. If true, the signal will be enveloped.
    :param envelope_threshold: threshold parameter for the envelope.
    :param envelope_distance: distance parameter for the envelope
    :param elements_to_plot: a list of the different elements needed to plot. Each element must be a dictionary
        following the format element = {'dataframe': '' (str), 'x_feature': '' (str), 'y_feature': '' (str),
        'title': '' (str, can be empty), 'x_label': '' (str, can be empty), 'y_label': '' (str, can be empty),
        'color' : ''}
    :param max_plot_per_row: number of maximum graphs in a row.
    :param group_n_plots: number of graphs grouped by column.
    :param sharey: sharey parameter of the matplotlib.pyplot.plot() function
    :param sharex: sharex parameter of the matplotlib.pyplot.plot() function
    :param save_plot: Boolean. If true, the plot will be saved in files.
    :param plot_path: Str. If save_plot is true, it is the path where the plot will be saved.
    :param show_plot: Boolean. If True, the plot will be displayed.
    :return:
    """

    if len(elements_to_plot) >= max_plot_per_row:
        subplots_in_width = max_plot_per_row
    else:
        subplots_in_width = len(elements_to_plot)
    subplots_in_height = math.ceil(len(elements_to_plot) / subplots_in_width) * group_n_plots

    fig, axes = plt.subplots(subplots_in_height, subplots_in_width, sharey=sharey, sharex=sharex,
                             figsize=(ratio_width * subplots_in_width, ratio_height * subplots_in_height),
                             constrained_layout=True)
    r = 0
    c = 0

    for element in elements_to_plot:
        if envelope:
            enveloped = spr.make_envelope(element['y_feature'], envelope_threshold, envelope_distance)[0]
            axes[r, c].plot(element['x_feature'][enveloped], element['y_feature'][enveloped], color=element['color'])
        else:
            axes[r, c].plot(element['x_feature'], element['y_feature'], color=element['color'])
        axes[r, c].set_title(element['title'])
        axes[r, c].set_xlabel(element['x_label'])
        axes[r, c].set_ylabel(element['y_label'])

        if c < subplots_in_width - 1:
            c += 1
        elif c == subplots_in_width - 1:
            c = 0
            r += 1
    if save_plot:
        plt.savefig(plot_path)
    if show_plot:
        plt.show()


def fft_evolution_through_time(low_window, high_window, PATHS_HDD):
    """
    x axis: for each day, inf or non inf
    y axis:  in frequencies domain, mean all channels, then AUC / mean of a certain portion in frequencies.
    3 repetitions = 3 points, indicate std bar and mean
    :param PATHS_HDD: GLOBAL VARIABLE
    :param low_window: low limit of windowing (Hz)
    :param high_window: high limit of windowing (Hz)
    :return:
    """
    plt.figure(figsize=(10, 5))
    all_cov_values = []
    mean_cov_values = []
    std_cov_values = []
    all_ni_values = []
    mean_ni_values = []
    std_ni_values = []

    for p in PATHS_HDD:
        df1 = pd.read_csv(p + "/freq_data_1_mean.csv")
        df1_mean = df1[(df1['frequency'] >= low_window) & (df1['frequency'] <= high_window)]["mean"].mean()
        df2 = pd.read_csv(p + "/freq_data_2_mean.csv")
        df2_mean = df2[(df2['frequency'] >= low_window) & (df2['frequency'] <= high_window)]["mean"].mean()
        df3 = pd.read_csv(p + "/freq_data_3_mean.csv")
        df3_mean = df3[(df3['frequency'] >= low_window) & (df3['frequency'] <= high_window)]["mean"].mean()

        current_value = [df1_mean, df2_mean, df3_mean]

        if p.split("/")[3] == "COV":
            all_cov_values.append(current_value)
            mean_cov_values.append(np.mean(current_value))
            std_cov_values.append(np.std(current_value))
        else:
            all_ni_values.append(current_value)
            mean_ni_values.append(np.mean(current_value))
            std_ni_values.append(np.std(current_value))

    days = ("T=0", "T=30m", "T=4H", "T=24H", "T=48H", "T=6J", "T=7J",)

    N = 7
    ind = np.arange(N)

    width = 0.2
    plt.bar(ind, mean_cov_values, width=width, color="dodgerblue", label="infected", yerr=std_cov_values)
    plt.bar(ind + width, mean_ni_values, width=width, color="turquoise", label="non infected", yerr=std_ni_values)

    plt.xlabel("Timestamp")
    plt.ylabel("Amplitude - \nmean of the band [" + str(low_window) + "-" + str(high_window) + "] Hz from the FFT")
    plt.title("Amplitude for infected and healthy cells through time")

    plt.xticks(ind + width / 2, days)
    plt.legend(loc='best')
    print()

    plt.show()

def count_spikes_by_std_all_channels(df, threshold):
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

    return np.mean(all_channels_spikes)

def count_spikes_and_channel_std_by_std_all_channels(df, threshold):
    channels = df.columns
    all_channels_spikes = []
    all_channels_std = []
    for ch in channels[1:]:
        channel_spikes_count = []
        std = np.std(df[ch])
        all_channels_std.append(std)
        for amp in df[ch]:
            if amp > threshold * std:
                index = df[df[ch] == amp].index.tolist()
                channel_spikes_count.append(index)
        all_channels_spikes.append(len(channel_spikes_count))

    return np.mean(all_channels_std), np.mean(all_channels_spikes)


def plot_models_scores_by_days(cov_test_scores: tuple, ni_test_scores: tuple, title: str):
    days = ("T=0", "T=30m", "T=4H", "T=24H", "T=48H", "T=6J", "T=7J",)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cov = ax.plot(days, cov_test_scores, "-")
    ni = ax.plot(days, ni_test_scores, ":")

    # add some
    ax.set_ylabel('test scores')
    ax.set_xlabel('timestamp')
    ax.set_title(title)

    ax.legend((cov, ni), ('COV', 'NI'))

    plt.show()
