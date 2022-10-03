from findpeaks import findpeaks
from scipy.signal import butter, filtfilt, freqz, iirnotch, find_peaks, savgol_filter
from scipy.fft import fft
from scipy.signal import butter, filtfilt, freqz, iirnotch
import numpy as np
import pandas as pd


def make_envelope(y, threshold, distance):
    peaks = find_peaks(y, threshold=threshold, distance=distance)
    return peaks


def butter_filter(signal, order, lowcut):
    freq = 10000
    nyq = 0.5 * freq
    low = lowcut / nyq
    high = nyq / nyq

    b, a = butter(order, low, btype='highpass')
    w, h = freqz(b, a)

    #xanswer = (w / (2 * np.pi)) * freq
    #yanswer = 20 * np.log10(abs(h))

    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal  # , xanswer, yanswer


def fast_fourier(dfy, freq):
    fft_df = np.fft.fft(dfy)
    freqs = np.fft.fftfreq(len(dfy), d=1 / freq)

    clean_fft_df = abs(fft_df)
    clean_freqs = abs(freqs[0:len(freqs // 2)])
    return clean_fft_df[:int(len(clean_fft_df) / 2)], clean_freqs[:int(len(clean_freqs) / 2)]


def filter_noise_by_fft_peak(path, channel, peak_threshold=9e10, peak_distance=50, filter_order=4, filter_margin=5):
    df = pd.read_csv(path)

    """FFT section"""
    duration = df["TimeStamp [µs]"][len(df) - 1] * 1e-6
    freq = len(df) / duration
    df_fft, freqs_fft = fast_fourier(df[channel], freq)
    df_fft = df_fft[:int(len(df_fft) / 2)]

    """Finding peaks we have to filter"""
    peaks = find_peaks(df_fft, threshold=peak_threshold, distance=peak_distance)
    peaks_freq = freqs_fft[peaks[0]]

    unfiltered_signal = df[channel]
    filtered_signal = unfiltered_signal
    filtered_fft = df_fft
    filtered_fft_freq = freqs_fft

    for p in peaks_freq:
        unfiltered_fft, unfiltered_freqs_fft = fast_fourier(unfiltered_signal, freq)

        filtered_signal, xanswer, yanswer = butter_filter(unfiltered_signal, p, filter_margin, filter_order, duration)

        filtered_fft, filtered_fft_freq = fast_fourier(filtered_signal, freq)

        unfiltered_signal = filtered_signal
        unfiltered_fft = filtered_fft

    return filtered_signal, filtered_fft, filtered_fft_freq
