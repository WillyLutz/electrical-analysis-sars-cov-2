import pickle
import os
from pathlib import Path
import PATHS as P
import pandas as pd
from sklearn.model_selection import train_test_split
import fiiireflyyy.firelearn as fl
import fiiireflyyy.firefiles as ff
import signal_processing as spr
import data_processing as dpr
import numpy as np
import machine_learning as ml
import matplotlib.pyplot as plt
import statistics
import data_analysis as dan
from random import randint
import get_plots as gp
from sklearn.model_selection import train_test_split, KFold, cross_val_score


def fig2c_Amplitude_for_Mock_CoV_Stachel_in_region_Hz_at_T_24H_for_all_organoids(min_freq=0, max_freq=500, batch="all organoids"):
    show = False
    batches = {"batch 1": [1, 2, 3, 4], "batch 2": [5, 6, 7], "all organoids": [1, 2, 3, 4, 5, 6, 7]}

    percentiles = 0.1
    min_feat = int(min_freq * 300 / 5000)
    max_feat = int(max_freq * 300 / 5000)
    cov = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                           to_include=("freq_50hz_sample", "T=24H"),
                                           to_exclude=("TTX", "STACHEL", "NI"),
                                           verbose=False,
                                           save=False,
                                           select_organoids=batches[batch],
                                           separate_organoids=False,
                                           freq_range=(min_freq, max_freq),
                                           label_comment="")

    discarded_cov = dpr.discard_outliers_by_iqr(cov, low_percentile=percentiles,
                                                high_percentile=1 - percentiles,
                                                mode='capping')

    ni = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                          to_include=("freq_50hz_sample", "T=24H"),
                                          to_exclude=("TTX", "STACHEL", "INF"),
                                          verbose=False,
                                          save=False,
                                          select_organoids=batches[batch],
                                          freq_range=(min_freq, max_freq),
                                          separate_organoids=False,
                                          label_comment="")

    discarded_ni = dpr.discard_outliers_by_iqr(ni, low_percentile=percentiles,
                                               high_percentile=1 - percentiles,
                                               mode='capping')

    stachel = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                           to_include=("freq_50hz_sample", "T=24H"),
                                           to_exclude=("TTX", "STACHEL", "NI"),
                                           verbose=False,
                                           save=False,
                                           select_organoids=batches["batch 2"],
                                           separate_organoids=False,
                                           freq_range=(min_freq, max_freq),
                                           label_comment="")

    discarded_stachel = dpr.discard_outliers_by_iqr(stachel, low_percentile=percentiles,
                                                high_percentile=1 - percentiles,
                                                mode='capping')

    discarded_ni.replace("NI", "Mock", inplace=True)
    discarded_cov.replace("INF", "SARS-CoV-2", inplace=True)
    discarded_stachel.replace("INF", "Stachel-treated SARS-CoV-2", inplace=True)
    global_df = pd.concat([discarded_ni, discarded_cov, discarded_stachel], ignore_index=True)

    data = pd.DataFrame(columns=["label", "mean amplitude [pV]", "std amplitude [pV]"])
    plt.figure(figsize=(9, 8))
    mock_df = global_df.loc[global_df["label"] == "Mock"]
    mock_region = mock_df.loc[:, mock_df.columns != "label"]
    plt.bar(0, np.mean(np.array(mock_region)), color='dimgray',
            yerr=np.std(np.array(mock_region)))
    data.loc[len(data)] = ["Mock", np.mean(np.array(mock_region)), np.std(np.array(mock_region))]

    cov_df = global_df.loc[global_df["label"] == "SARS-CoV-2"]
    cov_region = cov_df.loc[:, cov_df.columns != "label"]
    plt.bar(1, np.mean(np.array(cov_region)), color='darkgray',
            yerr=np.std(np.array(cov_region)))
    data.loc[len(data)] = ["SARS-CoV-2", np.mean(np.array(cov_region)), np.std(np.array(cov_region))]

    stachel_df = global_df.loc[global_df["label"] == "Stachel-treated SARS-CoV-2"]
    stachel_region = stachel_df.loc[:, stachel_df.columns != "label"]
    plt.bar(2, np.mean(np.array(stachel_region)), color='gray',
            yerr=np.std(np.array(stachel_region)))
    data.loc[len(data)] = ["Stachel-treated SARS-CoV-2", np.mean(np.array(stachel_region)), np.std(np.array(stachel_region))]

    plt.xticks([0, 1, 2], ["Mock", "SARS-CoV-2", "Stachel-treated SARS-CoV-2"], rotation=7, fontsize=20)
    plt.ylabel("Mean amplitude [pV]", fontsize=25)

    plt.savefig(
        os.path.join(P.FIGURES_PAPER, f"Fig2c Amplitude for Mock,CoV in {min_freq}-{max_freq} Hz at T=24H for {batch}.png"),
        dpi=1200)
    data.to_csv(
        os.path.join(P.FIGURES_PAPER,
                     f"Fig2c Amplitude for Mock,CoV,Stachel in {min_freq}-{max_freq} Hz at T=24H for {batch}.csv"),
        index=False)
    if show:
        plt.show()


def fig2b_Smoothened_frequencies_regionHz_Mock_CoV_Stachel_on_batch(min_freq=0, max_freq=500, batch="all organoids"):
    percentiles = 0.1
    batches = {"batch 1": [1, 2, 3, 4], "batch 2": [5, 6, 7], "all organoids": [1, 2, 3, 4, 5, 6, 7]}
    percentiles = 0.1
    show=False
    cov_nostachel = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                     to_include=("freq_50hz_sample", "T=24H"),
                                                     to_exclude=("TTX", "STACHEL", "NI"),
                                                     verbose=False,
                                                     save=False,
                                                     freq_range=(min_freq, max_freq),
                                                     select_organoids=batches[batch],
                                                     separate_organoids=False,
                                                     label_comment=" NOSTACHEL")

    discarded_cov_nostachel = dpr.discard_outliers_by_iqr(cov_nostachel, low_percentile=percentiles,
                                                          high_percentile=1 - percentiles,
                                                          mode='capping')

    ni_nostachel = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                    to_include=("freq_50hz_sample", "T=24H"),
                                                    to_exclude=("TTX", "STACHEL", "INF"),
                                                    verbose=False,
                                                    freq_range=(min_freq, max_freq),
                                                    save=False,
                                                    select_organoids=batches[batch],
                                                    separate_organoids=False,
                                                    label_comment=" NOSTACHEL")

    discarded_ni_nostachel = dpr.discard_outliers_by_iqr(ni_nostachel, low_percentile=percentiles,
                                                         high_percentile=1 - percentiles,
                                                         mode='capping')

    cov_stachel = dpr.make_dataset_from_freq_files(parent_dir=P.STACHEL,
                                                   to_include=("freq_50hz_sample", "T=24H",),
                                                   to_exclude=("TTX", "NI"),
                                                   verbose=False,
                                                   freq_range=(min_freq, max_freq),
                                                   save=False,
                                                   separate_organoids=False,
                                                   label_comment=" STACHEL0"
                                                   )
    discarded_cov_stachel = dpr.discard_outliers_by_iqr(cov_stachel, low_percentile=percentiles,
                                                        high_percentile=1 - percentiles,
                                                        mode='capping')

    discarded_ni_nostachel.replace("NI NOSTACHEL", "Mock", inplace=True)
    discarded_cov_nostachel.replace("INF NOSTACHEL", "SARS-CoV-2", inplace=True)
    discarded_cov_stachel.replace("INF STACHEL0", "Stachel-treated SARS-CoV-2", inplace=True)

    global_df = pd.concat([discarded_cov_nostachel, discarded_cov_stachel, discarded_ni_nostachel], ignore_index=True)

    plt.figure(figsize=(8, 8))
    plt.plot(global_df.loc[global_df["label"] == "Mock"].mean(axis=0), label="Mock", linewidth=1, color='g')
    plt.fill_between([x for x in range(0, 300)],
                     global_df.loc[global_df["label"] == "Mock"].mean(axis=0).subtract(
                         global_df.loc[global_df["label"] == "Mock"].std(axis=0)),
                     global_df.loc[global_df["label"] == "Mock"].mean(axis=0).add(
                         global_df.loc[global_df["label"] == "Mock"].std(axis=0)),
                     color='g', alpha=.5)
    plt.plot(global_df.loc[global_df["label"] == "SARS-CoV-2"].mean(axis=0), label="SARS-CoV-2", linewidth=1, color='b')
    plt.fill_between([x for x in range(0, 300)],
                     global_df.loc[global_df["label"] == "SARS-CoV-2"].mean(axis=0).subtract(
                         global_df.loc[global_df["label"] == "SARS-CoV-2"].std(axis=0)),
                     global_df.loc[global_df["label"] == "SARS-CoV-2"].mean(axis=0).add(
                         global_df.loc[global_df["label"] == "SARS-CoV-2"].std(axis=0)),
                     color='b', alpha=.5)
    plt.plot(global_df.loc[global_df["label"] == "Stachel-treated SARS-CoV-2"].mean(axis=0),
             label="Stachel-treated SARS-CoV-2", linewidth=1, color='r')
    plt.fill_between([x for x in range(0, 300)],
                     global_df.loc[global_df["label"] == "Stachel-treated SARS-CoV-2"].mean(axis=0).subtract(
                         global_df.loc[global_df["label"] == "Stachel-treated SARS-CoV-2"].std(axis=0)),
                     global_df.loc[global_df["label"] == "Stachel-treated SARS-CoV-2"].mean(axis=0).add(
                         global_df.loc[global_df["label"] == "Stachel-treated SARS-CoV-2"].std(axis=0)),
                     color='r', alpha=.5)
    ratio = (max_freq - min_freq) / 300
    x_ds = [x for x in range(0, 301, 15)]
    x_freq = [int(x * ratio + min_freq) for x in x_ds]
    plt.xticks(x_ds, x_freq, rotation=45)
    plt.xlabel("Smoothened frequencies [Hz]", fontsize=25)
    plt.ylabel("Amplitude [pV]", fontsize=25)
    plt.legend(prop={'size': 20})
    plt.savefig(os.path.join(P.FIGURES_PAPER,
                             f"Fig2b Smoothened frequencies Mock-CoV-Stachel on {min_freq}-{max_freq}Hz {batch}.png"),
                dpi=1200)
    if show:
        plt.show()


def fig2a_PCA_on_regionHz_all_organoids_for_Mock_CoV_test_stachel(min_freq, max_freq, batch="all organoids"):
    percentiles = 0.1
    n_components = 2
    batches = {"batch 1": [1, 2, 3, 4], "batch 2": [5, 6, 7], "all organoids": [1, 2, 3, 4, 5, 6, 7]}

    covni = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                             to_include=("freq_50hz_sample", "T=24H"),
                                             to_exclude=("TTX", "STACHEL",),
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_organoids=batches[batch],
                                             separate_organoids=False,
                                             label_comment="")

    discarded_covni = dpr.discard_outliers_by_iqr(covni, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    covni_stachel = dpr.make_dataset_from_freq_files(parent_dir=P.STACHEL,
                                                     to_include=("freq_50hz_sample", "T=24H"),
                                                     to_exclude=("TTX", "NI"),
                                                     verbose=False,
                                                     save=False,
                                                     freq_range=(min_freq, max_freq),
                                                     select_organoids=batches["batch 2"],
                                                     separate_organoids=False,
                                                     label_comment="")

    discarded_covni_stachel = dpr.discard_outliers_by_iqr(covni_stachel, low_percentile=percentiles,
                                                          high_percentile=1 - percentiles,
                                                          mode='capping')

    discarded_covni.replace("NI", "Mock", inplace=True)
    discarded_covni.replace("INF", "SARS-CoV-2", inplace=True)
    discarded_covni_stachel.replace("NI", "Stachel-treated Mock", inplace=True)
    discarded_covni_stachel.replace("INF", "Stachel-treated SARS-CoV-2", inplace=True)

    pca, pcdf, _ = fl.fit_pca(discarded_covni, n_components=n_components)
    stachel_pcdf = fl.apply_pca(pca, discarded_covni_stachel)
    global_df = pd.concat([pcdf, stachel_pcdf], ignore_index=True)

    ml.plot_pca(global_df, n_components=2, show=False,
                title=f"Fig2a PCA on {min_freq}-{max_freq}Hz {batch} for Mock,CoV, applied on stachel",
                points=True, metrics=True, savedir=P.FIGURES_PAPER, )


def fig1h_Confusion_matrix_train_on_batch_Mock_CoV_in_region_Hz(min_freq=300, max_freq=5000,
                                                                batch="all organoids",
                                                                ):
    percentiles = 0.1
    batches = {"batch 1": [1, 2, 3, 4], "batch 2": [5, 6, 7], "all organoids": [1, 2, 3, 4, 5, 6, 7]}

    covni_24 = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                to_include=("freq_50hz_sample", "T=24H"),
                                                to_exclude=("TTX", "STACHEL",),
                                                verbose=False,
                                                save=False,
                                                freq_range=(min_freq, max_freq),
                                                select_organoids=batches[batch],
                                                separate_organoids=False,
                                                label_comment=f"")

    discarded_covni_24 = dpr.discard_outliers_by_iqr(covni_24, low_percentile=percentiles,
                                                     high_percentile=1 - percentiles,
                                                     mode='capping')

    covni_30 = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                to_include=("freq_50hz_sample", "T=30MIN",),
                                                to_exclude=("TTX", "STACHEL"),
                                                verbose=False,
                                                freq_range=(min_freq, max_freq),
                                                save=False,
                                                separate_organoids=False,
                                                select_organoids=batches[batch],
                                                label_comment=f""
                                                )
    discarded_covni_30 = dpr.discard_outliers_by_iqr(covni_30, low_percentile=percentiles,
                                                     high_percentile=1 - percentiles,
                                                     mode='capping')

    covni_0 = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                               to_include=("freq_50hz_sample", "T=0MIN",),
                                               to_exclude=("TTX", "STACHEL"),
                                               verbose=False,
                                               freq_range=(min_freq, max_freq),
                                               save=False,
                                               separate_organoids=False,
                                               select_organoids=batches[batch],
                                               label_comment=f""
                                               )
    discarded_covni_0 = dpr.discard_outliers_by_iqr(covni_0, low_percentile=percentiles,
                                                    high_percentile=1 - percentiles,
                                                    mode='capping')
    discarded_covni_24["label"].replace(f'INF', f'SARS-CoV-2 24H', inplace=True)
    discarded_covni_24["label"].replace(f'NI', f'Mock 24H', inplace=True)
    discarded_covni_30["label"].replace(f'INF', f'SARS-CoV-2 30MIN', inplace=True)
    discarded_covni_30["label"].replace(f'NI', f'Mock 30MIN', inplace=True)
    discarded_covni_0["label"].replace(f'INF', f'SARS-CoV-2 0MIN', inplace=True)
    discarded_covni_0["label"].replace(f'NI', f'Mock 0MIN', inplace=True)
    rfc, _ = fl.train_RFC_from_dataset(discarded_covni_24)

    global_df = pd.concat([discarded_covni_24, discarded_covni_30, discarded_covni_0], ignore_index=True)

    fl.test_model(rfc, global_df, training_targets=(f'Mock 24H', f'SARS-CoV-2 24H'),
                  testing_targets=tuple(set(list((
                      f'Mock 24H', f'SARS-CoV-2 24H', f'Mock 30MIN', f'SARS-CoV-2 30MIN', f'Mock 0MIN',
                      f'SARS-CoV-2 0MIN')))),
                  show=False, verbose=False, savepath=P.FIGURES_PAPER,
                  title=f"Fig1h Confusion matrix train on T=24H, test on T=24H, 30MIN, 0MIN for {batch} {min_freq}-{max_freq}Hz Mock,CoV",
                  iterations=5)


def fig1g_Confusion_matrix_train_on_batch_Mock_CoV_in_region_Hz(min_freq=300, max_freq=5000,
                                                                train_batch="all organoids",
                                                                test_batch="all organoids"):
    fig1e_Confusion_matrix_train_on_batch_Mock_CoV_in_region_Hz(min_freq=min_freq, max_freq=max_freq,
                                                                train_batch=train_batch,
                                                                test_batch=test_batch, fig="Fig1g")


def fig1f_PCA_on_regionHz_all_organoids_for_Mock_CoV(min_freq, max_freq, batch="all organoids"):
    percentiles = 0.1
    n_components = 2
    batches = {"batch 1": [1, 2, 3, 4], "batch 2": [5, 6, 7], "all organoids": [1, 2, 3, 4, 5, 6, 7]}

    covni = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                             to_include=("freq_50hz_sample", "T=24H"),
                                             to_exclude=("TTX", "STACHEL",),
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_organoids=batches[batch],
                                             separate_organoids=False,
                                             label_comment="")

    discarded_covni = dpr.discard_outliers_by_iqr(covni, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    discarded_covni.replace("NI", "Mock", inplace=True)
    discarded_covni.replace("INF", "SARS-CoV-2", inplace=True)

    pca, pcdf, _ = fl.fit_pca(discarded_covni, n_components=n_components)
    ml.plot_pca(pcdf, n_components=2, show=False, title=f"Fig1f PCA on {min_freq}-{max_freq}Hz {batch} for Mock,CoV",
                points=True, metrics=True, savedir=P.FIGURES_PAPER, )


def fig1e_Confusion_matrix_train_on_batch_Mock_CoV_in_region_Hz(min_freq=0, max_freq=500, train_batch="all organoids",
                                                                test_batch="all organoids", fig="Fig1e"):
    percentiles = 0.1
    batches = {"batch 1": [1, 2, 3, 4], "batch 2": [5, 6, 7], "all organoids": [1, 2, 3, 4, 5, 6, 7]}

    covni_train_batch = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                         to_include=("freq_50hz_sample", "T=24H"),
                                                         to_exclude=("TTX", "STACHEL",),
                                                         verbose=False,
                                                         save=False,
                                                         freq_range=(min_freq, max_freq),
                                                         select_organoids=batches[train_batch],
                                                         separate_organoids=False,
                                                         label_comment=f" {train_batch}")

    discarded_covni_train_batch = dpr.discard_outliers_by_iqr(covni_train_batch, low_percentile=percentiles,
                                                              high_percentile=1 - percentiles,
                                                              mode='capping')

    covni_test_batch = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                        to_include=("freq_50hz_sample", "T=24H",),
                                                        to_exclude=("TTX", "STACHEL"),
                                                        verbose=False,
                                                        freq_range=(min_freq, max_freq),
                                                        save=False,
                                                        separate_organoids=False,
                                                        select_organoids=batches[test_batch],
                                                        label_comment=f" {test_batch}"
                                                        )
    discarded_covni_test_batch = dpr.discard_outliers_by_iqr(covni_test_batch, low_percentile=percentiles,
                                                             high_percentile=1 - percentiles,
                                                             mode='capping')
    discarded_covni_train_batch["label"].replace(f'INF {train_batch}', f'SARS-CoV-2 {train_batch}', inplace=True)
    discarded_covni_test_batch["label"].replace(f'INF {test_batch}', f'SARS-CoV-2 {test_batch}', inplace=True)
    discarded_covni_train_batch["label"].replace(f'NI {train_batch}', f'Mock {train_batch}', inplace=True)
    discarded_covni_test_batch["label"].replace(f'NI {test_batch}', f'Mock {test_batch}', inplace=True)
    rfc, _ = fl.train_RFC_from_dataset(discarded_covni_train_batch)

    global_df = pd.concat([discarded_covni_train_batch, discarded_covni_test_batch], ignore_index=True)

    fl.test_model(rfc, global_df, training_targets=(f'Mock {train_batch}', f'SARS-CoV-2 {train_batch}'),
                  testing_targets=tuple(set(list((
                      f'Mock {train_batch}', f'SARS-CoV-2 {train_batch}', f'Mock {test_batch}',
                      f'SARS-CoV-2 {test_batch}')))),
                  show=False, verbose=False, savepath=P.FIGURES_PAPER,
                  title=f"{fig} Confusion matrix train on {train_batch}, test on {test_batch} for {min_freq}-{max_freq}Hz Mock,CoV",
                  iterations=5)


def fig1d_Amplitude_for_Mock_CoV_in_region_Hz_at_T_24H_for_all_organoids(min_freq=0, max_freq=500):
    show = False
    percentiles = 0.1
    min_feat = int(min_freq * 300 / 5000)
    max_feat = int(max_freq * 300 / 5000)
    cov = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                           to_include=("freq_50hz_sample", "T=24H"),
                                           to_exclude=("TTX", "STACHEL", "NI"),
                                           verbose=False,
                                           save=False,
                                           select_organoids=[1, 2, 3, 4, 5, 6, 7],
                                           separate_organoids=False,
                                           freq_range=(min_freq, max_freq),
                                           label_comment="")

    discarded_cov = dpr.discard_outliers_by_iqr(cov, low_percentile=percentiles,
                                                high_percentile=1 - percentiles,
                                                mode='capping')

    ni = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                          to_include=("freq_50hz_sample", "T=24H"),
                                          to_exclude=("TTX", "STACHEL", "INF"),
                                          verbose=False,
                                          save=False,
                                          select_organoids=[1, 2, 3, 4, 5, 6, 7],
                                          freq_range=(min_freq, max_freq),
                                          separate_organoids=False,
                                          label_comment="")

    discarded_ni = dpr.discard_outliers_by_iqr(ni, low_percentile=percentiles,
                                               high_percentile=1 - percentiles,
                                               mode='capping')

    discarded_ni.replace("NI", "Mock", inplace=True)
    discarded_cov.replace("INF", "SARS-CoV-2", inplace=True)
    global_df = pd.concat([discarded_ni, discarded_cov], ignore_index=True)

    data = pd.DataFrame(columns=["label", "mean amplitude [pV]", "std amplitude [pV]"])
    plt.figure(figsize=(9, 8))
    mock_df = global_df.loc[global_df["label"] == "Mock"]
    mock_region = mock_df.loc[:, mock_df.columns != "label"]
    plt.bar(0, np.mean(np.array(mock_region)), color='dimgray',
            yerr=np.std(np.array(mock_region)))
    data.loc[len(data)] = ["Mock", np.mean(np.array(mock_region)), np.std(np.array(mock_region))]

    cov_df = global_df.loc[global_df["label"] == "SARS-CoV-2"]
    cov_region = cov_df.loc[:, cov_df.columns != "label"]
    plt.bar(1, np.mean(np.array(cov_region)), color='darkgray',
            yerr=np.std(np.array(cov_region)))
    data.loc[len(data)] = ["SARS-CoV-2", np.mean(np.array(cov_region)), np.std(np.array(cov_region))]

    plt.xticks([0, 1, ], ["Mock", "SARS-CoV-2"], rotation=0, fontsize=20)
    plt.ylabel("Mean amplitude [pV]", fontsize=25)

    plt.savefig(
        os.path.join(P.FIGURES_PAPER, f"Fig1d Amplitude for Mock,CoV in {min_freq}-{max_freq} Hz at T=24H for all "
                                      f"organoids.png"),
        dpi=1200)
    data.to_csv(
        os.path.join(P.FIGURES_PAPER,
                     f"Fig1d Amplitude for Mock,CoV in {min_freq}-{max_freq} Hz at T=24H for all organoids.csv"),
        index=False)
    if show:
        plt.show()


def fig1c_Feature_importance_for_regionHz_at_T_24H_batch_for_Mock_CoV(min_freq=0, max_freq=500, batch="all organoids"):
    percentiles = 0.1
    show = False
    batches = {"batch 1": [1, 2, 3, 4], "batch 2": [5, 6, 7], "all organoids": [1, 2, 3, 4, 5, 6, 7]}

    class1 = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                              to_include=("freq_50hz_sample", "T=24H"),
                                              to_exclude=("TTX", "STACHEL", "NI"),
                                              verbose=False,
                                              save=False,
                                              freq_range=(min_freq, max_freq),
                                              select_organoids=batches[batch],
                                              separate_organoids=False,
                                              label_comment=f" {batch}")

    discarded_class1 = dpr.discard_outliers_by_iqr(class1, low_percentile=percentiles,
                                                   high_percentile=1 - percentiles,
                                                   mode='capping')

    class2 = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                              to_include=("freq_50hz_sample", "T=24H"),
                                              to_exclude=("TTX", "STACHEL", "INF"),
                                              verbose=False,
                                              save=False,
                                              freq_range=(min_freq, max_freq),
                                              select_organoids=batches[batch],
                                              separate_organoids=False,
                                              label_comment=f" {batch}")

    discarded_class2 = dpr.discard_outliers_by_iqr(class2, low_percentile=percentiles,
                                                   high_percentile=1 - percentiles,
                                                   mode='capping')

    discarded_class2.replace(f"NI {batch}", "Mock", inplace=True)
    discarded_class1.replace(f"INF {batch}", "SARS-CoV-2", inplace=True)
    train_df = pd.concat([discarded_class1, discarded_class2], ignore_index=True)

    rfc, _ = fl.train_RFC_from_dataset(train_df)

    _, mean_importance, _ = fl.get_top_features_from_trained_RFC(rfc, percentage=1, show=False, save=False, title='',
                                                                 savepath='')
    plt.figure(figsize=(9, 8))
    plt.plot(mean_importance, color='b', linewidth=1)
    plt.fill_between([x for x in range(0, 300)], mean_importance, color='b', alpha=.5)

    hertz = []
    factor = 5000 / 300
    for i in range(300):
        hertz.append(int(i * factor))

    xticks = [x for x in range(0, 300, 50)]
    new_ticks = [hertz[x] for x in xticks]
    xticks.append(300)
    new_ticks.append(5000)
    plt.xticks(xticks, new_ticks, rotation=15, fontsize=15)
    plt.xlabel("Frequency-like features [Hz]", fontsize=25)
    plt.ylabel("Feature importance [AU]", fontsize=25)
    plt.savefig(
        os.path.join(P.FIGURES_PAPER,
                     f"Fig1c Feature importance for {min_freq}-{max_freq}Hz at T=24H {batch} for Mock,CoV.png"),
        dpi=1200)
    if show:
        plt.show()


def fig1b_Smoothened_frequencies_regionHz_Mock_CoV_on_batch(min_freq=0, max_freq=500, batch="all organoids"):
    percentiles = 0.1
    batches = {"batch 1": [1, 2, 3, 4], "batch 2": [5, 6, 7], "all organoids": [1, 2, 3, 4, 5, 6, 7]}

    show = False
    covni = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                             to_include=("freq_50hz_sample", "T=24H"),
                                             to_exclude=("TTX", "STACHEL", "NI"),
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_organoids=batches[batch],
                                             separate_organoids=False,
                                             label_comment="")

    discarded_covni = dpr.discard_outliers_by_iqr(covni, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    discarded_covni.replace("NI", "Mock", inplace=True)
    discarded_covni.replace("INF", "SARS-CoV-2", inplace=True)

    plt.figure(figsize=(8, 8))
    plt.plot(discarded_covni.loc[discarded_covni["label"] == "Mock"].mean(axis=0), label="Mock", linewidth=1, color='g')
    plt.fill_between([x for x in range(0, 300)],
                     discarded_covni.loc[discarded_covni["label"] == "Mock"].mean(axis=0).subtract(
                         discarded_covni.loc[discarded_covni["label"] == "Mock"].std(axis=0)),
                     discarded_covni.loc[discarded_covni["label"] == "Mock"].mean(axis=0).add(
                         discarded_covni.loc[discarded_covni["label"] == "Mock"].std(axis=0)),
                     color='g', alpha=.5)
    plt.plot(discarded_covni.loc[discarded_covni["label"] == "SARS-CoV-2"].mean(axis=0), label="SARS-CoV-2",
             linewidth=1, color='b')
    plt.fill_between([x for x in range(0, 300)],
                     discarded_covni.loc[discarded_covni["label"] == "SARS-CoV-2"].mean(axis=0).subtract(
                         discarded_covni.loc[discarded_covni["label"] == "SARS-CoV-2"].std(axis=0)),
                     discarded_covni.loc[discarded_covni["label"] == "SARS-CoV-2"].mean(axis=0).add(
                         discarded_covni.loc[discarded_covni["label"] == "SARS-CoV-2"].std(axis=0)),
                     color='b', alpha=.5)

    ratio = 5000 / 300
    x_ds = [x for x in range(0, 301, 15)]
    x_freq = [int(x * ratio) for x in x_ds]
    plt.xticks(x_ds, x_freq, rotation=45)
    plt.xlabel("Smoothened frequencies [Hz]", fontsize=25)
    plt.ylabel("Amplitude [pV]", fontsize=25)
    plt.legend(prop={'size': 20})
    plt.savefig(os.path.join(P.FIGURES_PAPER,
                             f"Fig1b Smoothened frequencies {min_freq}-{max_freq}Hz Mock-CoV on {batch}.png"), dpi=1200)
    if show:
        plt.show()


def fig1a_Confusion_matrix_train_test_on_batches_Mock_CoV(min_freq=0, max_freq=500, train_batch="all organoids",
                                                          test_batch="all organoids"):
    percentiles = 0.1
    batches = {"batch 1": [1, 2, 3, 4], "batch 2": [5, 6, 7], "all organoids": [1, 2, 3, 4, 5, 6, 7]}

    covni_train_batch = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                         to_include=("freq_50hz_sample", "T=24H"),
                                                         to_exclude=("TTX", "STACHEL",),
                                                         verbose=False,
                                                         save=False,
                                                         freq_range=(min_freq, max_freq),
                                                         select_organoids=batches[train_batch],
                                                         separate_organoids=False,
                                                         label_comment=f" {train_batch}")

    discarded_covni_train_batch = dpr.discard_outliers_by_iqr(covni_train_batch, low_percentile=percentiles,
                                                              high_percentile=1 - percentiles,
                                                              mode='capping')

    covni_test_batch = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                        to_include=("freq_50hz_sample", "T=24H",),
                                                        to_exclude=("TTX", "STACHEL"),
                                                        verbose=False,
                                                        freq_range=(min_freq, max_freq),
                                                        save=False,
                                                        separate_organoids=False,
                                                        select_organoids=batches[test_batch],
                                                        label_comment=f" {test_batch}"
                                                        )
    discarded_covni_test_batch = dpr.discard_outliers_by_iqr(covni_test_batch, low_percentile=percentiles,
                                                             high_percentile=1 - percentiles,
                                                             mode='capping')
    discarded_covni_train_batch["label"].replace(f'INF {train_batch}', f'SARS-CoV-2 {train_batch}', inplace=True)
    discarded_covni_test_batch["label"].replace(f'INF {test_batch}', f'SARS-CoV-2 {test_batch}', inplace=True)
    discarded_covni_train_batch["label"].replace(f'NI {train_batch}', f'Mock {train_batch}', inplace=True)
    discarded_covni_test_batch["label"].replace(f'NI {test_batch}', f'Mock {test_batch}', inplace=True)
    rfc, _ = fl.train_RFC_from_dataset(discarded_covni_train_batch)

    global_df = pd.concat([discarded_covni_train_batch, discarded_covni_test_batch], ignore_index=True)

    fl.test_model(rfc, global_df, training_targets=(f'Mock {train_batch}', f'SARS-CoV-2 {train_batch}'),
                  testing_targets=
                  tuple(set(list((
                      f'Mock {train_batch}', f'SARS-CoV-2 {train_batch}', f'Mock {test_batch}',
                      f'SARS-CoV-2 {test_batch}')))),
                  show=False, verbose=False, savepath=P.FIGURES_PAPER,
                  title=f"Fig1a Confusion matrix train on {train_batch}, test on {test_batch} Mock,CoV")


def amplitude_bar_plot_for_mock_cov_cov_stachel_at_T_24_without_outlier_01(min_feat, max_feat):
    # roi: 0-25 (0-416 Hz) and 230-250 (~3800-4200 Hz)
    n_components = 2
    percentiles = 0.1
    cov_nostachel = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                     to_include=("freq_50hz_sample", "T=24H"),
                                                     to_exclude=("TTX", "STACHEL", "NI"),
                                                     verbose=False,
                                                     save=False,
                                                     select_organoids=[1, 2, 3, 4, ],
                                                     separate_organoids=False,
                                                     label_comment=" NOSTACHEL")

    discarded_cov_nostachel = dpr.discard_outliers_by_iqr(cov_nostachel, low_percentile=percentiles,
                                                          high_percentile=1 - percentiles,
                                                          mode='capping')

    ni_nostachel = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                    to_include=("freq_50hz_sample", "T=24H"),
                                                    to_exclude=("TTX", "STACHEL", "INF"),
                                                    verbose=False,
                                                    save=False,
                                                    select_organoids=[1, 2, 3, 4, ],
                                                    separate_organoids=False,
                                                    label_comment=" NOSTACHEL")

    discarded_ni_nostachel = dpr.discard_outliers_by_iqr(ni_nostachel, low_percentile=percentiles,
                                                         high_percentile=1 - percentiles,
                                                         mode='capping')

    cov_stachel = dpr.make_dataset_from_freq_files(parent_dir=P.STACHEL,
                                                   to_include=("freq_50hz_sample", "T=24H",),
                                                   to_exclude=("TTX", "NI"),
                                                   verbose=False,
                                                   save=False,
                                                   separate_organoids=False,
                                                   label_comment=" STACHEL0"
                                                   )
    discarded_cov_stachel = dpr.discard_outliers_by_iqr(cov_stachel, low_percentile=percentiles,
                                                        high_percentile=1 - percentiles,
                                                        mode='capping')

    discarded_ni_nostachel.replace("NI NOSTACHEL", "Mock", inplace=True)
    discarded_cov_nostachel.replace("INF NOSTACHEL", "SARS-CoV-2", inplace=True)
    discarded_cov_stachel.replace("INF STACHEL0", "Stachel-treated SARS-CoV-2", inplace=True)
    train_df = pd.concat([discarded_cov_nostachel, discarded_cov_stachel], ignore_index=True)

    commentary = f"outliers percentile={str(percentiles).replace('.', '')}"
    rfc, _ = fl.train_RFC_from_dataset(train_df)
    global_df = pd.concat([train_df, discarded_ni_nostachel], ignore_index=True)

    plt.figure(figsize=(9, 8))
    mock_df = global_df.loc[global_df["label"] == "Mock"]
    mock_region = mock_df.loc[:, [mock_df[x].name for x in range(min_feat, max_feat + 1)]]
    plt.bar(0, np.mean(np.array(mock_region)), color='dimgray',
            yerr=np.std(np.array(mock_region)))

    cov_df = global_df.loc[global_df["label"] == "SARS-CoV-2"]
    cov_region = cov_df.loc[:, [cov_df[x].name for x in range(min_feat, max_feat + 1)]]
    plt.bar(1, np.mean(np.array(cov_region)), color='darkgray',
            yerr=np.std(np.array(cov_region)))

    stachel_df = global_df.loc[global_df["label"] == "Stachel-treated SARS-CoV-2"]
    stachel_region = cov_df.loc[:, [stachel_df[x].name for x in range(min_feat, max_feat + 1)]]
    plt.bar(2, np.mean(np.array(stachel_region)), color='lightgray',
            yerr=np.std(np.array(stachel_region)))

    plt.xticks([0, 1, 2], ["Mock", "SARS-CoV-2", "Stachel-treated SARS-CoV-2"], rotation=5, fontsize=20)
    plt.ylabel("Mean amplitude [pV]", fontsize=25)

    plt.savefig(os.path.join(P.RESULTS, f"Amplitude barplot for Mock,CoV,CoV Stachel at T=24H restricted between "
                                        f"{int(min_feat * 5000 / 300)}Hz and {int(max_feat * 5000 / 300)}Hz organoids1,2,3,4 for Ni,CoV.png"),
                dpi=1200)
    plt.show()


def smoothened_frequencies_for_Mock_CoV_CoV_Stachel_at_T_24H_without_outliers_01():
    percentiles = 0.1

    cov_nostachel = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                     to_include=("freq_50hz_sample", "T=24H"),
                                                     to_exclude=("TTX", "STACHEL", "NI"),
                                                     verbose=False,
                                                     save=False,
                                                     select_organoids=[5, 6, 7],
                                                     separate_organoids=False,
                                                     label_comment=" NOSTACHEL")

    discarded_cov_nostachel = dpr.discard_outliers_by_iqr(cov_nostachel, low_percentile=percentiles,
                                                          high_percentile=1 - percentiles,
                                                          mode='capping')

    ni_nostachel = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                    to_include=("freq_50hz_sample", "T=24H"),
                                                    to_exclude=("TTX", "STACHEL", "INF"),
                                                    verbose=False,
                                                    save=False,
                                                    select_organoids=[5, 6, 7],
                                                    separate_organoids=False,
                                                    label_comment=" NOSTACHEL")

    discarded_ni_nostachel = dpr.discard_outliers_by_iqr(ni_nostachel, low_percentile=percentiles,
                                                         high_percentile=1 - percentiles,
                                                         mode='capping')

    cov_stachel = dpr.make_dataset_from_freq_files(parent_dir=P.STACHEL,
                                                   to_include=("freq_50hz_sample", "T=24H",),
                                                   to_exclude=("TTX", "NI"),
                                                   verbose=False,
                                                   save=False,
                                                   separate_organoids=False,
                                                   label_comment=" STACHEL0"
                                                   )
    discarded_cov_stachel = dpr.discard_outliers_by_iqr(cov_stachel, low_percentile=percentiles,
                                                        high_percentile=1 - percentiles,
                                                        mode='capping')

    discarded_ni_nostachel.replace("NI NOSTACHEL", "Mock", inplace=True)
    discarded_cov_nostachel.replace("INF NOSTACHEL", "SARS-CoV-2", inplace=True)
    discarded_cov_stachel.replace("INF STACHEL0", "Stachel-treated SARS-CoV-2", inplace=True)
    train_df = pd.concat([discarded_cov_nostachel, discarded_cov_stachel], ignore_index=True)

    rfc, _ = fl.train_RFC_from_dataset(train_df)
    global_df = pd.concat([train_df, discarded_ni_nostachel], ignore_index=True)

    plt.figure(figsize=(8, 8))
    plt.plot(global_df.loc[global_df["label"] == "Mock"].mean(axis=0), label="Mock", linewidth=1, color='g')
    plt.fill_between([x for x in range(0, 300)],
                     global_df.loc[global_df["label"] == "Mock"].mean(axis=0).subtract(
                         global_df.loc[global_df["label"] == "Mock"].std(axis=0)),
                     global_df.loc[global_df["label"] == "Mock"].mean(axis=0).add(
                         global_df.loc[global_df["label"] == "Mock"].std(axis=0)),
                     color='g', alpha=.5)
    plt.plot(global_df.loc[global_df["label"] == "SARS-CoV-2"].mean(axis=0), label="SARS-CoV-2", linewidth=1, color='b')
    plt.fill_between([x for x in range(0, 300)],
                     global_df.loc[global_df["label"] == "SARS-CoV-2"].mean(axis=0).subtract(
                         global_df.loc[global_df["label"] == "SARS-CoV-2"].std(axis=0)),
                     global_df.loc[global_df["label"] == "SARS-CoV-2"].mean(axis=0).add(
                         global_df.loc[global_df["label"] == "SARS-CoV-2"].std(axis=0)),
                     color='b', alpha=.5)
    plt.plot(global_df.loc[global_df["label"] == "Stachel-treated SARS-CoV-2"].mean(axis=0),
             label="Stachel-treated SARS-CoV-2", linewidth=1, color='r')
    plt.fill_between([x for x in range(0, 300)],
                     global_df.loc[global_df["label"] == "Stachel-treated SARS-CoV-2"].mean(axis=0).subtract(
                         global_df.loc[global_df["label"] == "Stachel-treated SARS-CoV-2"].std(axis=0)),
                     global_df.loc[global_df["label"] == "Stachel-treated SARS-CoV-2"].mean(axis=0).add(
                         global_df.loc[global_df["label"] == "Stachel-treated SARS-CoV-2"].std(axis=0)),
                     color='r', alpha=.5)
    ratio = 5000 / 300
    x_ds = [x for x in range(0, 301, 15)]
    x_freq = [int(x * ratio) for x in x_ds]
    plt.xticks(x_ds, x_freq, rotation=45)
    plt.xlabel("Smoothened frequencies [Hz]", fontsize=25)
    plt.ylabel("Amplitude [pV]", fontsize=25)
    plt.legend(prop={'size': 20})
    plt.savefig(os.path.join(P.RESULTS,
                             "Smoothened frequencies for Mock,CoV,CoV Stachel at T=24H organoids5,6,7 for Ni CoV.png"),
                dpi=1200)
    plt.show()


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
