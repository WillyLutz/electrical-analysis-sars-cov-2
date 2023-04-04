import os
from pathlib import Path

import fiiireflyyy.firefiles as ff
import fiiireflyyy.firelearn as fl
import fiiireflyyy.fireprocess as fp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import PATHS as P
import data_processing as dpr
import machine_learning as ml
import signal_processing as spr


def fig2d_Confusion_matrix_train_on_batch_Mock_CoV_in_region_Hz_test_on_stachel(min_freq=0, max_freq=5000,
                                                                                batch="batch 2",
                                                                                ):
    show = False
    percentiles = 0.1
    batches = {"batch 1": ["1", "2", "3", "4"], "batch 2": ["5", "6", "7"], "all organoids": ["1", "2", "3", "4", "5",
                                                                                              "6", "7"]}

    covni = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                            to_include=("freq_50hz_sample", "T=24H"),
                                            to_exclude=("TTX", "STACHEL",),
                                            verbose=False,
                                            save=False,
                                            freq_range=(min_freq, max_freq),
                                            select_samples=batches[batch],
                                            separate_samples=False,
                                            label_comment=f"")

    discarded_covni = fp.discard_outliers_by_iqr(covni, low_percentile=percentiles,
                                                 high_percentile=1 - percentiles,
                                                 mode='capping')

    cov_stachel = fp.make_dataset_from_freq_files(parent_dir=P.STACHEL,
                                                  to_include=("freq_50hz_sample", "T=24H"),
                                                  to_exclude=("TTX", "NI",),
                                                  verbose=False,
                                                  save=False,
                                                  freq_range=(min_freq, max_freq),
                                                  select_samples=batches[batch],
                                                  separate_samples=False,
                                                  label_comment=f"")

    discarded_cov_stachel = fp.discard_outliers_by_iqr(cov_stachel, low_percentile=percentiles,
                                                 high_percentile=1 - percentiles,
                                                 mode='capping')

    discarded_covni["label"].replace(f'INF', f'SARS-CoV-2', inplace=True)
    discarded_covni["label"].replace(f'NI', f'Mock', inplace=True)
    discarded_cov_stachel["label"].replace(f'INF', f'Stachel-treated\nSARS-CoV-2', inplace=True)
    rfc, _ = fl.train_RFC_from_dataset(discarded_covni)

    global_df = pd.concat([discarded_covni, discarded_cov_stachel, ], ignore_index=True)
    fl.test_model_by_confusion(rfc, global_df, training_targets=(f'Mock', f'SARS-CoV-2'),
                               testing_targets=tuple(set(list((
                                   f'Mock', f'SARS-CoV-2', f'SARS-CoV-2', f'Stachel-treated\nSARS-CoV-2',)))),
                               show=show, verbose=False, savepath=P.FIGURES_PAPER,
                               title=f"Fig2d Confusion matrix train on T=24H CoV,Mock, test on CpV,Mock,Stachel for {batch} "
                                     f"{min_freq}-{max_freq}Hz",
                               iterations=5, )


def fig2c_Amplitude_for_Mock_CoV_Stachel_in_region_Hz_at_T_24H_for_all_organoids(min_freq=0, max_freq=5000,
                                                                                 batch="all organoids"):
    show = False
    batches = {"batch 1": ["1", "2", "3", "4"], "batch 2": ["5", "6", "7"],
               "all organoids": ["1", "2", "3", "4", "5", "6", "7"]}

    percentiles = 0.1
    cov = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                          to_include=("freq_50hz_sample", "T=24H"),
                                          to_exclude=("TTX", "STACHEL", "NI"),
                                          verbose=False,
                                          save=False,
                                          select_samples=batches[batch],
                                          separate_samples=False,
                                          freq_range=(min_freq, max_freq),
                                          label_comment="")

    discarded_cov = fp.discard_outliers_by_iqr(cov, low_percentile=percentiles,
                                               high_percentile=1 - percentiles,
                                               mode='capping')

    ni = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                         to_include=("freq_50hz_sample", "T=24H"),
                                         to_exclude=("TTX", "STACHEL", "INF"),
                                         verbose=False,
                                         save=False,
                                         select_samples=batches[batch],
                                         freq_range=(min_freq, max_freq),
                                         separate_samples=False,
                                         label_comment="")

    discarded_ni = fp.discard_outliers_by_iqr(ni, low_percentile=percentiles,
                                              high_percentile=1 - percentiles,
                                              mode='capping')

    cov_stachel = fp.make_dataset_from_freq_files(parent_dir=P.STACHEL,
                                                  to_include=("freq_50hz_sample", "T=24H",),
                                                  to_exclude=("TTX", "NI"),
                                                  verbose=False,
                                                  freq_range=(min_freq, max_freq),
                                                  save=False,
                                                  separate_samples=False,
                                                  label_comment=""
                                                  )
    discarded_stachel = fp.discard_outliers_by_iqr(cov_stachel, low_percentile=percentiles,
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
    data.loc[len(data)] = ["Stachel-treated SARS-CoV-2", np.mean(np.array(stachel_region)),
                           np.std(np.array(stachel_region))]

    plt.xticks([0, 1, 2], ["Mock", "SARS-CoV-2", "Stachel-treated SARS-CoV-2"], rotation=7, fontsize=20)
    plt.ylabel("Mean amplitude [pV]", fontsize=25)

    plt.savefig(
        os.path.join(P.FIGURES_PAPER,
                     f"Fig2c Amplitude for Mock,CoV in {min_freq}-{max_freq} Hz at T=24H for {batch}.png"),
        dpi=1200)
    data.to_csv(
        os.path.join(P.FIGURES_PAPER,
                     f"Fig2c Amplitude for Mock,CoV,Stachel in {min_freq}-{max_freq} Hz at T=24H for {batch}.csv"),
        index=False)
    if show:
        plt.show()


def fig2b_Smoothened_frequencies_regionHz_Mock_CoV_Stachel_on_batch(min_freq=0, max_freq=500, batch="all organoids"):
    batches = {"batch 1": ["1", "2", "3", "4"], "batch 2": ["5", "6", "7"],
               "all organoids": ["1", "2", "3", "4", "5", "6", "7"]}
    percentiles = 0.1
    show = False
    cov_nostachel = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                    to_include=("freq_50hz_sample", "T=24H"),
                                                    to_exclude=("TTX", "STACHEL", "NI"),
                                                    verbose=False,
                                                    save=False,
                                                    freq_range=(min_freq, max_freq),
                                                    select_samples=batches[batch],
                                                    separate_samples=False,
                                                    label_comment=" NOSTACHEL")

    discarded_cov_nostachel = fp.discard_outliers_by_iqr(cov_nostachel, low_percentile=percentiles,
                                                         high_percentile=1 - percentiles,
                                                         mode='capping')

    ni_nostachel = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                   to_include=("freq_50hz_sample", "T=24H"),
                                                   to_exclude=("TTX", "STACHEL", "INF"),
                                                   verbose=False,
                                                   freq_range=(min_freq, max_freq),
                                                   save=False,
                                                   select_samples=batches[batch],
                                                   separate_samples=False,
                                                   label_comment=" NOSTACHEL")

    discarded_ni_nostachel = fp.discard_outliers_by_iqr(ni_nostachel, low_percentile=percentiles,
                                                        high_percentile=1 - percentiles,
                                                        mode='capping')

    cov_stachel = fp.make_dataset_from_freq_files(parent_dir=P.STACHEL,
                                                  to_include=("freq_50hz_sample", "T=24H",),
                                                  to_exclude=("TTX", "NI"),
                                                  verbose=False,
                                                  freq_range=(min_freq, max_freq),
                                                  save=False,
                                                  separate_samples=False,
                                                  label_comment=" STACHEL0"
                                                  )
    discarded_cov_stachel = fp.discard_outliers_by_iqr(cov_stachel, low_percentile=percentiles,
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
    batches = {"batch 1": ["1", "2", "3", "4"], "batch 2": ["5", "6", "7"],
               "all organoids": ["1", "2", "3", "4", "5", "6", "7"]}

    covni = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                            to_include=("freq_50hz_sample", "T=24H"),
                                            to_exclude=("TTX", "STACHEL",),
                                            verbose=False,
                                            save=False,
                                            freq_range=(min_freq, max_freq),
                                            select_samples=batches[batch],
                                            separate_samples=False,
                                            label_comment="")
    discarded_covni = fp.discard_outliers_by_iqr(covni, low_percentile=percentiles,
                                                 high_percentile=1 - percentiles,
                                                 mode='capping')

    covni_stachel = fp.make_dataset_from_freq_files(parent_dir=P.STACHEL,
                                                    to_include=("freq_50hz_sample", "T=24H"),
                                                    to_exclude=("TTX", "NI"),
                                                    verbose=False,
                                                    save=False,
                                                    freq_range=(min_freq, max_freq),
                                                    select_samples=batches["batch 2"],
                                                    separate_samples=False,
                                                    label_comment="")

    discarded_covni_stachel = fp.discard_outliers_by_iqr(covni_stachel, low_percentile=percentiles,
                                                         high_percentile=1 - percentiles,
                                                         mode='capping')

    discarded_covni.replace("NI", "Mock", inplace=True)
    discarded_covni.replace("INF", "SARS-CoV-2", inplace=True)
    discarded_covni_stachel.replace("NI", "Stachel-treated Mock", inplace=True)
    discarded_covni_stachel.replace("INF", "Stachel-treated SARS-CoV-2", inplace=True)

    pca, pcdf, ratios = fl.fit_pca(discarded_covni, n_components=n_components)
    stachel_pcdf = fl.apply_pca(pca, discarded_covni_stachel)
    global_df = pd.concat([pcdf, stachel_pcdf], ignore_index=True)

    rounded_ratio = [round(r * 100, 1) for r in ratios]
    fl.plot_pca(global_df, n_components=n_components, show=True,
                title=f"Fig2a PCA on {min_freq}-{max_freq}Hz {batch} for Mock,CoV, applied on stachel",
                points=True, metrics=True, savedir='', ratios=rounded_ratio)


def fig1h_Confusion_matrix_train_on_batch_Mock_CoV_in_region_Hz(min_freq=300, max_freq=5000,
                                                                batch="all organoids",
                                                                ):
    show = True
    percentiles = 0.1
    batches = {"batch 1": ["1", "2", "3", "4"], "batch 2": ["5", "6", "7"], "all organoids": ["1", "2", "3", "4", "5",
                                                                                              "6", "7"]}

    covni_24 = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                               to_include=("freq_50hz_sample", "T=24H"),
                                               to_exclude=("TTX", "STACHEL",),
                                               verbose=False,
                                               save=False,
                                               freq_range=(min_freq, max_freq),
                                               select_samples=batches[batch],
                                               separate_samples=False,
                                               label_comment=f"")

    discarded_covni_24 = fp.discard_outliers_by_iqr(covni_24, low_percentile=percentiles,
                                                    high_percentile=1 - percentiles,
                                                    mode='capping')

    covni_30 = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                               to_include=("freq_50hz_sample", "T=30MIN",),
                                               to_exclude=("TTX", "STACHEL"),
                                               verbose=False,
                                               freq_range=(min_freq, max_freq),
                                               save=False,
                                               separate_samples=False,
                                               select_samples=batches[batch],
                                               label_comment=f""
                                               )
    discarded_covni_30 = fp.discard_outliers_by_iqr(covni_30, low_percentile=percentiles,
                                                    high_percentile=1 - percentiles,
                                                    mode='capping')

    covni_0 = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                              to_include=("freq_50hz_sample", "T=0MIN",),
                                              to_exclude=("TTX", "STACHEL"),
                                              verbose=False,
                                              freq_range=(min_freq, max_freq),
                                              save=False,
                                              separate_samples=False,
                                              select_samples=batches[batch],
                                              label_comment=f""
                                              )
    discarded_covni_0 = fp.discard_outliers_by_iqr(covni_0, low_percentile=percentiles,
                                                   high_percentile=1 - percentiles,
                                                   mode='capping')
    discarded_covni_24["label"].replace(f'INF', f'Cov 24h', inplace=True)
    discarded_covni_24["label"].replace(f'NI', f'Mock 24h', inplace=True)
    discarded_covni_30["label"].replace(f'INF', f'Cov 30 min', inplace=True)
    discarded_covni_30["label"].replace(f'NI', f'Mock 30 min', inplace=True)
    discarded_covni_0["label"].replace(f'INF', f'Cov 0 min', inplace=True)
    discarded_covni_0["label"].replace(f'NI', f'Mock 0 min', inplace=True)
    rfc, _ = fl.train_RFC_from_dataset(discarded_covni_24)

    global_df = pd.concat([discarded_covni_24, discarded_covni_30, discarded_covni_0], ignore_index=True)
    fl.test_model_by_confusion(rfc, global_df, training_targets=(f'Mock 24h', f'Cov 24h'),
                               testing_targets=tuple(set(list((
                                   f'Mock 24h', f'Cov 24h', f'Mock 30 min', f'Cov 30 min', f'Mock 0 min',
                                   f'Cov 0 min')))),
                               show=show, verbose=False, savepath=P.FIGURES_PAPER,
                               title=f"Fig1h Confusion matrix train on T=24H, test on T=24H, 30MIN, 0MIN for {batch} "
                                     f"{min_freq}-{max_freq}Hz Mock,CoV",
                               iterations=5, )


def fig1g_Confusion_matrix_train_on_batch_Mock_CoV_in_region_Hz(min_freq=300, max_freq=5000,
                                                                train_batch="all organoids",
                                                                test_batch="all organoids"):
    fig1e_Confusion_matrix_train_on_batch_Mock_CoV_in_region_Hz(min_freq=min_freq, max_freq=max_freq,
                                                                train_batch=train_batch,
                                                                test_batch=test_batch, fig="Fig1g")


def fig1f_PCA_on_regionHz_all_organoids_for_Mock_CoV(min_freq, max_freq, batch="all organoids"):
    show = False
    percentiles = 0.1
    n_components = 2
    batches = {"batch 1": ["1", "2", "3", "4"], "batch 2": ["5", "6", "7"],
               "all organoids": ["1", "2", "3", "4", "5", "6", "7"]}

    covni = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                            to_include=("freq_50hz_sample", "T=24H"),
                                            to_exclude=("TTX", "STACHEL",),
                                            verbose=False,
                                            save=False,
                                            freq_range=(min_freq, max_freq),
                                            select_samples=batches[batch],
                                            separate_samples=False,
                                            label_comment="")

    discarded_covni = fp.discard_outliers_by_iqr(covni, low_percentile=percentiles,
                                                 high_percentile=1 - percentiles,
                                                 mode='capping')

    discarded_covni.replace("NI", "Mock", inplace=True)
    discarded_covni.replace("INF", "SARS-CoV-2", inplace=True)

    pca, pcdf, ratio = fl.fit_pca(discarded_covni, n_components=n_components)
    rounded_ratio = [round(r * 100, 1) for r in ratio]
    fl.plot_pca(pcdf, n_components=2, show=show, title=f"Fig1f PCA on {min_freq}-{max_freq}Hz {batch} for Mock,CoV",
                points=True, metrics=True, savedir=P.FIGURES_PAPER, ratios=rounded_ratio)


def fig1e_Confusion_matrix_train_on_batch_Mock_CoV_in_region_Hz(min_freq=0, max_freq=500, train_batch="all organoids",
                                                                test_batch="all organoids", fig="Fig1e"):
    show = False
    percentiles = 0.1
    batches = {"batch 1": ["1", "2", "3", "4"], "batch 2": ["5", "6", "7"],
               "all organoids": ["1", "2", "3", "4", "5", "6", "7"]}

    covni_train_batch = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                        to_include=("freq_50hz_sample", "T=24H"),
                                                        to_exclude=("TTX", "STACHEL",),
                                                        verbose=False,
                                                        save=False,
                                                        freq_range=(min_freq, max_freq),
                                                        select_samples=batches[train_batch],
                                                        separate_samples=False,
                                                        label_comment=f" {train_batch}")

    discarded_covni_train_batch = fp.discard_outliers_by_iqr(covni_train_batch, low_percentile=percentiles,
                                                             high_percentile=1 - percentiles,
                                                             mode='capping')

    covni_test_batch = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                       to_include=("freq_50hz_sample", "T=24H",),
                                                       to_exclude=("TTX", "STACHEL"),
                                                       verbose=False,
                                                       freq_range=(min_freq, max_freq),
                                                       save=False,
                                                       separate_samples=False,
                                                       select_samples=batches[test_batch],
                                                       label_comment=f" {test_batch}"
                                                       )
    discarded_covni_test_batch = fp.discard_outliers_by_iqr(covni_test_batch, low_percentile=percentiles,
                                                            high_percentile=1 - percentiles,
                                                            mode='capping')
    discarded_covni_train_batch["label"].replace(f'INF {train_batch}', f'SARS-CoV-2 {train_batch}', inplace=True)
    discarded_covni_test_batch["label"].replace(f'INF {test_batch}', f'SARS-CoV-2 {test_batch}', inplace=True)
    discarded_covni_train_batch["label"].replace(f'NI {train_batch}', f'Mock {train_batch}', inplace=True)
    discarded_covni_test_batch["label"].replace(f'NI {test_batch}', f'Mock {test_batch}', inplace=True)
    rfc, _ = fl.train_RFC_from_dataset(discarded_covni_train_batch)

    global_df = pd.concat([discarded_covni_train_batch, discarded_covni_test_batch], ignore_index=True)

    fl.test_model_by_confusion(rfc, global_df, training_targets=(f'Mock {train_batch}', f'SARS-CoV-2 {train_batch}'),
                               testing_targets=tuple(set(list((
                                   f'Mock {train_batch}', f'SARS-CoV-2 {train_batch}', f'Mock {test_batch}',
                                   f'SARS-CoV-2 {test_batch}')))),
                               show=show, verbose=False, savepath=P.FIGURES_PAPER,
                               title=f"{fig} Confusion matrix train on {train_batch}, test on {test_batch} for "
                                     f"{min_freq}-{max_freq}Hz Mock,CoV",
                               iterations=5)


def fig1d_Amplitude_for_Mock_CoV_in_region_Hz_at_T_24H_for_all_organoids(min_freq=0, max_freq=500):
    show = False
    percentiles = 0.1
    cov = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                          to_include=("freq_50hz_sample", "T=24H"),
                                          to_exclude=("TTX", "STACHEL", "NI"),
                                          verbose=False,
                                          save=False,
                                          select_samples=[1, 2, 3, 4, 5, 6, 7],
                                          separate_samples=False,
                                          freq_range=(min_freq, max_freq),
                                          label_comment="")

    discarded_cov = fp.discard_outliers_by_iqr(cov, low_percentile=percentiles,
                                               high_percentile=1 - percentiles,
                                               mode='capping')

    ni = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                         to_include=("freq_50hz_sample", "T=24H"),
                                         to_exclude=("TTX", "STACHEL", "INF"),
                                         verbose=False,
                                         save=False,
                                         select_samples=[1, 2, 3, 4, 5, 6, 7],
                                         freq_range=(min_freq, max_freq),
                                         separate_samples=False,
                                         label_comment="")

    discarded_ni = fp.discard_outliers_by_iqr(ni, low_percentile=percentiles,
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
    batches = {"batch 1": ["1", "2", "3", "4"], "batch 2": ["5", "6", "7"],
               "all organoids": ["1", "2", "3", "4", "5", "6", "7"]}

    class1 = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                             to_include=("freq_50hz_sample", "T=24H"),
                                             to_exclude=("TTX", "STACHEL", "NI"),
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_samples=batches[batch],
                                             separate_samples=False,
                                             label_comment=f" {batch}")

    discarded_class1 = fp.discard_outliers_by_iqr(class1, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    class2 = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                             to_include=("freq_50hz_sample", "T=24H"),
                                             to_exclude=("TTX", "STACHEL", "INF"),
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_samples=batches[batch],
                                             separate_samples=False,
                                             label_comment=f" {batch}")

    discarded_class2 = fp.discard_outliers_by_iqr(class2, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    discarded_class2.replace(f"NI {batch}", "Mock", inplace=True)
    discarded_class1.replace(f"INF {batch}", "SARS-CoV-2", inplace=True)
    train_df = pd.concat([discarded_class1, discarded_class2], ignore_index=True)

    rfc, _ = fl.train_RFC_from_dataset(train_df)

    _, mean_importance, _ = fl.get_top_features_from_trained_RFC(rfc, percentage=1, show=show, save=False, title='',
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
    batches = {"batch 1": ["1", "2", "3", "4"], "batch 2": ["5", "6", "7"],
               "all organoids": ["1", "2", "3", "4", "5", "6", "7"]}

    show = False
    covni = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                            to_include=("freq_50hz_sample", "T=24H"),
                                            to_exclude=("TTX", "STACHEL", "NI"),
                                            verbose=False,
                                            save=False,
                                            freq_range=(min_freq, max_freq),
                                            select_samples=batches[batch],
                                            separate_samples=False,
                                            label_comment="")

    discarded_covni = fp.discard_outliers_by_iqr(covni, low_percentile=percentiles,
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
    show = False
    percentiles = 0.1
    batches = {"batch 1": ["1", "2", "3", "4"], "batch 2": ["5", "6", "7"],
               "all organoids": ["1", "2", "3", "4", "5", "6", "7"]}

    covni_train_batch = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                        to_include=("freq_50hz_sample", "T=24H"),
                                                        to_exclude=("TTX", "STACHEL",),
                                                        verbose=False,
                                                        save=False,
                                                        freq_range=(min_freq, max_freq),
                                                        select_samples=batches[train_batch],
                                                        separate_samples=False,
                                                        label_comment=f" {train_batch}")

    discarded_covni_train_batch = fp.discard_outliers_by_iqr(covni_train_batch, low_percentile=percentiles,
                                                             high_percentile=1 - percentiles,
                                                             mode='capping')

    covni_test_batch = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                       to_include=("freq_50hz_sample", "T=24H",),
                                                       to_exclude=("TTX", "STACHEL"),
                                                       verbose=False,
                                                       freq_range=(min_freq, max_freq),
                                                       save=False,
                                                       separate_samples=False,
                                                       select_samples=batches[test_batch],
                                                       label_comment=f" {test_batch}"
                                                       )
    discarded_covni_test_batch = fp.discard_outliers_by_iqr(covni_test_batch, low_percentile=percentiles,
                                                            high_percentile=1 - percentiles,
                                                            mode='capping')
    discarded_covni_train_batch["label"].replace(f'INF {train_batch}', f'SARS-CoV-2 {train_batch}', inplace=True)
    discarded_covni_test_batch["label"].replace(f'INF {test_batch}', f'SARS-CoV-2 {test_batch}', inplace=True)
    discarded_covni_train_batch["label"].replace(f'NI {train_batch}', f'Mock {train_batch}', inplace=True)
    discarded_covni_test_batch["label"].replace(f'NI {test_batch}', f'Mock {test_batch}', inplace=True)
    rfc, _ = fl.train_RFC_from_dataset(discarded_covni_train_batch)

    global_df = pd.concat([discarded_covni_train_batch, discarded_covni_test_batch], ignore_index=True)

    fl.test_model_by_confusion(rfc, global_df, training_targets=(f'Mock {train_batch}', f'SARS-CoV-2 {train_batch}'),
                               testing_targets=
                               tuple(set(list((
                                   f'Mock {train_batch}', f'SARS-CoV-2 {train_batch}', f'Mock {test_batch}',
                                   f'SARS-CoV-2 {test_batch}')))),
                               show=show, verbose=False, savepath=P.FIGURES_PAPER,
                               title=f"Fig1a Confusion matrix train on {train_batch}, test on {test_batch} Mock,CoV")


def amplitude_bar_plot_for_mock_cov_cov_stachel_at_T_24_without_outlier_01(min_feat, max_feat):
    show = False
    percentiles = 0.1
    cov_nostachel = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                    to_include=("freq_50hz_sample", "T=24H"),
                                                    to_exclude=("TTX", "STACHEL", "NI"),
                                                    verbose=False,
                                                    save=False,
                                                    select_samples=[1, 2, 3, 4, ],
                                                    separate_samples=False,
                                                    label_comment=" NOSTACHEL")

    discarded_cov_nostachel = fp.discard_outliers_by_iqr(cov_nostachel, low_percentile=percentiles,
                                                         high_percentile=1 - percentiles,
                                                         mode='capping')

    ni_nostachel = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                   to_include=("freq_50hz_sample", "T=24H"),
                                                   to_exclude=("TTX", "STACHEL", "INF"),
                                                   verbose=False,
                                                   save=False,
                                                   select_samples=[1, 2, 3, 4, ],
                                                   separate_samples=False,
                                                   label_comment=" NOSTACHEL")

    discarded_ni_nostachel = fp.discard_outliers_by_iqr(ni_nostachel, low_percentile=percentiles,
                                                        high_percentile=1 - percentiles,
                                                        mode='capping')

    cov_stachel = fp.make_dataset_from_freq_files(parent_dir=P.STACHEL,
                                                  to_include=("freq_50hz_sample", "T=24H",),
                                                  to_exclude=("TTX", "NI"),
                                                  verbose=False,
                                                  save=False,
                                                  separate_samples=False,
                                                  label_comment=" STACHEL0"
                                                  )
    discarded_cov_stachel = fp.discard_outliers_by_iqr(cov_stachel, low_percentile=percentiles,
                                                       high_percentile=1 - percentiles,
                                                       mode='capping')

    discarded_ni_nostachel.replace("NI NOSTACHEL", "Mock", inplace=True)
    discarded_cov_nostachel.replace("INF NOSTACHEL", "SARS-CoV-2", inplace=True)
    discarded_cov_stachel.replace("INF STACHEL0", "Stachel-treated SARS-CoV-2", inplace=True)
    train_df = pd.concat([discarded_cov_nostachel, discarded_cov_stachel], ignore_index=True)

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
                                        f"{int(min_feat * 5000 / 300)}Hz and {int(max_feat * 5000 / 300)}Hz "
                                        f"organoids1,2,3,4 for Ni,CoV.png"),
                dpi=1200)
    if show:
        plt.show()


def smoothened_frequencies_for_Mock_CoV_CoV_Stachel_at_T_24H_without_outliers_01():
    percentiles = 0.1

    cov_nostachel = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                    to_include=("freq_50hz_sample", "T=24H"),
                                                    to_exclude=("TTX", "STACHEL", "NI"),
                                                    verbose=False,
                                                    save=False,
                                                    select_samples=[5, 6, 7],
                                                    separate_samples=False,
                                                    label_comment=" NOSTACHEL")

    discarded_cov_nostachel = fp.discard_outliers_by_iqr(cov_nostachel, low_percentile=percentiles,
                                                         high_percentile=1 - percentiles,
                                                         mode='capping')

    ni_nostachel = fp.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                   to_include=("freq_50hz_sample", "T=24H"),
                                                   to_exclude=("TTX", "STACHEL", "INF"),
                                                   verbose=False,
                                                   save=False,
                                                   select_samples=[5, 6, 7],
                                                   separate_samples=False,
                                                   label_comment=" NOSTACHEL")

    discarded_ni_nostachel = fp.discard_outliers_by_iqr(ni_nostachel, low_percentile=percentiles,
                                                        high_percentile=1 - percentiles,
                                                        mode='capping')

    cov_stachel = fp.make_dataset_from_freq_files(parent_dir=P.STACHEL,
                                                  to_include=("freq_50hz_sample", "T=24H",),
                                                  to_exclude=("TTX", "NI"),
                                                  verbose=False,
                                                  save=False,
                                                  separate_samples=False,
                                                  label_comment=" STACHEL0"
                                                  )
    discarded_cov_stachel = fp.discard_outliers_by_iqr(cov_stachel, low_percentile=percentiles,
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
