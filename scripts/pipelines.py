import pickle

import matplotlib.pyplot as plt
import pandas as pd

import fiiireflyyy.learn as fl
import os
import fiiireflyyy.process as fp
import numpy as np
from sklearn.model_selection import train_test_split


def confusion(train, test=None, merge_path="example_data/merge.csv", savepath='', title=''):
    percentiles = 0.1
    
    merge = pd.read_csv(os.path.join(os.getcwd(), merge_path))
    
    # random split train test for train labels
    df_train_labels = merge[merge["label"].isin(train)]
    X = df_train_labels[df_train_labels.columns[:-1]]
    y = df_train_labels["label"]
    X_tr_train, X_te_train, y_tr_train, y_te_train = train_test_split(X, y, train_size=0.5)
    
    X_tr_train["label"] = y_tr_train
    X_te_train["label"] = y_te_train
    
    # discarding outliers
    _, iqr_metrics = fp.discard_outliers_by_iqr(X_tr_train, low_percentile=percentiles,
                                                high_percentile=1 - percentiles,
                                                mode='capping', metrics=None)
    X_tr_train = fp.discard_outliers_by_iqr(X_tr_train, low_percentile=percentiles,
                                            high_percentile=1 - percentiles,
                                            mode='capping', metrics=iqr_metrics)
    X_te_train = fp.discard_outliers_by_iqr(X_te_train, low_percentile=percentiles,
                                            high_percentile=1 - percentiles,
                                            mode='capping', metrics=iqr_metrics)
    # random split train test for test labels
    df_test_labels = pd.DataFrame()
    if test:
        df_test_labels = merge[merge["label"].isin(test)]
        df_test_labels = fp.discard_outliers_by_iqr(df_test_labels, low_percentile=percentiles,
                                                    high_percentile=1 - percentiles,
                                                    mode='capping', metrics=iqr_metrics)
    
    X_tr_train = X_tr_train[X_tr_train.columns[:-1]]
    rfc = fl.train_RFC_from_dataset(X_tr_train, y_tr_train)
    
    df = pd.concat([X_te_train, df_test_labels], ignore_index=True)
    fl.test_clf_by_confusion(rfc, df,
                             training_targets=train,
                             testing_targets=train + test,
                             show=True,
                             verbose=False,
                             savepath=savepath,
                             title=title,
                             iterations=10, )


def pca(train, test=None, merge_path="example_data/merge.csv", savepath='', title=''):
    percentiles = 0.1
    merge = pd.read_csv(os.path.join(os.getcwd(), merge_path))
    
    # random split train test for train labels
    df_train_labels = merge[merge["label"].isin(train)]
    # discarding outliers
    _, iqr_metrics = fp.discard_outliers_by_iqr(df_train_labels, low_percentile=percentiles,
                                                high_percentile=1 - percentiles,
                                                mode='capping', metrics=None)
    df_train_labels = fp.discard_outliers_by_iqr(df_train_labels, low_percentile=percentiles,
                                                 high_percentile=1 - percentiles,
                                                 mode='capping', metrics=iqr_metrics)
    
    pca, pcdf, ratio = fl.fit_pca(df_train_labels, 2)
    
    # random split train test for test labels
    df_test_labels = pd.DataFrame()
    test_pcdf = pd.DataFrame()
    if test:
        df_test_labels = merge[merge["label"].isin(test)]
        df_test_labels = fp.discard_outliers_by_iqr(df_test_labels, low_percentile=percentiles,
                                                    high_percentile=1 - percentiles,
                                                    mode='capping', metrics=iqr_metrics)
        test_pcdf = fl.apply_pca(pca, df_test_labels)
    
    fl.plot_pca(pd.concat([pcdf, test_pcdf], ignore_index=True), n_components=2,
                show=True,
                metrics=True,
                savedir=savepath,
                title=title,
                ratios=[round(x, 2) for x in ratio],
                dpi=300)