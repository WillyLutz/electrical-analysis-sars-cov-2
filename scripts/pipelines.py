import pickle

import matplotlib.pyplot as plt
import pandas as pd

import fiiireflyyy.learn as fl
import os
import fiiireflyyy.process as fp
import numpy as np
from sklearn.model_selection import train_test_split


def confusion(train, test, merge_path="example_data/merge.csv"):
    percentiles = 0.1
    
    
    
    merge = pd.read_csv(os.path.join(os.getcwd(), merge_path), index_col=False)
    
    # random split train test for train labels
    df_train = merge[merge["label"].isin(train)]
    X = df_train[df_train.columns[:-1]]
    y = df_train["label"]
    X_train_tr, X_test_tr, y_train_tr, y_test_tr = train_test_split(X, y, train_size=0.7)
    
    # random split train test for test labels
    # todo : handle case where there is not test
    df_test = merge[merge["label"].isin(test)]
    X = df_test[df_test.columns[:-1]]
    y = df_test["label"]
    X_train_te, X_test_te, y_train_te, y_test_te = train_test_split(X, y, train_size=0.7)
    
    # aggregating label column for outlier removal
    X_train_tr["label"] = y_train_tr
    X_train_te["label"] = y_train_te
    X_test_tr["label"] = y_test_tr
    X_test_te["label"] = y_test_te
    
    # get iqr metrics on training labels only
    _, iqr_metrics = fp.discard_outliers_by_iqr(X_train_tr, low_percentile=percentiles,
                                                high_percentile=1 - percentiles,
                                                mode='capping', metrics=None)
    
    # apply outlier removal to all sub datasets
    X_train_tr = fp.discard_outliers_by_iqr(X_train_tr, low_percentile=percentiles,
                                            high_percentile=1 - percentiles,
                                            mode='capping', metrics=iqr_metrics)
    X_train_te = fp.discard_outliers_by_iqr(X_train_te, low_percentile=percentiles,
                                            high_percentile=1 - percentiles,
                                            mode='capping', metrics=iqr_metrics)
    X_test_tr = fp.discard_outliers_by_iqr(X_test_tr, low_percentile=percentiles,
                                           high_percentile=1 - percentiles,
                                           mode='capping', metrics=iqr_metrics)
    X_test_te = fp.discard_outliers_by_iqr(X_test_te, low_percentile=percentiles,
                                           high_percentile=1 - percentiles,
                                           mode='capping', metrics=iqr_metrics)
    
    # concatenating post outliers removal for model training
    processed_df_train = pd.concat([X_train_tr, X_test_tr], ignore_index=True)
    processed_df_test = pd.concat([X_train_te, X_test_te], ignore_index=True)
    
    
    
    rfc, _ = fl.train_RFC_from_dataset(processed_df_train)
    fl.test_clf_by_confusion(rfc, pd.concat([processed_df_train, processed_df_test], ignore_index=True),
                             training_targets=train,
                             testing_targets=train + test,
                             show=True, verbose=False, savepath="",
                             title=f"",
                             iterations=10, )


def pca(train, test, n_component, merge_path):
    percentiles = 0.1
    
    merge = pd.read_csv(os.path.join(os.getcwd(), merge_path), index_col=False)
    train_dataframes = []
    test_dataframes = []
    
    for tr in train:
        df = merge[merge["label"] == tr]
        df, _ = fp.discard_outliers_by_iqr(df, low_percentile=percentiles,
                                        high_percentile=1 - percentiles,
                                        mode='capping')
        train_dataframes.append(df)
    
    for te in test:
        if te not in train:
            df = merge[merge["label"] == te]
            df, _ = fp.discard_outliers_by_iqr(df, low_percentile=percentiles,
                                            high_percentile=1 - percentiles,
                                            mode='capping')
            test_dataframes.append(df)
    
    pca, pcdf, ratios = fl.fit_pca(pd.concat(train_dataframes, ignore_index=True), n_components=n_component)
    pcdf_applied = fl.apply_pca(pca, pd.concat(test_dataframes, ignore_index=True))
    
    fl.plot_pca(pd.concat([pcdf, pcdf_applied], ignore_index=True),
                n_components=n_component,
                show=True,
                metrics=True,
                ratios=[round(x, 2) for x in ratios])
