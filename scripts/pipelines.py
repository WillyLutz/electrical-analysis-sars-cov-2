import pickle

import matplotlib.pyplot as plt
import pandas as pd

import fiiireflyyy.learn as fl
import os
import fiiireflyyy.process as fp
import numpy as np


def confusion(train, test):
    percentiles = 0.1
    
    merge = pd.read_csv(os.path.join(os.getcwd(), 'example_data/merge.csv'), index_col=False)
    train_dataframes = []
    test_dataframes = []
    for tr in train:
        df = merge[merge["label"] == tr]
        df = fp.discard_outliers_by_iqr(df, low_percentile=percentiles,
                                        high_percentile=1 - percentiles,
                                        mode='capping')
        train_dataframes.append(df)
        test_dataframes.append(df)
    
    for te in test:
        if te not in train:
            df = merge[merge["label"] == te]
            df = fp.discard_outliers_by_iqr(df, low_percentile=percentiles,
                                            high_percentile=1 - percentiles,
                                            mode='capping')
            test_dataframes.append(df)
    
    rfc, _ = fl.train_RFC_from_dataset(pd.concat(train_dataframes, ignore_index=True))
    
    fl.test_clf_by_confusion(rfc, pd.concat(test_dataframes, ignore_index=True),
                             training_targets=train,
                             testing_targets=train + test,
                             show=True, verbose=False, savepath="",
                             title=f"",
                             iterations=1, )


def pca(train, test, n_component):
    percentiles = 0.1
    
    merge = pd.read_csv(os.path.join(os.getcwd(), 'example_data/merge.csv'), index_col=False)
    train_dataframes = []
    test_dataframes = []
    for tr in train:
        df = merge[merge["label"] == tr]
        df = fp.discard_outliers_by_iqr(df, low_percentile=percentiles,
                                        high_percentile=1 - percentiles,
                                        mode='capping')
        train_dataframes.append(df)
        test_dataframes.append(df)
    
    for te in test:
        if te not in train:
            df = merge[merge["label"] == te]
            df = fp.discard_outliers_by_iqr(df, low_percentile=percentiles,
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


def feature_importance(train):
    percentiles = 0.1
    
    merge = pd.read_csv(os.path.join(os.getcwd(), 'example_data/merge.csv'), index_col=False)
    train_dataframes = []
    test_dataframes = []
    for tr in train:
        df = merge[merge["label"] == tr]
        df = fp.discard_outliers_by_iqr(df, low_percentile=percentiles,
                                        high_percentile=1 - percentiles,
                                        mode='capping')
        train_dataframes.append(df)
        test_dataframes.append(df)
    
    rfc, _ = fl.train_RFC_from_dataset(pd.concat(train_dataframes, ignore_index=True))
    
    _, mean_importance = fl.get_top_features_from_trained_RFC(rfc, percentage=1, show=False, save=False, title='',
                                                              savepath='')
    
    y_data = np.mean([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
    x_data = [i for i in range(len(y_data))]
    
    plt.plot(x_data, y_data, color="royalblue", )
    
    ylim = plt.gca().get_ylim()
    plt.fill_between(x_data, y_data, ylim[0], color="royalblue", alpha=0.5)
    
    plt.xlabel("Frequency [Hz]", fontdict={"fontsize": 16})
    plt.ylabel("Relative importance [AU]", fontdict={"fontsize": 16})
    
    hertz = []
    factor = 5000 / 300
    for i in range(300):
        hertz.append(int(i * factor))
    xticks = [x for x in range(0, 300, 50)]
    new_ticks = [hertz[x] for x in xticks]
    xticks.append(300)
    new_ticks.append(5000)
    plt.xticks(xticks, new_ticks)
    plt.show()
