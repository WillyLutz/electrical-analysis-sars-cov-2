import pickle

import matplotlib.pyplot as plt
import pandas as pd

import complete_procedures_deprecated as cp
import fiiireflyyy.learn as fl
import fiiireflyyy.process as fp


def main():
    percentiles = 0.1
    
    merge = pd.read_csv("/media/wlutz/TOSHIBA EXT/Electrical activity analysis/spike experiment young organoids/DATASET/merge.csv")
    ni = merge[merge["label"] == "Mock T24"]
    cov = merge[merge["label"] == "Inf T24"]
    spike = merge[merge["label"] == "Spike T24"]
    ni = fp.discard_outliers_by_iqr(ni, low_percentile=percentiles,
                                        high_percentile=1 - percentiles,
                                        mode='capping')
    cov = fp.discard_outliers_by_iqr(cov, low_percentile=percentiles,
                                    high_percentile=1 - percentiles,
                                    mode='capping')
    spike = fp.discard_outliers_by_iqr(spike, low_percentile=percentiles,
                                        high_percentile=1 - percentiles,
                                        mode='capping')
    
    rfc, _ = fl.train_RFC_from_dataset(pd.concat([ni, cov], ignore_index=True))
    df = pd.concat([ni, cov, spike],
                   ignore_index=True)
    
    fl.test_clf_by_confusion(rfc, df,
                             training_targets=(f'Mock T24', f'Inf T24',),
                             testing_targets=(f'Mock T24', 'Inf T24', f'Spike T24', ),
                             show=True, verbose=False, savepath="",
                             title=f"",
                             iterations=1, )
    
    # for more tests and possibilities please refer to the fiiireflyyy library


main()
