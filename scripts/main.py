import pickle

import matplotlib.pyplot as plt
import pandas as pd

import complete_procedures as cp
import fiiireflyyy.firelearn as fl
import fiiireflyyy.fireprocess as fp


def main():
    percentiles = 0.1
    base = "/home/wlutz/PycharmProjects/sars-cov-organoids/datasets"
    t0inf = pd.read_csv(f"{base}/DATASET_T=0 INF.csv", index_col=False)
    dt0inf = fp.discard_outliers_by_iqr(t0inf, low_percentile=percentiles,
                                        high_percentile=1 - percentiles,
                                        mode='capping')
    dt0inf["label"].replace(f'Sars-CoV', f'Sars-CoV\n0min', inplace=True)

    t30inf = pd.read_csv(f"{base}/DATASET_T=30MIN INF.csv", index_col=False)
    dt30inf = fp.discard_outliers_by_iqr(t30inf, low_percentile=percentiles,
                                         high_percentile=1 - percentiles,
                                         mode='capping')
    dt30inf["label"].replace(f'Sars-CoV', f'Sars-CoV\n30min', inplace=True)

    t24inf = pd.read_csv(f"{base}/DATASET_T=24H INF.csv", index_col=False)
    dt24inf = fp.discard_outliers_by_iqr(t24inf, low_percentile=percentiles,
                                         high_percentile=1 - percentiles,
                                         mode='capping')
    dt24inf["label"].replace(f'Sars-CoV', f'Sars-CoV\n24h', inplace=True)

    t0mock = pd.read_csv(f"{base}/DATASET_T=0 MOCK.csv", index_col=False)
    dt0mock = fp.discard_outliers_by_iqr(t0mock, low_percentile=percentiles,
                                         high_percentile=1 - percentiles,
                                         mode='capping')
    dt0mock["label"].replace(f'Mock', f'Mock\n0min', inplace=True)

    t30mock = pd.read_csv(f"{base}/DATASET_T=30MIN MOCK.csv", index_col=False)
    dt30mock = fp.discard_outliers_by_iqr(t30mock, low_percentile=percentiles,
                                          high_percentile=1 - percentiles,
                                          mode='capping')
    dt30mock["label"].replace(f'Mock', f'Mock\n30min', inplace=True)

    t24mock = pd.read_csv(f"{base}/DATASET_T=24H MOCK.csv", index_col=False)
    dt24mock = fp.discard_outliers_by_iqr(t24mock, low_percentile=percentiles,
                                          high_percentile=1 - percentiles,
                                          mode='capping')
    dt24mock["label"].replace(f'Mock', f'Mock\n24h', inplace=True)

    rfc, _ = fl.train_RFC_from_dataset(pd.concat([dt24mock, dt24inf], ignore_index=True))
    df = pd.concat([dt24mock, dt24inf, dt0mock, dt0inf, dt30mock, dt30inf],
                   ignore_index=True)

    fl.test_sequential_by_confusion(rfc, df,
                                    training_targets=(f'Mock\n24h', f'Sars-CoV\n24h',),
                                    testing_targets=tuple(set(list((
                                        f'Mock\n24h', 'Sars-CoV\n24h',
                                        f'Mock\n0min', 'Sars-CoV\n0min',
                                        f'Mock\n30min', 'Sars-CoV\n30min',)))),
                                    show=True, verbose=False, savepath="",
                                    title=f"",
                                    iterations=5, )


main()
