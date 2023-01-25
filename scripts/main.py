import datetime

import pandas as pd
import os
import fileinput
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import signal_processing as spr
import data_processing as dpr
import machine_learning as ml
import seaborn as sns
import PATHS as P
import data_processing as dp

import fiiireflyyy.firelearn as fl
import fiiireflyyy.firefiles as ff

import pickle
from pathlib import Path

import complete_procedures as cp
import get_plots as gp
import requests

BOT_TOKEN = "5852039858:AAFajWAuLZjoQM6Tm43EbPEnlkNgqpWCIYE"
CHAT_ID = 1988021253
def send_telegram_notification(text):
    text += f"\n\n {datetime.datetime.now()}"
    token = BOT_TOKEN
    url = f"https://api.telegram.org/bot{token}"
    params = {"chat_id": CHAT_ID, "text": text}
    r = requests.get(url + "/sendMessage", params=params)


def main():
    # create dataset : train model on T=24H, INF/NI, no Stachel
    # test on T=0/30min/24H and Stachel: None/0min, test on T=24H and stachel:24H
    n_components = 3
    df24_nostachel = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                                      to_include=("freq_50hz_sample", "T=24H"),
                                                      to_exclude=("TTX", "STACHEL"),
                                                      verbose=False,
                                                      save=False,
                                                      separate_organoids=False,
                                                      label_comment=" NOSTACHEL")

    pca, pc_df24_nostachel, _ = fl.fit_pca(df24_nostachel, n_components=n_components)

    df24_stachel0 = dpr.make_dataset_from_freq_files(parent_dir=P.STACHEL,
                                                     to_include=("freq_50hz_sample", "T=24H",),
                                                     to_exclude=("TTX",),
                                                     verbose=False,
                                                     save=False,
                                                     separate_organoids=False,
                                                     label_comment=" STACHEL0"
                                                     )
    pc_df24_stachel0 = fl.apply_pca(pca, df24_stachel0)

    global_pc_df = pd.concat([pc_df24_nostachel,
                              pc_df24_stachel0], ignore_index=True)
    fl.plot_pca(global_pc_df, n_components=n_components, show=True, commentary="fitted NOSTACHEL", points=True,
                 metrics=True, savedir="")

    # rfc, _ = fl.train_RFC_from_dataset(pc_df24_nostachel)
    # fl.test_model(rfc, global_pc_df, training_targets=("NI NOSTACHEL", "INF NOSTACHEL",),
    #               testing_targets=("NI NOSTACHEL", "INF NOSTACHEL", f"NI STACHEL0", "INF STACHEL0"), show=False,
    #               savepath=P.RESULTS, commentary="")



def test_from_scratch():
    df24 = dpr.make_dataset_from_freq_files(parent_dir=P.NOSTACHEL,
                                            to_include=("freq_50hz_sample", "T=24H"),
                                            to_exclude=("TTX", "STACHEL"),
                                            verbose=True,
                                            save=False, )
    clf = ml.train_model_from_dataset(df24)
    foi = ml.get_features_of_interest_from_trained_model(clf)
    del clf
    hdf24 = dpr.make_highest_features_dataset_from_complete_dataset(foi, df24)
    clf = ml.train_model_from_dataset(hdf24)

    scores = ml.test_model(clf, hdf24, iterations=15, verbose=True)
    print(scores)


def test_with_loaded_model():
    clf = pickle.load(open(os.path.join(P.MODELS, "T=24H mixed organoids - base foi - no stachel.sav"), "rb"))
    df24 = dpr.make_dataset_from_freq_files(parent_dir=P.STACHEL,
                                            to_include=("freq_50hz_sample", "T=24H", "TTX",),
                                            to_exclude=("STACHEL",),
                                            verbose=True,
                                            save=False, )

    hdf24 = dpr.make_highest_features_dataset_from_complete_dataset(clf.feature_names, df24)

    scores = ml.test_model(clf, hdf24, iterations=15, verbose=True, show_metrics=True)
    print(scores)


main()
# try:
#     start = datetime.datetime.now()
#     main()
#     send_telegram_notification(f"Execution finished.\nRuntime: {datetime.datetime.now() - start}.")
# except Exception as e:
#     send_telegram_notification(str(e))
