import datetime
import os
from pathlib import Path

import fiiireflyyy.firefiles as ff
import matplotlib.pyplot as plt
import pandas as pd
import requests

import PATHS as P
import complete_procedures as cp
import data_processing as dpr

BOT_TOKEN = "5852039858:AAFajWAuLZjoQM6Tm43EbPEnlkNgqpWCIYE"
CHAT_ID = 1988021253


def send_telegram_notification(text):
    text += f"\n\n {datetime.datetime.now()}"
    token = BOT_TOKEN
    url = f"https://api.telegram.org/bot{token}"
    params = {"chat_id": CHAT_ID, "text": text}
    r = requests.get(url + "/sendMessage", params=params)


def main():
    cp.fig2a_PCA_on_regionHz_all_organoids_for_Mock_CoV_test_stachel(300, 5000, batch='batch 2')


main()



# try:
#     start = datetime.datetime.now()
#     main()
#     send_telegram_notification(f"Execution finished.\nRuntime: {datetime.datetime.now() - start}.")
# except Exception as e:
#     send_telegram_notification(str(e))


def for_later_spikes():
    percentiles = 0.1
    min_freq = 0
    max_freq = 500
    threshold = 0.6e10
    parent_dir = P.NOSTACHEL
    select_organoids = [1, 2, 3, 4, 5, 6, 7]
    to_include = ("pr_", "T=24H")
    to_exclude = ("TTX", "STACHEL", "INF")
    if select_organoids is False:
        select_organoids = [1, 2, 3, 4, 5, 6, 7]

    files = ff.get_all_files(os.path.join(parent_dir))
    temp_files = []
    for f in files:
        if all(i in f for i in to_include) and (not any(e in f for e in to_exclude)) and int(
                os.path.basename(Path(f).parent)) in select_organoids:
            temp_files.append(f)

    columns = list(range(0, 300))
    dataset = pd.DataFrame(columns=columns)
    target = pd.DataFrame(columns=["label", ])

    n_processed_files = 0
    for t in temp_files:
        df = pd.read_csv(t)
        df_mean = dpr.merge_all_columns_to_mean(df, except_column="TimeStamp [µs]")
        plt.plot(df_mean["mean"])
        plt.show()
        # todo : count spikes
        # todo: how to define threshold ? mail to raph and ganesh
