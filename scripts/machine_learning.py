import os
import pickle
import time
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm, linear_model
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import data_processing as dpr
import data_analysis as dan
import FireFiles as ff





def test_model(model_path, model_type, data_path, class_column, train_size=0.7, spec_path="", commentary=""):

    dir_path = os.path.dirname(model_path)
    report_path = dir_path + r"/" + os.path.basename(model_path).split(".")[0] + spec_path + "_report.txt"
    print(report_path)
    df = pd.read_csv(data_path)
    X = df.loc[:, df.columns != class_column]
    X = X.loc[:, X.columns != "organoid number"]
    X.columns = [str(11),str(6),str(245),str(239),str(243),str(15),str(242),str(241),str(240),str(5),str(244),str(238),str(237),str(10),str(169)]
    y = df[class_column]
    print(len(X), len(y))
    with open(report_path, "w+") as f:
        text = "Model value prediction report: {} \n" \
               " ---------------------------------\n" \
               "|      Classification metrics     |\n" \
               " ---------------------------------\n".format(models_types[model_type])
        f.write(text)

        f.write(commentary)
        if commentary != "":
            f.write("\n")

        model = load_model(model_path)
        train_scores = []
        test_scores = []
        n_iter = 10
        importances_iterations = []
        for i in range(0, n_iter):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
            train = model.score(X_train, y_train)
            test = model.score(X_test, y_test)
            train_scores.append(train)
            test_scores.append(test)


        text = "Mean scores of {} iterations: \n Training: {}\n Testing: {}\n\n".format(n_iter,
                                                                                        round(np.mean(train_scores), 3),
                                                                                        round(np.mean(test_scores), 3))
        f.write(text)
        predicted = model.predict(X_test)
        matrix = confusion_matrix(y_test, predicted)
        text = "---------------------------------\n" \
               "Confusion matrix:\n---\n{}\n ".format(matrix)
        f.write(text)

        report = classification_report(y_test, predicted)
        text = "---------------------------------\n" \
               "Classification report :\n---\n{}\n ".format(report)
        f.write(text)

def test_model_across_organoids(model_path, model_type, data_path, class_column, train_size=0.7, spec_path="", commentary=""):

    dir_path = os.path.dirname(model_path)
    report_path = dir_path + r"/" + os.path.basename(model_path).split(".")[0] + spec_path + "_report.txt"
    print(report_path)
    df = pd.read_csv(data_path)

    orga_inf1 = df[df["organoid number"] == "INF1"] \
        .drop("organoid number", axis=1)
    orga_inf2 = df[df["organoid number"] == "INF2"] \
        .drop("organoid number", axis=1)
    orga_inf3 = df[df["organoid number"] == "INF3"] \
        .drop("organoid number", axis=1)
    orga_inf4 = df[df["organoid number"] == "INF4"] \
        .drop("organoid number", axis=1)
    orga_ni1 = df[df["organoid number"] == "NI1"] \
        .drop("organoid number", axis=1)
    orga_ni2 = df[df["organoid number"] == "NI2"] \
        .drop("organoid number", axis=1)
    orga_ni3 = df[df["organoid number"] == "NI3"] \
        .drop("organoid number", axis=1)
    orga_ni4 = df[df["organoid number"] == "NI4"] \
        .drop("organoid number", axis=1)
    scores_orga = []
    for orga in (orga_inf1, orga_inf2, orga_inf3, orga_inf4, orga_ni1, orga_ni2, orga_ni3, orga_ni4):
        model = load_model(model_path)
        test_scores = []
        n_iter = 10
        for i in range(0, n_iter):
            X = orga[orga.columns[:-1]]
            y = orga[orga.columns[-1]]
            X.columns = [str(11), str(6), str(245), str(239), str(243), str(15), str(242), str(241), str(240), str(5),
                         str(244), str(238), str(237), str(10), str(169)]

            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
            test = model.score(X_test, y_test)
            test_scores.append(test)
        scores_orga.append(np.mean(test_scores))

    with open(report_path, "w+") as f:
        text = "Model value prediction report: {} \n" \
               " ---------------------------------\n" \
               "|      Classification metrics     |\n" \
               " ---------------------------------\n".format(models_types[model_type])
        f.write(text)

        f.write(commentary)
        if commentary != "":
            f.write("\n")

        text = f"Mean scores of {n_iter} iterations: \nTesting across organoids {np.mean(scores_orga)}\n" \
               f"Std testing across organoids: {np.std(scores_orga)}\n"
        
        f.write(text)
    return (scores_orga)

def get_feature_weight_svm_linear(path):
    clf = load_model(path)
    coef = clf.coef_[0]
    plt.bar(range(0, 60), coef)
    plt.title("Feature weights using of SVM linear model")
    plt.xlabel("features")
    plt.ylabel("weights")
    plt.show()


def alpha_beta_kfold_calculation(paths_pr, mono_time, top_n=35, min_spike_thresh=3, max_spike_thresh=3,
                                 step_spike_thresh=1, truncate=30, data_processing=True, numbered_path="",
                                 learning=True, save=True):
    if data_processing:
        spikes_thresholds = []
        columns = ["organoid number", "std", ]

        for threshold in range(min_spike_thresh, max_spike_thresh + step_spike_thresh, step_spike_thresh):
            spikes_thresholds.append(threshold)
            columns.append("n_spikes_t" + str(threshold))

        dataset = pd.DataFrame(columns=columns)

        target = pd.DataFrame(columns=["status", ])
        for p in paths_pr:
            if p.split("\\")[3] == mono_time:
                print("path = ", p)
                df = pd.read_csv(p)
                # selecting top channels by their std
                df_top = dpr.top_N_electrodes(df, top_n, "TimeStamp [µs]")

                # divide by sample
                samples = dpr.equal_samples(df_top, truncate)
                for df_s in samples:
                    # number of spikes and mean spikes amplitude per sample
                    dataset_line = []
                    sample_std = 0
                    for threshold in range(min_spike_thresh, max_spike_thresh + step_spike_thresh, step_spike_thresh):
                        std, spikes, = dan.count_spikes_and_channel_std_by_std_all_channels(df_s, threshold)
                        dataset_line.append(spikes.round(3))
                        sample_std = std.round(3)
                    dataset_line.insert(0, p.split("\\")[5])
                    dataset_line.insert(1, sample_std)

                    # construct the dataset with n features
                    dataset.loc[len(dataset)] = dataset_line
                    if p.split("\\")[4] == "NI":
                        target.loc[len(target)] = 0
                    elif p.split("\\")[4] == "INF":
                        target.loc[len(target)] = 1
                else:
                    continue
            else:
                continue

        dataset["status"] = target["status"]
        folder = "Four organoids\\datasets\\"
        ff.verify_dir(folder)
        title = folder + f"std_range_spikes_{str(max_spike_thresh)}_{mono_time}_numbered_organoids.csv"
        dataset.to_csv(title, index=False)
    else:
        title = numbered_path

    if learning:
        # training
        print("learning")
        dataset = pd.read_csv(title)
        X = dataset[dataset.columns[1:-1]]
        y = dataset["status"]

        folder = "Four organoids\\models\\"
        ff.verify_dir(folder)
        modelpath = folder + f"std_range_spikes_{str(max_spike_thresh)}_{mono_time}_numbered_organoids"

        if save:
            isExist = os.path.exists(modelpath)
            if not isExist:
                os.makedirs(modelpath)

            orgainf1 = dataset[dataset["organoid number"] == "INF1"].drop("organoid number", axis=1)
            organi1 = dataset[dataset["organoid number"] == "NI1"].drop("organoid number", axis=1)
            orga1 = pd.concat([orgainf1, organi1])
            X1 = orga1[orgainf1.columns[:-1]]
            y1 = orga1["status"]
            X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1)

            orgainf2 = dataset[dataset["organoid number"] == "INF2"].drop("organoid number", axis=1)
            organi2 = dataset[dataset["organoid number"] == "NI2"].drop("organoid number", axis=1)
            orga2 = pd.concat([orgainf2, organi2])
            X2 = orga2[orgainf2.columns[:-1]]
            y2 = orga2["status"]
            X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2)

            orgainf3 = dataset[dataset["organoid number"] == "INF3"].drop("organoid number", axis=1)
            organi3 = dataset[dataset["organoid number"] == "NI3"].drop("organoid number", axis=1)
            orga3 = pd.concat([orgainf3, organi3])
            X3 = orga3[orgainf3.columns[:-1]]
            y3 = orga3["status"]
            X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3)

            orgainf4 = dataset[dataset["organoid number"] == "INF4"].drop("organoid number", axis=1)
            organi4 = dataset[dataset["organoid number"] == "NI4"].drop("organoid number", axis=1)
            orga4 = pd.concat([orgainf4, organi4])
            X4 = orga4[orgainf4.columns[:-1]]
            y4 = orga4["status"]
            X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4)

            X_train = pd.concat([X1_train, X2_train, X3_train, X4_train])
            X_test = pd.concat([X1_test, X2_test, X3_test, X4_test])
            y_train = pd.concat([y1_train, y2_train, y3_train, y4_train])
            y_test = pd.concat([y1_test, y2_test, y3_test, y4_test])

            modelname = "rfc1000"
            model_perf = modelpath + "\\" + modelname + ".sav"

            clf = RandomForestClassifier(n_estimators=1000)
            clf.fit(X_train, y_train)
            save_model(clf, str(modelpath + "\\" + modelname + ".sav"))
            print("orga1", clf.score(X1_test, y1_test))
            print("orga2", clf.score(X2_test, y2_test))
            print("orga3", clf.score(X3_test, y3_test))
            print("orga4", clf.score(X4_test, y4_test))
            print("mean", np.mean([clf.score(X1_test, y1_test), clf.score(X2_test, y2_test), clf.score(X3_test, y3_test), clf.score(X4_test, y4_test)]))
            print("std ", np.std([clf.score(X1_test, y1_test), clf.score(X2_test, y2_test), clf.score(X3_test, y3_test), clf.score(X4_test, y4_test)]))

            # alpha calculation
            n_orga = 1
            scores_kfs = []
            accuracies = []
            for orga in (orgainf1, orgainf2, orgainf3, orgainf4):
                n_splits = 10
                random_state = 1
                shuffle = True
                cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)

                Xorga = orga[orga.columns[:-1]]
                yorga = orga["status"]
                scores_kf = cross_val_score(clf, Xorga, yorga, scoring='accuracy', cv=cv, n_jobs=-1)
                scores_kfs.append(scores_kf)
                result = clf.score(Xorga, yorga)
                accuracies.append(result)
                print(
                    f"INF_orga={n_orga}, mean K fold score={np.mean(scores_kf)}, std score={np.std(scores_kf)}, Accuracy={round(result, 3)}")
                n_orga += 1
            print(f"alpha={np.mean(scores_kfs)}, std={np.std(scores_kfs)}, acc={np.mean(accuracies)}")

            # beta calculation
            n_orga = 1
            scores_kfs = []
            accuracies = []
            for orga in (organi1, organi2, organi3, organi4):
                n_splits = 10
                random_state = 1
                shuffle = True
                cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)

                Xorga = orga[orga.columns[:-1]]
                yorga = orga["status"]
                scores_kf = cross_val_score(clf, Xorga, yorga, scoring='accuracy', cv=cv)
                scores_kfs.append(scores_kf)
                result = clf.score(Xorga, yorga)
                accuracies.append(result)

                print(
                    f"NI_orga={n_orga}, mean K fold score={np.mean(scores_kf)}, std score={np.std(scores_kf)}, Accuracy={round(result, 3)}")
                n_orga += 1
            print(f"beta={1 - np.mean(scores_kfs)}, std={np.std(scores_kfs)}, acc={np.mean(accuracies)}")
            # ml.model_performance_analysis(model_perf, "rfc", X, y, train_size=0.7)


