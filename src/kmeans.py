from data_load import *
from zeugma.embeddings import EmbeddingTransformer
import json
from sklearn.cluster import KMeans
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import argparse


def get_data(dataset):
    if dataset == "nela":
        data_train = NELADataset().df
        data_valid = NELADataset(split="valid").df
        data_test = NELADataset(split="test").df
    if dataset == "kaggle_fake_news":
        data_train = KaggleFakeNewsDataset().df
        data_valid = KaggleFakeNewsDataset(split="valid").df
        data_test = KaggleFakeNewsDataset(split="test").df
    if dataset == "isot":
        data_train = ISOTDataset().df
        data_valid = ISOTDataset(split="valid").df
        data_test = ISOTDataset(split="test").df
    if dataset == "liar":
        data_train = LIARDataset().df
        data_valid = LIARDataset(split="valid").df
        data_test = LIARDataset(split="test").df
    if dataset == "tfg":
        data_train = TFGDataset().df
        data_valid = TFGDataset(split="valid").df
        data_test = TFGDataset(split="test").df
    if dataset == "ti_cnn":
        data_train = TICNNDataset().df
        data_valid = TICNNDataset(split="valid").df
        data_test = TICNNDataset(split="test").df

    return data_train, data_valid, data_test


def project_pca(X, comps="mle"):
    pca = PCA(n_components=comps, svd_solver="full")
    pca.fit(X)
    return pca.transform(X)


def get_list(a):
    l = []

    for i in range(a.shape[0]):
        l.append(int(a[i]))

    return l


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset",
        type=str,
        default="nela",
        choices=["nela", "kaggle_fake_news", "isot", "liar", "tfg", "ti_cnn"],
    )
    args.add_argument(
        "--embeddings", type=str, default="fasttext", choices=["glove", "fasttext"]
    )
    args = args.parse_args()

    data_train, data_valid, data_test = get_data(args.dataset)
    glove = EmbeddingTransformer(args.embeddings)

    x_train = glove.transform(data_train["content"])
    y_train = data_train["label"]

    x_test = glove.transform(data_test["content"])
    y_test = data_test["label"]

    x_valid = glove.transform(data_valid["content"])
    y_valid = data_valid["label"]

    data = x_train

    out = []
    for n_components in range(2, 26):
        for k in range(2, 11, 2):
            print("# components = ", n_components)
            print("K = ", k)
            data = project_pca(x_train, n_components)

            kmeans = KMeans(n_clusters=k, random_state=0, init="k-means++").fit(
                data, y=data_train["label"]
            )

            labels = kmeans.predict(data)
            print("Labels:", labels)

            db_score = davies_bouldin_score(data, labels)

            silhouette_avg = silhouette_score(data, labels)

            train_acc = (
                np.sum(np.array(labels) == np.array(data_train["label"]))
                / np.array(labels).shape[0]
            )

            data_v = project_pca(x_test, n_components)

            labels_valid = kmeans.predict(data_v)

            valid_acc = (
                np.sum(np.array(labels_valid) == np.array(y_valid))
                / np.array(labels_valid).shape[0]
            )

            out.append(
                {
                    "n_components": int(n_components),
                    "K": int(k),
                    "DB_Score": float(db_score),
                    "Silhouette": float(silhouette_avg),
                    "Train Accuracy": float(train_acc),
                    "Train Predicted": get_list(labels),
                    "Valid Accuracy": float(valid_acc),
                    "Valid Predicted": get_list(labels_valid),
                }
            )

            train_acc = (
                np.sum(np.array(labels) == np.array(data_train["label"]))
                / np.array(labels).shape[0]
            )
            print(db_score, silhouette_avg, train_acc)

            with open(f"{args.embeddings}_output_{args.dataset}.json", "w") as f:
                obj = json.dumps(out, indent=4)
                f.write(obj)
