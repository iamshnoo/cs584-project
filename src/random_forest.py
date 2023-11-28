import json
import random
import sys

from data_load import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score


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


def get_scores(y_true, y_pred):
    return {
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
    }


dataset_name = sys.argv[1]

data_train, data_valid, data_test = get_data(dataset_name)

vectorizer = TfidfVectorizer()
vect = vectorizer.fit(data_train["content"])
X_train = vect.transform(data_train["content"])
y_train = data_train["label"]

X_valid = vect.transform(data_valid["content"])
y_valid = data_valid["label"]

X_valid = vect.transform(data_test["content"])
y_valid = data_test["label"]

# Trying depth values between 2 and 100

seen = []
out = []
for k in range(2, 101):
    print("K ", k)
    rf = RandomForestClassifier(max_depth=k, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_valid)
    scores = get_scores(y_valid, y_pred)
    out.append(
        {
            "k": k,
            "f1": scores["f1"],
            "recall": scores["recall"],
            "precision": scores["precision"],
        }
    )

with open(f"valid_scores_RF_{dataset_name}.json", "w") as f:
    obj = json.dumps(out, indent=4)
    f.write(obj)
