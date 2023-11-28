import json

import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filename):
    with open(filename, "r") as f:
        return json.load(f)


data1 = load_data("../../outputs/knn/valid_scores_KNN_isot.json")
data2 = load_data("../../outputs/knn/valid_scores_KNN_kaggle_fake_news.json")
data3 = load_data("../../outputs/knn/valid_scores_KNN_liar.json")
data4 = load_data("../../outputs/knn/valid_scores_KNN_nela.json")
data5 = load_data("../../outputs/knn/valid_scores_KNN_tfg.json")
data6 = load_data("../../outputs/knn/valid_scores_KNN_ti_cnn.json")


def extract_k_f1(data):
    k_values = [item["k"] for item in data]
    f1_scores = [item["f1"] for item in data]
    return k_values, f1_scores


k1, f1_1 = extract_k_f1(data1)
k2, f1_2 = extract_k_f1(data2)
k3, f1_3 = extract_k_f1(data3)
k4, f1_4 = extract_k_f1(data4)
k5, f1_5 = extract_k_f1(data5)
k6, f1_6 = extract_k_f1(data6)

datasets = ["ISOT", "KAGGLE", "LIAR", "NELA", "TFG", "TICNN"]

plt.figure(figsize=(12, 7))

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
sns.set_style("whitegrid")

datasets = [
    ("ISOT", k1, f1_1),
    ("KAGGLE", k2, f1_2),
    ("LIAR", k3, f1_3),
    ("NELA", k4, f1_4),
    ("TFG", k5, f1_5),
    ("TICNN", k6, f1_6),
]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

for (dataset_name, k, f1), ax in zip(datasets, axes.flatten()):
    sns.lineplot(x=k, y=f1, ax=ax, color=colors[datasets.index((dataset_name, k, f1))])
    max_f1 = max(f1)
    best_k = k[f1.index(max_f1)]
    sns.scatterplot(x=[best_k], y=[max_f1], ax=ax, color="black", marker="o")
    ax.set_title(dataset_name)
    ax.set_xlabel("k Value")
    ax.set_ylabel("F1 Score")

plt.tight_layout()
plt.savefig("../../figs/knn_graph.pdf", dpi=600)


best_values = {}

for dataset_name, k, f1 in datasets:
    max_f1 = max(f1)
    best_k = k[f1.index(max_f1)]
    best_values[dataset_name] = {"Best K": best_k, "Best F1 Score": max_f1}

for dataset, values in best_values.items():
    print(f"Dataset: {dataset}")
    print(f"Best K Value: {values['Best K']}")
    print(f"Best F1 Score: {values['Best F1 Score']}\n")
