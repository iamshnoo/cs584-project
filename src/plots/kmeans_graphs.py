import json
from tqdm import tqdm
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_data(filename):
    with open(filename, "r") as f:
        return json.load(f)


data1a = load_data("../../outputs/kmeans/fasttext_output_isot.json")
data1b = load_data("../../outputs/kmeans/glove_output_isot.json")
data2a = load_data("../../outputs/kmeans/fasttext_output_kaggle_fake_news.json")
data2b = load_data("../../outputs/kmeans/glove_output_kaggle_fake_news.json")
data3a = load_data("../../outputs/kmeans/fasttext_output_liar.json")
data3b = load_data("../../outputs/kmeans/glove_output_liar.json")
data4a = load_data("../../outputs/kmeans/fasttext_output_nela.json")
data4b = load_data("../../outputs/kmeans/glove_output_nela.json")
data5a = load_data("../../outputs/kmeans/fasttext_output_tfg.json")
data5b = load_data("../../outputs/kmeans/glove_output_tfg.json")
data6a = load_data("../../outputs/kmeans/fasttext_output_ti_cnn.json")
data6b = load_data("../../outputs/kmeans/glove_output_ti_cnn.json")

datasets = ["ISOT", "KAGGLE", "LIAR", "NELA", "TFG", "TICNN"]
data_map = {
    "ISOT": {"fasttext": data1a, "glove": data1b},
    "KAGGLE": {"fasttext": data2a, "glove": data2b},
    "LIAR": {"fasttext": data3a, "glove": data3b},
    "NELA": {"fasttext": data4a, "glove": data4b},
    "TFG": {"fasttext": data5a, "glove": data5b},
    "TICNN": {"fasttext": data6a, "glove": data6b},
}


def extract_info(data):
    k = []
    silhouette = []
    db_score = []
    for d in data:
        k.append(d["K"])
        silhouette.append(d["Silhouette"])
        db_score.append(d["DB_Score"])

    values = [k, silhouette, db_score]
    return values


def plot_combined_embeddings(datasets, data_map, metric_index, title, ylabel):
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle(title, fontsize=16)

    for i, dataset_name in enumerate(datasets):
        row, col = divmod(i, 2)
        fasttext_metrics = extract_info(data_map[dataset_name]["fasttext"])[
            metric_index
        ]
        glove_metrics = extract_info(data_map[dataset_name]["glove"])[metric_index]

        axes[row, col].plot(
            fasttext_metrics, marker="o", color="blue", label="FastText"
        )
        axes[row, col].plot(glove_metrics, marker="o", color="red", label="GloVe")

        axes[row, col].set_title(f"{dataset_name} Dataset")
        axes[row, col].set_xlabel("Number of Clusters (K)")
        axes[row, col].set_ylabel(ylabel)
        axes[row, col].grid(True)
        axes[row, col].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"../../figs/kmeans_{title}.pdf", dpi=600)


SILHOUETTE_INDEX = 1
DB_SCORE_INDEX = 2
plot_combined_embeddings(
    datasets, data_map, DB_SCORE_INDEX, "Elbow Curve (DB Score)", "DB Score"
)
plot_combined_embeddings(
    datasets, data_map, SILHOUETTE_INDEX, "Silhouette Score", "Silhouette Score"
)
