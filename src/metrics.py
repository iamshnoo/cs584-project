import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    help="Name of the dataset",
    choices=["liar", "ti_cnn", "isot", "nela", "tfg", "kaggle_fake_news"],
    default="liar",
)
args = parser.parse_args()

DATASET = args.dataset
PATH = f"../outputs/ft-bert/{DATASET}_test.csv"

# Main execution
if __name__ == "__main__":
    # Load the data
    df = pd.read_csv(PATH)

    # Extract true labels and predicted labels
    y_true = df["true_labels"]
    y_pred = df["predicted_labels"]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # For AUC-ROC, labels need to be binary
    auc_roc = roc_auc_score(y_true, y_pred)

    # Print the results
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUC-ROC:", auc_roc)
