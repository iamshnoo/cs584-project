import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix

# Argument Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    help="Name of the dataset",
    choices=["liar", "ti_cnn", "isot", "nela", "tfg", "kaggle_fake_news"],
    default="liar",
)
args = parser.parse_args()

DATASET_NAME = args.dataset
DATA_PATH = f"../outputs/attack/{DATASET_NAME}_test.csv"
OUTPUT_PATH = f"../outputs/attack/results/{DATASET_NAME}_results.csv"


# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    return {
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1-score": f1 * 100,
    }


def calculate_additional_metrics(df, attack_type):
    original_error = (df["true_labels"] != df["predicted_labels"]).mean()
    attack_error = (
        df["true_labels"] != df[f"predicted_label_after_{attack_type}"]
    ).mean()
    delta_error_rate = (attack_error - original_error) * 100

    attack_success = (
        (df["true_labels"] == df["predicted_labels"])
        & (df["true_labels"] != df[f"predicted_label_after_{attack_type}"])
    ).mean()
    attack_success_rate = attack_success * 100

    return delta_error_rate, attack_success_rate


# Function to read data and compute metrics
def evaluate_attack(attack_type, df):
    original_metrics = calculate_metrics(df["true_labels"], df["predicted_labels"])
    attack_metrics = calculate_metrics(
        df["true_labels"], df[f"predicted_label_after_{attack_type}"]
    )
    delta_error_rate, attack_success_rate = calculate_additional_metrics(
        df, attack_type
    )
    return original_metrics, attack_metrics, delta_error_rate, attack_success_rate


df = pd.read_csv(DATA_PATH)

# Adverb Attack Evaluation
(
    adverb_original_metrics,
    adverb_attack_metrics,
    adverb_delta_error,
    adverb_success_rate,
) = evaluate_attack("adverb_intensity_attack", df)
# print("Adverb Attack Metrics:")
# print("Original:", adverb_original_metrics)
# print("After Attack:", adverb_attack_metrics)
# print("Delta Error Rate:", adverb_delta_error)
# print("Attack Success Rate:", adverb_success_rate)

# Negation Attack Evaluation
(
    negation_original_metrics,
    negation_attack_metrics,
    negation_delta_error,
    negation_success_rate,
) = evaluate_attack("negation_attack", df)
# print("Negation Attack Metrics:")
# print("Original:", negation_original_metrics)
# print("After Attack:", negation_attack_metrics)
# print("Delta Error Rate:", negation_delta_error)
# print("Attack Success Rate:", negation_success_rate)

# Save Results to CSV
results_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "Delta Error Rate",
            "Attack Success Rate",
        ],
        "Adverb Original": [
            adverb_original_metrics[k]
            for k in ["Accuracy", "Precision", "Recall", "F1-score"]
        ]
        + [None, None],
        "Adverb After Attack": [
            adverb_attack_metrics[k]
            for k in ["Accuracy", "Precision", "Recall", "F1-score"]
        ]
        + [adverb_delta_error, adverb_success_rate],
        "Negation Original": [
            negation_original_metrics[k]
            for k in ["Accuracy", "Precision", "Recall", "F1-score"]
        ]
        + [None, None],
        "Negation After Attack": [
            negation_attack_metrics[k]
            for k in ["Accuracy", "Precision", "Recall", "F1-score"]
        ]
        + [negation_delta_error, negation_success_rate],
    }
)
results_df.to_csv(OUTPUT_PATH, index=False)
print(results_df.head())
print(f"Results written to CSV file at {OUTPUT_PATH}")
