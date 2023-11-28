import pandas as pd
import torch
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm

from bert_model import BertClassifier

# Set seeds
SEED = 42
torch.manual_seed(SEED)

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

# Define constants
DATASET_NAME = args.dataset
MODEL_PATH = f"../models/{DATASET_NAME}.pt"
TEST_DATA_PATH = f"../outputs/attack/{DATASET_NAME}_test.csv"
OUTPUT_PATH = f"../outputs/attack/{DATASET_NAME}_test.csv"

# Model and Tokenizer settings
MODEL_NAME = "distilbert-base-uncased"
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 32

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    model = BertClassifier(name=MODEL_NAME)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


def run_predictions(model, df, attack_type):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sentences = df[attack_type].tolist()
    sentences = [str(s) for s in sentences]
    encoded = tokenizer.batch_encode_plus(
        sentences, max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding="max_length"
    )
    input_ids = torch.tensor(encoded["input_ids"]).to(device)
    attention_mask = torch.tensor(encoded["attention_mask"]).to(device)

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_ids, attention_mask), batch_size=BATCH_SIZE
    )
    predicted_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            inputs, masks = batch
            outputs = model(inputs, masks)
            _, preds = torch.max(outputs, dim=1)
            predicted_labels.extend(preds.cpu().numpy())

    df[f"predicted_label_after_{attack_type}"] = predicted_labels
    return df


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    df = pd.read_csv(TEST_DATA_PATH)
    df = run_predictions(model, df, "negation_attack")
    df = run_predictions(model, df, "adverb_intensity_attack")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_PATH}")
