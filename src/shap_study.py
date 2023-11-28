import json
import random
import re
import nltk
import numpy as np
import pandas as pd
import shap
import spacy
import torch
from nltk.corpus import stopwords
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse

from bert_model import BertClassifier

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    help="Name of the dataset",
    choices=["liar", "ti_cnn"],
    default="liar",
)
args = parser.parse_args()

DATASET = args.dataset

# Initialize NLTK and spaCy
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

# Set random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Define tokenizer and model parameters
MAX_SEQUENCE_LENGTH = 512
NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(NAME, use_fast=True)
model = BertClassifier(name=NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load model checkpoint
MODEL_CKPT = f"../models/{DATASET}.pt"
model.load_state_dict(torch.load(MODEL_CKPT, map_location=device))
model.eval()

# Define file path
TEST_PATH = f"../outputs/case-study/{DATASET}_casestudy.csv"


# Tokenization function
def tokenize(tokenizer, sentences, padding="max_length"):
    encoded = tokenizer.batch_encode_plus(
        sentences, max_length=MAX_SEQUENCE_LENGTH, truncation=True, padding=padding
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Convert to tensors and move to the correct device
    input_ids = torch.tensor(input_ids).to(device)
    attention_mask = torch.tensor(attention_mask).to(device)

    return input_ids, attention_mask


# Get model output
def get_model_output(sentences):
    # Convert NumPy array to a list if necessary
    if isinstance(sentences, np.ndarray):
        sentences = sentences.tolist()

    input_ids, attention_mask = tokenize(tokenizer, sentences)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probabilities = torch.softmax(output, dim=-1)
    return probabilities.cpu().numpy()


# Preprocess text
def preprocess_text(text):
    text = re.sub(r"#", "", text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    text = " ".join(tokens)
    return text


# SHAP analysis
def shapper(sentence, output_class):
    explainer = shap.Explainer(
        lambda x: get_model_output(x),
        shap.maskers.Text(tokenizer),
        silent=False,
    )
    shap_values = explainer([sentence])
    importance_values = shap_values[:, :, output_class].values
    tokenized_sentence = tokenizer.tokenize(sentence)
    token_importance = list(zip(tokenized_sentence, importance_values[0]))

    # Perform NER
    doc = nlp(sentence)
    aggregated_token_importance = []
    token_scores = {
        token: score for token, score in token_importance if not token.startswith("##")
    }
    token_scores_aggregated = token_scores.copy()

    for ent in doc.ents:
        scores = [token_scores.get(token, 0) for token in ent.text.split()]
        aggregated_score = sum(scores)
        average_score = aggregated_score / len(scores) if scores else 0
        aggregated_token_importance.append((ent.text, average_score))

        for token in ent.text.split():
            if token in token_scores_aggregated:
                del token_scores_aggregated[token]

    for token, score in token_scores_aggregated.items():
        aggregated_token_importance.append((token, score))

    shap_neg_outs = [item for item in aggregated_token_importance if item[1] < 0]
    shap_neg_outs = sorted(shap_neg_outs, key=lambda x: x[1])

    shap_pos_outs = [item for item in aggregated_token_importance if item[1] > 0]
    shap_pos_outs = sorted(shap_pos_outs, key=lambda x: x[1], reverse=True)

    return shap_neg_outs, shap_pos_outs


# Main execution
if __name__ == "__main__":
    df = pd.read_csv(TEST_PATH)
    df = df.dropna(subset=["content"])
    df["processed_content"] = df["content"].apply(preprocess_text)

    print(f"Running SHAP on {len(df)} samples...")
    print("-" * 80)

    results = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        neg_outs, pos_outs = shapper(row["processed_content"], row["predicted_labels"])
        results.append([neg_outs, pos_outs])

    df[["shap_neg_outs", "shap_pos_outs"]] = pd.DataFrame(
        results, columns=["shap_neg_outs", "shap_pos_outs"]
    )
    print(df.head())
    print("-" * 80)

    df.to_csv(TEST_PATH, index=False)
