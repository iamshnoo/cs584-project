import argparse
import pandas as pd
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

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

DATASET = args.dataset
PATH = f"../outputs/ft-bert/{DATASET}_test.csv"

# Negation dictionary
negate_dict = {
    "isn't": "is",
    "isn't": "is",
    "is not ": "is ",
    "is ": "is not ",
    "didn't": "did",
    "didn't": "did",
    "did not ": "did",
    "does not have": "has",
    "doesn't have": "has",
    "doesn't have": "has",
    "has ": "does not have ",
    "shouldn't": "should",
    "shouldn't": "should",
    "should not": "should",
    "should": "should not",
    "wouldn't": "would",
    "wouldn't": "would",
    "would not": "would",
    "would": "would not",
    "mustn't": "must",
    "mustn't": "must",
    "must not": "must",
    "must ": "must not ",
    "can't": "can",
    "can't": "can",
    "cannot": "can",
    " can ": " cannot ",
}

# Adverb List
BOOSTER_DICT = [
    "absolutely",
    "amazingly",
    "awfully",
    "barely",
    "completely",
    "considerably",
    "decidedly",
    "deeply",
    "enormously",
    "entirely",
    "especially",
    "exceptionally",
    "exclusively",
    "extremely",
    "fully",
    "greatly",
    "hardly",
    "hella",
    "highly",
    "hugely",
    "incredibly",
    "intensely",
    "majorly",
    "overwhelmingly",
    "really",
    "remarkably",
    "substantially",
    "thoroughly",
    "totally",
    "tremendously",
    "unbelievably",
    "unusually",
    "utterly",
    "very",
]

IRREGULAR_ES_VERB_ENDINGS = ["ss", "x", "ch", "sh", "o"]


# Negation Attack
def negate(sentence):
    original = sentence
    sentence = str(sentence)
    for key in negate_dict.keys():
        if sentence.find(key) > -1:
            return sentence.replace(key, negate_dict[key])
    doesnt_regex = r"(doesn't|doesn\\'t|does not) (?P<verb>\w+)"
    if re.search(doesnt_regex, sentence):
        return re.sub(doesnt_regex, replace_doesnt, sentence, 1)
    return " ".join([w for w in str(sentence).split() if w.lower() not in stop_words])


def __is_consonant(letter):
    return letter not in ["a", "e", "i", "o", "u", "y"]


def replace_doesnt(matchobj):
    verb = matchobj.group(2)
    if verb.endswith("y") and __is_consonant(verb[-2]):
        return "{0}ies".format(verb[0:-1])
    for ending in IRREGULAR_ES_VERB_ENDINGS:
        if verb.endswith(ending):
            return "{0}es".format(verb)
    return "{0}s".format(verb)


# Adverb intensity attack
def reduce_intensity(sentence):
    sentence = str(sentence)
    sentence = " ".join(
        [w for w in str(sentence).split() if w.lower() not in BOOSTER_DICT]
    )
    return " ".join([w for w in str(sentence).split() if w.lower() not in stop_words])


# Main execution
if __name__ == "__main__":
    # Load Dataset
    df = pd.read_csv(PATH)

    # Apply Negation Attack and Adverb Intensity Attack
    df["negation_attack"] = df["content"].apply(negate)
    df["adverb_intensity_attack"] = df["content"].apply(reduce_intensity)

    print(df.head())

    # Save Results
    df.to_csv(f"../outputs/attack/{DATASET}_test.csv", index=False)
