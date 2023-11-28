# Project Files

Categorized by task.

## Classification using KNN, Random Forest

- `knn.py` - Sklearn KNN classifier
- `random_forest.py` - Sklearn Random Forest classifier

## Classification using BERT

- `data_load.py` - PyTorch data loading utility for all 6 datasets
- `bert_model.py` - PyTorch wrapper class for a BERT model that adds a linear
classifier on top of the BERT model
- `train.py` - PyTorch training script for a BERT model
- `metrics.py` - Calculates test set metrics for BERT model

## Clustering using K-Means

- `kmeans.py` - Sklearn K-Means clustering

## Case study 1: Interpretability

- `shap_study.py` - SHAP based interpretability study for examples incorrectly
predicted by BERT

## Case study 2: Adversarial examples

- `attack.py` - Defines negation attack and adversarial attacks
- `attack_pred.py` - Predicts labels for adversarial examples
- `attack_eval.py` - Evaluates success metrics of the adversarial attacks

## Plotting utils

Files inside src/plots.
Contains utility functions used for plotting graphs presented in the reports.

- `eda.py` - Data analysis (article lengths, word clouds, etc.)
- `knn_graphs.py` - Graphs for KNN classifier grid search
- `rf_graphs.py` - Graphs for Random Forest classifier grid search
- `kmeans_graphs.py` - Graphs for K-Means clustering (DB score and Silhouette
  score vs Number of clusters)

## Setup

- `setup.py` - Present in the root folder, used to setup src and plots and the
  main folders as packages

## Main folder structure

```bash
.
├── __init__.py
├── data
│   ├── isot
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── valid.csv
│   ├── kaggle_fake_news
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── valid.csv
│   ├── liar
│   │   ├── README.md
│   │   ├── false_negatives_val.csv
│   │   ├── false_positives_val.csv
│   │   ├── sentence_length_distribution_liar.png
│   │   ├── test.tsv
│   │   ├── train.tsv
│   │   └── valid.tsv
│   ├── nela
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── valid.csv
│   ├── tfg
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── valid.csv
│   └── ti_cnn
│       ├── test.csv
│       ├── train.csv
│       └── valid.csv
├── figs
│   ├── article_lengths.pdf
│   ├── class_dists.pdf
│   ├── kmeans_Elbow Curve (DB Score).pdf
│   ├── kmeans_Silhouette Score.pdf
│   ├── knn_graph.pdf
│   ├── random_forest_graph.pdf
│   └── tfidf_combined.pdf
├── logs
│   ├── ft.1286939.err.txt
│   ├── ft.1286939.out.txt
│   ├── ft.1286941.err.txt
│   ├── ft.1286941.out.txt
│   ├── ft.1286942.err.txt
│   ├── ft.1286942.out.txt
│   ├── ft.1286947.err.txt
│   ├── ft.1286947.out.txt
│   ├── ft.1286948.err.txt
│   ├── ft.1286948.out.txt
│   ├── ft.1286949.err.txt
│   └── ft.1286949.out.txt
├── models
│   ├── isot.pt
│   ├── kaggle_fake_news.pt
│   ├── liar.pt
│   ├── nela.pt
│   ├── tfg.pt
│   └── ti_cnn.pt
├── outputs
│   ├── attack
│   │   ├── isot_test.csv
│   │   ...
│   │   ├── results
│   │   │   ├── isot_results.csv
│   │   │   ...
│   ├── case-study
│   │   ├── liar_casestudy.csv
│   │   └── ti_cnn_casestudy.csv
│   ├── ft-bert
│   │   ├── isot_test.csv
│   │   ...
│   │   ├── metrics.xlsx
│   ├── kmeans
│   │   ├── fasttext_output_isot.json
│   │   ...
│   │   ├── glove_output_isot.json
│   │   ...
│   ├── knn
│   │   ├── valid_scores_KNN_isot.json
│   │   ...
│   └── random_forest
│       ├── valid_scores_RF_isot.json
│       ...
├── setup.py
└── src
    ├── README.md
    ├── __init__.py
    ├── attack.py
    ├── attack_eval.py
    ├── attack_pred.py
    ├── bert_model.py
    ├── data_load.py
    ├── kmeans.py
    ├── knn.py
    ├── metrics.py
    ├── plots
    │   ├── __init__.py
    │   ├── eda.py
    │   ├── kmeans_graphs.py
    │   ├── knn_graphs.py
    │   └── rf_graphs.py
    ├── random_forest.py
    ├── shap_study.py
    └── train.py

```
