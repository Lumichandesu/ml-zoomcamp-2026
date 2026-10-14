# Homework 4: Evaluation Metrics for Classification

This folder contains a reproducible implementation of the [official ML Zoomcamp 2026 Homework 4](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/04-evaluation/homework.md).

## Dataset

The course dataset is not included. Download `course_lead_scoring_2026.csv` from the [official 2026 data release](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv). The script checks that the file's SHA-256 is `2d9da196bdefd2a45aa0a17ca21c19f06a5d9d56af628be1f04bbb3531cc12e9` before calculating any results.

## Run locally

```sh
python -m pip install -r requirements.txt
python calculate.py /path/to/course_lead_scoring_2026.csv
```

The script follows the assigned preprocessing and split instructions, evaluates the model on the validation data, computes the five-fold AUC statistics, and keeps the held-out test data out of model selection. It writes `results.json` and `precision_recall_thresholds.svg` next to the script.
