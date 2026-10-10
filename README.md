# ML Zoomcamp 2026 coursework

Coursework for the DataTalksClub ML Zoomcamp 2026 cohort.

## Homework 03

The executable calculation, dataset fingerprint, package versions, and full
numerical results are in [homework03](homework03/).

Specification: https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/03-classification/homework.md

Dataset: https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv

The CSV is not redistributed here. Download the official dataset separately and run:

```sh
python -m pip install -r homework03/requirements.txt
python homework03/calculate.py /path/to/course_lead_scoring_2026.csv
```

The script uses the exact two seeded splits, fits its encoder only on training
data, excludes the target from features, and reports validation metrics without
using the test set for model selection. It applies the prescribed missing-value
rules, signed feature-removal differences, and rounded regularization tie-break.

The submission status is recorded by the course portal, separately from this
reproducible calculation. No course certificate is claimed by this repository.
