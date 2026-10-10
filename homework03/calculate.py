"""HW3 calculation, executed on the official pinned data.

Source: DataTalksClub/machine-learning-zoomcamp, cohorts/2026/homework/
03-classification/homework.md (read 11 October 2026).
Use only the actual pinned CSV; this script never downloads data or posts results.
Usage: python calculate.py PATH_TO_PINNED_CSV
Requires pandas, numpy, scikit-learn. See README for setup and usage.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error('The actual pinned course CSV is required; no synthetic answers.')
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, mutual_info_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(args.csv)
    required = {'converted', 'industry', 'location', 'lead_source',
                'employment_status', 'interaction_count', 'lead_score',
                'number_of_courses_viewed', 'annual_income'}
    if required - set(df.columns):
        raise ValueError(f'Missing expected columns: {sorted(required - set(df.columns))}')
    if df['converted'].isna().any():
        raise ValueError('Missing target labels must not be silently imputed')
    features = [c for c in df.columns if c != 'converted']
    categorical = df[features].select_dtypes(include=['object', 'string', 'category']).columns
    numerical = [c for c in features if c not in categorical]
    missing_before = df[features].isna().sum().to_dict()
    for column in categorical:
        df[column] = df[column].astype('string').fillna('NA').astype(str)
    df[numerical] = df[numerical].fillna(0.0)

    pairs = [('interaction_count', 'lead_score'),
             ('number_of_courses_viewed', 'lead_score'),
             ('number_of_courses_viewed', 'interaction_count'),
             ('annual_income', 'interaction_count')]
    correlations = {f'{a} / {b}': float(df[a].corr(df[b])) for a, b in pairs}
    if not all(np.isfinite(v) for v in correlations.values()):
        raise ValueError('A candidate correlation is undefined')
    full_train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, validation = train_test_split(full_train, test_size=0.25, random_state=42)
    y_train, y_val = train['converted'].to_numpy(), validation['converted'].to_numpy()
    x_train, x_val = train[features], validation[features]

    def fit_accuracy(columns, regularization=1.0):
        encoder = DictVectorizer(sparse=False)
        encoded_train = encoder.fit_transform(x_train[columns].to_dict(orient='records'))
        encoded_val = encoder.transform(x_val[columns].to_dict(orient='records'))
        model = LogisticRegression(solver='liblinear', C=regularization,
                                   max_iter=1000, random_state=42)
        model.fit(encoded_train, y_train)
        return float(accuracy_score(y_val, model.predict(encoded_val)))

    mutual_information = {
        c: round(float(mutual_info_score(train[c], train['converted'])), 2)
        for c in ['industry', 'location', 'lead_source', 'employment_status']}
    baseline = fit_accuracy(features)
    differences = {c: baseline - fit_accuracy([f for f in features if f != c])
                   for c in ['lead_source', 'number_of_courses_viewed', 'interaction_count']}
    regularization_scores = {c: round(fit_accuracy(features, c), 3)
                             for c in [0.000001, 0.00001, 0.0001, 0.001]}
    best_score = max(regularization_scores.values())
    best_c = min(c for c, score in regularization_scores.items() if score == best_score)
    report = {
        'dataset_sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        'versions': {'pandas': pd.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'rows': len(df), 'feature_columns': features, 'missing_before': missing_before,
        'split_sizes': {'train': len(train), 'validation': len(validation), 'test': len(test)},
        'q1_industry_modes': df['industry'].mode().tolist(),
        'q2_candidate_correlations': correlations,
        'q2_max_correlation_pair': max(correlations, key=correlations.get),
        'q3_training_mutual_information_rounded_2': mutual_information,
        'q4_validation_accuracy_unrounded': baseline,
        'q4_validation_accuracy_rounded_2': round(baseline, 2),
        'q5_signed_accuracy_differences': differences,
        'q5_smallest_difference_feature': min(differences, key=differences.get),
        'q6_accuracy_rounded_3_by_C': regularization_scores,
        'q6_best_C_smallest_on_tie': best_c,
    }
    destination = Path(__file__).with_name('results.json')
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
