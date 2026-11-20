# Results

Output of `src/run_experiments.py` and `src/extra_analyses.py` with the library versions in
`requirements.txt` (Python 3.12, scikit-learn 1.5.1, XGBoost 2.1.4, TensorFlow 2.18.0,
NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1); see `environment.json`.

| File | Content | Paper |
|---|---|---|
| `folds_sensor_disjoint.csv` | Test sensor and row of every record in the 9 outer folds (seeds 42/52/62) | Protocols (ii), (iii) |
| `folds_record_cv.csv` | Fold of every record in the record-level 5-fold CV (seed 42) | Protocol (i) |
| `predictions/fixed_benchmark.csv.gz` | Out-of-fold scores, fixed configurations | Table 6 |
| `predictions/tuned_conventional.csv.gz` | Out-of-fold scores, tuned conventional models and threshold rules | Table 7 |
| `predictions/neural_tuned.csv.gz` | Out-of-fold scores, tuned networks (3 initialisation seeds) | Table 7, Fig. 11 |
| `predictions/record_cv.csv.gz`, `record_cv_oof.npy` | Out-of-fold scores, record-level CV (10-epoch networks) | Table 5 |
| `fixed_benchmark.json`, `tuned_conventional.json`, `neural_tuned.json`, `record_cv.json`, `record_cv_tuned.json`, `tree_diagnostics.json` | Fold-level metrics, summaries and selected settings | Tables 5-8 |

`neural_tuned` comes from the complete rerun of the tuned neural stage described in
Section 4.7 (accuracy 97.43 / 97.32 / 97.52% for CNN / RNN / LSTM, within 0.01 points of
Table 7). Recurrent training on a multi-threaded CPU is not bit-for-bit reproducible, so a
new run can differ in the second decimal.
