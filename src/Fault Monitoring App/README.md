# Node fault detection in WSNs: sensor-disjoint comparison of deep and conventional classifiers

Code for the manuscript "Comparing Deep and Conventional Classifiers for Node Fault Detection
in Wireless Sensor Networks Using Validation on Unseen Sensors" (Discover Applied Sciences).

## Data
`data/WsnData.csv` is the Node Status in WSN dataset (10,000 records, 100 sensors), from
Kaggle: https://www.kaggle.com/datasets/smilikasangam/node-status-in-wsn

## Environment
Python 3.12 with the pinned versions in `requirements.txt`:

    python -m venv wsn312
    wsn312\Scripts\activate          (Linux/macOS: source wsn312/bin/activate)
    pip install -r requirements.txt

Use exactly these versions to reproduce the reported numbers; newer scikit-learn versions
can change the sensor-disjoint folds.

## Running (from the repository folder)
1. Main experiments (benchmarks, statistics, learning curves, order test, ablation,
   autoencoder and runtime):

       python src/run_experiments.py

   Stages: environment, data_summary, record_cv, fixed_benchmark, tuned_conventional,
   neural_fixed, neural_tuned, stats, order_test, ablation, autoencoder, training_size,
   runtime, report. Run selected stages with `--stages record_cv,report`.
   `--quick` is a 5-10 minute smoke test. Stages are resumable; `--force` reruns them.

2. Analyses added in revision (record-level tuning, tree diagnostics and figures):

       python src/extra_analyses.py --stages trees,record,record_figs,eda

   Requires `results/tuned_conventional.json` from step 1.

The full run takes several hours on a CPU (the neural tuning stage dominates).

## Outputs
Everything is written to `results/`: tables, figures, fold assignments
(`folds_record_cv.csv`, `folds_sensor_disjoint.csv`) and out-of-fold predictions
(`predictions/*.csv.gz`). `results/environment.json` records the library versions used.

## Monitoring application
`app/fault_monitor_app.py` is the Streamlit dashboard described in Sections 3.6 and 4.12 of
the paper (Figs. 12-15). It trains HGB and the three 10-epoch networks on 80% of the sensors
and replays the records of the 20 held-out sensors one by one.

    pip install streamlit seaborn altair
    streamlit run app/fault_monitor_app.py

Upload `data/WsnData.csv` and keep the default settings (20% held out, seed 42, 200 records).
With these settings the held-out accuracies are CNN 93.75%, RNN 97.20% and LSTM 96.95%.
HGB gives 99.15% with scikit-learn 1.9.1, the version used for the screenshots in the paper,
and 99.05% with the pinned scikit-learn 1.5.1; the held-out sensors are the same in both.
