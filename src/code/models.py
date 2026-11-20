"""Model definitions: neural networks, conventional classifiers and threshold rules."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# ---------------------------------------------------------------- neural networks
# Base (untuned) configurations, as in Table 2/3 of the manuscript.
BASE_CFG = {
    "CNN": dict(filters=32, dense=0, lr=1e-3),
    "RNN": dict(units=50, lr=1e-3),
    "LSTM": dict(units=50, lr=1e-3),
}

# Training-only tuning grids. "full" = 6 settings per model (the conventional
# models have 5-9); "reduced" = 4 settings per model.
NN_GRID = {
    "full": {
        "CNN": [dict(filters=f, dense=h, lr=lr) for (f, h) in [(32, 0), (64, 32)]
                for lr in (1e-3, 3e-3, 1e-2)],
        "RNN": [dict(units=u, lr=lr) for u in (32, 64) for lr in (1e-3, 3e-3, 1e-2)],
        "LSTM": [dict(units=u, lr=lr) for u in (32, 64) for lr in (1e-3, 3e-3, 1e-2)],
    },
    "reduced": {
        "CNN": [dict(filters=f, dense=h, lr=lr) for (f, h) in [(32, 0), (64, 32)]
                for lr in (3e-3, 1e-2)],
        "RNN": [dict(units=u, lr=lr) for u in (32, 64) for lr in (3e-3, 1e-2)],
        "LSTM": [dict(units=u, lr=lr) for u in (32, 64) for lr in (3e-3, 1e-2)],
    },
}

# Early-stopping budget for the tuned neural models
TUNED_BATCH = 128
TUNED_MAX_EPOCHS = 400
TUNED_PATIENCE = 20


def n_params(kind, cfg, n_feat=3):
    m = build_nn(kind, cfg, n_feat)
    return int(sum(np.prod(w.shape) for w in m.trainable_weights))


def build_nn(kind, cfg, n_feat=3):
    import tensorflow as tf
    L = tf.keras.layers
    inp = L.Input(shape=(n_feat, 1))
    if kind == "CNN":
        x = L.Conv1D(cfg["filters"], 2, activation="relu")(inp)
        x = L.Flatten()(x)
        if cfg.get("dense", 0):
            x = L.Dense(cfg["dense"], activation="relu")(x)
    elif kind == "RNN":
        x = L.SimpleRNN(cfg["units"], return_sequences=True)(inp)
        x = L.SimpleRNN(cfg["units"])(x)
    elif kind == "LSTM":
        x = L.LSTM(cfg["units"], return_sequences=True)(inp)
        x = L.LSTM(cfg["units"])(x)
    else:
        raise ValueError(kind)
    out = L.Dense(1, activation="sigmoid")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr"]), loss="binary_crossentropy",
              steps_per_execution=cfg.get("spe", 1))
    return m


def fit_nn(kind, cfg, Xtr, ytr, seed, epochs=10, batch=16, Xva=None, yva=None,
           patience=None):
    """Train one network. With validation data and patience: early stopping on
    validation loss, best weights restored."""
    import tensorflow as tf
    from common import seed_all
    seed_all(seed)
    m = build_nn(kind, cfg, Xtr.shape[1])
    cb = []
    if patience is not None:
        cb.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                                   restore_best_weights=True))
    h = m.fit(Xtr[..., None], ytr, epochs=epochs, batch_size=batch, verbose=0,
              validation_data=None if Xva is None else (Xva[..., None], yva),
              callbacks=cb, shuffle=True)
    hist = {k: [float(v) for v in vals] for k, vals in h.history.items()}
    return m, hist


def nn_predict(m, X):
    return m.predict(X[..., None], verbose=0, batch_size=2048).ravel()


# ----------------------------------------------------------- conventional models
def fixed_conventional(seed=0):
    """Prespecified settings (fixed-configuration benchmark, Table 5)."""
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "DT": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=500,
                                                             random_state=seed)),
        "RF": RandomForestClassifier(200, n_jobs=-1, random_state=seed),
        "HGB": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31,
                                              random_state=seed),
    }


def tuned_conventional(seed=0):
    """(estimator, parameter grid) pairs for nested training-only tuning."""
    from xgboost import XGBClassifier
    return {
        "LR": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
               {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}),
        "DT": (DecisionTreeClassifier(random_state=seed),
               {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5]}),
        "SVM": (make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.1, 1.0]}),
        "MLP": (make_pipeline(StandardScaler(), MLPClassifier(max_iter=500, random_state=seed)),
                {"mlpclassifier__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
                 "mlpclassifier__alpha": [1e-4, 1e-2]}),
        "RF": (RandomForestClassifier(200, n_jobs=-1, random_state=seed),
               {"max_depth": [None, 10, 20], "min_samples_leaf": [1, 3]}),
        "HGB": (HistGradientBoostingClassifier(random_state=seed),
                {"learning_rate": [0.05, 0.1, 0.2], "max_leaf_nodes": [15, 31]}),
        "XGBoost": (XGBClassifier(n_estimators=300, n_jobs=-1, random_state=seed,
                                  eval_metric="logloss", tree_method="hist"),
                    {"learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 6]}),
    }


def score_of(est, X):
    """Continuous score and its decision threshold."""
    if hasattr(est, "predict_proba"):
        try:
            return est.predict_proba(X)[:, 1], 0.5
        except AttributeError:
            pass
    return est.decision_function(X), 0.0


# ---------------------------------------------------------------- threshold rules
def _f1_cols(y, F):
    tp = (F & (y[:, None] == 1)).sum(0)
    fp = (F & (y[:, None] == 0)).sum(0)
    fn = ((~F) & (y[:, None] == 1)).sum(0)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def fit_threshold1(X, y, names):
    """Best single-attribute cut-off (x <= t or x >= t) by training F1."""
    best = (-1.0, None)
    for j in range(X.shape[1]):
        ts = np.unique(np.quantile(X[:, j], np.linspace(0.005, 0.995, 400)))
        for direction in ("le", "ge"):
            col = X[:, j][:, None]
            F = col <= ts[None, :] if direction == "le" else col >= ts[None, :]
            f = _f1_cols(y, F)
            i = int(np.argmax(f))
            if f[i] > best[0]:
                best = (float(f[i]), (j, direction, float(ts[i])))
    j, d, t = best[1]
    return dict(attr=names[j], j=j, dir=d, t=t, train_f1=best[0])


def apply_threshold1(r, X):
    x = X[:, r["j"]]
    return (x <= r["t"]) if r["dir"] == "le" else (x >= r["t"])


def fit_threshold2(X, y):
    """Flag a record if BatteryLife <= a or Temperature is outside [l, u].
    a, l and u are chosen by grid search on training F1 (columns: SensorData,
    BatteryLife, Temperature)."""
    bl, T = X[:, 1], X[:, 2]
    A = np.arange(5.0, 30.01, 0.5)
    Ls = np.arange(0.0, 25.01, 0.5)
    Us = np.arange(55.0, 90.01, 0.5)
    best = (-1.0, None)
    upper = T[:, None] > Us[None, :]
    for a in A:
        base = bl <= a
        for l in Ls:
            F = (base | (T < l))[:, None] | upper
            f = _f1_cols(y, F)
            i = int(np.argmax(f))
            if f[i] > best[0]:
                best = (float(f[i]), (float(a), float(l), float(Us[i])))
    a, l, u = best[1]
    return dict(a=a, l=l, u=u, train_f1=best[0])


def apply_threshold2(r, X):
    return (X[:, 1] <= r["a"]) | (X[:, 2] < r["l"]) | (X[:, 2] > r["u"])
