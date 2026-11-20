"""Shared utilities for the WSN node-level fault benchmark.

All experiments read data/WsnData.csv, use SensorData, BatteryLife and
Temperature as inputs, IsFaulty as the target and SensorID as the grouping
variable. Every random process is seeded.
"""
import os, json, time, random, platform
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "WsnData.csv")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
TAB = os.path.join(RES, "tables")
PRED = os.path.join(RES, "predictions")
PART = os.path.join(RES, "partial")
for _p in (RES, FIG, TAB, PRED, PART):
    os.makedirs(_p, exist_ok=True)

FEATURES = ["SensorData", "BatteryLife", "Temperature"]
OUTER_SEEDS = [42, 52, 62]
INIT_SEEDS = [0, 1, 2]
METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc"]

# Overridden by run_experiments.py --quick (smoke test)
QUICK = {"on": False}


def load():
    if not os.path.exists(DATA):
        raise SystemExit(f"Dataset not found: {DATA}\nPut WsnData.csv in the data/ folder.")
    d = pd.read_csv(DATA)
    X = d[FEATURES].to_numpy(dtype=np.float64)
    y = d["IsFaulty"].to_numpy(dtype=np.int64)
    g = d["SensorID"].to_numpy()
    return d, X, y, g


def outer_folds(y, g):
    """Repeated 3-fold StratifiedGroupKFold by SensorID (seeds 42/52/62)."""
    seeds = OUTER_SEEDS[:1] if QUICK["on"] else OUTER_SEEDS
    out = []
    for s in seeds:
        sgk = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=s)
        for k, (tr, te) in enumerate(sgk.split(np.zeros(len(y)), y, g)):
            out.append(dict(seed=s, fold=k, train=tr, test=te))
            if QUICK["on"]:
                return out
    return out


def inner_cv(seed):
    return StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)


def val_split(g_train, seed):
    """Sensor-disjoint 20% validation split inside a training part."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, va = next(gss.split(np.zeros(len(g_train)), groups=g_train))
    return tr, va


def metrics(y, score, thr=0.5, has_score=True):
    pred = (np.asarray(score) >= thr).astype(int)
    m = dict(
        accuracy=100 * accuracy_score(y, pred),
        precision=100 * precision_score(y, pred, zero_division=0),
        recall=100 * recall_score(y, pred, zero_division=0),
        f1=100 * f1_score(y, pred, zero_division=0),
        mcc=float(matthews_corrcoef(y, pred)),
    )
    try:
        m["roc_auc"] = float(roc_auc_score(y, score)) if has_score else float("nan")
    except ValueError:
        m["roc_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return m


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(s)


_TF = {}


def tf_setup(threads=0):
    """Configure TensorFlow once: thread count and deterministic kernels."""
    if "tf" in _TF:
        return _TF["tf"]
    import tensorflow as tf
    if threads:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    _TF["tf"] = tf
    return tf


def save_json(obj, name):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=_conv)


def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def exists(name):
    return os.path.exists(os.path.join(RES, name))


def _conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def environment():
    import sklearn, xgboost, scipy
    info = dict(python=platform.python_version(), platform=platform.platform(),
                processor=platform.processor() or platform.machine(),
                cpu_count=os.cpu_count(), numpy=np.__version__, pandas=pd.__version__,
                sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                scipy=scipy.__version__)
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
        info["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception as e:
        info["tensorflow"] = f"not available ({e})"
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:  # CPU model name on Linux
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    return info


def summarize(rows, keys=METRICS):
    """rows: list of dicts with 'seed' and metric keys (one per fold or per
    fold x init seed). Returns mean and SD over the seed-level means."""
    df = pd.DataFrame(rows)
    seed_means = df.groupby("seed")[keys].mean()
    out = {}
    for k in keys:
        v = seed_means[k].to_numpy()
        out[k] = float(np.nanmean(v))
        out[k + "_sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
    return out


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)
