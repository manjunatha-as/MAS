"""Run every experiment of the manuscript and write tables, figures, fold
assignments and out-of-fold predictions to results/.

Usage (from the project folder):
    python src/run_experiments.py --quick            # 5-10 min smoke test
    python src/run_experiments.py                    # full run, all stages
    python src/run_experiments.py --grid reduced     # smaller neural grid
    python src/run_experiments.py --stages record_cv,report

Stages are resumable: a finished stage is skipped unless --force is given,
and the long neural tuning stage saves each finished fold to results/partial/.
"""
import os, sys, argparse, itertools, pickle, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", message="Mean of empty slice")

import common as C
from common import log
import models as M

STAGES = ["environment", "data_summary", "record_cv", "fixed_benchmark",
          "tuned_conventional", "neural_fixed", "neural_tuned", "stats",
          "order_test", "ablation", "autoencoder", "training_size", "runtime",
          "report"]

ARGS = None  # filled in main()


# ------------------------------------------------------------------ helpers
def q(full, quick):
    return quick if C.QUICK["on"] else full


def save_preds(frames, name):
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(C.PRED, name + ".csv.gz"), index=False, compression="gzip")


def pred_frame(stage, model, seed, fold, init, idx, y, score, thr):
    return pd.DataFrame(dict(stage=stage, model=model, outer_seed=seed, fold=fold,
                             init_seed=init, row=idx, y=y, score=score,
                             pred=(score >= thr).astype(int)))


def minmax(Xtr, *others):
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler().fit(Xtr)
    return [sc.transform(Xtr)] + [sc.transform(o) for o in others]


# ------------------------------------------------------------------ stages
def stage_environment():
    env = C.environment()
    env["grid"] = ARGS.grid
    env["quick"] = C.QUICK["on"]
    C.save_json(env, "environment.json")
    log(f"environment: {env}")


def stage_data_summary():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d, X, y, g = C.load()
    s = {}
    s["rows"] = len(d); s["sensors"] = int(d.SensorID.nunique())
    s["rows_per_sensor"] = d.groupby("SensorID").size().agg(["min", "max"]).tolist()
    s["normal"] = int((y == 0).sum()); s["faulty"] = int((y == 1).sum())
    s["missing_values"] = int(d.isna().sum().sum())
    s["timestamps_per_sensor_max"] = int(d.groupby("SensorID").Timestamp.nunique().max())
    s["sensor_type_counts"] = d.SensorType.value_counts().to_dict()
    s["fault_rate_by_type"] = d.groupby("SensorType").IsFaulty.mean().round(4).to_dict()
    s["fault_rate_per_sensor"] = d.groupby("SensorID").IsFaulty.mean().describe().round(4).to_dict()
    n = d[d.IsFaulty == 0]
    s["normal_ranges"] = {c: [float(n[c].min()), float(n[c].max())] for c in C.FEATURES}
    out = ((d.SensorData < 0) | (d.SensorData > 100) | (d.BatteryLife < 15) |
           (d.Temperature < 5.2) | (d.Temperature > 75))
    s["range_rule"] = dict(tp=int((out & (y == 1)).sum()), fp=int((out & (y == 0)).sum()),
                           accuracy=float(100 * (out.astype(int) == y).mean()),
                           median_temp_inrange_faulty=float(
                               d[(~out) & (y == 1)].Temperature.median()))
    s["correlation"] = d[C.FEATURES + ["IsFaulty"]].corr().round(4).to_dict()
    # Figures 9 and 10: fault share per bin
    for col, edges, labels, fname in [
        ("BatteryLife", [-np.inf, 20, 40, 60, 80, np.inf],
         ["<20", "20–40", "40–60", "60–80", "80–100"], "fig_battery_bins.png"),
        ("Temperature", [-np.inf, 10, 20, 30, 40, 50, 60, 70, np.inf],
         ["<10", "10–20", "20–30", "30–40", "40–50", "50–60", "60–70", "≥70"],
         "fig_temperature_bins.png")]:
        b = pd.cut(d[col], edges, labels=labels, right=False)
        tab = d.groupby(b, observed=False).IsFaulty.agg(["sum", "count"])
        s[f"bins_{col}"] = {str(k): [int(v["sum"]), int(v["count"])] for k, v in tab.iterrows()}
        fig, ax = plt.subplots(figsize=(6, 3.6))
        share = 100 * tab["sum"] / tab["count"]
        ax.bar(range(len(tab)), share, color="#c0504d", edgecolor="black", linewidth=0.5)
        for i, (k, v) in enumerate(tab.iterrows()):
            ax.text(i, share.iloc[i] + 1.5, f"{v['sum']}/{v['count']}", ha="center", fontsize=7)
        ax.set_xticks(range(len(tab))); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 100); ax.set_ylabel("Faulty records in bin (%)")
        ax.set_xlabel(f"{col} (dataset units)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(C.FIG, fname), dpi=200); plt.close(fig)
    C.save_json(s, "data_summary.json")
    log(f"data summary: {s['rows']} rows, {s['sensors']} sensors, "
        f"{s['normal']} normal / {s['faulty']} faulty")


def stage_record_cv():
    """Record-level stratified 5-fold CV of CNN/RNN/LSTM (Table 3 settings),
    scaler fitted per training fold, initialisation seed = fold index."""
    from sklearn.model_selection import StratifiedKFold
    from scipy.stats import friedmanchisquare
    C.tf_setup(ARGS.threads)
    d, X, y, g = C.load()
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rows, oof, preds, folds = [], {k: np.zeros(len(y)) for k in M.BASE_CFG}, [], []
    for k, (tr, te) in enumerate(skf.split(X, y)):
        folds.append(pd.DataFrame(dict(row=te, fold=k)))
        Xtr, Xte = minmax(X[tr], X[te])
        for name, cfg in M.BASE_CFG.items():
            m, _ = M.fit_nn(name, cfg, Xtr, y[tr], seed=k, epochs=q(10, 1), batch=16)
            p = M.nn_predict(m, Xte)
            oof[name][te] = p
            r = C.metrics(y[te], p); r.update(model=name, fold=k, seed=0)
            rows.append(r)
            preds.append(pred_frame("record_cv", name, 42, k, k, te, y[te], p, 0.5))
            log(f"record_cv fold {k} {name}: acc {r['accuracy']:.2f} recall {r['recall']:.2f}")
    pd.concat(folds).sort_values("row").to_csv(os.path.join(C.RES, "folds_record_cv.csv"), index=False)
    save_preds(preds, "record_cv")
    df = pd.DataFrame(rows)
    out = {"folds": rows, "summary": {}, "pooled": {}, "friedman": {}}
    for name in M.BASE_CFG:
        sub = df[df.model == name]
        out["summary"][name] = {mm: [float(sub[mm].mean()), float(sub[mm].std(ddof=1))]
                                for mm in C.METRICS}
        out["pooled"][name] = C.metrics(y, oof[name])
    for mm in C.METRICS:
        vals = [df[df.model == n].sort_values("fold")[mm].to_numpy() for n in M.BASE_CFG]
        st, p = friedmanchisquare(*vals)
        out["friedman"][mm] = dict(chi2=float(st), p=float(p))
    C.save_json(out, "record_cv.json")
    np.save(os.path.join(C.RES, "record_cv_oof.npy"), np.vstack([oof[n] for n in M.BASE_CFG]))
    _record_figures(y, oof)


def _record_figures(y, oof):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (name, p) in zip(axes, oof.items()):
        cm = confusion_matrix(y, (p >= 0.5).astype(int))
        ax.imshow(cm, cmap="Greys")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Faulty"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Faulty"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{name} pooled out-of-fold confusion matrix", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG, "fig_record_confusion.png"), dpi=200)
    plt.close(fig)
    for name, p in oof.items():
        fpr, tpr, _ = roc_curve(y, p); auc = roc_auc_score(y, p)
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.plot(fpr, tpr, color="black", label=f"Pooled OOF AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(f"{name} pooled out-of-fold ROC", fontsize=9); ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(C.FIG, f"fig_record_roc_{name}.png"), dpi=200)
        plt.close(fig)


def stage_fixed_benchmark():
    """Sensor-disjoint benchmark, prespecified settings (Table 5)."""
    C.tf_setup(ARGS.threads)
    d, X, y, g = C.load()
    folds = C.outer_folds(y, g)
    fa = []
    for f in folds:
        fa.append(pd.DataFrame(dict(outer_seed=f["seed"], fold=f["fold"], row=f["test"],
                                    sensor=g[f["test"]])))
    pd.concat(fa).to_csv(os.path.join(C.RES, "folds_sensor_disjoint.csv"), index=False)
    rows, preds = [], []
    for f in folds:
        tr, te, s, k = f["train"], f["test"], f["seed"], f["fold"]
        for name, est in M.fixed_conventional(seed=0).items():
            est.fit(X[tr], y[tr]); sc, thr = M.score_of(est, X[te])
            r = C.metrics(y[te], sc, thr); r.update(model=name, seed=s, fold=k); rows.append(r)
            preds.append(pred_frame("fixed", name, s, k, 0, te, y[te], sc, thr))
        Xtr, Xte = minmax(X[tr], X[te])
        for name, cfg in M.BASE_CFG.items():
            m, _ = M.fit_nn(name, cfg, Xtr, y[tr], seed=0, epochs=q(10, 1), batch=16)
            p = M.nn_predict(m, Xte)
            r = C.metrics(y[te], p); r.update(model=name, seed=s, fold=k); rows.append(r)
            preds.append(pred_frame("fixed", name, s, k, 0, te, y[te], p, 0.5))
        log(f"fixed benchmark seed {s} fold {k} done")
    save_preds(preds, "fixed_benchmark")
    df = pd.DataFrame(rows)
    summ = {n: C.summarize(df[df.model == n].to_dict("records")) for n in df.model.unique()}
    C.save_json({"folds": rows, "summary": summ}, "fixed_benchmark.json")


def stage_tuned_conventional():
    """Threshold rules + conventional models with nested, sensor-grouped,
    training-only grid search (F1 of the faulty class)."""
    from sklearn.model_selection import GridSearchCV
    d, X, y, g = C.load()
    rows, preds, chosen = [], [], []
    for f in C.outer_folds(y, g):
        tr, te, s, k = f["train"], f["test"], f["seed"], f["fold"]
        r1 = M.fit_threshold1(X[tr], y[tr], C.FEATURES)
        sc = M.apply_threshold1(r1, X[te]).astype(float)
        r = C.metrics(y[te], sc, 0.5, has_score=False); r.update(model="Threshold-1", seed=s, fold=k)
        rows.append(r); chosen.append(dict(model="Threshold-1", seed=s, fold=k, params=r1))
        preds.append(pred_frame("tuned", "Threshold-1", s, k, 0, te, y[te], sc, 0.5))
        r2 = M.fit_threshold2(X[tr], y[tr])
        sc = M.apply_threshold2(r2, X[te]).astype(float)
        r = C.metrics(y[te], sc, 0.5, has_score=False); r.update(model="Threshold-2", seed=s, fold=k)
        rows.append(r); chosen.append(dict(model="Threshold-2", seed=s, fold=k, params=r2))
        preds.append(pred_frame("tuned", "Threshold-2", s, k, 0, te, y[te], sc, 0.5))
        for name, (est, grid) in M.tuned_conventional(seed=0).items():
            gs = GridSearchCV(est, grid, scoring="f1", cv=C.inner_cv(s),
                              n_jobs=1 if name in ("RF", "XGBoost") else -1, refit=True)
            gs.fit(X[tr], y[tr], groups=g[tr])
            sc, thr = M.score_of(gs.best_estimator_, X[te])
            r = C.metrics(y[te], sc, thr); r.update(model=name, seed=s, fold=k); rows.append(r)
            chosen.append(dict(model=name, seed=s, fold=k, params=gs.best_params_,
                               inner_f1=float(gs.best_score_)))
            preds.append(pred_frame("tuned", name, s, k, 0, te, y[te], sc, thr))
        log(f"tuned conventional seed {s} fold {k}: Thr2 {r2}, "
            f"HGB acc {[x for x in rows if x['model']=='HGB'][-1]['accuracy']:.2f}")
    save_preds(preds, "tuned_conventional")
    df = pd.DataFrame(rows)
    summ = {n: C.summarize(df[df.model == n].to_dict("records")) for n in df.model.unique()}
    C.save_json({"folds": rows, "summary": summ, "selected": chosen}, "tuned_conventional.json")


def stage_neural_fixed():
    """CNN/RNN/LSTM with the fixed 10-epoch setting, three init seeds per fold."""
    C.tf_setup(ARGS.threads)
    d, X, y, g = C.load()
    rows, preds = [], []
    for f in C.outer_folds(y, g):
        tr, te, s, k = f["train"], f["test"], f["seed"], f["fold"]
        Xtr, Xte = minmax(X[tr], X[te])
        for name, cfg in M.BASE_CFG.items():
            for init in q(C.INIT_SEEDS, [0]):
                m, _ = M.fit_nn(name, cfg, Xtr, y[tr], seed=init, epochs=q(10, 1), batch=16)
                p = M.nn_predict(m, Xte)
                r = C.metrics(y[te], p); r.update(model=f"{name}, 10 epochs", seed=s, fold=k, init=init)
                rows.append(r)
                preds.append(pred_frame("neural_fixed", name, s, k, init, te, y[te], p, 0.5))
        log(f"neural fixed seed {s} fold {k} done")
    save_preds(preds, "neural_fixed")
    df = pd.DataFrame(rows)
    summ = {n: C.summarize(df[df.model == n].to_dict("records")) for n in df.model.unique()}
    C.save_json({"folds": rows, "summary": summ}, "neural_fixed.json")


def stage_neural_tuned():
    """Training-only tuning of CNN/RNN/LSTM on each outer fold.
    Selection: each grid setting is trained (init seed 0) on 80% of the training
    sensors with early stopping on the remaining 20% (sensor-disjoint); the
    setting with the lowest validation loss is chosen. The chosen setting is then
    trained with init seeds 0, 1, 2 (seed 0 reuses the selection run) and
    evaluated once on the outer test fold."""
    C.tf_setup(ARGS.threads)
    d, X, y, g = C.load()
    grid = M.NN_GRID[ARGS.grid]
    seeds = q(C.INIT_SEEDS if ARGS.grid == "full" else C.INIT_SEEDS[:2], [0])
    max_ep, pat = q(M.TUNED_MAX_EPOCHS, 5), q(M.TUNED_PATIENCE, 2)
    folds = C.outer_folds(y, g)
    n_total = len(folds) * len(M.BASE_CFG)
    done = 0
    t_start = time.time()
    for name in M.BASE_CFG:
        for f in folds:
            tr, te, s, k = f["train"], f["test"], f["seed"], f["fold"]
            part = os.path.join(C.PART, f"neural_tuned_{ARGS.grid}_{name}_{s}_{k}.json")
            done += 1
            if os.path.exists(part) and not ARGS.force:
                continue
            itr, iva = C.val_split(g[tr], seed=s * 10 + k)
            A, V, T = minmax(X[tr][itr], X[tr][iva], X[te])
            ya, yv = y[tr][itr], y[tr][iva]
            sel = []
            models0 = {}
            for ci, cfg in enumerate(grid[name]):
                cfg = dict(cfg, spe=8)
                m, h = M.fit_nn(name, cfg, A, ya, seed=0, epochs=max_ep, batch=M.TUNED_BATCH,
                                Xva=V, yva=yv, patience=pat)
                best = int(np.argmin(h["val_loss"]))
                sel.append(dict(cfg=grid[name][ci], val_loss=float(h["val_loss"][best]),
                                best_epoch=best + 1, epochs_run=len(h["loss"])))
                models0[ci] = (m, h)
            ci = int(np.argmin([x["val_loss"] for x in sel]))
            cfg = dict(grid[name][ci], spe=8)
            res = dict(model=name, seed=s, fold=k, grid=sel, chosen=grid[name][ci],
                       params=M.n_params(name, grid[name][ci]), runs=[])
            for init in seeds:
                if init == 0:
                    m, h = models0[ci]
                else:
                    m, h = M.fit_nn(name, cfg, A, ya, seed=init, epochs=max_ep,
                                    batch=M.TUNED_BATCH, Xva=V, yva=yv, patience=pat)
                p = M.nn_predict(m, T)
                r = C.metrics(y[te], p)
                res["runs"].append(dict(init=init, metrics=r, history=h,
                                        best_epoch=int(np.argmin(h["val_loss"])) + 1,
                                        epochs_run=len(h["loss"]), score=p.tolist()))
            with open(part, "w") as fh:
                json.dump(res, fh, default=C._conv)
            el = (time.time() - t_start) / 60
            log(f"neural tuned {name} seed {s} fold {k}: chose {grid[name][ci]}, "
                f"acc {np.mean([x['metrics']['accuracy'] for x in res['runs']]):.2f} "
                f"({done}/{n_total} done, {el:.0f} min elapsed)")
    # collect
    rows, preds, selected = [], [], []
    for name in M.BASE_CFG:
        for f in folds:
            s, k, te = f["seed"], f["fold"], f["test"]
            with open(os.path.join(C.PART, f"neural_tuned_{ARGS.grid}_{name}_{s}_{k}.json")) as fh:
                res = json.load(fh)
            selected.append(dict(model=name, seed=s, fold=k, chosen=res["chosen"],
                                 params=res["params"], grid=res["grid"],
                                 best_epochs=[r["best_epoch"] for r in res["runs"]],
                                 epochs_run=[r["epochs_run"] for r in res["runs"]]))
            for run in res["runs"]:
                r = dict(run["metrics"]); r.update(model=f"{name}, tuned", seed=s, fold=k, init=run["init"])
                rows.append(r)
                preds.append(pred_frame("neural_tuned", name, s, k, run["init"], te, y[te],
                                        np.array(run["score"]), 0.5))
    save_preds(preds, "neural_tuned")
    df = pd.DataFrame(rows)
    summ = {n: C.summarize(df[df.model == n].to_dict("records")) for n in df.model.unique()}
    C.save_json({"folds": rows, "summary": summ, "selected": selected,
                 "budget": dict(grid=ARGS.grid, max_epochs=max_ep, patience=pat,
                                batch=M.TUNED_BATCH, init_seeds=seeds)}, "neural_tuned.json")
    _learning_curves(folds)


def _learning_curves(folds):
    """Training and validation loss per epoch of the tuned models (init seed 0,
    every outer fold)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, name in zip(axes, M.BASE_CFG):
        for f in folds:
            with open(os.path.join(C.PART, f"neural_tuned_{ARGS.grid}_{name}_{f['seed']}_{f['fold']}.json")) as fh:
                res = json.load(fh)
            h = res["runs"][0]["history"]
            ep = np.arange(1, len(h["loss"]) + 1)
            ax.plot(ep, h["loss"], color="#1f77b4", lw=0.8, alpha=0.7)
            ax.plot(ep, h["val_loss"], color="#d62728", lw=0.8, alpha=0.7)
        ax.set_title(f"{name} (tuned)", fontsize=10); ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Binary cross-entropy (log scale)")
    axes[0].plot([], [], color="#1f77b4", label="Training loss")
    axes[0].plot([], [], color="#d62728", label="Validation loss (held-out sensors)")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG, "fig_learning_curves.png"), dpi=200)
    plt.close(fig)


def _fold_table():
    """Per-fold accuracy/recall (averaged over init seeds) for every model of the
    tuned benchmark."""
    rows = []
    for fn in ("tuned_conventional.json", "neural_fixed.json", "neural_tuned.json"):
        if C.exists(fn):
            rows += C.load_json(fn)["folds"]
    df = pd.DataFrame(rows)
    return df.groupby(["model", "seed", "fold"])[C.METRICS].mean().reset_index()


def stage_stats():
    """Paired differences against HGB (corrected resampled t-test, Holm, Wilcoxon)
    and sensor-level cluster-bootstrap intervals."""
    from scipy import stats
    d, X, y, g = C.load()
    folds = C.outer_folds(y, g)
    n_te = np.mean([len(f["test"]) for f in folds]); n_tr = np.mean([len(f["train"]) for f in folds])
    ft = _fold_table()
    ref = ft[ft.model == "HGB"].set_index(["seed", "fold"])
    comps = []
    for name in ft.model.unique():
        if name == "HGB":
            continue
        other = ft[ft.model == name].set_index(["seed", "fold"]).loc[ref.index]
        for metric in ("accuracy", "recall"):
            dlt = ref[metric].to_numpy() - other[metric].to_numpy()
            J = len(dlt); mean = dlt.mean(); var = dlt.var(ddof=1) if J > 1 else 0.0
            se = np.sqrt((1 / J + n_te / n_tr) * var) if var > 0 else 0.0
            tcrit = stats.t.ppf(0.975, J - 1) if J > 1 else np.nan
            tval = mean / se if se > 0 else np.inf * np.sign(mean)
            p = 2 * stats.t.sf(abs(tval), J - 1) if (se > 0 and J > 1) else (0.0 if mean != 0 else 1.0)
            try:
                pw = float(stats.wilcoxon(dlt).pvalue) if J > 1 and np.any(dlt != 0) else 1.0
            except ValueError:
                pw = 1.0
            comps.append(dict(model=name, metric=metric, mean_diff=float(mean),
                              ci_low=float(mean - tcrit * se), ci_high=float(mean + tcrit * se),
                              p=float(p), p_wilcoxon=pw, hgb_wins=int((dlt > 0).sum()),
                              ties=int((dlt == 0).sum()), folds=J))
    cd = pd.DataFrame(comps)
    for metric in ("accuracy", "recall"):  # Holm over all comparisons of one metric
        idx = cd.index[cd.metric == metric]
        p = cd.loc[idx, "p"].to_numpy(); order = np.argsort(p); m = len(p)
        adj = np.empty(m); run = 0
        for r, i in enumerate(order):
            run = max(run, min(1, (m - r) * p[i])); adj[i] = run
        cd.loc[idx, "p_holm"] = adj
    # cluster bootstrap over sensors
    rng = np.random.default_rng(2024)
    B = q(1000, 50)
    sensors = np.unique(g); sid = np.searchsorted(sensors, g)
    W = rng.multinomial(len(sensors), np.full(len(sensors), 1 / len(sensors)), size=B)
    boots = []
    allp = []
    for fn in ("tuned_conventional", "neural_fixed", "neural_tuned"):
        pth = os.path.join(C.PRED, fn + ".csv.gz")
        if os.path.exists(pth):
            allp.append(pd.read_csv(pth))
    P = pd.concat(allp)
    suffix = np.where(P["stage"] == "neural_fixed", ", 10 epochs",
                      np.where(P["stage"] == "neural_tuned", ", tuned", ""))
    P["model"] = P["model"].astype(str) + suffix
    for name, sub in P.groupby("model"):
        acc_lo, acc_hi, rec_lo, rec_hi = [], [], [], []
        for (s, init), ss in sub.groupby(["outer_seed", "init_seed"]):
            yy, pp, ii = ss.y.to_numpy(), ss.pred.to_numpy(), sid[ss.row.to_numpy()]
            cor = np.bincount(ii, (yy == pp).astype(float), len(sensors))
            n = np.bincount(ii, None, len(sensors)).astype(float)
            tp = np.bincount(ii, ((yy == 1) & (pp == 1)).astype(float), len(sensors))
            pos = np.bincount(ii, (yy == 1).astype(float), len(sensors))
            acc = 100 * (W @ cor) / (W @ n); rec = 100 * (W @ tp) / np.maximum(W @ pos, 1)
            acc_lo.append(np.percentile(acc, 2.5)); acc_hi.append(np.percentile(acc, 97.5))
            rec_lo.append(np.percentile(rec, 2.5)); rec_hi.append(np.percentile(rec, 97.5))
        boots.append(dict(model=name, acc_low=float(np.mean(acc_lo)), acc_high=float(np.mean(acc_hi)),
                          recall_low=float(np.mean(rec_lo)), recall_high=float(np.mean(rec_hi))))
    C.save_json({"comparisons_vs_HGB": cd.to_dict("records"), "bootstrap": boots,
                 "n_test_mean": n_te, "n_train_mean": n_tr, "bootstrap_resamples": B,
                 "per_fold": ft.to_dict("records")}, "stats.json")
    log("stats done")


def _chosen_cfg(name, s, k):
    pth = os.path.join(C.PART, f"neural_tuned_{ARGS.grid}_{name}_{s}_{k}.json")
    if os.path.exists(pth):
        with open(pth) as fh:
            return json.load(fh)["chosen"]
    return None


def stage_order_test():
    """Retrain the neural models with all six input orders on the seed-42 folds,
    (a) with the fixed 10-epoch setting and (b) with the tuned setting chosen for
    that fold (early stopping)."""
    C.tf_setup(ARGS.threads)
    d, X, y, g = C.load()
    folds = [f for f in C.outer_folds(y, g) if f["seed"] == 42]
    rows = []
    for perm in itertools.permutations(range(3)):
        order = " > ".join(C.FEATURES[i] for i in perm)
        for f in folds:
            tr, te, k = f["train"], f["test"], f["fold"]
            Xp = X[:, list(perm)]
            Xtr, Xte = minmax(Xp[tr], Xp[te])
            for name, cfg in M.BASE_CFG.items():
                m, _ = M.fit_nn(name, cfg, Xtr, y[tr], seed=0, epochs=q(10, 1), batch=16)
                r = C.metrics(y[te], M.nn_predict(m, Xte))
                r.update(model=name, setting="10 epochs", order=order, fold=k); rows.append(r)
                cfg_t = _chosen_cfg(name, 42, k)
                if cfg_t is None:
                    continue
                itr, iva = C.val_split(g[tr], seed=42 * 10 + k)
                A, V, T = minmax(Xp[tr][itr], Xp[tr][iva], Xp[te])
                m, _ = M.fit_nn(name, dict(cfg_t, spe=8), A, y[tr][itr], seed=0,
                                epochs=q(M.TUNED_MAX_EPOCHS, 5), batch=M.TUNED_BATCH,
                                Xva=V, yva=y[tr][iva], patience=q(M.TUNED_PATIENCE, 2))
                r = C.metrics(y[te], M.nn_predict(m, T))
                r.update(model=name, setting="tuned", order=order, fold=k); rows.append(r)
        log(f"order test {order} done")
    df = pd.DataFrame(rows)
    per_order = df.groupby(["model", "setting", "order"])[C.METRICS].mean().reset_index()
    rng = per_order.groupby(["model", "setting"]).agg(
        **{f"{mm}_{fn}": (mm, fn) for mm in ("accuracy", "recall", "f1", "mcc") for fn in ("min", "max")}
    ).reset_index()
    C.save_json({"rows": rows, "per_order": per_order.to_dict("records"),
                 "range": rng.to_dict("records")}, "order_test.json")


def stage_ablation():
    """HGB and RF (fixed settings) on every subset of the inputs, plus SensorType."""
    d, X, y, g = C.load()
    st = pd.get_dummies(d.SensorType).to_numpy(dtype=float)
    subsets = []
    for r in (1, 2, 3):
        subsets += list(itertools.combinations(range(3), r))
    rows = []
    for f in C.outer_folds(y, g):
        tr, te = f["train"], f["test"]
        for sub in subsets + ["all+type"]:
            if sub == "all+type":
                Xs, label = np.hstack([X, st]), "All three + SensorType"
            else:
                Xs = X[:, list(sub)]
                label = "All three" if len(sub) == 3 else " + ".join(C.FEATURES[i] for i in sub) + \
                    (" only" if len(sub) == 1 else "")
            for name in ("HGB", "RF"):
                est = M.fixed_conventional(seed=0)[name]
                est.fit(Xs[tr], y[tr]); sc, thr = M.score_of(est, Xs[te])
                r = C.metrics(y[te], sc, thr); r.update(model=name, inputs=label, seed=f["seed"], fold=f["fold"])
                rows.append(r)
    df = pd.DataFrame(rows)
    summ = df.groupby(["model", "inputs"])[C.METRICS].mean().reset_index()
    C.save_json({"rows": rows, "summary": summ.to_dict("records")}, "ablation.json")
    log("ablation done")


def stage_autoencoder():
    """Protocol-adapted LSTM autoencoder (reconstruction-based detector after
    Lakshmi et al.), on the same sensor-disjoint folds."""
    tf = C.tf_setup(ARGS.threads)
    from sklearn.preprocessing import MinMaxScaler
    d, X, y, g = C.load()
    Wn = 20
    # windows in the original row order within each sensor
    pos = {s: np.where(g == s)[0] for s in np.unique(g)}
    def windows(sensors):
        idx = []
        for s in sensors:
            r = pos[s]
            for i in range(len(r) - Wn + 1):
                idx.append(r[i:i + Wn])
        return np.array(idx)
    L = tf.keras.layers
    def build():
        inp = L.Input(shape=(Wn, 3))
        h = L.LSTM(64)(inp)
        h = L.RepeatVector(Wn)(h)
        h = L.LSTM(64, return_sequences=True)(h)
        out = L.TimeDistributed(L.Dense(3))(h)
        m = tf.keras.Model(inp, out); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        return m
    def obs_error(m, Xs, widx):
        rec = m.predict(Xs[widx], verbose=0, batch_size=512)
        e = ((rec - Xs[widx]) ** 2).mean(-1)            # (n_win, Wn)
        tot = np.bincount(widx.ravel(), e.ravel(), len(Xs))
        cnt = np.bincount(widx.ravel(), None, len(Xs))
        return tot / np.maximum(cnt, 1), cnt > 0
    rows, preds, n_par = [], [], None
    for f in C.outer_folds(y, g):
        tr, te, s, k = f["train"], f["test"], f["seed"], f["fold"]
        trs, tes = np.unique(g[tr]), np.unique(g[te])
        sc = MinMaxScaler().fit(X[tr]); Xs = sc.transform(X)
        rs = np.random.default_rng(s * 10 + k); va_s = rs.choice(trs, max(1, len(trs) // 5), replace=False)
        fit_s = np.setdiff1d(trs, va_s)
        wf, wv = windows(fit_s), windows(va_s)
        wf = wf[(y[wf] == 0).all(1)]; wv = wv[(y[wv] == 0).all(1)]
        C.seed_all(0); m = build(); n_par = int(sum(np.prod(w.shape) for w in m.trainable_weights))
        m.fit(Xs[wf], Xs[wf], validation_data=(Xs[wv], Xs[wv]), epochs=q(100, 2), batch_size=64,
              verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
        err_tr, _ = obs_error(m, Xs, windows(trs))
        thr = np.percentile(err_tr[tr][y[tr] == 0], 95)
        err_te, _ = obs_error(m, Xs, windows(tes))
        r = C.metrics(y[te], err_te[te], thr); r.update(model="LSTM-AE", seed=s, fold=k); rows.append(r)
        preds.append(pred_frame("autoencoder", "LSTM-AE", s, k, 0, te, y[te], err_te[te], thr))
        log(f"autoencoder seed {s} fold {k}: acc {r['accuracy']:.2f}")
    save_preds(preds, "autoencoder")
    C.save_json({"folds": rows, "summary": C.summarize(rows), "params": n_par, "window": Wn},
                "autoencoder.json")


def stage_training_size():
    """Learning curves over training-set size: models trained on 25-100% of the
    training sensors of the seed-42 folds, always tested on the full test fold."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.tf_setup(ARGS.threads)
    d, X, y, g = C.load()
    folds = [f for f in C.outer_folds(y, g) if f["seed"] == 42]
    fracs = q([0.25, 0.5, 0.75, 1.0], [0.5, 1.0])
    rows = []
    for f in folds:
        tr, te, k = f["train"], f["test"], f["fold"]
        trs = np.unique(g[tr]); rng = np.random.default_rng(100 + k); perm = rng.permutation(trs)
        for fr in fracs:
            keep = np.isin(g[tr], perm[:max(2, int(round(fr * len(trs))))]); sub = tr[keep]
            ns = len(np.unique(g[sub]))
            r2 = M.fit_threshold2(X[sub], y[sub])
            r = C.metrics(y[te], M.apply_threshold2(r2, X[te]).astype(float), has_score=False)
            r.update(model="Threshold-2", sensors=ns, fold=k); rows.append(r)
            for name in ("LR", "MLP", "RF", "HGB"):
                est = M.fixed_conventional(seed=0)[name]; est.fit(X[sub], y[sub])
                sc, thr = M.score_of(est, X[te])
                r = C.metrics(y[te], sc, thr); r.update(model=name, sensors=ns, fold=k); rows.append(r)
            for name in ("RNN", "LSTM"):
                cfg = _chosen_cfg(name, 42, k) or dict(units=64, lr=3e-3)
                itr, iva = C.val_split(g[sub], seed=42 * 10 + k)
                A, V, T = minmax(X[sub][itr], X[sub][iva], X[te])
                m, _ = M.fit_nn(name, dict(cfg, spe=8), A, y[sub][itr], seed=0,
                                epochs=q(M.TUNED_MAX_EPOCHS, 5), batch=M.TUNED_BATCH,
                                Xva=V, yva=y[sub][iva], patience=q(M.TUNED_PATIENCE, 2))
                r = C.metrics(y[te], M.nn_predict(m, T)); r.update(model=f"{name} (tuned)", sensors=ns, fold=k)
                rows.append(r)
        log(f"training-size curve fold {k} done")
    df = pd.DataFrame(rows)
    df["frac"] = df.groupby("fold")["sensors"].rank(method="dense")
    summ = df.groupby(["model", "frac"]).agg(sensors=("sensors", "mean"),
                                            accuracy=("accuracy", "mean"),
                                            recall=("recall", "mean")).reset_index()
    C.save_json({"rows": rows, "summary": summ.to_dict("records")}, "training_size.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, sub in summ.groupby("model"):
        axes[0].plot(sub.sensors, sub.accuracy, marker="o", label=name)
        axes[1].plot(sub.sensors, sub.recall, marker="o", label=name)
    for ax, lab in zip(axes, ("Accuracy (%)", "Recall (%)")):
        ax.set_xlabel("Number of training sensors"); ax.set_ylabel(lab)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG, "fig_training_size.png"), dpi=200); plt.close(fig)


def stage_runtime():
    """Training time, single-record latency, throughput and model size on the
    training part of the first outer fold."""
    tf = C.tf_setup(ARGS.threads)
    from xgboost import XGBClassifier
    d, X, y, g = C.load()
    f = C.outer_folds(y, g)[0]; tr, te = f["train"], f["test"]
    rows = []
    n_lat, n_thr = q(200, 20), q(30, 3)
    def bench(name, predict_one, predict_batch, train_s, size_kb):
        x1 = X[te][:1]; xb = X[te][:1000]
        for _ in range(10):
            predict_one(x1)
        lat = []
        for i in range(n_lat):
            xi = X[te][i % len(te):i % len(te) + 1]
            t = time.perf_counter(); predict_one(xi); lat.append(1000 * (time.perf_counter() - t))
        thr = []
        for _ in range(n_thr):
            t = time.perf_counter(); predict_batch(xb); thr.append(len(xb) / (time.perf_counter() - t))
        rows.append(dict(model=name, train_s=train_s, lat_median_ms=float(np.median(lat)),
                         lat_p95_ms=float(np.percentile(lat, 95)), throughput=float(np.median(thr)),
                         size_kb=size_kb))
        log(f"runtime {name}: {rows[-1]}")
    convs = M.fixed_conventional(seed=0)
    convs["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, n_jobs=-1,
                                     random_state=0, eval_metric="logloss", tree_method="hist")
    for name, est in convs.items():
        if hasattr(est, "n_jobs"):
            est.set_params(n_jobs=1)
        t = time.perf_counter(); est.fit(X[tr], y[tr]); ts = time.perf_counter() - t
        size = len(pickle.dumps(est)) / 1024
        fn = (lambda e: (lambda x: M.score_of(e, x)))(est)
        bench(name, fn, fn, ts, size)
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler().fit(X[tr]); Xtr = sc.transform(X[tr])
    nt = C.load_json("neural_tuned.json") if C.exists("neural_tuned.json") else None
    for name, cfg in M.BASE_CFG.items():
        variants = [("10 epochs", cfg, False)]
        if nt:
            ch = [json.dumps(s["chosen"], sort_keys=True) for s in nt["selected"] if s["model"] == name]
            top = json.loads(max(set(ch), key=ch.count))
            variants.append(("tuned", top, True))
        for lab, c, es in variants:
            t = time.perf_counter()
            if es:
                itr, iva = C.val_split(g[tr], seed=420)
                m, _ = M.fit_nn(name, dict(c, spe=8), Xtr[itr], y[tr][itr], seed=0,
                                epochs=q(M.TUNED_MAX_EPOCHS, 5), batch=M.TUNED_BATCH,
                                Xva=Xtr[iva], yva=y[tr][iva], patience=q(M.TUNED_PATIENCE, 2))
            else:
                m, _ = M.fit_nn(name, c, Xtr, y[tr], seed=0, epochs=q(10, 1), batch=16)
            ts = time.perf_counter() - t
            size = sum(np.prod(w.shape) for w in m.weights) * 4 / 1024
            fwd = tf.function(lambda t, m=m: m(t, training=False), reduce_retracing=True,
                              input_signature=[tf.TensorSpec([None, X.shape[1], 1], tf.float32)])
            one = lambda x, f=fwd: f(tf.constant(sc.transform(x)[..., None], tf.float32)).numpy()
            bench(f"{name}, {lab}", one, one, ts, float(size))
    C.save_json({"rows": rows, "train_rows": int(len(tr)),
                 "note": "latency: one call on one record (input scaling included, Keras models run "
                         "through a compiled tf.function), 10 warm-up calls; throughput: batches of 1,000"},
                "runtime.json")


# ------------------------------------------------------------------ report
def fmt(v, sd=None, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{d}f}" if sd is None else f"{v:.{d}f} ± {sd:.{d}f}"


def stage_report():
    lines = ["# Results summary", ""]
    env = C.load_json("environment.json") if C.exists("environment.json") else {}
    lines += ["## Environment", "", "```", json.dumps(env, indent=1), "```", ""]
    def table(title, df):
        nonlocal lines
        try:
            md = df.to_markdown(index=False)
        except ImportError:
            md = df.to_string(index=False)
        lines += [f"## {title}", "", md, ""]
        df.to_csv(os.path.join(C.TAB, title.split(":")[0].replace(" ", "_") + ".csv"), index=False)
    if C.exists("record_cv.json"):
        rc = C.load_json("record_cv.json")
        df = pd.DataFrame([{"Model": n, **{mm: fmt(*v, d=4 if mm in ("mcc", "roc_auc") else 2)
                                            for mm, v in s.items()}} for n, s in rc["summary"].items()])
        table("Table4: record-level 5-fold CV (mean ± sample SD)", df)
        pooled = pd.DataFrame([{"Model": n, **{k: p[k] for k in ("tn", "fp", "fn", "tp", "recall", "roc_auc")}}
                               for n, p in rc["pooled"].items()])
        table("Table4b: pooled out-of-fold confusion counts", pooled)
        table("Friedman: record-level", pd.DataFrame([{"metric": k, **v} for k, v in rc["friedman"].items()]))
    if C.exists("fixed_benchmark.json"):
        fb = C.load_json("fixed_benchmark.json")["summary"]
        df = pd.DataFrame([{"Model": n, **{mm: fmt(s[mm], d=4 if mm in ("mcc", "roc_auc") else 2)
                                            for mm in C.METRICS}} for n, s in fb.items()])
        table("Table5: sensor-disjoint fixed configurations (mean over seeds)", df)
    summ = {}
    for fn in ("tuned_conventional.json", "neural_fixed.json", "neural_tuned.json"):
        if C.exists(fn):
            summ.update(C.load_json(fn)["summary"])
    if summ:
        order = ["Threshold-1", "Threshold-2", "LR", "DT", "SVM", "MLP", "RF", "HGB", "XGBoost",
                 "CNN, 10 epochs", "RNN, 10 epochs", "LSTM, 10 epochs",
                 "CNN, tuned", "RNN, tuned", "LSTM, tuned"]
        df = pd.DataFrame([{"Model": n, **{mm: fmt(summ[n][mm], summ[n][mm + "_sd"],
                                                    d=3 if mm in ("mcc", "roc_auc") else 2)
                                            for mm in C.METRICS}} for n in order if n in summ])
        table("Table6: tuned sensor-disjoint benchmark (mean ± SD of seed means)", df)
    if C.exists("tuned_conventional.json"):
        sel = C.load_json("tuned_conventional.json")["selected"]
        table("Selected: conventional hyper-parameters", pd.DataFrame(
            [{"model": s["model"], "seed": s["seed"], "fold": s["fold"], "params": json.dumps(s["params"])} for s in sel]))
    if C.exists("neural_tuned.json"):
        nt = C.load_json("neural_tuned.json")
        table("Selected: neural settings", pd.DataFrame(
            [{"model": s["model"], "seed": s["seed"], "fold": s["fold"], "chosen": json.dumps(s["chosen"]),
              "params": s["params"], "best_epochs": s["best_epochs"], "epochs_run": s["epochs_run"]}
             for s in nt["selected"]]))
        lines += ["Budget: " + json.dumps(nt["budget"]), ""]
    if C.exists("stats.json"):
        st = C.load_json("stats.json")
        table("Stats: differences vs HGB", pd.DataFrame(st["comparisons_vs_HGB"]).round(4))
        table("Stats: cluster bootstrap 95% intervals", pd.DataFrame(st["bootstrap"]).round(2))
    if C.exists("order_test.json"):
        table("Table7: feature-order range (min-max over 6 orders)",
              pd.DataFrame(C.load_json("order_test.json")["range"]).round(2))
    if C.exists("ablation.json"):
        table("Table8: feature ablation", pd.DataFrame(C.load_json("ablation.json")["summary"]).round(3))
    if C.exists("autoencoder.json"):
        ae = C.load_json("autoencoder.json")
        table("Table9: LSTM autoencoder", pd.DataFrame([{k: round(v, 4) for k, v in ae["summary"].items()}]))
        lines += [f"Autoencoder trainable parameters: {ae['params']}", ""]
    if C.exists("training_size.json"):
        table("Training-size curve", pd.DataFrame(C.load_json("training_size.json")["summary"]).round(2))
    if C.exists("runtime.json"):
        table("Table10: runtime", pd.DataFrame(C.load_json("runtime.json")["rows"]).round(3))
    with open(os.path.join(C.RES, "SUMMARY.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"report written to {os.path.join(C.RES, 'SUMMARY.md')}")


# ------------------------------------------------------------------ main
def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="all", help="comma-separated list or 'all': " + ",".join(STAGES))
    ap.add_argument("--grid", choices=["full", "reduced"], default="full",
                    help="neural tuning grid: full (6 settings, 3 seeds) or reduced (4 settings, 2 seeds)")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 fold, very short training")
    ap.add_argument("--threads", type=int, default=0, help="TensorFlow threads (0 = all cores)")
    ap.add_argument("--force", action="store_true", help="rerun stages that already have results")
    ARGS = ap.parse_args()
    if ARGS.quick:
        C.QUICK["on"] = True
        C.RES = os.path.join(C.ROOT, "results_quick")
        for name in ("FIG", "TAB", "PRED", "PART"):
            setattr(C, name, os.path.join(C.RES, {"FIG": "figures", "TAB": "tables",
                                                  "PRED": "predictions", "PART": "partial"}[name]))
        for p in (C.RES, C.FIG, C.TAB, C.PRED, C.PART):
            os.makedirs(p, exist_ok=True)
    todo = STAGES if ARGS.stages == "all" else [s.strip() for s in ARGS.stages.split(",")]
    for s in todo:
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s}")
    t0 = time.time()
    for s in todo:
        marker = s + ".json"
        if s not in ("report", "environment", "neural_tuned") and C.exists(marker) and not ARGS.force:
            log(f"skip {s} (results/{marker} exists; use --force to rerun)")
            continue
        log(f"=== stage {s} ===")
        t = time.time()
        globals()["stage_" + s]()
        log(f"=== stage {s} finished in {(time.time() - t) / 60:.1f} min ===")
    log(f"all done in {(time.time() - t0) / 60:.1f} min; see {C.RES}")


if __name__ == "__main__":
    main()
