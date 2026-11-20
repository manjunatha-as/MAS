"""Additional analyses requested in the third review round.

Stages (run after run_experiments.py has produced tuned_conventional.json):
  trees       Tree-model diagnostics on the tuned sensor-disjoint folds:
              selected hyper-parameters, pooled out-of-fold ROC curves and
              confusion matrices, permutation feature importance (reviewer 4).
  record      Record-level five-fold CV re-run with tuned networks: the same
              training-only search, batch size and early stopping as the tuned
              sensor-disjoint benchmark, plus tuned tree models and the
              two-attribute rule as references (reviewer 6).
  record_figs Figures and tables of the record-level re-run.
  eda         Exploratory figures (correlation, fault share per bin,
              class-wise distributions) in the common figure style (reviewer 3).

Usage:  python src/extra_analyses.py --stages trees,record,record_figs,eda
The 'record' stage is resumable; each finished (model, fold) is stored in
results/partial/.
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import common as C
import models as M
from common import log
import plot_style as PS
import matplotlib.pyplot as plt

TREES = ["DT", "RF", "HGB", "XGBoost"]
TREE_COLORS = {"DT": "#7f7f7f", "RF": "#2ca02c", "HGB": "#1f4e9c", "XGBoost": "#d1495b"}
NN = ["CNN", "RNN", "LSTM"]
ARGS = None


def minmax(Xtr, *others):
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler().fit(Xtr)
    return [sc.transform(Xtr)] + [sc.transform(o) for o in others]


def _estimator(name, params, seed=0):
    est, _ = M.tuned_conventional(seed=seed)[name]
    return est.set_params(**params)


# ============================================================ tree diagnostics
def stage_trees():
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
    from sklearn.inspection import permutation_importance
    d, X, y, g = C.load()
    tc = C.load_json("tuned_conventional.json")
    pr = pd.read_csv(os.path.join(C.PRED, "tuned_conventional.csv.gz"))
    out = {"selected": {}, "pooled": {}, "importance": {}, "intrinsic": {}}

    # 1. selected hyper-parameters (how often each setting won on the 9 folds)
    for name in TREES:
        sel = [json.dumps(s["params"], sort_keys=True) for s in tc["selected"] if s["model"] == name]
        out["selected"][name] = pd.Series(sel).value_counts().to_dict()

    # 2. pooled out-of-fold predictions per outer seed (each seed covers all 10,000 records once)
    for name in TREES + ["Threshold-2"]:
        sub = pr[pr.model == name]
        per_seed = []
        for s, ss in sub.groupby("outer_seed"):
            ss = ss.sort_values("row")
            m = C.metrics(ss.y.to_numpy(), ss.score.to_numpy(), 0.5,
                          has_score=name != "Threshold-2")
            m["seed"] = int(s); per_seed.append(m)
        df = pd.DataFrame(per_seed)
        out["pooled"][name] = dict(
            per_seed=per_seed,
            auc_mean=float(df.roc_auc.mean()), auc_sd=float(df.roc_auc.std(ddof=1)),
            cm_mean={k: float(df[k].mean()) for k in ("tn", "fp", "fn", "tp")})

    # 3. permutation importance on the outer test folds (refit with the selected setting)
    rows, intr = [], []
    for f in C.outer_folds(y, g):
        tr, te, s, k = f["train"], f["test"], f["seed"], f["fold"]
        for name in TREES:
            p = [x for x in tc["selected"] if x["model"] == name and x["seed"] == s and x["fold"] == k][0]
            est = _estimator(name, p["params"]).fit(X[tr], y[tr])
            for metric in ("accuracy", "f1"):
                pi = permutation_importance(est, X[te], y[te], scoring=metric, n_repeats=20,
                                            random_state=0, n_jobs=1)
                for j, feat in enumerate(C.FEATURES):
                    rows.append(dict(model=name, seed=s, fold=k, metric=metric, feature=feat,
                                     drop=100 * pi.importances_mean[j]))
            fi = getattr(est, "feature_importances_", None)
            if fi is not None:
                for j, feat in enumerate(C.FEATURES):
                    intr.append(dict(model=name, seed=s, fold=k, feature=feat, value=float(fi[j])))
        log(f"trees: permutation importance seed {s} fold {k} done")
    imp = pd.DataFrame(rows)
    imp.to_csv(os.path.join(C.TAB, "tree_permutation_importance_folds.csv"), index=False)
    agg = imp.groupby(["model", "metric", "feature"]).drop.agg(["mean", "std"]).reset_index()
    out["importance"] = agg.to_dict("records")
    if intr:
        ia = pd.DataFrame(intr).groupby(["model", "feature"]).value.agg(["mean", "std"]).reset_index()
        out["intrinsic"] = ia.to_dict("records")

    # 4. first splits of the tuned decision tree on the first fold (interpretability)
    from sklearn.tree import export_text
    f0 = C.outer_folds(y, g)[0]
    p = [x for x in tc["selected"] if x["model"] == "DT" and x["seed"] == f0["seed"] and x["fold"] == f0["fold"]][0]
    dt = _estimator("DT", p["params"]).fit(X[f0["train"]], y[f0["train"]])
    out["dt_rules_depth3"] = export_text(dt, feature_names=C.FEATURES, max_depth=3, decimals=1)
    out["dt_size"] = dict(depth=int(dt.get_depth()), leaves=int(dt.get_n_leaves()))
    C.save_json(out, "tree_diagnostics.json")
    _tree_figures(pr, imp, y)


def _tree_figures(pr, imp, y):
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
    # ROC: full range and zoom on the upper-left corner, outer seed 42 pooled
    fig, axes = plt.subplots(1, 2, figsize=(PS.FULL, 2.7))
    for name in TREES:
        ss = pr[(pr.model == name) & (pr.outer_seed == 42)].sort_values("row")
        fpr, tpr, _ = roc_curve(ss.y, ss.score); auc = roc_auc_score(ss.y, ss.score)
        for ax in axes:
            ax.plot(fpr, tpr, color=TREE_COLORS[name], label=f"{name} (AUC = {auc:.4f})")
    t2 = pr[(pr.model == "Threshold-2") & (pr.outer_seed == 42)]
    tp = ((t2.pred == 1) & (t2.y == 1)).sum() / (t2.y == 1).sum()
    fp = ((t2.pred == 1) & (t2.y == 0)).sum() / (t2.y == 0).sum()
    for ax in axes:
        ax.plot([fp], [tp], marker="s", ms=4, color="black", ls="none", label="Threshold-2 rule")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.7)
    axes[0].set_title("(a) Full range")
    axes[1].set_xlim(0, 0.05); axes[1].set_ylim(0.85, 1.0)
    axes[1].set_title("(b) Upper-left corner")
    axes[1].legend(loc="lower right", frameon=False)
    fig.tight_layout(); PS.save(fig, os.path.join(C.FIG, "fig_tree_roc"))

    # Confusion matrices, outer seed 42 pooled (10,000 records)
    fig, axes = plt.subplots(1, 4, figsize=(PS.FULL, 1.9))
    for ax, name in zip(axes, TREES):
        ss = pr[(pr.model == name) & (pr.outer_seed == 42)]
        cm = confusion_matrix(ss.y, ss.pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.6)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9,
                    color="white" if v > cm.max() * 0.8 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Faulty"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Faulty"] if ax is axes[0] else ["", ""])
        ax.set_xlabel("Predicted"); ax.set_title(name)
        if ax is axes[0]:
            ax.set_ylabel("Actual")
    fig.tight_layout(); PS.save(fig, os.path.join(C.FIG, "fig_tree_confusion"))

    # Permutation importance (accuracy drop), mean +- SD over the 9 outer folds
    a = imp[imp.metric == "accuracy"].groupby(["model", "feature"]).drop.agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(PS.FULL, 2.2))
    w = 0.2; xs = np.arange(len(C.FEATURES))
    for i, name in enumerate(TREES):
        m = [a.loc[(name, f), "mean"] for f in C.FEATURES]
        sd = [a.loc[(name, f), "std"] for f in C.FEATURES]
        ax.bar(xs + (i - 1.5) * w, m, w, yerr=sd, capsize=2, color=TREE_COLORS[name],
               label=name, error_kw=dict(lw=0.6))
    ax.set_xticks(xs); ax.set_xticklabels(C.FEATURES)
    ax.set_ylabel("Accuracy drop when\npermuted (points)")
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); PS.save(fig, os.path.join(C.FIG, "fig_tree_importance"))


# ============================================================ record-level, tuned
def stage_record():
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
    C.tf_setup(ARGS.threads)
    d, X, y, g = C.load()
    skf = StratifiedKFold(5, shuffle=True, random_state=42)   # same folds as Table 4
    grid = M.NN_GRID["full"]
    t0 = time.time()
    for k, (tr, te) in enumerate(skf.split(X, y)):
        # ---- references: tuned trees and the two-attribute rule (training records only)
        part = os.path.join(C.PART, f"record_tuned_trees_{k}.json")
        if not os.path.exists(part):
            res = {}
            r2 = M.fit_threshold2(X[tr], y[tr])
            sc = M.apply_threshold2(r2, X[te]).astype(float)
            res["Threshold-2"] = dict(metrics=C.metrics(y[te], sc, 0.5, has_score=False),
                                      params=r2, score=sc.tolist())
            for name in TREES:
                est, prm = M.tuned_conventional(seed=0)[name]
                gs = GridSearchCV(est, prm, scoring="f1",
                                  cv=StratifiedKFold(3, shuffle=True, random_state=42),
                                  n_jobs=1, refit=True).fit(X[tr], y[tr])
                sc, thr = M.score_of(gs.best_estimator_, X[te])
                res[name] = dict(metrics=C.metrics(y[te], sc, thr), params=gs.best_params_,
                                 score=np.asarray(sc).tolist())
            with open(part, "w") as fh:
                json.dump(res, fh, default=C._conv)
            log(f"record tuned trees fold {k}: HGB acc {res['HGB']['metrics']['accuracy']:.2f}")
        # ---- tuned networks: same search and budget as the sensor-disjoint benchmark;
        #      validation = stratified 20% of the training records (record-level, like the protocol)
        sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=100 + k)
        ia, iv = next(sss.split(X[tr], y[tr]))
        A, V, T = minmax(X[tr][ia], X[tr][iv], X[te])
        ya, yv = y[tr][ia], y[tr][iv]
        for name in NN:
            part = os.path.join(C.PART, f"record_tuned_{name}_{k}.json")
            if os.path.exists(part):
                continue
            sel, models0 = [], {}
            for ci, cfg in enumerate(grid[name]):
                m, h = M.fit_nn(name, dict(cfg, spe=8), A, ya, seed=0, epochs=M.TUNED_MAX_EPOCHS,
                                batch=M.TUNED_BATCH, Xva=V, yva=yv, patience=M.TUNED_PATIENCE)
                b = int(np.argmin(h["val_loss"]))
                sel.append(dict(cfg=cfg, val_loss=float(h["val_loss"][b]), best_epoch=b + 1,
                                epochs_run=len(h["loss"])))
                models0[ci] = (m, h)
            ci = int(np.argmin([s["val_loss"] for s in sel]))
            res = dict(model=name, fold=k, grid=sel, chosen=grid[name][ci],
                       params=M.n_params(name, grid[name][ci]), runs=[])
            for init in C.INIT_SEEDS:
                if init == 0:
                    m, h = models0[ci]
                else:
                    m, h = M.fit_nn(name, dict(grid[name][ci], spe=8), A, ya, seed=init,
                                    epochs=M.TUNED_MAX_EPOCHS, batch=M.TUNED_BATCH,
                                    Xva=V, yva=yv, patience=M.TUNED_PATIENCE)
                p = M.nn_predict(m, T)
                res["runs"].append(dict(init=init, metrics=C.metrics(y[te], p),
                                        best_epoch=int(np.argmin(h["val_loss"])) + 1,
                                        epochs_run=len(h["loss"]), score=p.tolist()))
            with open(part, "w") as fh:
                json.dump(res, fh, default=C._conv)
            log(f"record tuned {name} fold {k}: chose {grid[name][ci]}, acc "
                f"{np.mean([r['metrics']['accuracy'] for r in res['runs']]):.2f} "
                f"({(time.time() - t0) / 60:.0f} min)")
    pd.DataFrame([dict(row=i, fold=k) for k, (_, te) in enumerate(skf.split(X, y)) for i in te]
                 ).sort_values("row").to_csv(os.path.join(C.RES, "folds_record_cv.csv"), index=False)


def stage_record_figs():
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
    from scipy.stats import friedmanchisquare
    d, X, y, g = C.load()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    rows, oof, oof_by_seed, chosen = [], {}, {}, []
    for name in NN:
        oof[name] = np.zeros(len(y))
        oof_by_seed[name] = {i: np.zeros(len(y)) for i in C.INIT_SEEDS}
        for k, (tr, te) in enumerate(folds):
            res = json.load(open(os.path.join(C.PART, f"record_tuned_{name}_{k}.json")))
            chosen.append(dict(model=name, fold=k, chosen=res["chosen"], params=res["params"],
                               best_epochs=[r["best_epoch"] for r in res["runs"]]))
            fold_m = pd.DataFrame([r["metrics"] for r in res["runs"]])[C.METRICS].mean()
            rows.append(dict(model=f"{name}, tuned", fold=k, **fold_m.to_dict()))
            for r in res["runs"]:
                oof_by_seed[name][r["init"]][te] = np.array(r["score"])
            oof[name][te] = np.array(res["runs"][0]["score"])    # init seed 0 for figures
    for k, (tr, te) in enumerate(folds):
        res = json.load(open(os.path.join(C.PART, f"record_tuned_trees_{k}.json")))
        for name, v in res.items():
            rows.append(dict(model=name, fold=k, **{m: v["metrics"][m] for m in C.METRICS}))
            chosen.append(dict(model=name, fold=k, chosen=v["params"]))
    df = pd.DataFrame(rows)
    summ = {n: {m: [float(df[df.model == n][m].mean()), float(df[df.model == n][m].std(ddof=1))]
                for m in C.METRICS} for n in df.model.unique()}
    pooled = {n: [C.metrics(y, oof_by_seed[n][i]) for i in C.INIT_SEEDS] for n in NN}
    fried = {}
    for m in C.METRICS:
        vals = [df[df.model == f"{n}, tuned"].sort_values("fold")[m].to_numpy() for n in NN]
        st, p = friedmanchisquare(*vals)
        fried[m] = dict(chi2=float(st), p=float(p))
    old = C.load_json("record_cv.json") if C.exists("record_cv.json") else None
    C.save_json(dict(folds=rows, summary=summ, pooled=pooled, friedman=fried, chosen=chosen,
                     fixed_10_epochs=(old or {}).get("summary")), "record_cv_tuned.json")
    df.to_csv(os.path.join(C.TAB, "record_cv_tuned_folds.csv"), index=False)

    colors = {"CNN": "#7f7f7f", "RNN": "#d1495b", "LSTM": "#1f4e9c"}
    fig, ax = plt.subplots(figsize=(PS.HALF, 2.8))
    for n in NN:
        fpr, tpr, _ = roc_curve(y, oof[n])
        ax.plot(fpr, tpr, color=colors[n], label=f"{n}, tuned (AUC = {roc_auc_score(y, oof[n]):.4f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.7)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); PS.save(fig, os.path.join(C.FIG, "fig_record_tuned_roc"))
    fig, axes = plt.subplots(1, 3, figsize=(PS.FULL, 2.0))
    for ax, n in zip(axes, NN):
        cm = confusion_matrix(y, (oof[n] >= 0.5).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.6)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9,
                    color="white" if v > cm.max() * 0.8 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Faulty"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Faulty"] if ax is axes[0] else ["", ""])
        ax.set_xlabel("Predicted"); ax.set_title(f"{n}, tuned")
        if ax is axes[0]:
            ax.set_ylabel("Actual")
    fig.tight_layout(); PS.save(fig, os.path.join(C.FIG, "fig_record_tuned_confusion"))


# ============================================================ exploratory figures
def stage_eda():
    d, X, y, g = C.load()
    cols = C.FEATURES + ["IsFaulty"]
    corr = d[cols].corr()
    fig, ax = plt.subplots(figsize=(PS.HALF, 2.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    for (i, j), v in np.ndenumerate(corr.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.6 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(cols)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)
    fig.tight_layout(); PS.save(fig, os.path.join(C.FIG, "fig_eda_correlation"))

    specs = [("BatteryLife", [-np.inf, 20, 40, 60, 80, np.inf],
              ["<20", "20–40", "40–60", "60–80", "≥80"], "(a) BatteryLife bin"),
             ("Temperature", [-np.inf, 10, 20, 30, 40, 50, 60, 70, np.inf],
              ["<10", "10–20", "20–30", "30–40", "40–50", "50–60", "60–70", "≥70"],
              "(b) Temperature bin")]
    fig, axes = plt.subplots(1, 2, figsize=(PS.FULL, 2.4), gridspec_kw=dict(width_ratios=[5, 8]))
    tabs = {}
    for ax, (col, edges, labels, xlabel) in zip(axes, specs):
        b = pd.cut(d[col], edges, right=False, labels=labels)
        t = d.groupby(b, observed=False).IsFaulty.agg(["sum", "count"])
        tabs[col] = t
        ax.bar(range(len(t)), 100 * t["sum"] / t["count"], color="#1f4e9c", width=0.7)
        for i, (s_, n) in enumerate(zip(t["sum"], t["count"])):
            ax.text(i, 100 * s_ / n + 2, f"{100 * s_ / n:.1f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(len(t)))
        ax.set_xticklabels([f"{lab}\n{n:,}" for lab, n in zip(labels, t["count"])])
        ax.set_xlabel(xlabel); ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Records labelled faulty (%)")
    fig.tight_layout(); PS.save(fig, os.path.join(C.FIG, "fig_eda_fault_share"))
    tb, tt = tabs["BatteryLife"], tabs["Temperature"]
    fig, axes = plt.subplots(1, 3, figsize=(PS.FULL, 2.0))
    for ax, col in zip(axes, C.FEATURES):
        lo, hi = np.percentile(d[col], [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        for lab, c, name in ((0, "#2ca02c", "Normal"), (1, "#d1495b", "Faulty")):
            ax.hist(d.loc[d.IsFaulty == lab, col].clip(lo, hi), bins=bins, density=True,
                    alpha=0.55, color=c, label=name)
        ax.set_xlabel(col); ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Density")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(); PS.save(fig, os.path.join(C.FIG, "fig_eda_distributions"))
    C.save_json(dict(battery=tb.to_dict(), temperature=tt.to_dict()), "eda_bins.json")


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="trees,record,record_figs,eda")
    ap.add_argument("--threads", type=int, default=0)
    ARGS = ap.parse_args()
    for s in ARGS.stages.split(","):
        log(f"=== {s} ===")
        globals()[f"stage_{s}"]()


if __name__ == "__main__":
    main()
