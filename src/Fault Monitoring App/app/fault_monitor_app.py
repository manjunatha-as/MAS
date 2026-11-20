"""WSN node-level fault monitoring dashboard (CSV replay).

Run with:   streamlit run fault_monitor_app.py

The app reads recorded sensor records from a CSV file (e.g. WsnData.csv),
trains the classifiers on the records of 80% of the sensors, and replays the
records of the remaining 20% (sensors never seen in training) one by one. It is
a software demonstration and is not connected to live sensor nodes.
"""
import base64
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler

FEATURES = ["SensorData", "BatteryLife", "Temperature"]
TARGET = "IsFaulty"
NEURAL = ["CNN", "RNN", "LSTM"]

st.set_page_config(page_title="WSN Fault Monitoring", page_icon="📡", layout="wide")
st.title("📡 WSN Fault Monitoring (CSV replay)")
st.caption("Replays recorded sensor records from an uploaded CSV file. "
           "The application is not connected to live sensor nodes.")

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    test_share = st.slider("Share of sensors held out for replay", 0.1, 0.5, 0.2, 0.05)
    seed = st.number_input("Random seed", value=42, step=1)
    chosen = st.multiselect("Models", ["HGB"] + NEURAL, default=["HGB"] + NEURAL)
    n_replay = st.number_input("Records to replay", min_value=10, value=200, step=10)
    delay = st.slider("Delay per replay step (s)", 0.0, 1.0, 0.05, 0.05)
    refresh = st.number_input("Update charts every N steps", min_value=1, value=5, step=1)
    alert_on = st.checkbox("Sound alert when a record is classified as faulty", value=False)
    alert_file = st.file_uploader("Alert sound (mp3, optional)", type=["mp3"]) if alert_on else None


# ----------------------------------------------------------------- models
def build_nn(kind, n_feat):
    import tensorflow as tf
    L = tf.keras.layers
    inp = L.Input(shape=(n_feat, 1))
    if kind == "CNN":
        x = L.Flatten()(L.Conv1D(32, 2, activation="relu")(inp))
    elif kind == "RNN":
        x = L.SimpleRNN(50)(L.SimpleRNN(50, return_sequences=True)(inp))
    else:
        x = L.LSTM(50)(L.LSTM(50, return_sequences=True)(inp))
    m = tf.keras.Model(inp, L.Dense(1, activation="sigmoid")(x))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m


@st.cache_resource(show_spinner=False)
def train_models(csv_bytes, test_share, seed, model_names):
    """Split by sensor, fit the scaler on training sensors only, train models,
    and return held-out probabilities. Cached so the replay does not retrain."""
    import io
    import tensorflow as tf
    df = pd.read_csv(io.BytesIO(csv_bytes))
    X = df[FEATURES].to_numpy(float)
    y = df[TARGET].to_numpy(int)
    if "SensorID" in df.columns:  # sensor-disjoint split
        gss = GroupShuffleSplit(n_splits=1, test_size=test_share, random_state=seed)
        tr, te = next(gss.split(X, y, groups=df["SensorID"]))
        split = "sensor-disjoint"
    else:  # fall back to a stratified record split
        tr, te = train_test_split(np.arange(len(y)), test_size=test_share,
                                  stratify=y, random_state=seed)
        split = "record-level (no SensorID column)"
    scaler = MinMaxScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
    probs, timings = {}, {}
    for name in model_names:
        t0 = time.perf_counter()
        if name == "HGB":
            m = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
            probs[name] = m.predict_proba(X[te])[:, 1]
        else:
            tf.keras.utils.set_random_seed(int(seed))
            m = build_nn(name, len(FEATURES))
            m.fit(Xtr[..., None], y[tr], epochs=10, batch_size=16, verbose=0)
            probs[name] = m.predict(Xte[..., None], verbose=0).ravel()
        timings[name] = time.perf_counter() - t0
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(te))  # replay order of the held-out records
    return dict(test_idx=te[order], y=y[te][order],
                probs={k: v[order] for k, v in probs.items()},
                n_train_sensors=int(df.iloc[tr]["SensorID"].nunique()) if "SensorID" in df else None,
                n_test_sensors=int(df.iloc[te]["SensorID"].nunique()) if "SensorID" in df else None,
                n_train=len(tr), n_test=len(te), split=split, timings=timings)


def scores(y, p, undefined_as_nan=False):
    """Accuracy, precision, recall, F1 in %. During replay, precision/recall/F1
    are left undefined (NaN) until a faulty record has been predicted/seen."""
    pred = (p >= 0.5).astype(int)
    nan = float("nan") if undefined_as_nan else 0.0
    has_pos, has_pred = (y == 1).any(), (pred == 1).any()
    return dict(Accuracy=100 * accuracy_score(y, pred),
                Precision=100 * precision_score(y, pred, zero_division=0) if has_pred else nan,
                Recall=100 * recall_score(y, pred, zero_division=0) if has_pos else nan,
                F1=100 * f1_score(y, pred, zero_division=0) if (has_pos or has_pred) else nan)


def metric_chart(cdf):
    """Line chart with a fixed 0-102% axis so lines at 100% stay visible."""
    import altair as alt
    long = cdf.reset_index().melt("Replay step", var_name="Metric", value_name="Value").dropna()
    return (alt.Chart(long).mark_line(strokeWidth=2)
            .encode(x=alt.X("Replay step:Q", title="Replay step"),
                    y=alt.Y("Value:Q", title="Running metric (%)", scale=alt.Scale(domain=[0, 102])),
                    color=alt.Color("Metric:N", sort=["Accuracy", "Precision", "Recall", "F1"]),
                    strokeDash=alt.StrokeDash("Metric:N", sort=["Accuracy", "Precision", "Recall", "F1"]))
            .properties(height=260))


def play_alert(data):
    b64 = base64.b64encode(data).decode()
    st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------- main
uploaded = st.file_uploader("📂 Upload a sensor data CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV with the columns SensorData, BatteryLife, Temperature and IsFaulty "
            "(and SensorID for a sensor-disjoint split).")
    st.stop()

csv_bytes = uploaded.getvalue()
df = pd.read_csv(uploaded)
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()
if not chosen:
    st.warning("Select at least one model in the sidebar.")
    st.stop()

st.subheader("🔍 Preview of uploaded data")
st.write(df.head())
st.write(f"{len(df):,} records" + (f" from {df['SensorID'].nunique()} sensors" if "SensorID" in df else "")
         + f"; {int(df[TARGET].sum()):,} labelled faulty.")

# ---- exploratory plots
st.subheader("📊 Attribute distributions (normal vs. faulty)")
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, feat in zip(axes, FEATURES):
    sns.histplot(df, x=feat, hue=TARGET, kde=True, ax=ax, palette=["green", "red"])
    ax.set_title(f"{feat} distribution")
st.pyplot(fig)
plt.close(fig)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**📈 Correlation matrix**")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig); plt.close(fig)


def fault_share_plot(col, edges, labels, title):
    b = pd.cut(df[col], edges, labels=labels, right=False)
    tab = df.groupby(b, observed=False)[TARGET].agg(["sum", "count"])
    share = 100 * tab["sum"] / tab["count"].where(tab["count"] > 0)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(range(len(tab)), share.fillna(0), color="#d6604d", edgecolor="black", linewidth=0.5)
    for i, (_, v) in enumerate(tab.iterrows()):
        ax.text(i, (share.iloc[i] if v["count"] else 0) + 1.5, f"{int(v['sum'])}/{int(v['count'])}",
                ha="center", fontsize=7)
    ax.set_xticks(range(len(tab))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100); ax.set_ylabel("Faulty records in bin (%)")
    ax.set_xlabel(f"{col} (dataset units)"); ax.set_title(title)
    st.pyplot(fig); plt.close(fig)


with c2:
    st.markdown("**🌡️ Fault share by Temperature**")
    fault_share_plot("Temperature", [-np.inf, 10, 20, 30, 40, 50, 60, 70, np.inf],
                     ["<10", "10–20", "20–30", "30–40", "40–50", "50–60", "60–70", "≥70"], "")
with c3:
    st.markdown("**🔋 Fault share by BatteryLife**")
    fault_share_plot("BatteryLife", [-np.inf, 20, 40, 60, 80, np.inf],
                     ["<20", "20–40", "40–60", "60–80", "80–100"], "")

# ---- training
with st.spinner("Training models on the training sensors..."):
    R = train_models(csv_bytes, float(test_share), int(seed), tuple(chosen))
st.success(f"Models trained ({R['split']} split): {R['n_train']:,} training records"
           + (f" from {R['n_train_sensors']} sensors" if R["n_train_sensors"] else "")
           + f"; {R['n_test']:,} held-out records"
           + (f" from {R['n_test_sensors']} sensors" if R["n_test_sensors"] else "") + ".")

st.subheader("🧪 Held-out performance (all held-out records)")
st.dataframe(pd.DataFrame({m: scores(R["y"], R["probs"][m]) for m in chosen}).T.round(2))

# ---- replay
st.subheader("▶️ Record replay (held-out sensors)")
st.caption("Running metrics are computed from the recorded labels in the CSV file. In a live network "
           "the true fault state is usually not known immediately, so such metrics could only be "
           "updated after the labels are confirmed.")
if not st.button("Start replay"):
    st.stop()

n = int(min(n_replay, len(R["y"])))
prob_ph = {m: st.empty() for m in chosen}
chart_ph = {}
for m in chosen:
    st.markdown(f"**{m}: running metrics**")
    chart_ph[m] = st.empty()
cm_cols = st.columns(len(chosen))
cm_ph = {m: c.empty() for m, c in zip(chosen, cm_cols)}

alert_bytes = alert_file.getvalue() if (alert_on and alert_file is not None) else None
hist = {m: [] for m in chosen}
for i in range(n):
    yi = int(R["y"][i])
    any_fault = False
    for m in chosen:
        p = float(R["probs"][m][i])
        status = "FAULTY" if p >= 0.5 else "NORMAL"
        any_fault |= status == "FAULTY"
        prob_ph[m].markdown(f"**{m}**: fault probability {p:.2f} ({status}), recorded label: {yi}")
        hist[m].append(scores(R["y"][:i + 1], R["probs"][m][:i + 1], undefined_as_nan=True))
    if any_fault and alert_bytes:
        play_alert(alert_bytes)
    if (i + 1) % int(refresh) == 0 or i == n - 1:
        for m in chosen:
            cdf = pd.DataFrame(hist[m]); cdf.index = cdf.index + 1; cdf.index.name = "Replay step"
            chart_ph[m].altair_chart(metric_chart(cdf), use_container_width=True)
            cm = confusion_matrix(R["y"][:i + 1], (R["probs"][m][:i + 1] >= 0.5).astype(int), labels=[0, 1])
            fig, ax = plt.subplots(figsize=(3.2, 2.8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Normal", "Faulty"], yticklabels=["Normal", "Faulty"])
            ax.set_title(f"{m} (first {i + 1} records)", fontsize=9)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            cm_ph[m].pyplot(fig); plt.close(fig)
    if delay:
        time.sleep(float(delay))

st.success(f"Replay finished: {n} held-out records.")
