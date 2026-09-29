"""Interactive Neural Network Learning Demo (Streamlit + TensorFlow/Keras)."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Interactive Neural Network Learning Demo",
    page_icon="🧠",
    layout="wide",
)

C0, C1 = "#F97316", "#2563EB"  # class colours (orange / blue)
LIM = 1.6
GRID_N = 70

# ----------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1350px; }
.hero { background: linear-gradient(120deg,#4F46E5 0%,#7C3AED 60%,#2563EB 100%);
        border-radius: 18px; padding: 26px 32px; color: #fff; margin-bottom: 18px; }
.hero h1 { margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -.02em; color:#fff; }
.hero p { margin: 6px 0 0 0; opacity: .92; font-size: 1rem; color:#fff; }
.card-title { font-weight: 700; font-size: 1.05rem; color: #0F172A; margin: 4px 0 2px 0; }
.card-sub { color: #64748B; font-size: .85rem; margin-bottom: 6px; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #E2E8F0; border-radius:14px;
        padding: 14px 18px; box-shadow: 0 1px 2px rgba(15,23,42,.04); }
div[data-testid="stMetricLabel"] p { color:#64748B; font-weight:600; }
.result { border-radius: 14px; padding: 14px 18px; font-weight: 700; font-size: 1.05rem; color:#fff; }
section[data-testid="stSidebar"] { background:#fff; border-right:1px solid #E2E8F0; }
.stButton>button { width:100%; border-radius:12px; font-weight:700; padding:.65rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ data
def make_data(name: str, n: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    if name == "Moons":
        h = n // 2
        t = rng.uniform(0, np.pi, h)
        a = np.c_[np.cos(t), np.sin(t)]
        b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
        X = np.vstack([a, b]) - [0.5, 0.25]
        y = np.r_[np.zeros(h), np.ones(h)]
        X = X * 1.2
    elif name == "Circles":
        h = n // 2
        t0, t1 = rng.uniform(0, 2 * np.pi, h), rng.uniform(0, 2 * np.pi, h)
        X = np.vstack([np.c_[np.cos(t0), np.sin(t0)] * 1.1, np.c_[np.cos(t1), np.sin(t1)] * 0.45])
        y = np.r_[np.zeros(h), np.ones(h)]
    elif name == "XOR":
        X = rng.uniform(-1.2, 1.2, (n, 2))
        y = ((X[:, 0] * X[:, 1]) > 0).astype(float)
        X = X + rng.normal(0, noise * 0.4, X.shape)
        noise = 0.0
    else:  # Two clusters (linearly separable)
        h = n // 2
        X = np.vstack([rng.normal([-0.7, -0.5], 0.35, (h, 2)), rng.normal([0.7, 0.5], 0.35, (h, 2))])
        y = np.r_[np.zeros(h), np.ones(h)]
        X = X * (1 + noise)
        noise = 0.0
    X = X + rng.normal(0, noise, X.shape)
    idx = rng.permutation(len(X))
    X, y = X[idx].astype("float32"), y[idx].astype("float32")
    k = int(0.8 * len(X))
    return X[:k], y[:k], X[k:], y[k:]


# --------------------------------------------------------------- training
class Snapshots(tf.keras.callbacks.Callback):
    """Stores the decision surface at chosen epochs so it can be replayed."""

    def __init__(self, grid, epochs_to_keep, store):
        super().__init__()
        self.grid, self.keep, self.store = grid, set(epochs_to_keep), store

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 in self.keep:
            self.store[epoch + 1] = self.surface(self.model, self.grid)

    @staticmethod
    def surface(model, grid):
        return model(grid, training=False).numpy().reshape(GRID_N, GRID_N)


def train_model(cfg: dict):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(cfg["seed"])
    Xtr, ytr, Xte, yte = make_data(cfg["dataset"], cfg["samples"], cfg["noise"], cfg["seed"])

    layers = [tf.keras.layers.Input(shape=(2,))]
    for _ in range(cfg["layers"]):
        layers.append(tf.keras.layers.Dense(cfg["neurons"], activation=cfg["act"]))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    gx = np.linspace(-LIM, LIM, GRID_N)
    xx, yy = np.meshgrid(gx, gx)
    grid = np.c_[xx.ravel(), yy.ravel()].astype("float32")

    keep = sorted(set(np.linspace(1, cfg["epochs"], 12).astype(int).tolist()))
    snaps = {0: Snapshots.surface(model, grid)}
    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xte, yte),
        epochs=cfg["epochs"], batch_size=32, verbose=0,
        callbacks=[Snapshots(grid, keep, snaps)],
    )
    return {
        "cfg": cfg, "model": model, "hist": hist.history, "snaps": snaps,
        "gx": gx, "data": (Xtr, ytr, Xte, yte),
        "run_id": st.session_state.get("run_counter", 0) + 1,
    }


# ------------------------------------------------------------------ plots
COLORSCALE = [[0, "#FDBA74"], [0.5, "#FFFFFF"], [1, "#93C5FD"]]


def boundary_fig(res, epoch, user_pt):
    Z = res["snaps"][epoch]
    gx = res["gx"]
    Xtr, ytr, Xte, yte = res["data"]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=gx, y=gx, z=Z, zmin=0, zmax=1, colorscale=COLORSCALE,
                             showscale=False, zsmooth="best",
                             hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<br>P(class 1)=%{z:.2f}<extra></extra>"))
    fig.add_trace(go.Contour(x=gx, y=gx, z=Z, showscale=False, hoverinfo="skip",
                             contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
                             line=dict(color="#0F172A", width=2.5)))
    for cls, col, nm in [(0, C0, "Class 0"), (1, C1, "Class 1")]:
        m = ytr == cls
        fig.add_trace(go.Scatter(x=Xtr[m, 0], y=Xtr[m, 1], mode="markers", name=f"{nm} (train)",
                                 marker=dict(color=col, size=6, line=dict(color="white", width=.8))))
    for cls, col, nm in [(0, C0, "Class 0"), (1, C1, "Class 1")]:
        m = yte == cls
        fig.add_trace(go.Scatter(x=Xte[m, 0], y=Xte[m, 1], mode="markers", name=f"{nm} (test)",
                                 marker=dict(color=col, size=8, symbol="diamond",
                                             line=dict(color="#0F172A", width=1))))
    fig.add_trace(go.Scatter(x=[user_pt[0]], y=[user_pt[1]], mode="markers", name="Your input",
                             marker=dict(color="#FACC15", size=17, symbol="star",
                                         line=dict(color="#0F172A", width=1.5))))
    fig.update_layout(
        height=470, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-LIM, LIM], title="x₁", zeroline=False, showgrid=False),
        yaxis=dict(range=[-LIM, LIM], title="x₂", zeroline=False, showgrid=False,
                   scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
    )
    return fig


def arch_fig(sizes, weights=None):
    n_layers = len(sizes)
    xs = np.linspace(0, 1, n_layers)
    ys = [np.linspace(-1, 1, s) * (0.09 * (s - 1) + 0.0) if s > 1 else np.array([0.0]) for s in sizes]
    ys = [y - y.mean() for y in ys]
    fig = go.Figure()

    if weights:
        Ws = weights[0::2]
        mx = max(np.abs(W).max() for W in Ws) or 1.0
        bins = {(s, b): ([], []) for s in (1, -1) for b in range(3)}
        for li, W in enumerate(Ws):
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    w = W[i, j]
                    b = min(int(abs(w) / mx * 3), 2)
                    bx, by = bins[(1 if w >= 0 else -1, b)]
                    bx += [xs[li], xs[li + 1], None]
                    by += [ys[li][i], ys[li + 1][j], None]
        for (s, b), (bx, by) in bins.items():
            if bx:
                fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", hoverinfo="skip", showlegend=False,
                                         line=dict(color=C1 if s > 0 else C0, width=[0.6, 1.8, 3.6][b]),
                                         opacity=0.55))
    else:
        bx, by = [], []
        for li in range(n_layers - 1):
            for a in ys[li]:
                for b_ in ys[li + 1]:
                    bx += [xs[li], xs[li + 1], None]
                    by += [a, b_, None]
        fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color="#CBD5E1", width=1)))

    colors = ["#10B981"] + ["#7C3AED"] * (n_layers - 2) + ["#4F46E5"]
    for li in range(n_layers):
        fig.add_trace(go.Scatter(x=[xs[li]] * sizes[li], y=ys[li], mode="markers", showlegend=False,
                                 hoverinfo="skip",
                                 marker=dict(size=22, color=colors[li], line=dict(color="white", width=2))))
    names = ["Input"] + [f"Hidden {i}" for i in range(1, n_layers - 1)] + ["Output"]
    top = max(y.max() for y in ys) + 0.2
    fig.add_trace(go.Scatter(x=xs, y=[top] * n_layers, mode="text", text=names, showlegend=False,
                             hoverinfo="skip", textfont=dict(size=12, color="#475569")))
    fig.update_layout(
        height=340, margin=dict(l=5, r=5, t=5, b=5),
        xaxis=dict(visible=False, range=[-0.1, 1.1]), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
    )
    return fig


def curves_fig(hist, marker_epoch):
    ep = np.arange(1, len(hist["loss"]) + 1)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss (binary cross-entropy)", "Accuracy"))
    for key, name, col, dash in [("loss", "Train", "#4F46E5", "solid"), ("val_loss", "Test", "#F97316", "dash")]:
        fig.add_trace(go.Scatter(x=ep, y=hist[key], name=name, line=dict(color=col, width=2.5, dash=dash),
                                 legendgroup=name), row=1, col=1)
    for key, name, col, dash in [("accuracy", "Train", "#4F46E5", "solid"), ("val_accuracy", "Test", "#F97316", "dash")]:
        fig.add_trace(go.Scatter(x=ep, y=hist[key], name=name, line=dict(color=col, width=2.5, dash=dash),
                                 legendgroup=name, showlegend=False), row=1, col=2)
    if marker_epoch > 0:
        for c in (1, 2):
            fig.add_vline(x=marker_epoch, line=dict(color="#94A3B8", dash="dot"), row=1, col=c)
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(range=[0, 1.02], row=1, col=2)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", font=dict(family="Inter"))
    return fig


# ---------------------------------------------------------------- sidebar
st.markdown(
    """<div class="hero"><h1>🧠 Interactive Neural Network Learning Demo</h1>
<p>Change the network, press <b>Train</b>, and watch a real TensorFlow model learn to separate two classes.</p></div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.markdown("**1 · Dataset**")
    dataset = st.selectbox("Dataset", ["Moons", "Circles", "XOR", "Two clusters"], label_visibility="collapsed")
    noise = st.slider("Noise", 0.0, 0.5, 0.2, 0.05)
    samples = st.slider("Number of points", 200, 600, 400, 50)
    st.markdown("**2 · Network**")
    layers = st.slider("Hidden layers", 1, 4, 2)
    neurons = st.slider("Neurons per hidden layer", 1, 12, 6)
    act_label = st.selectbox("Activation function", ["ReLU", "Tanh", "Sigmoid", "Linear (no activation)"])
    st.markdown("**3 · Training**")
    lr = st.select_slider("Learning rate", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], value=0.03)
    epochs = st.slider("Epochs", 10, 300, 100, 10)
    seed = st.number_input("Random seed", 0, 9999, 42, 1)
    train_clicked = st.button("🚀 Train network", type="primary")

ACT = {"ReLU": "relu", "Tanh": "tanh", "Sigmoid": "sigmoid", "Linear (no activation)": "linear"}
cfg = dict(dataset=dataset, noise=noise, samples=samples, layers=layers, neurons=neurons,
           act=ACT[act_label], lr=lr, epochs=epochs, seed=int(seed))

if train_clicked or "res" not in st.session_state:
    with st.spinner("Training the neural network…"):
        res = train_model(cfg)
    st.session_state["run_counter"] = res["run_id"]
    st.session_state["res"] = res

res = st.session_state["res"]
if res["cfg"] != cfg:
    st.info("Settings changed — press **🚀 Train network** in the sidebar to apply them. "
            "The results below are from the previous run.")

rc = res["cfg"]
model, hist = res["model"], res["hist"]
sizes = [2] + [rc["neurons"]] * rc["layers"] + [1]

# ---------------------------------------------------------------- metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Train accuracy", f"{hist['accuracy'][-1] * 100:.1f}%")
m2.metric("Test accuracy", f"{hist['val_accuracy'][-1] * 100:.1f}%")
m3.metric("Final training loss", f"{hist['loss'][-1]:.3f}")
m4.metric("Trainable parameters", f"{model.count_params():,}")
st.write("")

# ------------------------------------------------------------ main columns
left, right = st.columns([1.15, 1], gap="large")

with right:
    st.markdown('<div class="card-title">Network architecture</div>'
                '<div class="card-sub">Blue connections = positive weights, orange = negative. '
                'Thicker = stronger.</div>', unsafe_allow_html=True)
    st.plotly_chart(arch_fig(sizes, model.get_weights()), width="stretch",
                    config={"displayModeBar": False})

    st.markdown('<div class="card-title">Try your own input</div>'
                '<div class="card-sub">Move the star on the decision map and see what the trained '
                'network predicts.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    x1 = c1.slider("x₁", -1.5, 1.5, 0.0, 0.05)
    x2 = c2.slider("x₂", -1.5, 1.5, 0.0, 0.05)
    p = float(model(np.array([[x1, x2]], dtype="float32"), training=False).numpy()[0][0])
    cls = 1 if p >= 0.5 else 0
    conf = p if cls == 1 else 1 - p
    st.markdown(
        f'<div class="result" style="background:{C1 if cls else C0}">Prediction: Class {cls} '
        f'({"blue" if cls else "orange"}) — {conf * 100:.1f}% confident</div>',
        unsafe_allow_html=True,
    )

with left:
    st.markdown('<div class="card-title">Decision boundary</div>'
                '<div class="card-sub">The black line is where the network switches from Class 0 to '
                'Class 1. Drag the slider to replay learning.</div>', unsafe_allow_html=True)
    options = sorted(res["snaps"].keys())
    snap_ep = st.select_slider("Epoch shown", options=options, value=options[-1],
                               key=f"snap_{res['run_id']}")
    st.plotly_chart(boundary_fig(res, snap_ep, (x1, x2)), width="stretch",
                    config={"displayModeBar": False})

st.markdown('<div class="card-title">Training progress</div>'
            '<div class="card-sub">Loss should fall and accuracy rise as the network learns. '
            'A big gap between train and test means overfitting.</div>', unsafe_allow_html=True)
st.plotly_chart(curves_fig(hist, snap_ep), width="stretch", config={"displayModeBar": False})

with st.expander("💡 What to try (for the demo)"):
    st.markdown(
        "- **Moons/Circles + Linear activation** → the boundary stays a straight line and accuracy is poor.\n"
        "- Switch to **ReLU or Tanh** → the boundary bends and accuracy jumps.\n"
        "- Set **neurons = 1** or **learning rate = 0.001** → learning is slow or fails.\n"
        "- Increase **noise** → test accuracy drops; watch train vs. test curves.\n"
        "- **XOR** needs at least one hidden layer with non-linear activation."
    )
