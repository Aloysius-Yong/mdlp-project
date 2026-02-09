import os
import warnings
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# =========================
# Config
# =========================
APP_TITLE = "Smartphone Price Predictor"
APP_TAGLINE = "Estimate phone prices and get model-driven recommendations — simple, fast, and user-friendly."
DATA_PATH = "smartphones.csv"

# ✅ Model files (Index vs MLDP)
MODEL_FILES = {
    "Index model (current)": "final_price_model.joblib",
    "MLDP model (test)": "final_price_model_mldp.joblib",
}

POSSIBLE_PRICE_COLS = ["Final Price", "final_price", "price", "Price", "Final_Price", "finalPrice"]
POSSIBLE_BRAND_COLS = ["Brand", "brand", "Make", "make"]
POSSIBLE_MODEL_COLS = ["Model", "model", "Phone Model", "phone_model"]
POSSIBLE_COLOR_COLS = ["Color", "color", "Colour", "colour"]
POSSIBLE_RAM_COLS = ["RAM (GB)", "RAM", "ram", "Ram", "ram_gb", "RAM_GB"]
POSSIBLE_STORAGE_COLS = ["Storage (GB)", "Storage", "storage", "storage_gb", "Storage_GB"]
POSSIBLE_UNLOCKED_COLS = ["Free / Unlocked?", "Unlocked", "unlocked", "Free", "free", "Free_Unlocked"]

ENGINEERED_COLS = ["Storage_per_RAM", "RAM_x_Storage", "Free_bin"]


# =========================
# Helpers
# =========================
def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def norm_text(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip().lower()


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
    return df


# ✅ Cache per-model-path so switching models works
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def format_currency(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"${v:,.2f}"


def field_label(title: str, desc: str):
    st.markdown(f"**{title}:** {desc}")


def normalize_yes_no(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "No"
    s = str(v).strip().lower()
    if s in ["yes", "y", "true", "1", "unlocked", "free"]:
        return "Yes"
    if s in ["no", "n", "false", "0", "locked"]:
        return "No"
    if "unlocked" in s or "free" in s or "yes" in s:
        return "Yes"
    return "No"


def unlocked_label_to_bool(val: str) -> bool:
    return str(val).lower().startswith("yes")


def build_feature_row(
    brand: str,
    model: str,
    ram_gb: float,
    storage_gb: float,
    color: str,
    unlocked_yes: bool,
    colmap: Dict[str, str],
) -> pd.DataFrame:
    row = {}

    brand_col = colmap.get("brand")
    model_col = colmap.get("model")
    color_col = colmap.get("color")
    ram_col = colmap.get("ram")
    storage_col = colmap.get("storage")
    unlocked_col = colmap.get("unlocked")

    row[brand_col or "Brand"] = brand
    row[model_col or "Model"] = model
    row[color_col or "Color"] = color
    row[ram_col or "RAM (GB)"] = ram_gb
    row[storage_col or "Storage (GB)"] = storage_gb
    row[unlocked_col or "Free / Unlocked?"] = "Yes" if unlocked_yes else "No"

    df_row = pd.DataFrame([row])

    ram_val = safe_float(ram_gb, np.nan)
    storage_val = safe_float(storage_gb, np.nan)

    # These are safe even if the model ignores them
    df_row["Free_bin"] = 1 if unlocked_yes else 0
    df_row["RAM_x_Storage"] = ram_val * storage_val
    df_row["Storage_per_RAM"] = (storage_val / ram_val) if (not np.isnan(ram_val) and ram_val != 0) else np.nan

    return df_row


def predict_price(model, X: pd.DataFrame) -> float:
    y = model.predict(X)
    return float(y[0])


def unique_sorted(df: pd.DataFrame, col: Optional[str]) -> List[str]:
    if df.empty or not col or col not in df.columns:
        return []
    vals = df[col].dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    return sorted(vals.unique().tolist())


def unique_sorted_numeric(df: pd.DataFrame, col: Optional[str]) -> List[int]:
    if df.empty or not col or col not in df.columns:
        return []
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if x.empty:
        return []
    return sorted(set(int(v) for v in x.tolist()))


# =========================
# Page setup + CSS
# =========================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",  # ✅ need expanded so user can switch models
)

CUSTOM_CSS = """
<style>
.block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1200px; }
.hero {
  background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(236,72,153,0.10));
  border: 1px solid rgba(148,163,184,0.35);
  border-radius: 18px;
  padding: 18px 18px 14px 18px;
  margin-bottom: 1.1rem;
}
.hero h1 { margin: 0; font-size: 2.0rem; letter-spacing: -0.02em; }
.hero p { margin: 6px 0 0 0; color: rgba(15,23,42,0.70); font-size: 1.02rem; }
.card {
  border: 1px solid rgba(148,163,184,0.35);
  border-radius: 16px;
  padding: 16px;
  background: #ffffff;
  box-shadow: 0 1px 10px rgba(15,23,42,0.04);
}
.card h2 { margin: 0 0 8px 0; font-size: 1.25rem; }
.muted { color: rgba(15,23,42,0.60); font-size: 0.95rem; }
.small { font-size: 0.9rem; color: rgba(15,23,42,0.65); }
div[data-testid="stMarkdownContainer"] p { margin-bottom: 0.25rem; }
div.stButton > button, div.stDownloadButton > button {
  border-radius: 12px !important;
  padding: 0.75rem 1rem !important;
  font-weight: 650 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# Load dataset
# =========================
df = load_dataset(DATA_PATH)

colmap = {
    "price": pick_first_existing(df, POSSIBLE_PRICE_COLS) if not df.empty else None,
    "brand": pick_first_existing(df, POSSIBLE_BRAND_COLS) if not df.empty else None,
    "model": pick_first_existing(df, POSSIBLE_MODEL_COLS) if not df.empty else None,
    "color": pick_first_existing(df, POSSIBLE_COLOR_COLS) if not df.empty else None,
    "ram": pick_first_existing(df, POSSIBLE_RAM_COLS) if not df.empty else None,
    "storage": pick_first_existing(df, POSSIBLE_STORAGE_COLS) if not df.empty else None,
    "unlocked": pick_first_existing(df, POSSIBLE_UNLOCKED_COLS) if not df.empty else None,
}

# =========================
# ✅ Sidebar: Model selector
# =========================
st.sidebar.title("⚙️ Model Selection")
model_choice = st.sidebar.selectbox("Choose model to use", list(MODEL_FILES.keys()))
MODEL_PATH = MODEL_FILES[model_choice]

model = load_model(MODEL_PATH)

st.sidebar.write("")
st.sidebar.markdown("**Loaded model file:**")
st.sidebar.code(MODEL_PATH)

if model is None:
    st.sidebar.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

# =========================
# Basic dataset fallback lists
# =========================
brands = unique_sorted(df, colmap["brand"]) or ["Apple", "Samsung", "Xiaomi", "Oppo", "Vivo"]
colors_all = unique_sorted(df, colmap["color"]) or ["Black", "White", "Blue", "Red", "Green"]
all_models = unique_sorted(df, colmap["model"]) if colmap["model"] else []

# =========================
# Session state
# =========================
if "brand" not in st.session_state:
    st.session_state.brand = brands[0]
if "model" not in st.session_state:
    st.session_state.model = "(Select a model)"
if "ram" not in st.session_state:
    st.session_state.ram = None
if "storage" not in st.session_state:
    st.session_state.storage = None
if "color" not in st.session_state:
    st.session_state.color = colors_all[0]
if "sim_status" not in st.session_state:
    st.session_state.sim_status = "Yes (Any SIM)"
if "budget" not in st.session_state:
    st.session_state.budget = 1200

if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
    st.session_state.last_error = None


def on_brand_change():
    st.session_state.model = "(Select a model)"
    st.session_state.ram = None
    st.session_state.storage = None


def on_model_change():
    st.session_state.ram = None
    st.session_state.storage = None


# =========================
# UI Header
# =========================
st.markdown(
    f"""
<div class="hero">
  <h1>📱 {APP_TITLE}</h1>
  <p>{APP_TAGLINE}</p>
</div>
""",
    unsafe_allow_html=True,
)

st.info(f"✅ Currently using: **{model_choice}** (`{MODEL_PATH}`)")

tab_predict, tab_compare = st.tabs(["🔮 Estimate Price", "🆚 Compare Phones"])

# =========================
# TAB 1: Estimate Price
# =========================
with tab_predict:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## Enter phone details")
        st.markdown(
            '<div class="muted">Prediction uses the currently selected model.</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        field_label("Brand", "Choose the smartphone brand.")
        st.selectbox(
            "Brand",
            brands,
            index=brands.index(st.session_state.brand) if st.session_state.brand in brands else 0,
            key="brand",
            on_change=on_brand_change,
            label_visibility="collapsed",
        )

        model_options = ["(Select a model)"]
        if not df.empty and colmap["brand"] and colmap["model"]:
            bcol = colmap["brand"]
            mcol = colmap["model"]
            subset = df.copy()
            subset["_brand_norm"] = subset[bcol].apply(norm_text)
            subset = subset[subset["_brand_norm"] == norm_text(st.session_state.brand)]
            mlist = sorted(subset[mcol].dropna().astype(str).str.strip().unique().tolist())
            model_options += mlist
        else:
            model_options += (all_models or [])

        field_label("Model", "Choose the model for the selected brand.")
        st.selectbox(
            "Model",
            model_options,
            index=model_options.index(st.session_state.model) if st.session_state.model in model_options else 0,
            key="model",
            on_change=on_model_change,
            label_visibility="collapsed",
        )

        # RAM options
        ram_options = unique_sorted_numeric(df, colmap["ram"]) or [2, 4, 6, 8, 12, 16]
        ram_labels = [str(r) for r in ram_options]
        ram_index = (
            0
            if (st.session_state.ram is None or str(st.session_state.ram) not in ram_labels)
            else ram_labels.index(str(st.session_state.ram))
        )
        field_label("RAM (Memory)", "More RAM helps multitasking (GB).")
        st.selectbox("RAM", ram_labels, index=ram_index, key="ram", label_visibility="collapsed")

        # Storage options
        storage_options = unique_sorted_numeric(df, colmap["storage"]) or [32, 64, 128, 256, 512]
        storage_labels = [str(s) for s in storage_options]
        storage_index = (
            0
            if (st.session_state.storage is None or str(st.session_state.storage) not in storage_labels)
            else storage_labels.index(str(st.session_state.storage))
        )
        field_label("Storage", "How much space you have for apps and photos (GB).")
        st.selectbox("Storage", storage_labels, index=storage_index, key="storage", label_visibility="collapsed")

        # Color
        color_options = unique_sorted(df, colmap["color"]) or colors_all
        if st.session_state.color not in color_options:
            st.session_state.color = color_options[0]
        field_label("Color", "Choose the phone color.")
        st.selectbox(
            "Color",
            color_options,
            index=color_options.index(st.session_state.color),
            key="color",
            label_visibility="collapsed",
        )

        # SIM
        field_label("SIM Status", "Does it work with any SIM card, or is it locked to a telco?")
        st.radio(
            "Works with any SIM card?",
            ["Yes (Any SIM)", "No (Locked to a telco)"],
            horizontal=True,
            key="sim_status",
            label_visibility="collapsed",
        )

        st.write("")
        field_label("Budget", "Optional: set your max budget to see if the estimate fits.")
        st.slider("Budget", min_value=0, max_value=5000, step=50, key="budget", label_visibility="collapsed")

        st.write("")
        predict_clicked = st.button("✨ Estimate price", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## Your result")

        if predict_clicked:
            user_errors = []
            if not st.session_state.brand or str(st.session_state.brand).strip() == "":
                user_errors.append("Please choose a brand.")
            if not st.session_state.model or st.session_state.model == "(Select a model)":
                user_errors.append("Please choose a model.")
            if st.session_state.ram is None:
                user_errors.append("Please choose a RAM option.")
            if st.session_state.storage is None:
                user_errors.append("Please choose a storage option.")

            if user_errors:
                st.session_state.last_pred = None
                st.session_state.last_error = " ".join(user_errors)
            else:
                try:
                    ram_val = float(st.session_state.ram)
                    storage_val = float(st.session_state.storage)
                    unlocked_yes = unlocked_label_to_bool(st.session_state.sim_status)

                    X = build_feature_row(
                        brand=st.session_state.brand,
                        model=st.session_state.model,
                        ram_gb=ram_val,
                        storage_gb=storage_val,
                        color=st.session_state.color,
                        unlocked_yes=unlocked_yes,
                        colmap=colmap,
                    )
                    pred = predict_price(model, X)

                    st.session_state.last_pred = pred
                    st.session_state.last_error = None
                except Exception as e:
                    st.session_state.last_pred = None
                    st.session_state.last_error = f"Prediction failed: {e}"

        if st.session_state.last_error:
            st.error(st.session_state.last_error)

        pred = st.session_state.last_pred
        if pred is None:
            st.markdown('<div class="small">Fill specs and click <b>Estimate price</b>.</div>', unsafe_allow_html=True)
        else:
            st.metric("Estimated price", format_currency(pred))

            if st.session_state.budget > 0:
                if pred <= st.session_state.budget:
                    st.success(f"Within your budget ({format_currency(st.session_state.budget)}).")
                else:
                    st.warning(f"Above your budget by {format_currency(pred - st.session_state.budget)}.")

            low = pred * 0.9
            high = pred * 1.1
            st.info(f"Typical range: {format_currency(low)} to {format_currency(high)} (estimate).")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TAB 2: Compare Phones (kept minimal to avoid huge changes)
# =========================
with tab_compare:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Compare two phones")
    st.markdown(
        '<div class="muted">Compare predicted prices using the selected model.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    colA, colB = st.columns(2, gap="large")

    def mini_phone(prefix: str, default_brand_idx: int = 0):
        b = st.selectbox(f"{prefix} Brand", brands, index=default_brand_idx, key=f"{prefix}_b")
        m = st.text_input(f"{prefix} Model (type)", key=f"{prefix}_m")
        r = st.selectbox(f"{prefix} RAM", [str(x) for x in (unique_sorted_numeric(df, colmap["ram"]) or [2,4,6,8,12,16])], key=f"{prefix}_r")
        s = st.selectbox(f"{prefix} Storage", [str(x) for x in (unique_sorted_numeric(df, colmap["storage"]) or [32,64,128,256,512])], key=f"{prefix}_s")
        c = st.selectbox(f"{prefix} Color", (unique_sorted(df, colmap["color"]) or colors_all), key=f"{prefix}_c")
        u = st.radio(f"{prefix} SIM", ["Yes (Any SIM)", "No (Locked to a telco)"], horizontal=True, key=f"{prefix}_u")
        return b, m, float(r), float(s), c, unlocked_label_to_bool(u)

    with colA:
        st.subheader("Phone A")
        A = mini_phone("A", 0)

    with colB:
        st.subheader("Phone B")
        A2 = mini_phone("B", min(1, len(brands)-1))

    st.write("")
    if st.button("Compare prices", use_container_width=True):
        try:
            XA = build_feature_row(A[0], A[1], A[2], A[3], A[4], A[5], colmap)
            XB = build_feature_row(A2[0], A2[1], A2[2], A2[3], A2[4], A2[5], colmap)
            pA = predict_price(model, XA)
            pB = predict_price(model, XB)

            c1, c2, c3 = st.columns(3)
            c1.metric("Phone A", format_currency(pA))
            c2.metric("Phone B", format_currency(pB))
            c3.metric("Difference", format_currency(abs(pA - pB)))
        except Exception as e:
            st.error(f"Compare failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.caption(
    "Built with Streamlit • Model switching enabled • "
    "All dropdown options come from your dataset (`smartphones.csv`)."
)
