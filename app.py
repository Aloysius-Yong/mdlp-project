import os
import warnings
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# =========================
# Config
# =========================
APP_TITLE = "Smartphone Price Predictor"
APP_TAGLINE = "Instant estimated price based on your phone specs — simple, fast, and user-friendly."
MODEL_PATH = "final_price_model.joblib"
DATA_PATH = "smartphones.csv"

# If your training used different column names, we try to infer them.
POSSIBLE_PRICE_COLS = ["Final Price", "final_price", "price", "Price", "Final_Price", "finalPrice"]
POSSIBLE_BRAND_COLS = ["Brand", "brand", "Make", "make"]
POSSIBLE_MODEL_COLS = ["Model", "model", "Phone Model", "phone_model"]
POSSIBLE_COLOR_COLS = ["Color", "color", "Colour", "colour"]
POSSIBLE_RAM_COLS = ["RAM (GB)", "RAM", "ram", "Ram", "ram_gb", "RAM_GB"]
POSSIBLE_STORAGE_COLS = ["Storage (GB)", "Storage", "storage", "storage_gb", "Storage_GB"]
POSSIBLE_UNLOCKED_COLS = ["Free / Unlocked?", "Unlocked", "unlocked", "Free", "free", "Free_Unlocked"]

# Engineered features expected by your model
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


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)

    # Clean whitespace
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    return df


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

    # engineered features expected by the model
    ram_val = safe_float(ram_gb, np.nan)
    storage_val = safe_float(storage_gb, np.nan)

    df_row["Free_bin"] = 1 if unlocked_yes else 0
    df_row["RAM_x_Storage"] = ram_val * storage_val
    df_row["Storage_per_RAM"] = (storage_val / ram_val) if ram_val not in [0, np.nan] else np.nan

    return df_row


def predict_price(model, X: pd.DataFrame) -> float:
    y = model.predict(X)
    return float(y[0])


def find_similar_phones(
    df: pd.DataFrame,
    colmap: Dict[str, str],
    brand: str,
    ram_gb: float,
    storage_gb: float,
    unlocked_yes: bool,
    top_k: int = 8,
) -> pd.DataFrame:
    if df.empty:
        return df

    bcol = colmap.get("brand")
    rcol = colmap.get("ram")
    scol = colmap.get("storage")
    ucol = colmap.get("unlocked")

    work = df.copy()
    if rcol not in work.columns or scol not in work.columns:
        return pd.DataFrame()

    work[rcol] = pd.to_numeric(work[rcol], errors="coerce")
    work[scol] = pd.to_numeric(work[scol], errors="coerce")
    work = work.dropna(subset=[rcol, scol])

    if bcol in work.columns:
        work = work[work[bcol].astype(str).str.strip() == str(brand).strip()]

    if ucol in work.columns:
        want = "Yes" if unlocked_yes else "No"
        work = work[work[ucol].astype(str).str.strip().str.lower() == want.lower()]

    if work.empty:
        return pd.DataFrame()

    r = safe_float(ram_gb, np.nan)
    s = safe_float(storage_gb, np.nan)
    work["__dist__"] = ((work[rcol] - r) ** 2 + (work[scol] - s) ** 2) ** 0.5

    keep_cols = []
    for key in ["brand", "model", "ram", "storage", "color", "unlocked", "price"]:
        col = colmap.get(key)
        if col and col in work.columns:
            keep_cols.append(col)

    out = work.sort_values("__dist__").head(top_k)
    if keep_cols:
        out = out[keep_cols + ["__dist__"]]
    else:
        out = out.head(top_k)

    out = out.rename(columns={"__dist__": "Similarity (lower is closer)"})
    return out


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
    x = sorted(set(int(v) for v in x.tolist()))
    return x


def filter_df_for_choices(df: pd.DataFrame, colmap: Dict[str, str], brand: str, model: str) -> pd.DataFrame:
    """
    Filter rows by brand + model (when columns exist).
    This ensures RAM/Storage dropdown options come ONLY from dataset
    and match the chosen brand/model.
    """
    if df.empty:
        return df

    bcol = colmap.get("brand")
    mcol = colmap.get("model")

    out = df.copy()
    if bcol and bcol in out.columns and brand:
        out = out[out[bcol].astype(str).str.strip() == str(brand).strip()]
    if mcol and mcol in out.columns and model and model != "(Select a model)":
        out = out[out[mcol].astype(str).str.strip() == str(model).strip()]
    return out


# =========================
# Page setup + CSS
# =========================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
.block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1150px; }
section[data-testid="stSidebar"] { display: none; }

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

div.stButton > button, div.stDownloadButton > button {
  border-radius: 12px !important;
  padding: 0.7rem 1rem !important;
  font-weight: 600 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# Load artifacts
# =========================
df = load_dataset(DATA_PATH)
model = load_model(MODEL_PATH)

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
# Guardrails
# =========================
if model is None:
    st.error(f"Model file not found: `{MODEL_PATH}`. Make sure it is in the same folder as `app.py`.")
    st.stop()

if df.empty:
    st.warning(
        f"Dataset file not found (or empty): `{DATA_PATH}`. "
        f"Dropdown lists will use safe fallback options."
    )

# Dropdown base options from dataset
brands = unique_sorted(df, colmap["brand"]) or ["Apple", "Samsung", "Xiaomi", "Oppo", "Vivo"]
colors_all = unique_sorted(df, colmap["color"]) or ["Black", "White", "Blue", "Red", "Green"]
all_models = unique_sorted(df, colmap["model"]) if colmap["model"] else []

# =========================
# Session state (for reactive resetting)
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
if "unlocked" not in st.session_state:
    st.session_state.unlocked = "Yes"
if "budget" not in st.session_state:
    st.session_state.budget = 1200

if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
    st.session_state.last_input = None
    st.session_state.last_error = None


def on_brand_change():
    # When brand changes, model must reset
    st.session_state.model = "(Select a model)"
    st.session_state.ram = None
    st.session_state.storage = None


def on_model_change():
    # When model changes, reset RAM/Storage so user must pick valid ones for that model
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

tab_predict, tab_compare = st.tabs(["🔮 Predict Price", "🆚 Compare Phones"])

# =========================
# Predict Tab
# =========================
with tab_predict:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## Enter phone details")
        st.markdown(
            '<div class="muted">Choose specs from the dataset. The model will estimate the price instantly.</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        # Brand (reactive)
        st.selectbox(
            "Brand",
            brands,
            index=brands.index(st.session_state.brand) if st.session_state.brand in brands else 0,
            key="brand",
            on_change=on_brand_change,
        )

        # Models filtered by brand
        model_options = ["(Select a model)"]
        if not df.empty and colmap["brand"] and colmap["model"]:
            subset = df[df[colmap["brand"]].astype(str).str.strip() == str(st.session_state.brand).strip()]
            mlist = sorted(subset[colmap["model"]].dropna().astype(str).str.strip().unique().tolist())
            model_options += mlist
        else:
            model_options += (all_models or [])

        st.selectbox(
            "Model",
            model_options,
            index=model_options.index(st.session_state.model) if st.session_state.model in model_options else 0,
            key="model",
            on_change=on_model_change,
        )

        # Filter for dependent dropdowns (RAM/Storage/Color)
        filtered = filter_df_for_choices(df, colmap, st.session_state.brand, st.session_state.model)

        # RAM dropdown ONLY from dataset (filtered)
        ram_options = unique_sorted_numeric(filtered, colmap["ram"])
        if not ram_options:
            ram_options = unique_sorted_numeric(df, colmap["ram"]) or [2, 4, 6, 8, 12, 16]

        # Convert to strings for nicer dropdown, but keep mapping back to int
        ram_labels = [str(r) for r in ram_options]
        if st.session_state.ram is None or str(st.session_state.ram) not in ram_labels:
            ram_index = 0
        else:
            ram_index = ram_labels.index(str(st.session_state.ram))

        st.selectbox(
            "RAM (GB)",
            ram_labels,
            index=ram_index,
            key="ram",
        )

        # Storage dropdown ONLY from dataset (filtered)
        storage_options = unique_sorted_numeric(filtered, colmap["storage"])
        if not storage_options:
            storage_options = unique_sorted_numeric(df, colmap["storage"]) or [32, 64, 128, 256, 512]

        storage_labels = [str(s) for s in storage_options]
        if st.session_state.storage is None or str(st.session_state.storage) not in storage_labels:
            storage_index = 0
        else:
            storage_index = storage_labels.index(str(st.session_state.storage))

        st.selectbox(
            "Storage (GB)",
            storage_labels,
            index=storage_index,
            key="storage",
        )

        # Color dropdown ONLY from dataset (filtered)
        color_options = unique_sorted(filtered, colmap["color"])
        if not color_options:
            color_options = colors_all

        if st.session_state.color not in color_options:
            st.session_state.color = color_options[0]

        st.selectbox(
            "Color",
            color_options,
            index=color_options.index(st.session_state.color),
            key="color",
        )

        st.radio("Unlocked?", ["Yes", "No"], horizontal=True, key="unlocked")

        st.write("")
        st.slider("Optional: your budget", min_value=0, max_value=5000, step=50, key="budget")

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

                    X = build_feature_row(
                        brand=st.session_state.brand,
                        model=st.session_state.model,
                        ram_gb=ram_val,
                        storage_gb=storage_val,
                        color=st.session_state.color,
                        unlocked_yes=(st.session_state.unlocked == "Yes"),
                        colmap=colmap,
                    )
                    pred = predict_price(model, X)

                    st.session_state.last_pred = pred
                    st.session_state.last_input = (
                        st.session_state.brand,
                        st.session_state.model,
                        ram_val,
                        storage_val,
                        st.session_state.color,
                        st.session_state.unlocked,
                        st.session_state.budget,
                    )
                    st.session_state.last_error = None
                except Exception:
                    st.session_state.last_pred = None
                    st.session_state.last_error = (
                        "We couldn’t estimate the price right now. Please try again."
                    )

        if st.session_state.last_error:
            st.error(st.session_state.last_error)

        pred = st.session_state.last_pred
        if pred is None:
            st.markdown(
                '<div class="small">Choose specs and click <b>Estimate price</b> to see the result.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.metric("Estimated price", format_currency(pred))

            last_budget = st.session_state.last_input[-1] if st.session_state.last_input else 0
            if last_budget > 0:
                if pred <= last_budget:
                    st.success(f"✅ Within your budget ({format_currency(last_budget)}).")
                else:
                    st.warning(f"⚠️ Above your budget by {format_currency(pred - last_budget)}.")

            low = pred * 0.9
            high = pred * 1.1
            st.info(f"Typical range: {format_currency(low)} to {format_currency(high)} (estimate).")

            similar = find_similar_phones(
                df=df,
                colmap=colmap,
                brand=st.session_state.last_input[0],
                ram_gb=float(st.session_state.last_input[2]),
                storage_gb=float(st.session_state.last_input[3]),
                unlocked_yes=(st.session_state.last_input[5] == "Yes"),
            )
            if not similar.empty:
                st.write("")
                st.markdown("### Similar phones in the dataset")
                st.dataframe(similar, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    with st.expander("ℹ️ About this estimator"):
        st.write(
            "This app uses a trained ML model saved in `final_price_model.joblib`. "
            "All dropdown options are loaded from your dataset (`smartphones.csv`) where possible."
        )

# =========================
# Compare Tab
# =========================
with tab_compare:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Compare two phones")
    st.markdown(
        '<div class="muted">Pick two sets of specs (all from the dataset) — compare estimated prices.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    colA, colB = st.columns(2, gap="large")

    def phone_form(prefix: str, default_brand_idx: int = 0) -> Dict[str, object]:
        # state keys
        b_key = f"{prefix}_brand"
        m_key = f"{prefix}_model"
        r_key = f"{prefix}_ram"
        s_key = f"{prefix}_storage"
        c_key = f"{prefix}_color"
        u_key = f"{prefix}_unlock"

        # init defaults
        if b_key not in st.session_state:
            st.session_state[b_key] = brands[min(default_brand_idx, len(brands) - 1)]
        if m_key not in st.session_state:
            st.session_state[m_key] = "(Select a model)"
        if r_key not in st.session_state:
            st.session_state[r_key] = None
        if s_key not in st.session_state:
            st.session_state[s_key] = None
        if c_key not in st.session_state:
            st.session_state[c_key] = colors_all[0]
        if u_key not in st.session_state:
            st.session_state[u_key] = "Yes"

        def _on_brand_change():
            st.session_state[m_key] = "(Select a model)"
            st.session_state[r_key] = None
            st.session_state[s_key] = None

        def _on_model_change():
            st.session_state[r_key] = None
            st.session_state[s_key] = None

        st.selectbox(
            f"{prefix} Brand",
            brands,
            index=brands.index(st.session_state[b_key]) if st.session_state[b_key] in brands else 0,
            key=b_key,
            on_change=_on_brand_change,
        )

        # model options filtered by brand
        mopts = ["(Select a model)"]
        if not df.empty and colmap["brand"] and colmap["model"]:
            subset = df[df[colmap["brand"]].astype(str).str.strip() == str(st.session_state[b_key]).strip()]
            mlist = sorted(subset[colmap["model"]].dropna().astype(str).str.strip().unique().tolist())
            mopts += mlist
        else:
            mopts += (all_models or [])

        st.selectbox(
            f"{prefix} Model",
            mopts,
            index=mopts.index(st.session_state[m_key]) if st.session_state[m_key] in mopts else 0,
            key=m_key,
            on_change=_on_model_change,
        )

        filtered = filter_df_for_choices(df, colmap, st.session_state[b_key], st.session_state[m_key])

        ram_opts = unique_sorted_numeric(filtered, colmap["ram"])
        if not ram_opts:
            ram_opts = unique_sorted_numeric(df, colmap["ram"]) or [2, 4, 6, 8, 12, 16]
        ram_labels = [str(v) for v in ram_opts]
        if st.session_state[r_key] is None or str(st.session_state[r_key]) not in ram_labels:
            r_index = 0
        else:
            r_index = ram_labels.index(str(st.session_state[r_key]))
        st.selectbox(f"{prefix} RAM (GB)", ram_labels, index=r_index, key=r_key)

        storage_opts = unique_sorted_numeric(filtered, colmap["storage"])
        if not storage_opts:
            storage_opts = unique_sorted_numeric(df, colmap["storage"]) or [32, 64, 128, 256, 512]
        storage_labels = [str(v) for v in storage_opts]
        if st.session_state[s_key] is None or str(st.session_state[s_key]) not in storage_labels:
            s_index = 0
        else:
            s_index = storage_labels.index(str(st.session_state[s_key]))
        st.selectbox(f"{prefix} Storage (GB)", storage_labels, index=s_index, key=s_key)

        color_opts = unique_sorted(filtered, colmap["color"])
        if not color_opts:
            color_opts = colors_all
        if st.session_state[c_key] not in color_opts:
            st.session_state[c_key] = color_opts[0]
        st.selectbox(f"{prefix} Color", color_opts, index=color_opts.index(st.session_state[c_key]), key=c_key)

        st.radio(f"{prefix} Unlocked?", ["Yes", "No"], horizontal=True, key=u_key)

        return {
            "brand": st.session_state[b_key],
            "model": st.session_state[m_key],
            "ram": st.session_state[r_key],
            "storage": st.session_state[s_key],
            "color": st.session_state[c_key],
            "unlock": st.session_state[u_key],
        }

    with colA:
        st.subheader("Phone A")
        A = phone_form("A")

    with colB:
        st.subheader("Phone B")
        B = phone_form("B", default_brand_idx=min(1, len(brands) - 1))

    st.write("")
    compare_clicked = st.button("🆚 Compare prices", use_container_width=True)

    if compare_clicked:
        def validate_phone(P: Dict[str, object]) -> Optional[str]:
            if not P["brand"] or str(P["brand"]).strip() == "":
                return "Please choose a brand."
            if not P["model"] or str(P["model"]).strip() == "" or P["model"] == "(Select a model)":
                return "Please choose a model."
            if P["ram"] is None or P["storage"] is None:
                return "Please choose RAM and storage options."
            return None

        errA = validate_phone(A)
        errB = validate_phone(B)

        if errA or errB:
            st.error(f"{'Phone A: ' + errA if errA else ''} {'Phone B: ' + errB if errB else ''}".strip())
        else:
            try:
                XA = build_feature_row(
                    A["brand"], A["model"], float(A["ram"]), float(A["storage"]),
                    A["color"], A["unlock"] == "Yes", colmap
                )
                XB = build_feature_row(
                    B["brand"], B["model"], float(B["ram"]), float(B["storage"]),
                    B["color"], B["unlock"] == "Yes", colmap
                )

                pA = predict_price(model, XA)
                pB = predict_price(model, XB)

                c1, c2, c3 = st.columns(3)
                c1.metric("Phone A", format_currency(pA))
                c2.metric("Phone B", format_currency(pB))
                c3.metric("Difference", format_currency(abs(pA - pB)))

                if pA < pB:
                    st.success("✅ Phone A is estimated to be cheaper.")
                elif pB < pA:
                    st.success("✅ Phone B is estimated to be cheaper.")
                else:
                    st.info("Both phones have the same estimated price.")

            except Exception:
                st.error("We couldn’t compare these phones right now. Please try again.")

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.caption(
    "Built with Streamlit • Predictions generated by your saved ML model (`final_price_model.joblib`) • "
    "All dropdown options loaded from your dataset (`smartphones.csv`)."
)
