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
MODEL_PATH = "final_price_model.joblib"
DATA_PATH = "smartphones.csv"

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


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)

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
    x = sorted(set(int(v) for v in x.tolist()))
    return x


def filter_df_for_choices(df: pd.DataFrame, colmap: Dict[str, str], brand: str, model: str) -> pd.DataFrame:
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
        work = work[work[ucol].astype(str).apply(normalize_yes_no).str.lower() == want.lower()]

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


def nice_phone_name(row: pd.Series, colmap: Dict[str, str]) -> str:
    b = str(row.get(colmap.get("brand") or "Brand", "")).strip()
    m = str(row.get(colmap.get("model") or "Model", "")).strip()
    if b and m:
        return f"{b} {m}"
    return (m or b or "Phone")


# IMPORTANT: Do NOT cache this (it takes model object; Streamlit can’t hash it reliably)
def predict_for_rows(df_subset: pd.DataFrame, colmap: Dict[str, str], model_obj) -> pd.DataFrame:
    if df_subset.empty:
        return df_subset

    bcol = colmap.get("brand")
    mcol = colmap.get("model")
    ccol = colmap.get("color")
    rcol = colmap.get("ram")
    scol = colmap.get("storage")
    ucol = colmap.get("unlocked")
    pcol = colmap.get("price")

    needed = [bcol, mcol, ccol, rcol, scol, ucol]
    if any(col is None for col in needed):
        return pd.DataFrame()

    work = df_subset.copy()

    work[rcol] = pd.to_numeric(work[rcol], errors="coerce")
    work[scol] = pd.to_numeric(work[scol], errors="coerce")

    work["_unlocked_norm"] = work[ucol].apply(normalize_yes_no)
    work["_free_bin"] = (work["_unlocked_norm"].str.lower() == "yes").astype(int)

    work = work.dropna(subset=[rcol, scol, bcol, mcol, ccol])
    if work.empty:
        return pd.DataFrame()

    X = pd.DataFrame(
        {
            bcol: work[bcol].astype(str),
            mcol: work[mcol].astype(str),
            ccol: work[ccol].astype(str),
            rcol: work[rcol].astype(float),
            scol: work[scol].astype(float),
            ucol: work["_unlocked_norm"].astype(str),
        }
    )
    X["Free_bin"] = work["_free_bin"].astype(int)
    X["RAM_x_Storage"] = (work[rcol].astype(float) * work[scol].astype(float)).astype(float)
    X["Storage_per_RAM"] = (work[scol].astype(float) / work[rcol].replace(0, np.nan).astype(float)).astype(float)

    preds = model_obj.predict(X)
    work["_predicted_price"] = preds.astype(float)

    if pcol and pcol in work.columns:
        work[pcol] = pd.to_numeric(work[pcol], errors="coerce")
        work["_value_gap"] = work["_predicted_price"] - work[pcol]
    else:
        work["_value_gap"] = np.nan

    return work


def minmax01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().all():
        return pd.Series([0.0] * len(s), index=s.index)
    lo = float(np.nanmin(s.values))
    hi = float(np.nanmax(s.values))
    if hi - lo == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


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
.block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1200px; }
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

if model is None:
    st.error(f"Model file not found: `{MODEL_PATH}`. Make sure it is in the same folder as `app.py`.")
    st.stop()

if df.empty:
    st.warning(
        f"Dataset file not found (or empty): `{DATA_PATH}`. "
        f"Dropdown lists will use safe fallback options."
    )

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
    st.session_state.last_input = None
    st.session_state.last_error = None

# Recommendations state
if "rec_brand_selected" not in st.session_state:
    st.session_state.rec_brand_selected = []  # multiselect dropdown
if "rec_budget_min" not in st.session_state:
    st.session_state.rec_budget_min = 0
if "rec_budget_max" not in st.session_state:
    st.session_state.rec_budget_max = 1500
if "rec_sim" not in st.session_state:
    st.session_state.rec_sim = "Any"
if "rec_min_ram" not in st.session_state:
    st.session_state.rec_min_ram = "Any"
if "rec_min_storage" not in st.session_state:
    st.session_state.rec_min_storage = "Any"
if "rec_top_n" not in st.session_state:
    st.session_state.rec_top_n = 10

# Ranking checkboxes state
RANK_CHOICES = [
    "Best overall",
    "Cheapest (predicted)",
    "Highest value (best deal)",
    "Most storage",
    "Most RAM",
]
for k in ["rank_overall", "rank_cheap", "rank_value", "rank_storage", "rank_ram"]:
    if k not in st.session_state:
        st.session_state[k] = False
if not any([st.session_state.rank_overall, st.session_state.rank_cheap, st.session_state.rank_value, st.session_state.rank_storage, st.session_state.rank_ram]):
    st.session_state.rank_overall = True


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

tab_predict, tab_compare, tab_reco = st.tabs(["🔮 Estimate Price", "🆚 Compare Phones", "⭐ Recommendations"])

# =========================
# TAB 1: Estimate Price
# =========================
with tab_predict:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## Enter phone details")
        st.markdown('<div class="muted">All options come from your dataset. The prediction comes from your trained model.</div>', unsafe_allow_html=True)
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
            subset = df[df[colmap["brand"]].astype(str).str.strip() == str(st.session_state.brand).strip()]
            mlist = sorted(subset[colmap["model"]].dropna().astype(str).str.strip().unique().tolist())
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

        filtered = filter_df_for_choices(df, colmap, st.session_state.brand, st.session_state.model)

        ram_options = unique_sorted_numeric(filtered, colmap["ram"]) or unique_sorted_numeric(df, colmap["ram"]) or [2, 4, 6, 8, 12, 16]
        ram_labels = [str(r) for r in ram_options]
        ram_index = 0 if (st.session_state.ram is None or str(st.session_state.ram) not in ram_labels) else ram_labels.index(str(st.session_state.ram))

        field_label("RAM (Memory)", "More RAM helps multitasking (GB).")
        st.selectbox("RAM", ram_labels, index=ram_index, key="ram", label_visibility="collapsed")

        storage_options = unique_sorted_numeric(filtered, colmap["storage"]) or unique_sorted_numeric(df, colmap["storage"]) or [32, 64, 128, 256, 512]
        storage_labels = [str(s) for s in storage_options]
        storage_index = 0 if (st.session_state.storage is None or str(st.session_state.storage) not in storage_labels) else storage_labels.index(str(st.session_state.storage))

        field_label("Storage", "How much space you have for apps and photos (GB).")
        st.selectbox("Storage", storage_labels, index=storage_index, key="storage", label_visibility="collapsed")

        color_options = unique_sorted(filtered, colmap["color"]) or colors_all
        if st.session_state.color not in color_options:
            st.session_state.color = color_options[0]

        field_label("Color", "Choose the phone color.")
        st.selectbox("Color", color_options, index=color_options.index(st.session_state.color), key="color", label_visibility="collapsed")

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
                    st.session_state.last_input = (
                        st.session_state.brand,
                        st.session_state.model,
                        ram_val,
                        storage_val,
                        st.session_state.color,
                        st.session_state.sim_status,
                        st.session_state.budget,
                    )
                    st.session_state.last_error = None
                except Exception:
                    st.session_state.last_pred = None
                    st.session_state.last_error = "We couldn’t estimate the price right now. Please try again."

        if st.session_state.last_error:
            st.error(st.session_state.last_error)

        pred = st.session_state.last_pred
        if pred is None:
            st.markdown('<div class="small">Choose specs and click <b>Estimate price</b> to see the result.</div>', unsafe_allow_html=True)
        else:
            st.metric("Estimated price", format_currency(pred))

            last_budget = st.session_state.last_input[-1] if st.session_state.last_input else 0
            if last_budget > 0:
                if pred <= last_budget:
                    st.success(f"Within your budget ({format_currency(last_budget)}).")
                else:
                    st.warning(f"Above your budget by {format_currency(pred - last_budget)}.")

            low = pred * 0.9
            high = pred * 1.1
            st.info(f"Typical range: {format_currency(low)} to {format_currency(high)} (estimate).")

            similar = find_similar_phones(
                df=df,
                colmap=colmap,
                brand=st.session_state.last_input[0],
                ram_gb=float(st.session_state.last_input[2]),
                storage_gb=float(st.session_state.last_input[3]),
                unlocked_yes=unlocked_label_to_bool(st.session_state.last_input[5]),
            )
            if not similar.empty:
                st.write("")
                st.markdown("### Similar phones in the dataset")
                st.dataframe(similar, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TAB 2: Compare Phones
# =========================
with tab_compare:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Compare two phones")
    st.markdown('<div class="muted">Pick two sets of specs (from the dataset) — compare predicted prices.</div>', unsafe_allow_html=True)
    st.write("")

    colA, colB = st.columns(2, gap="large")

    def phone_form(prefix: str, default_brand_idx: int = 0) -> Dict[str, object]:
        b_key = f"{prefix}_brand"
        m_key = f"{prefix}_model"
        r_key = f"{prefix}_ram"
        s_key = f"{prefix}_storage"
        c_key = f"{prefix}_color"
        u_key = f"{prefix}_sim"

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
            st.session_state[u_key] = "Yes (Any SIM)"

        def _on_brand_change():
            st.session_state[m_key] = "(Select a model)"
            st.session_state[r_key] = None
            st.session_state[s_key] = None

        def _on_model_change():
            st.session_state[r_key] = None
            st.session_state[s_key] = None

        field_label("Brand", "Choose the smartphone brand.")
        st.selectbox(
            "Brand",
            brands,
            index=brands.index(st.session_state[b_key]) if st.session_state[b_key] in brands else 0,
            key=b_key,
            on_change=_on_brand_change,
            label_visibility="collapsed",
        )

        mopts = ["(Select a model)"]
        if not df.empty and colmap["brand"] and colmap["model"]:
            subset = df[df[colmap["brand"]].astype(str).str.strip() == str(st.session_state[b_key]).strip()]
            mlist = sorted(subset[colmap["model"]].dropna().astype(str).str.strip().unique().tolist())
            mopts += mlist
        else:
            mopts += (all_models or [])

        field_label("Model", "Choose the model for the selected brand.")
        st.selectbox(
            "Model",
            mopts,
            index=mopts.index(st.session_state[m_key]) if st.session_state[m_key] in mopts else 0,
            key=m_key,
            on_change=_on_model_change,
            label_visibility="collapsed",
        )

        filtered_local = filter_df_for_choices(df, colmap, st.session_state[b_key], st.session_state[m_key])

        ram_opts = unique_sorted_numeric(filtered_local, colmap["ram"]) or unique_sorted_numeric(df, colmap["ram"]) or [2, 4, 6, 8, 12, 16]
        ram_labels = [str(v) for v in ram_opts]
        r_index = 0 if (st.session_state[r_key] is None or str(st.session_state[r_key]) not in ram_labels) else ram_labels.index(str(st.session_state[r_key]))

        field_label("RAM (Memory)", "More RAM helps multitasking (GB).")
        st.selectbox("RAM", ram_labels, index=r_index, key=r_key, label_visibility="collapsed")

        storage_opts = unique_sorted_numeric(filtered_local, colmap["storage"]) or unique_sorted_numeric(df, colmap["storage"]) or [32, 64, 128, 256, 512]
        storage_labels = [str(v) for v in storage_opts]
        s_index = 0 if (st.session_state[s_key] is None or str(st.session_state[s_key]) not in storage_labels) else storage_labels.index(str(st.session_state[s_key]))

        field_label("Storage", "Space for apps/photos (GB).")
        st.selectbox("Storage", storage_labels, index=s_index, key=s_key, label_visibility="collapsed")

        color_opts = unique_sorted(filtered_local, colmap["color"]) or colors_all
        if st.session_state[c_key] not in color_opts:
            st.session_state[c_key] = color_opts[0]

        field_label("Color", "Choose the phone color.")
        st.selectbox("Color", color_opts, index=color_opts.index(st.session_state[c_key]), key=c_key, label_visibility="collapsed")

        field_label("SIM Status", "Works with any SIM (unlocked) or locked to a telco.")
        st.radio(
            "Works with any SIM card?",
            ["Yes (Any SIM)", "No (Locked to a telco)"],
            horizontal=True,
            key=u_key,
            label_visibility="collapsed",
        )

        return {
            "brand": st.session_state[b_key],
            "model": st.session_state[m_key],
            "ram": st.session_state[r_key],
            "storage": st.session_state[s_key],
            "color": st.session_state[c_key],
            "sim": st.session_state[u_key],
        }

    with colA:
        st.subheader("Phone A")
        A = phone_form("A")

    with colB:
        st.subheader("Phone B")
        B = phone_form("B", default_brand_idx=min(1, len(brands) - 1))

    st.write("")
    compare_clicked = st.button("Compare prices", use_container_width=True)

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
                XA = build_feature_row(A["brand"], A["model"], float(A["ram"]), float(A["storage"]), A["color"], unlocked_label_to_bool(A["sim"]), colmap)
                XB = build_feature_row(B["brand"], B["model"], float(B["ram"]), float(B["storage"]), B["color"], unlocked_label_to_bool(B["sim"]), colmap)

                pA = predict_price(model, XA)
                pB = predict_price(model, XB)

                c1, c2, c3 = st.columns(3)
                c1.metric("Phone A", format_currency(pA))
                c2.metric("Phone B", format_currency(pB))
                c3.metric("Difference", format_currency(abs(pA - pB)))

                if pA < pB:
                    st.success("Phone A is estimated to be cheaper.")
                elif pB < pA:
                    st.success("Phone B is estimated to be cheaper.")
                else:
                    st.info("Both phones have the same estimated price.")
            except Exception:
                st.error("We couldn’t compare these phones right now. Please try again.")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TAB 3: Recommendations
# =========================
with tab_reco:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Recommended phones for you")
    st.markdown('<div class="muted">We run your trained model on phones in your dataset, then rank results based on what you pick.</div>', unsafe_allow_html=True)
    st.write("")

    # --- budget range derived from dataset price (if available) ---
    pcol = colmap.get("price")
    if not df.empty and pcol and pcol in df.columns:
        pvals = pd.to_numeric(df[pcol], errors="coerce").dropna()
        ds_min = int(max(0, np.floor(pvals.min()))) if not pvals.empty else 0
        ds_max = int(max(1000, np.ceil(pvals.max()))) if not pvals.empty else 5000
    else:
        ds_min, ds_max = 0, 5000

    f1, f2, f3 = st.columns([1.1, 1.1, 1.0], gap="large")

    # ---------- Brand preference: multiselect dropdown ----------
    with f1:
        field_label("Brand preference", "Pick one or more brands. Leave empty to allow any brand.")
        st.multiselect(
            "Brands",
            options=brands,
            default=st.session_state.rec_brand_selected,
            key="rec_brand_selected",
            label_visibility="collapsed",
            help="You can choose multiple brands here.",
        )

        st.write("")
        field_label("SIM preference", "Do you need it to work with any SIM?")
        st.selectbox(
            "SIM preference",
            ["Any", "Yes (Any SIM)", "No (Locked to a telco)"],
            key="rec_sim",
            label_visibility="collapsed",
        )

    # ---------- Budget + minimum specs ----------
    with f2:
        field_label("Budget range", "Show phones whose model-predicted price falls within your budget.")
        bmin, bmax = st.slider(
            "Budget range",
            min_value=int(ds_min),
            max_value=int(max(ds_max, ds_min + 100)),
            value=(int(st.session_state.rec_budget_min), int(st.session_state.rec_budget_max)),
            step=50,
            label_visibility="collapsed",
        )
        st.session_state.rec_budget_min = bmin
        st.session_state.rec_budget_max = bmax

        ram_all = unique_sorted_numeric(df, colmap["ram"]) or [2, 4, 6, 8, 12, 16]
        sto_all = unique_sorted_numeric(df, colmap["storage"]) or [32, 64, 128, 256, 512]

        field_label("Minimum RAM", "Filter out phones below this RAM (optional).")
        st.selectbox("Minimum RAM", ["Any"] + [str(v) for v in ram_all], key="rec_min_ram", label_visibility="collapsed")

        field_label("Minimum Storage", "Filter out phones below this storage (optional).")
        st.selectbox("Minimum Storage", ["Any"] + [str(v) for v in sto_all], key="rec_min_storage", label_visibility="collapsed")

    # ---------- Ranking style (checkbox multi-select) ----------
    with f3:
        field_label("Ranking style", "Tick one or more ranking options. We combine them into one score.")

        # show checkboxes
        st.checkbox("Best overall", key="rank_overall")
        st.checkbox("Cheapest (predicted)", key="rank_cheap")
        st.checkbox("Highest value (best deal)", key="rank_value")
        st.checkbox("Most storage", key="rank_storage")
        st.checkbox("Most RAM", key="rank_ram")

        # If user unticks everything, we won't crash—we will show a friendly message on run
        st.write("")
        field_label("Number of results", "How many phones to show.")
        st.selectbox("Number of results", [5, 10, 15, 20], key="rec_top_n", label_visibility="collapsed")

        st.write("")
        run_reco = st.button("⭐ Get recommendations", use_container_width=True)

    st.write("")

    if run_reco:
        # Friendly validation FIRST (prevents confusing errors)
        missing_cols = [k for k in ["brand", "model", "color", "ram", "storage", "unlocked"] if colmap.get(k) is None]

        if df.empty:
            st.error("We can’t load your dataset right now. Please make sure `smartphones.csv` is in the app folder.")
        elif missing_cols:
            st.error(
                "Recommendations are unavailable because we couldn’t detect required columns in your dataset. "
                "Please ensure your CSV includes Brand, Model, RAM, Storage, Color, and Unlocked/Free."
            )
        else:
            # Ranking selection validation
            any_rank = any([
                st.session_state.rank_overall,
                st.session_state.rank_cheap,
                st.session_state.rank_value,
                st.session_state.rank_storage,
                st.session_state.rank_ram,
            ])
            if not any_rank:
                st.error("Please choose at least 1 ranking option (for example: Best overall).")
            else:
                work = df.copy()

                bcol = colmap["brand"]
                mcol = colmap["model"]
                ccol = colmap["color"]
                rcol = colmap["ram"]
                scol = colmap["storage"]
                ucol = colmap["unlocked"]

                # Brand filter: if user selected brands, filter; if empty, allow any brand
                selected_brands = [str(x).strip() for x in (st.session_state.rec_brand_selected or [])]
                if selected_brands:
                    work = work[work[bcol].astype(str).str.strip().isin(selected_brands)]

                # Minimum spec filters
                work[rcol] = pd.to_numeric(work[rcol], errors="coerce")
                work[scol] = pd.to_numeric(work[scol], errors="coerce")

                min_ram = None if st.session_state.rec_min_ram == "Any" else safe_float(st.session_state.rec_min_ram, np.nan)
                min_sto = None if st.session_state.rec_min_storage == "Any" else safe_float(st.session_state.rec_min_storage, np.nan)

                if min_ram is not None and not np.isnan(min_ram):
                    work = work[work[rcol] >= float(min_ram)]
                if min_sto is not None and not np.isnan(min_sto):
                    work = work[work[scol] >= float(min_sto)]

                # SIM filter
                if st.session_state.rec_sim != "Any":
                    want_yes = unlocked_label_to_bool(st.session_state.rec_sim)
                    want = "Yes" if want_yes else "No"
                    work = work[work[ucol].apply(normalize_yes_no).str.lower() == want.lower()]

                work = work.dropna(subset=[bcol, mcol, ccol, rcol, scol, ucol]).copy()

                if work.empty:
                    # Very important: no traceback, just clear guidance
                    st.warning(
                        "No phones matched your filters.\n\n"
                        "Try one of these:\n"
                        "- Clear the brand selection (allow any brand)\n"
                        "- Widen your budget range\n"
                        "- Set Minimum RAM/Storage to 'Any'\n"
                        "- Set SIM preference to 'Any'"
                    )
                else:
                    if len(work) > 4000:
                        work = work.sample(4000, random_state=42)

                    with st.spinner("Running the model to generate recommendations..."):
                        scored = predict_for_rows(work, colmap, model)

                    if scored.empty or "_predicted_price" not in scored.columns:
                        st.error("We couldn’t generate recommendations with the current dataset/model. Please check your dataset columns.")
                    else:
                        # Budget filter uses model prediction
                        scored = scored[
                            (scored["_predicted_price"] >= float(st.session_state.rec_budget_min))
                            & (scored["_predicted_price"] <= float(st.session_state.rec_budget_max))
                        ]

                        if scored.empty:
                            st.warning("No phones matched your budget range. Try widening your budget.")
                        else:
                            display = pd.DataFrame(
                                {
                                    "Phone": scored.apply(lambda r: nice_phone_name(r, colmap), axis=1),
                                    "Predicted price": scored["_predicted_price"].astype(float),
                                    "RAM (GB)": pd.to_numeric(scored[rcol], errors="coerce"),
                                    "Storage (GB)": pd.to_numeric(scored[scol], errors="coerce"),
                                    "Color": scored[ccol].astype(str),
                                    "Works with any SIM": scored[ucol].apply(normalize_yes_no).map(
                                        lambda x: "Yes" if str(x).lower() == "yes" else "No"
                                    ),
                                }
                            )

                            has_actual = colmap.get("price") and colmap["price"] in scored.columns
                            if has_actual:
                                actual = pd.to_numeric(scored[colmap["price"]], errors="coerce")
                                display["Dataset price"] = actual
                                display["Deal score"] = scored["_value_gap"].astype(float)

                            # ranking components
                            cheap_score = 1.0 - minmax01(display["Predicted price"])
                            ram_score = minmax01(display["RAM (GB)"])
                            sto_score = minmax01(display["Storage (GB)"])
                            if "Deal score" in display.columns and display["Deal score"].notna().any():
                                value_score = minmax01(display["Deal score"])
                            else:
                                value_score = pd.Series([0.0] * len(display), index=display.index)

                            overall_score = (
                                0.45 * cheap_score
                                + 0.20 * ram_score
                                + 0.20 * sto_score
                                + 0.15 * value_score
                            )

                            score = pd.Series([0.0] * len(display), index=display.index)
                            weights_used = 0

                            if st.session_state.rank_overall:
                                score += overall_score
                                weights_used += 1
                            if st.session_state.rank_cheap:
                                score += cheap_score
                                weights_used += 1
                            if st.session_state.rank_value:
                                score += value_score
                                weights_used += 1
                            if st.session_state.rank_storage:
                                score += sto_score
                                weights_used += 1
                            if st.session_state.rank_ram:
                                score += ram_score
                                weights_used += 1

                            score = score / float(weights_used) if weights_used > 0 else overall_score
                            display["Recommendation score"] = score.astype(float)

                            display = display.sort_values(["Recommendation score", "Predicted price"], ascending=[False, True])
                            display = display.head(int(st.session_state.rec_top_n)).reset_index(drop=True)

                            st.write("")
                            st.markdown("### Your recommended phones")
                            st.dataframe(
                                display,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Predicted price": st.column_config.NumberColumn(format="$%.2f"),
                                    "Dataset price": st.column_config.NumberColumn(format="$%.2f") if "Dataset price" in display.columns else None,
                                    "Recommendation score": st.column_config.NumberColumn(format="%.3f"),
                                    "Deal score": st.column_config.NumberColumn(format="%.2f") if "Deal score" in display.columns else None,
                                },
                            )

                            st.info(
                                "This list is model-driven: we run your trained model on phones in your dataset, "
                                "then rank them using the ranking options you selected."
                            )

                            csv = display.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download recommendations (CSV)",
                                data=csv,
                                file_name="phone_recommendations.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.caption(
    "Built with Streamlit • Predictions and recommendations are generated by your ML model (`final_price_model.joblib`) • "
    "All dropdown options come from your dataset (`smartphones.csv`)."
)
