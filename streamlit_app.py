"""
Streamlit dashboard: adjust scoring weights and view score distributions.
Upload CSVs in the sidebar or use default paths (test.csv, vagas-exclusivas).
Select which company/job to use for scoring; scores are recomputed for that job.
"""
import ast
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Default paths (used when no file is uploaded)
TEST_CSV = "test.csv"
VAGAS_CSV = "vagas-exclusivas-2026-02-22.csv"

SCORE_COLS = ["language_score", "role_score", "tech_score", "seniority_score"]
INFERRED_SENIORITY_COL = "inferred_seniority_score"

ENGLISH_LEVEL_MAP = {"basic": 0, "intermediate": 1, "fluent": 2}
SENIORITY_MAP = {"junior": 1, "mid": 2, "senior": 3, "staff_plus": 4}


def _parse_tech_stack(val):
    """Parse techStack from CSV (list string or comma-separated)."""
    if pd.isna(val) or val == "":
        return []
    if isinstance(val, list):
        return val
    s = str(val).strip()
    if not s:
        return []
    try:
        out = ast.literal_eval(s)
        return list(out) if isinstance(out, (list, tuple)) else [s]
    except (ValueError, SyntaxError):
        return [x.strip() for x in s.split(",") if x.strip()]


def _get_match_tech_stack(company_stack, candidate_stack, process_extract_one):
    total_score = 0
    for tech in company_stack:
        if not candidate_stack:
            continue
        _, score = process_extract_one(tech, candidate_stack)
        if score > 90:
            total_score += 1
    return total_score


def compute_scores_for_vaga(df: pd.DataFrame, vaga: pd.Series, use_inferred_seniority: bool):
    """Recompute language, role, tech, seniority scores for all profiles against one vaga."""
    from thefuzz import fuzz, process

    def extract_one(tech, stack):
        return process.extractOne(tech, stack) if stack else (None, 0)

    df = df.copy()
    # Tech stack: ensure lists
    req_tech = vaga.get("requiredTechStack")
    nice_tech = vaga.get("niceToHaveTechStack")
    if isinstance(req_tech, str):
        req_tech = [x.strip() for x in req_tech.split(",") if x.strip()]
    if isinstance(nice_tech, str):
        nice_tech = [x.strip() for x in nice_tech.split(",") if x.strip()]
    if not isinstance(req_tech, list):
        req_tech = []
    if not isinstance(nice_tech, list):
        nice_tech = []

    tech_scores = []
    for _, row in df.iterrows():
        cand_stack = _parse_tech_stack(row.get("techStack"))
        score_req = _get_match_tech_stack(req_tech, cand_stack, extract_one) if req_tech else 0
        score_nice = _get_match_tech_stack(nice_tech, cand_stack, extract_one) if nice_tech else 0
        denom_req = len(req_tech) or 1
        denom_nice = len(nice_tech) or 1
        tech_scores.append(score_req / denom_req * 0.9 + score_nice / denom_nice * 0.1)
    df["tech_score"] = tech_scores

    title = str(vaga.get("title", ""))
    df["role_score"] = df["jobRole"].fillna("").apply(lambda x: fuzz.partial_ratio(title, str(x)) / 100)

    req_eng = str(vaga.get("requiredEnglishLevel", "fluent")).strip().lower()
    eng_min = ENGLISH_LEVEL_MAP.get(req_eng, 2)
    df["language_score"] = (df["englishLevel"].fillna("").str.lower().map(ENGLISH_LEVEL_MAP).fillna(0) >= eng_min) * 2 - 1

    exp_level = str(vaga.get("experienceLevel", "senior")).strip().lower()
    target_sen = SENIORITY_MAP.get(exp_level, 3)
    seniority_col = "inferred_seniority" if use_inferred_seniority else "seniority"
    if seniority_col not in df.columns:
        seniority_col = "seniority"
    df["seniority_score"] = 1 - abs(df[seniority_col].fillna("").str.lower().map(SENIORITY_MAP).fillna(0) - target_sen) * (2 / 3)
    if use_inferred_seniority and "inferred_seniority_score" not in df.columns:
        df["inferred_seniority_score"] = df["seniority_score"]
    elif "inferred_seniority" in df.columns:
        df["inferred_seniority_score"] = 1 - abs(df["inferred_seniority"].fillna("").str.lower().map(SENIORITY_MAP).fillna(0) - target_sen) * (2 / 3)
    return df


def load_data(profiles_file=None, vagas_file=None):
    """Load profiles and vagas from uploaded files or default paths."""
    if profiles_file is not None:
        df = pd.read_csv(profiles_file)
    else:
        try:
            df = pd.read_csv(TEST_CSV)
        except FileNotFoundError:
            st.error(f"Default profiles file not found: {TEST_CSV}. Upload a CSV in the sidebar.")
            return None, None
    if vagas_file is not None:
        vagas = pd.read_csv(vagas_file)
    else:
        try:
            vagas = pd.read_csv(VAGAS_CSV)
        except FileNotFoundError:
            st.error(f"Default vagas file not found: {VAGAS_CSV}. Upload a CSV in the sidebar.")
            return None, None
    if vagas.empty:
        st.error("Vagas CSV is empty.")
        return None, None
    return df, vagas


def combined_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted combination: (weighted_avg + 1) / 2 -> [0, 1]."""
    cols = [c for c in weights if c in df.columns]
    w = {c: weights[c] for c in cols}
    total_w = sum(w.values())
    if total_w == 0:
        return pd.Series(0.5, index=df.index)
    weighted = sum(df[c] * w[c] for c in cols) / total_w
    return (weighted + 1) / 2


def main():
    st.set_page_config(page_title="Score weights dashboard", layout="wide")
    st.title("Matching score weights dashboard")

    # CSV inputs: upload or use defaults
    st.sidebar.subheader("Data")
    profiles_upload = st.sidebar.file_uploader(
        "Profiles (scores) CSV",
        type=["csv"],
        help="Precomputed scores per profile. If not set, uses default path.",
    )
    vagas_upload = st.sidebar.file_uploader(
        "Vagas CSV",
        type=["csv"],
        help="Jobs list for context. If not set, uses default path.",
    )
    df, vagas = load_data(profiles_upload, vagas_upload)
    if df is None or vagas is None:
        return

    # Prepare vagas: split tech stacks for recomputation
    vagas = vagas.copy()
    for col in ("requiredTechStack", "niceToHaveTechStack"):
        if col in vagas.columns and vagas[col].dtype == object:
            vagas[col] = vagas[col].fillna("").apply(lambda s: [x.strip() for x in str(s).split(",") if x.strip()] if s else [])

    # Company / job selector
    st.sidebar.subheader("Job")
    options = []
    for i, row in vagas.iterrows():
        company = row.get("companyName", "")
        title = row.get("title", "")
        options.append(f"{company} — {title}")
    job_index = st.sidebar.selectbox(
        "Company / job for scoring",
        range(len(vagas)),
        format_func=lambda i: options[i],
        index=0,
    )
    vaga = vagas.iloc[job_index]
    use_inferred = st.sidebar.checkbox("Use inferred seniority score", value=False)

    # Recompute scores for the selected job (requires techStack, jobRole, englishLevel, seniority)
    needed = ["techStack", "jobRole", "englishLevel"]
    if not use_inferred:
        needed.append("seniority")
    if use_inferred and "inferred_seniority" not in df.columns:
        use_inferred = False
    if all(c in df.columns for c in needed):
        df = compute_scores_for_vaga(df, vaga, use_inferred)
    # else: keep precomputed scores (e.g. from test.csv for first job only)

    st.caption(f"Scoring for: **{vaga['title']}** @ {vaga['companyName']} ({vaga['experienceLevel']})")

    # Ensure numeric score columns
    for c in SCORE_COLS + [INFERRED_SENIORITY_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    seniority_col = INFERRED_SENIORITY_COL if use_inferred else "seniority_score"
    if seniority_col not in df.columns:
        seniority_col = "seniority_score"

    # Weights
    st.sidebar.subheader("Weights")
    weight_keys = ["language_score", "role_score", "tech_score", seniority_col]
    weights = {}
    for k in weight_keys:
        if k not in df.columns:
            continue
        weights[k] = st.sidebar.slider(k.replace("_", " ").title(), 0.0, 5.0, 1.0, 0.25)

    if not weights:
        st.error("No score columns found in data.")
        return

    df = df.copy()
    df["combined_score"] = combined_score(df, weights)

    # Distributions: one histogram per component + combined
    st.header("Distributions")
    dist_cols = [c for c in weight_keys if c in df.columns] + ["combined_score"]
    n = len(dist_cols)
    fig = make_subplots(
        rows=2,
        cols=(n + 1) // 2,
        subplot_titles=[c.replace("_", " ").title() for c in dist_cols],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )
    for i, col in enumerate(dist_cols):
        row, c = divmod(i, (n + 1) // 2)
        row, c = row + 1, c + 1
        fig.add_trace(
            go.Histogram(x=df[col].dropna(), nbinsx=30, name=col),
            row=row,
            col=c,
        )
    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(t=40),
    )
    fig.update_xaxes(title_text="Score")
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    st.subheader("Score summary")
    summary = df[dist_cols].describe().loc[["mean", "std", "min", "max"]]
    st.dataframe(summary.style.format("{:.3f}"), use_container_width=True)

    # Combined score distribution (standalone)
    st.subheader("Combined score distribution")
    fig2 = px.histogram(
        df,
        x="combined_score",
        nbins=40,
        labels={"combined_score": "Combined score"},
    )
    fig2.update_layout(height=350, margin=dict(t=20))
    st.plotly_chart(fig2, use_container_width=True)

    # Top / bottom table
    st.subheader("Ranking (by combined score)")
    n_show = st.slider("Number of rows", 5, 100, 20)
    cols_display = ["jobRole", "currentCompany", "seniority", "combined_score"] + [
        c for c in dist_cols if c != "combined_score"
    ]
    cols_display = [c for c in cols_display if c in df.columns]
    ranked = df.nlargest(n_show, "combined_score")[cols_display].reset_index(drop=True)
    ranked.index = ranked.index + 1
    st.dataframe(ranked.style.format({c: "{:.3f}" for c in dist_cols if c in ranked.columns}), use_container_width=True)


if __name__ == "__main__":
    main()
