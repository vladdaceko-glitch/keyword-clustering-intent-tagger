import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import openpyxl
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Editable Brand List, Niche Markers & Intent Modifiers
# ---------------------------------------------------------
BRANDS = [
    "midjourney", "testfit", "archistar", "hypar", "autodesk",
    "adobe firefly", "chaos veras", "casetext", "co-counsel",
    "harvey", "robin ai", "magic school", "curipod", "quizizz",
    "canva", "chatgpt"
]

# Note: Marker lists below are configured for my current topics and must be edited for other niches.
NICHE_MARKERS = {
    "teachers": ["teacher", "teachers", "lesson", "classroom", "student", "students", "educator", "educators", "grading", "quiz", "school"],
    "architects": ["architect", "architects", "architecture", "architectural", "floor plan", "bim", "rendering", "interior design", "building"],
    "lawyers": ["legal", "lawyer", "lawyers", "law firm", "law firms", "paralegal", "paralegals", "contract", "contracts", "counsel"],
    "accountants": ["accountant", "accountants", "accounting", "bookkeeping", "bookkeeper", "tax", "taxes", "invoice", "invoicing", "audit"],
    "real estate": ["real estate", "realtor", "realtors", "listing", "listings", "property", "mortgage", "broker", "brokers"]
}

INTENT_MODIFIERS = {
    "transactional": ["buy", "price", "pricing", "free trial", "discount", "download"],
    "commercial": ["best", "top", "review", "vs", "alternatives", "comparison", "recommended", "popular", "picks", "ratings", "rated", "free"],
    "informational": ["how", "what", "why", "guide", "tutorial", "examples", "for beginners"],
    "navigational": ["login", "sign in", "official site", "official website"]
}

# Threshold constants for 'check manually' note rule
NOTE_GAP_THRESHOLD = 0.05
NOTE_LOW_SIMILARITY_THRESHOLD = 0.65

# Default articles prefilled in the sidebar
DEFAULT_ARTICLES = [
    "ChatGPT for Accountants: 10 Practical Ways to Work Smarter in 2026 | chatgpt for accountants | https://workflowaihub.blogspot.com/2026/06/chatgpt-for-accountants-2026.html",
    "10 Best AI Tools for Real Estate Agents in 2026 | ai tools for real estate agents | https://workflowaihub.blogspot.com/2026/06/best-ai-tools-for-real-estate-agents-2026.html",
    "Best AI Tools for Architects in 2026 (Tested & Ranked) | ai tools for architects | https://workflowaihub.blogspot.com/2026/06/best-ai-tools-for-architects-2026.html",
    "ChatGPT for Lawyers: 10 Practical Ways to Work Smarter in 2026 | chatgpt for lawyers | https://workflowaihub.blogspot.com/2026/06/chatgpt-for-lawyers-2026.html",
    "Best AI Tools for Lawyers in 2026 (Ranked by Practice Type) | ai tools for lawyers | https://workflowaihub.blogspot.com/2026/06/best-ai-tools-for-lawyers-2026.html",
    "ChatGPT for Real Estate Agents: 10 Practical Ways to Close More Deals in 2026 | chatgpt for real estate agents | https://workflowaihub.blogspot.com/2026/06/chatgpt-for-real-estate-agents.html",
    "Best AI Tools for Accountants in 2026 | ai tools for accountants | https://workflowaihub.blogspot.com/2026/05/best-ai-tools-for-accountants.html",
    "ChatGPT for Teachers: 15 Practical Ways to Save Time in 2026 | chatgpt for teachers | https://workflowaihub.blogspot.com/2026/05/chatgpt-for-teachers.html",
    "Best AI Tools for Teachers in 2026 (Tested & Ranked) | ai tools for teachers | https://workflowaihub.blogspot.com/2026/05/best-ai-tools-for-teachers.html"
]

def find_brands(keyword: str) -> set[str]:
    kw_lower = keyword.lower()
    return {b for b in BRANDS if b.lower() in kw_lower}

def find_niches(keyword: str) -> set[str]:
    kw_lower = keyword.lower()
    found = set()
    for niche, markers in NICHE_MARKERS.items():
        for m in markers:
            if re.search(r"\b" + re.escape(m.lower()) + r"\b", kw_lower):
                found.add(niche)
                break
    return found

def detect_keyword_intent(kw: str) -> str:
    kw_lower = kw.lower().strip()

    def has_phrase(phrase: str) -> bool:
        return bool(re.search(r"\b" + re.escape(phrase.lower()) + r"\b", kw_lower))

    # Navigational: exact brand name or explicit navigational phrase
    if kw_lower in {b.lower() for b in BRANDS}:
        return "navigational"
    for mod in INTENT_MODIFIERS["navigational"]:
        if has_phrase(mod):
            return "navigational"

    for mod in INTENT_MODIFIERS["transactional"]:
        if has_phrase(mod):
            return "transactional"
    for mod in INTENT_MODIFIERS["commercial"]:
        if has_phrase(mod):
            return "commercial"
    for mod in INTENT_MODIFIERS["informational"]:
        if has_phrase(mod):
            return "informational"

    return "informational"

def detect_cluster_intent(intents: list[str]) -> str:
    if not intents:
        return "informational"
    counts = pd.Series(intents).value_counts()
    top_intent = counts.index[0]
    top_share = counts.iloc[0] / len(intents)
    if top_share >= 0.60:
        return top_intent
    return "mixed"

def compute_cluster_note(cluster_intent: str, action: str, similarity: float, second_similarity: float) -> str:
    # Rule: for 'mixed' intent always check manually; for matched articles, also check if gap < 0.05 or similarity < 0.65
    if cluster_intent == "mixed":
        return "check manually"

    if action == "Добавить в существующую":
        is_close_gap = ((similarity - second_similarity) < NOTE_GAP_THRESHOLD)
        is_low_similarity = (similarity < NOTE_LOW_SIMILARITY_THRESHOLD)
        if is_close_gap or is_low_similarity:
            return "check manually"

    return ""

# ---------------------------------------------------------
# Configuration & Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Keyword Clustering and Intent Tagger",
    layout="wide"
)

st.title("Keyword Clustering and Intent Tagger")
st.caption("Internal SEO tool for grouping keywords, classifying search intent, and matching existing content.")

# ---------------------------------------------------------
# Sidebar: Existing Articles & Settings
# ---------------------------------------------------------
st.sidebar.header("1. Existing Articles")
default_articles_text = "\n".join(DEFAULT_ARTICLES) if isinstance(DEFAULT_ARTICLES, list) else DEFAULT_ARTICLES.strip()

articles_text = st.sidebar.text_area(
    "Paste articles (Format: title | main_keyword | url):",
    value=default_articles_text,
    height=200,
    help="One article per line. Used in Step 4 to match clusters to existing content."
)

st.sidebar.header("2. Pipeline Settings")
cluster_threshold = st.sidebar.slider(
    "Cluster Distance Threshold:",
    min_value=0.10,
    max_value=0.80,
    value=0.45,
    step=0.05,
    help="Distance limit for AgglomerativeClustering. Lower = tighter clusters, higher = broader clusters."
)

match_threshold = st.sidebar.slider(
    "Article Match Threshold:",
    min_value=0.10,
    max_value=0.90,
    value=0.55,
    step=0.05,
    help="Cosine similarity threshold between cluster centroid and article main keyword."
)

def parse_existing_articles(text) -> list[dict]:
    articles = []
    lines = text if isinstance(text, list) else str(text).strip().split("\n")
    for line in lines:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and parts[0] and parts[1]:
            articles.append({
                "title": parts[0],
                "main_keyword": parts[1],
                "url": parts[2]
            })
    return articles

parsed_articles = parse_existing_articles(articles_text)
st.sidebar.info(f"Loaded {len(parsed_articles)} existing articles")

# ---------------------------------------------------------
# Step 1: File Upload & Column Mapping
# ---------------------------------------------------------
st.header("Step 1: Upload Keywords and Map Columns")

uploaded_file = st.file_uploader(
    "Upload keyword export (CSV or XLSX from Ahrefs, GSC, or your own sheet):",
    type=["csv", "xlsx"]
)

def load_uploaded_file(file) -> pd.DataFrame:
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------
# Processing Functions: Clean, Cluster, Intent & Article Match
# ---------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def clean_keywords(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["keyword"] = cleaned["keyword"].astype(str).str.strip().str.lower()
    cleaned = cleaned[cleaned["keyword"] != ""]
    cleaned = cleaned.sort_values(by=["volume", "keyword"], ascending=[False, True])
    cleaned = cleaned.drop_duplicates(subset=["keyword"], keep="first")
    return cleaned.reset_index(drop=True)

def perform_clustering_and_intent(df: pd.DataFrame, threshold: float):
    if len(df) == 0:
        return df, np.array([])

    df["intent"] = df["keyword"].apply(detect_keyword_intent)

    embedder = load_embedder()
    embeddings = embedder.encode(df["keyword"].tolist(), show_progress_bar=False, normalize_embeddings=True)

    if len(df) == 1:
        df["cluster_id"] = 0
        df["cluster_name"] = df["keyword"].iloc[0]
        df["cluster_intent"] = df["intent"].iloc[0]
        return df, embeddings

    dist_matrix = cosine_distances(embeddings)

    brand_sets = [find_brands(kw) for kw in df["keyword"]]
    niche_sets = [find_niches(kw) for kw in df["keyword"]]
    n = len(df)
    for i in range(n):
        for j in range(i + 1, n):
            if len(brand_sets[i]) == 1 and len(brand_sets[j]) == 1 and brand_sets[i] != brand_sets[j]:
                dist_matrix[i, j] += 1.0
                dist_matrix[j, i] += 1.0
            if len(niche_sets[i]) == 1 and len(niche_sets[j]) == 1 and niche_sets[i] != niche_sets[j]:
                dist_matrix[i, j] += 1.0
                dist_matrix[j, i] += 1.0

    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=threshold,
        n_clusters=None
    )
    df["cluster_id"] = clustering.fit_predict(dist_matrix)

    cluster_names = {}
    for cid, group in df.groupby("cluster_id"):
        max_vol = group["volume"].max()
        if max_vol > 0:
            best_kw = group.loc[group["volume"].idxmax(), "keyword"]
        else:
            best_kw = min(group["keyword"].tolist(), key=len)
        cluster_names[cid] = best_kw
    df["cluster_name"] = df["cluster_id"].map(cluster_names)

    cluster_intents = {}
    for cid, group in df.groupby("cluster_id"):
        cluster_intents[cid] = detect_cluster_intent(group["intent"].tolist())
    df["cluster_intent"] = df["cluster_id"].map(cluster_intents)

    return df, embeddings

def match_clusters_to_articles(df: pd.DataFrame, embeddings: np.ndarray, articles: list[dict], threshold: float) -> pd.DataFrame:
    summary = (
        df.groupby(["cluster_id", "cluster_name", "cluster_intent"])
        .agg(
            keywords_count=("keyword", "count"),
            total_volume=("volume", "sum"),
            avg_kd=("kd", "mean")
        )
        .reset_index()
    )
    summary["avg_kd"] = summary["avg_kd"].round(1)

    if not articles or len(embeddings) == 0:
        summary["matched_article"] = ""
        summary["matched_url"] = ""
        summary["similarity"] = 0.0
        summary["action"] = "Новая статья"
        summary["final_article"] = ""
        summary["second_match"] = ""
        summary["second_similarity"] = 0.0
        summary["note"] = summary["cluster_intent"].apply(lambda x: "check manually" if x == "mixed" else "")
        summary["decision"] = ""
        summary["my_note"] = ""
        return summary.sort_values(by="total_volume", ascending=False)

    embedder = load_embedder()
    article_kws = [a["main_keyword"] for a in articles]
    art_embeddings = embedder.encode(article_kws, show_progress_bar=False, normalize_embeddings=True)

    match_rows = []
    for _, row in summary.iterrows():
        cid = row["cluster_id"]
        c_indices = df[df["cluster_id"] == cid].index.tolist()
        cluster_vecs = embeddings[c_indices]
        centroid = np.mean(cluster_vecs, axis=0, keepdims=True)

        sims = cosine_similarity(centroid, art_embeddings)[0]
        sorted_indices = np.argsort(sims)[::-1]

        best_idx = sorted_indices[0]
        best_sim = float(sims[best_idx])

        second_idx = sorted_indices[1] if len(sorted_indices) > 1 else None
        second_sim = float(sims[second_idx]) if second_idx is not None else 0.0

        if best_sim >= threshold:
            action = "Добавить в существующую"
            matched_article = articles[best_idx]["title"]
            matched_url = articles[best_idx]["url"]
            final_article = matched_article
        else:
            action = "Новая статья"
            matched_article = ""
            matched_url = ""
            final_article = ""

        second_title = articles[second_idx]["title"] if second_idx is not None else ""
        note = compute_cluster_note(row["cluster_intent"], action, best_sim, second_sim)

        match_rows.append({
            "matched_article": matched_article,
            "matched_url": matched_url,
            "similarity": round(best_sim, 3),
            "action": action,
            "final_article": final_article,
            "second_match": second_title,
            "second_similarity": round(second_sim, 3),
            "note": note,
            "decision": "",
            "my_note": ""
        })

    match_df = pd.DataFrame(match_rows)
    for col in match_df.columns:
        summary[col] = match_df[col]

    # Reorder columns as requested
    ordered_cols = [
        "cluster_id", "cluster_name", "cluster_intent", "keywords_count",
        "total_volume", "avg_kd", "matched_article", "matched_url",
        "similarity", "action", "final_article", "second_match",
        "second_similarity", "note", "decision", "my_note"
    ]
    summary = summary[ordered_cols]
    return summary.sort_values(by="total_volume", ascending=False)

def export_to_excel(clusters_df: pd.DataFrame, keywords_df: pd.DataFrame) -> bytes:
    # Generates XLSX file with frozen header rows and adjusted column widths
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        clusters_df.to_excel(writer, sheet_name="Clusters", index=False)
        keywords_df.to_excel(writer, sheet_name="Keywords", index=False)

        workbook = writer.book
        for sheet_name in ["Clusters", "Keywords"]:
            ws = workbook[sheet_name]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                col_letter = col[0].column_letter
                ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    return buffer.getvalue()

# ---------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------
if uploaded_file is not None:
    df_raw = load_uploaded_file(uploaded_file)

    if not df_raw.empty:
        st.success(f"File uploaded successfully! Loaded {len(df_raw)} rows and {len(df_raw.columns)} columns.")

        columns = list(df_raw.columns)

        def find_default_col(candidates, cols):
            for cand in candidates:
                for c in cols:
                    if cand.lower() == c.lower():
                        return cols.index(c)
            return 0

        col1, col2, col3 = st.columns(3)

        with col1:
            kw_default_idx = find_default_col(["keyword", "query", "top queries", "search term"], columns)
            kw_col = st.selectbox("Keyword column (Required):", options=columns, index=kw_default_idx)

        with col2:
            vol_options = ["(none)"] + columns
            vol_default_idx = find_default_col(["volume", "search volume", "clicks", "impressions"], columns)
            vol_idx = vol_default_idx + 1 if vol_default_idx < len(columns) and columns[vol_default_idx].lower() in ["volume", "search volume", "clicks"] else 0
            vol_col = st.selectbox("Search Volume column (Optional):", options=vol_options, index=vol_idx)

        with col3:
            kd_options = ["(none)"] + columns
            kd_default_idx = find_default_col(["kd", "keyword difficulty", "difficulty"], columns)
            kd_idx = kd_default_idx + 1 if kd_default_idx < len(columns) and columns[kd_default_idx].lower() in ["kd", "keyword difficulty", "difficulty"] else 0
            kd_col = st.selectbox("Difficulty / KD column (Optional):", options=kd_options, index=kd_idx)

        preview_df = pd.DataFrame()
        preview_df["keyword"] = df_raw[kw_col]

        if vol_col != "(none)":
            preview_df["volume"] = pd.to_numeric(df_raw[vol_col], errors="coerce").fillna(0).astype(int)
        else:
            preview_df["volume"] = 0

        if kd_col != "(none)":
            preview_df["kd"] = pd.to_numeric(df_raw[kd_col], errors="coerce").fillna(0).astype(int)
        else:
            preview_df["kd"] = 0

        st.subheader("Mapped Data Preview (First 5 rows)")
        st.dataframe(preview_df.head(5), use_container_width=True)

        # -----------------------------------------------------
        # Processing Trigger
        # -----------------------------------------------------
        st.header("Step 2-4: Process Pipeline")

        if st.button("Run Full Processing Pipeline", type="primary"):
            with st.spinner("Processing clustering, search intent, and article matching..."):
                cleaned_df = clean_keywords(preview_df)
                processed_df, embeddings = perform_clustering_and_intent(cleaned_df, cluster_threshold)
                clusters_summary = match_clusters_to_articles(processed_df, embeddings, parsed_articles, match_threshold)

                # Format Keywords table with requested columns and sorting
                keywords_export = processed_df.copy()
                keywords_export["decision"] = ""
                keywords_export["my_note"] = ""
                keywords_export = keywords_export.sort_values(by=["cluster_id", "volume"], ascending=[True, False])
                keywords_cols = ["keyword", "volume", "kd", "cluster_id", "cluster_name", "intent", "decision", "my_note"]
                keywords_export = keywords_export[keywords_cols]

                st.session_state["processed_df"] = keywords_export
                st.session_state["clusters_summary"] = clusters_summary

        if "processed_df" in st.session_state and "clusters_summary" in st.session_state:
            keywords_table = st.session_state["processed_df"]
            clusters_table = st.session_state["clusters_summary"]
            num_clusters = clusters_table["cluster_id"].nunique()

            st.success(f"Processing complete: {len(keywords_table)} keywords grouped into {num_clusters} clusters with article matching.")

            view_tab1, view_tab2 = st.tabs(["Clusters Overview", "All Keywords with Clusters"])
            with view_tab1:
                st.dataframe(clusters_table, use_container_width=True)
            with view_tab2:
                st.dataframe(keywords_table, use_container_width=True)

            # -------------------------------------------------
            # Step 5: Export Section
            # -------------------------------------------------
            st.header("Step 5: Export Results")

            excel_bytes = export_to_excel(clusters_table, keywords_table)
            clusters_csv = clusters_table.to_csv(index=False).encode("utf-8-sig")
            keywords_csv = keywords_table.to_csv(index=False).encode("utf-8-sig")

            dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1:
                st.download_button(
                    label="Download Full Excel (XLSX with 2 sheets)",
                    data=excel_bytes,
                    file_name="keyword_clusters_and_briefs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with dl_col2:
                st.download_button(
                    label="Download Clusters (CSV, UTF-8 BOM)",
                    data=clusters_csv,
                    file_name="clusters.csv",
                    mime="text/csv"
                )
            with dl_col3:
                st.download_button(
                    label="Download Keywords (CSV, UTF-8 BOM)",
                    data=keywords_csv,
                    file_name="keywords.csv",
                    mime="text/csv"
                )
else:
    st.info("Upload a CSV or XLSX file to begin. You can also test with sample_keywords.csv included in the project folder.")
