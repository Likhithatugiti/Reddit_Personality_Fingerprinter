"""
app.py — Streamlit web interface for Reddit Personality Fingerprinter

Run with:
    streamlit run app.py
"""

import logging
import streamlit as st
import os
from pathlib import Path

from config import MIN_COMMENTS_REQUIRED, TRAIT_NAMES, TRAITS
from src.scraper.reddit_scraper import RedditScraper
from src.features.feature_pipeline import run_pipeline
from src.models.predictor import Predictor
from src.utils.visualiser import radar_chart, bar_chart, reliability_gauge
from src.utils.report_generator import generate_pdf

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reddit Personality Fingerprinter",
    page_icon="🧠",
    layout="wide",
)

# ── Cached resource loaders ───────────────────────────────────────────────────
@st.cache_resource
def get_scraper():
    db_path = Path("data/database.sqlite")
    if db_path.exists():
        from src.scraper.dataset_loader import DatasetLoader
        return DatasetLoader(str(db_path))
    else:
        from src.scraper.reddit_scraper import RedditScraper
        return RedditScraper()

@st.cache_resource
def get_predictor():
    return Predictor()

@st.cache_data(show_spinner=False)
def fetch_and_predict(username: str, limit: int):
    scraper    = get_scraper()
    predictor  = get_predictor()

    result = scraper.fetch(username, limit=limit)
    if not result.success:
        return None, None, result.error

    feature_df = run_pipeline(username, result.comments)
    profile    = predictor.predict(feature_df)
    return profile, result, None


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Reddit Personality Fingerprinter")
st.markdown(
    "Estimates **Big-Five personality traits** (OCEAN) from a Reddit user's "
    "public comment history using psycholinguistic feature extraction and LightGBM models."
)
st.divider()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙ Settings")
    username = st.text_input(
        "Reddit username",
        placeholder="e.g. spez  (without u/)",
        help="The Reddit username to analyse. Only public comment history is fetched."
    )
    comment_limit = st.slider(
        "Max comments to fetch",
        min_value=50, max_value=500, value=200, step=50,
        help=f"More comments = higher reliability (minimum recommended: {MIN_COMMENTS_REQUIRED})"
    )
    compare_username = st.text_input(
        "Compare with (optional)",
        placeholder="Second username for overlay",
    )
    run_btn = st.button("🔍 Analyse", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "**Limitations**\n\n"
        "Scores are probabilistic estimates from public text only. "
        "Not a clinical assessment. Do not use for consequential decisions."
    )

# ── Main panel ────────────────────────────────────────────────────────────────
if run_btn and username:
    with st.spinner(f"Fetching comments for u/{username} …"):
        profile, scrape_result, error = fetch_and_predict(username, comment_limit)

    if error:
        st.error(f"Could not fetch data for u/{username}: {error}")
        st.stop()

    # Optional comparison user
    compare_profile = None
    if compare_username:
        with st.spinner(f"Fetching comparison user u/{compare_username} …"):
            compare_profile, _, _ = fetch_and_predict(compare_username, comment_limit)

    # ── Metrics row ───────────────────────────────────────────────────────
    st.subheader(f"Results for u/{username}")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    metric_cols = [col1, col2, col3, col4, col5]
    for i, trait in enumerate(TRAITS):
        score = profile.scores.get(trait, 4.0)
        delta = None
        if compare_profile:
            delta = round(score - compare_profile.scores.get(trait, 4.0), 2)
        metric_cols[i].metric(
            label=TRAIT_NAMES[trait],
            value=f"{score:.2f}",
            delta=f"{delta:+.2f} vs {compare_username}" if delta else None,
        )
    col6.metric("Comments", scrape_result.total_kept)

    if profile.low_data_warning:
        st.warning(
            f"⚠ Only {scrape_result.total_kept} comments available. "
            f"Scores may be unreliable. Aim for {MIN_COMMENTS_REQUIRED}+."
        )

    # ── Charts ────────────────────────────────────────────────────────────
    chart_col, bar_col, gauge_col = st.columns([2, 1.5, 1])

    with chart_col:
        fig = radar_chart(
            profile.scores,
            username,
            compare_scores=compare_profile.scores if compare_profile else None,
            compare_label=f"u/{compare_username}" if compare_username else "Comparison",
        )
        st.plotly_chart(fig, use_container_width=True)

    with bar_col:
        st.plotly_chart(bar_chart(profile.scores, username), use_container_width=True)

    with gauge_col:
        st.plotly_chart(
            reliability_gauge(scrape_result.total_kept),
            use_container_width=True,
        )

    # ── Subreddits ────────────────────────────────────────────────────────
    with st.expander("📋 Subreddits observed"):
        st.write(", ".join(scrape_result.subreddits_seen) or "None")

    # ── Download report ───────────────────────────────────────────────────
    st.divider()
    if st.button("📄 Generate PDF report"):
        with st.spinner("Generating PDF…"):
            pdf_path = generate_pdf(profile)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇ Download PDF",
                data=f,
                file_name=pdf_path.name,
                mime="application/pdf",
            )

elif run_btn and not username:
    st.warning("Please enter a Reddit username in the sidebar.")

else:
    # Landing state
    st.info(
        "Enter a Reddit username in the sidebar and click **Analyse** to get started.\n\n"
        "The app will fetch up to 500 public comments and estimate personality "
        "traits using psycholinguistic features."
    )
    with st.expander("How it works"):
        st.markdown("""
1. **Scraping** — Fetches public comments via the Reddit API (PRAW).
2. **Preprocessing** — Cleans markdown, URLs, and bot noise with spaCy.
3. **Feature extraction** — Computes 50+ psycholinguistic features:
   pronoun rates, hedging language, certainty words, VADER sentiment,
   NRC emotion lexicon scores, text complexity, tense distribution, and more.
4. **Prediction** — Five LightGBM regressors (one per trait) map the
   feature vector to a 1–7 score for each Big-Five dimension.
5. **Report** — Results displayed as a radar chart + downloadable PDF.

*Methodology based on Harrison et al. (2019), Strategic Management Journal.*
        """)
