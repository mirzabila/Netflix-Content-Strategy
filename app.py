import os
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from recommender import build_summary, filter_titles, get_recommendations, load_assets


load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()

st.set_page_config(
    page_title="Netflix Content Strategy",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading recommendation engine...")
def cached_assets():
    # Heavy model files cache ho jati hain taa ke har interaction par reload na hon.
    return load_assets()


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_poster(title, content_type):
    # TMDB key mil jaye to poster API se aata hai, warna app graceful fallback use karta hai.
    if not TMDB_API_KEY:
        return ""

    media_type = "movie" if content_type == "Movie" else "tv"
    url = f"https://api.themoviedb.org/3/search/{media_type}"
    params = {"api_key": TMDB_API_KEY, "query": title}

    try:
        response = requests.get(url, params=params, timeout=6)
        response.raise_for_status()
        results = response.json().get("results", [])
    except requests.RequestException:
        return ""

    if not results:
        return ""

    poster_path = results[0].get("poster_path")
    if not poster_path:
        return ""
    return f"https://image.tmdb.org/t/p/w500{poster_path}"


def poster_fallback(title):
    # Agar poster available na ho to readable placeholder title ke sath render hota hai.
    safe_title = quote_plus(title[:28])
    return f"https://placehold.co/420x620/151515/e50914?text={safe_title}"


def apply_styles():
    # Custom CSS Streamlit widgets ko Netflix style dashboard look deta hai.
    st.markdown(
        """
        <style>
        :root {
            --netflix-red: #e50914;
            --ink: #f8fafc;
            --muted: #a1a1aa;
            --panel: rgba(24, 24, 27, 0.78);
            --line: rgba(255, 255, 255, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 16%, rgba(229, 9, 20, 0.24), transparent 32rem),
                linear-gradient(135deg, #050505 0%, #121212 45%, #211113 100%);
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: rgba(7, 7, 8, 0.94);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] * {
            color: #f4f4f5;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }

        .hero {
            min-height: 250px;
            padding: 2rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background:
                linear-gradient(90deg, rgba(0, 0, 0, 0.92), rgba(0, 0, 0, 0.52)),
                url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=1800&q=80");
            background-size: cover;
            background-position: center;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            margin-bottom: 1.2rem;
        }

        .brand {
            color: var(--netflix-red);
            font-weight: 900;
            letter-spacing: 0;
            font-size: clamp(2rem, 5vw, 4.3rem);
            line-height: 1;
            margin: 0;
        }

        .tagline {
            color: #e4e4e7;
            max-width: 780px;
            font-size: 1.05rem;
            margin-top: 0.75rem;
        }

        .metric-card, .title-panel, .rec-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            backdrop-filter: blur(14px);
        }

        .metric-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: white;
            margin: 0;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.86rem;
            margin: 0.2rem 0 0;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            margin: 1rem 0 0.35rem;
        }

        .pill {
            display: inline-block;
            border: 1px solid rgba(229, 9, 20, 0.42);
            border-radius: 999px;
            padding: 0.22rem 0.62rem;
            margin: 0 0.35rem 0.35rem 0;
            color: #fecdd3;
            background: rgba(229, 9, 20, 0.1);
            font-size: 0.78rem;
        }

        .rec-title {
            font-size: 1rem;
            font-weight: 800;
            color: white;
            min-height: 2.5rem;
            margin-bottom: 0.25rem;
        }

        .small-muted {
            color: var(--muted);
            font-size: 0.86rem;
        }

        div.stButton > button,
        div.stDownloadButton > button {
            width: 100%;
            border-radius: 6px;
            border: 0;
            background: var(--netflix-red);
            color: white;
            font-weight: 800;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            background: #f6121d;
            color: white;
            border: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value):
    # Reusable metric block dashboard ke top numbers ko consistent banata hai.
    st.markdown(
        f"""
        <div class="metric-card">
            <p class="metric-value">{value}</p>
            <p class="metric-label">{label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_card(row):
    # Har recommendation poster, score aur metadata ke sath card mein render hoti hai.
    poster_url = fetch_poster(row["title"], row["type"]) or poster_fallback(row["title"])
    st.image(poster_url, use_container_width=True)
    st.markdown(
        f"""
        <div class="rec-card">
            <div class="rec-title">{row["title"]}</div>
            <div class="small-muted">{row["type"]} | {row["release_year"]} | {row["rating"]}</div>
            <span class="pill">{row["primary_genre"]}</span>
            <span class="pill">{row["match_score"]}% match</span>
            <p class="small-muted">{row["description"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


apply_styles()
df, tfidf_matrix, indices = cached_assets()
summary = build_summary(df)

with st.sidebar:
    st.title("Controls")
    keyword = st.text_input("Search title, genre, description")
    content_type = st.selectbox("Content type", ["All"] + sorted(df["type"].unique().tolist()))
    genre = st.selectbox("Primary genre", ["All"] + sorted(df["primary_genre"].unique().tolist()))
    rating = st.selectbox("Maturity rating", ["All"] + sorted(df["rating"].unique().tolist()))
    year_range = st.slider(
        "Release year",
        min_value=summary["year_min"],
        max_value=summary["year_max"],
        value=(summary["year_min"], summary["year_max"]),
    )
    recommendation_limit = st.slider("Recommendations", 4, 12, 8)

filtered_df = filter_titles(df, content_type, genre, rating, year_range, keyword)

st.markdown(
    """
    <section class="hero">
        <h1 class="brand">NETFLIX CONTENT STRATEGY</h1>
        <p class="tagline">Explore the catalog, spot content patterns, and get similarity-based recommendations from the embedded TF-IDF model.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(5)
with metric_columns[0]:
    metric_card("Filtered titles", f"{len(filtered_df):,}")
with metric_columns[1]:
    metric_card("Movies", f"{int((filtered_df['type'] == 'Movie').sum()):,}")
with metric_columns[2]:
    metric_card("TV shows", f"{int((filtered_df['type'] == 'TV Show').sum()):,}")
with metric_columns[3]:
    metric_card("Genres", f"{filtered_df['primary_genre'].nunique():,}")
with metric_columns[4]:
    metric_card("Full catalog", f"{summary['titles']:,}")

if filtered_df.empty:
    st.warning("No titles match the selected filters. Adjust the controls in the sidebar.")
    st.stop()

left_chart, right_chart = st.columns(2)
with left_chart:
    st.markdown('<div class="section-title">Genre Mix</div>', unsafe_allow_html=True)
    genre_counts = filtered_df["primary_genre"].value_counts().head(10)
    st.bar_chart(genre_counts)

with right_chart:
    st.markdown('<div class="section-title">Release Trend</div>', unsafe_allow_html=True)
    year_counts = filtered_df["release_year"].value_counts().sort_index()
    st.line_chart(year_counts)

title_options = filtered_df["title"].dropna().sort_values().unique().tolist()
selected_title = st.selectbox("Choose a Netflix title", title_options)
selected_row = filtered_df[filtered_df["title"] == selected_title].iloc[0]

st.markdown('<div class="section-title">Selected Title</div>', unsafe_allow_html=True)
selected_left, selected_right = st.columns([1, 2])
with selected_left:
    selected_poster = fetch_poster(selected_row["title"], selected_row["type"]) or poster_fallback(selected_row["title"])
    st.image(selected_poster, use_container_width=True)

with selected_right:
    st.markdown(
        f"""
        <div class="title-panel">
            <h2>{selected_row["title"]}</h2>
            <span class="pill">{selected_row["type"]}</span>
            <span class="pill">{selected_row["release_year"]}</span>
            <span class="pill">{selected_row["rating"]}</span>
            <span class="pill">Cluster {selected_row["cluster"]}</span>
            <p>{selected_row["description"]}</p>
            <p class="small-muted"><b>Genres:</b> {selected_row["listed_in"]}</p>
            <p class="small-muted"><b>Country:</b> {selected_row["country"]}</p>
            <p class="small-muted"><b>Duration:</b> {selected_row["duration"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Recommended For You</div>', unsafe_allow_html=True)
recommendations = get_recommendations(
    selected_title,
    df,
    tfidf_matrix,
    indices,
    limit=recommendation_limit,
)

if recommendations.empty:
    st.info("Recommendation engine could not find a match for this title.")
else:
    rows = list(recommendations.to_dict("records"))
    for start in range(0, len(rows), 4):
        columns = st.columns(4)
        for column, row in zip(columns, rows[start : start + 4]):
            with column:
                render_recommendation_card(row)

    download_columns = [
        "title",
        "type",
        "release_year",
        "rating",
        "primary_genre",
        "listed_in",
        "country",
        "duration",
        "match_score",
        "description",
    ]
    csv_data = recommendations[download_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download recommendations CSV",
        data=csv_data,
        file_name="netflix_recommendations.csv",
        mime="text/csv",
    )
