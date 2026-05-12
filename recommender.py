from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "netflix_data_with_cluster.csv"
TFIDF_MATRIX_PATH = BASE_DIR / "tfidf_matrix.pkl"


def _clean_text(value, fallback="Unknown"):
    # Yeh helper missing ya khaali text ko readable value mein convert karta hai.
    if pd.isna(value) or str(value).strip() == "":
        return fallback
    return str(value).strip()


def load_assets():
    # Dataset aur saved TF-IDF matrix yahin load hoti hai taa ke app ka backend centralized rahe.
    df = pd.read_csv(DATA_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)

    df["type"] = df["type"].apply(lambda value: _clean_text(value, "Unknown"))
    df["listed_in"] = df["listed_in"].apply(lambda value: _clean_text(value, "Unlisted"))
    df["primary_genre"] = df["listed_in"].str.split(",").str[0].str.strip()
    df["country"] = df["country"].apply(lambda value: _clean_text(value, "Unknown"))
    df["rating"] = df["rating"].apply(lambda value: _clean_text(value, "Unrated"))
    df["description"] = df["description"].apply(lambda value: _clean_text(value, "No description available."))

    if "cluster" not in df.columns:
        # Agar CSV mein cluster column na ho to primary genre se fast content segments ban jate hain.
        genre_codes = pd.factorize(df["primary_genre"])[0]
        df["cluster"] = (genre_codes + 1).astype(str)
    else:
        df["cluster"] = df["cluster"].astype(str)

    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    return df, tfidf_matrix, indices


def get_recommendations(title, df, tfidf_matrix, indices, limit=8):
    # Selected title ke nearest titles cosine similarity ki base par nikalte hain.
    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    similarity_row = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(similarity_row))
    sim_scores = sorted(sim_scores, key=lambda item: item[1], reverse=True)[1 : limit + 1]
    movie_indices = [item[0] for item in sim_scores]
    scores = [round(float(item[1]) * 100, 1) for item in sim_scores]

    recommendations = df.iloc[movie_indices].copy()
    recommendations["match_score"] = scores
    return recommendations


def filter_titles(df, content_type="All", genre="All", rating="All", year_range=None, keyword=""):
    # Sidebar filters dataset ko user ke selected criteria ke mutabiq narrow karte hain.
    filtered = df.copy()

    if content_type != "All":
        filtered = filtered[filtered["type"] == content_type]
    if genre != "All":
        filtered = filtered[filtered["primary_genre"] == genre]
    if rating != "All":
        filtered = filtered[filtered["rating"] == rating]
    if year_range:
        start_year, end_year = year_range
        filtered = filtered[filtered["release_year"].between(start_year, end_year)]
    if keyword:
        keyword = keyword.lower().strip()
        mask = (
            filtered["title"].str.lower().str.contains(keyword, na=False)
            | filtered["description"].str.lower().str.contains(keyword, na=False)
            | filtered["listed_in"].str.lower().str.contains(keyword, na=False)
        )
        filtered = filtered[mask]

    return filtered


def build_summary(df):
    # Dashboard cards ke liye basic content strategy metrics banaye jate hain.
    return {
        "titles": len(df),
        "movies": int((df["type"] == "Movie").sum()),
        "shows": int((df["type"] == "TV Show").sum()),
        "genres": df["primary_genre"].nunique(),
        "year_min": int(df["release_year"].min()),
        "year_max": int(df["release_year"].max()),
    }
