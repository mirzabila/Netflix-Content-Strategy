# Netflix Content Strategy Recommender

A Streamlit dashboard that explores Netflix catalog patterns and recommends similar titles using the embedded TF-IDF model.

## Features

- Attractive Netflix-style dashboard
- Search and filter by type, genre, rating, release year, and keyword
- Content strategy metrics for the filtered catalog
- Genre mix and release trend charts
- Selected-title detail panel
- Similarity-based recommendations with match scores
- CSV export for recommendations
- Optional TMDB poster integration through `.env`

## Project Structure

```text
app.py
recommender.py
requirements.txt
netflix_data_with_cluster.csv
tfidf_matrix.pkl
tfidf.pkl
.env.example
.streamlit/config.toml
```

## Setup

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The app runs locally at:

```text
http://localhost:8501
```

## Optional Posters

Create a `.env` file and add your TMDB key:

```text
TMDB_API_KEY=your_key_here
```

Without a TMDB key, the app still works and uses title placeholders for posters.
