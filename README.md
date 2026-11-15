# CineMatch

A content-based movie recommender built with Streamlit. Pick 1–3 movies you love and get
ranked matches **with reasons** (shared genres, director, cast, themes), plus posters,
trailers, cast and where-to-watch info from TMDB.

## Run it

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your TMDB key
streamlit run app.py
```

No key? The app still works – you just won't see posters, trailers or trending.
**Never commit `secrets.toml`** (it is in `.gitignore`). On Streamlit Community Cloud, put the key in
*App settings → Secrets* instead.

## Features
- Recommend from 1–3 movies (combined taste) · filters: genre, min rating, release year
- "Why this movie?" reasons on every card + calibrated match %
- Vibe search: describe a mood ("scary haunted house") instead of a title
- Details pop-up: trailer, cast photos, runtime, where to watch (region selectable), IMDb link
- Discover: trending (TMDB) and all-time top rated · Watchlist with CSV download
- Posters load in parallel and are cached; responsive dark UI with a consistent icon set

## How it works
| File | Purpose |
|---|---|
| `build_model.py` | One-time: builds weighted TF-IDF features (plot, keywords, genres, director, cast) → `data/` (~6 MB, replaces the old 184 MB `similarity.pkl`) |
| `engine.py` | Cosine similarity on the fly, quality nudge, filters, explanations, vibe search |
| `tmdb.py` | TMDB client (matches the right film by comparing plot text, not just the title) |
| `app.py` | Streamlit UI and recommendation-page composition |
| `styles.css` | Responsive dark-cinema visual system, component styling, and reduced-motion-aware transitions |

Rebuild the model only if you change the data or weights: `python build_model.py path/to/movies.pkl`

Data: TMDB 5000 dataset. This product uses the TMDB API but is not endorsed or certified by TMDB.
