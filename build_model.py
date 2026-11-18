"""
One-time script: converts the old movies.pkl (+ 184 MB similarity.pkl) into a
small feature matrix (a few MB). Run it only if you change the dataset/weights:

    python build_model.py path/to/movies.pkl

The app itself never needs the old similarity.pkl any more.
"""
import json                  # Saves metadata in JSON format
import pickle                # Loads the original movie dataset
import re                    # Performs text processing using regular expressions
import sys                   # Reads command-line arguments

import numpy as np           # Performs numerical calculations
import pandas as pd          # Handles movie data in DataFrame format
import scipy.sparse as sp    # Stores large matrices efficiently using sparse format

from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text into numerical features
from sklearn.preprocessing import normalize                  # Normalizes feature vectors



# DATASET PATH
# Use the terminal path or default to movies.pkl.
SRC = sys.argv[1] if len(sys.argv) > 1 else "movies.pkl"


# GENRE LIST
# Multi-word genres come first to avoid partial matches.
KNOWN_GENRES = [
    "Science Fiction", "TV Movie", "Action", "Adventure", "Animation", "Comedy",
    "Crime", "Documentary", "Drama", "Family", "Fantasy", "Foreign", "History",
    "Horror", "Music", "Mystery", "Romance", "Thriller", "War", "Western",
]


# FEATURE WEIGHTS
# Importance of each movie attribute.
W_OVERVIEW, W_KEYWORDS, W_GENRES, W_DIRECTOR, W_CAST = 0.90, 1.00, 0.90, 0.50, 0.45


# GENRE EXTRACTION
def parse_genres(s: str):
    # Extract matching genres.
    out, rest = [], s or ""

    for g in KNOWN_GENRES:
        if g in rest:
            out.append(g)
            rest = rest.replace(g, " ")

    return out


# CAST NAME BIGRAMS
def name_bigrams(s: str):
    # Split names into lowercase tokens.
    toks = re.findall(r"[^\s]+", (s or "").lower())

    # Create pairs of consecutive words.
    return [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


# LOAD DATASET
# Load the pickle file and reset the index.
movies = pickle.load(open(SRC, "rb")).reset_index(drop=True)


# HANDLE MISSING VALUES
# Replace missing values with empty strings.
for c in ["genres", "keywords", "overview", "cast", "crew"]:
    movies[c] = movies[c].fillna("").astype(str)


# EXTRACT GENRES
# Apply genre extraction to every movie.
movies["genre_list"] = movies["genres"].apply(parse_genres)


# DIRECTOR INFORMATION
# Currently stores the complete crew string, not just the director.
movies["director"] = movies["crew"].str.strip()


# FEATURE ENGINEERING
# Convert movie attributes into numerical features.


# 1. OVERVIEW VECTORIZER
v_over = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=40000
)


# 2. KEYWORD VECTORIZER
v_kw = TfidfVectorizer(
    stop_words="english",
    min_df=1,
    sublinear_tf=True
)


# 3. GENRE VECTORIZER
v_gen = TfidfVectorizer(
    analyzer=lambda g: g
)


# 4. DIRECTOR VECTORIZER
v_dir = TfidfVectorizer(
    analyzer=lambda d: [d] if d else []
)


# 5. CAST VECTORIZER
v_cast = TfidfVectorizer(
    analyzer=name_bigrams
)


# BUILD FEATURE BLOCKS
# Learn vocabulary and convert each attribute into vectors.
blocks = [
    (v_over.fit_transform(movies["overview"]), W_OVERVIEW),
    (v_kw.fit_transform(movies["keywords"]), W_KEYWORDS),
    (v_gen.fit_transform(movies["genre_list"]), W_GENRES),
    (v_dir.fit_transform(movies["director"].str.lower()), W_DIRECTOR),
    (v_cast.fit_transform(movies["cast"]), W_CAST),
]


# COMBINE FEATURE BLOCKS
# Normalize, apply weights and combine all features.
X = sp.hstack([normalize(b) * w for b, w in blocks]).tocsr()


# FINAL NORMALIZATION
# Normalize vectors and reduce memory usage.
X = normalize(X).astype(np.float32)


# QUALITY SCORE
# R = Average rating, v = Number of votes.
R, v = movies["vote_average"].astype(float), movies["vote_count"].astype(float)


# Calculate average rating and vote threshold.
C, m = R.mean(), v.quantile(0.60)


# Calculate Bayesian weighted ratings.
wr = v / (v + m) * R + m / (v + m) * C

movies["wr"] = wr


# MIN-MAX NORMALIZATION
# Scale quality scores between 0 and 1.
movies["quality"] = ((wr - wr.min()) / (wr.max() - wr.min())).astype(np.float32)


# FREE-TEXT SEARCH MATRIX
# Combine movie details for text-based searching.
docs = (
    movies["title"] + " " +
    movies["genres"] + " " +
    movies["keywords"] + " " +
    movies["overview"]
)


# TEXT VECTORIZER
v_txt = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=40000,
    dtype=np.float32
)


# Convert movie descriptions into numerical vectors.
T = v_txt.fit_transform(docs)


# SIMILARITY CALIBRATION
# Calculate cosine similarity between movies.
S = (X @ X.T).toarray()


# Ignore self-similarity.
np.fill_diagonal(S, 0)


# Find each movie's closest match.
top1 = S.max(axis=1)


# Use the 95th percentile as the reference score.
ref = float(np.percentile(top1, 95))


# KEYWORD IDF
# Store keyword importance scores.
kw_idf = dict(
    zip(
        v_kw.get_feature_names_out(),
        v_kw.idf_
    )
)


# LIGHTWEIGHT DATASET
# Keep only columns needed by the application.
lite = movies[
    [
        "title",
        "genre_list",
        "keywords",
        "overview",
        "director",
        "cast",
        "vote_average",
        "vote_count",
        "popularity",
        "quality"
    ]
].copy()


# Convert genre lists into pipe-separated strings.
lite["genre_list"] = lite["genre_list"].apply(lambda g: "|".join(g))


# SAVE MODEL FILES
# Save the lightweight dataset.
lite.to_csv("data/movies_lite.csv.gz", index=False)


# Save the movie feature matrix.
sp.save_npz("data/features.npz", X, compressed=True)


# Save the text search matrix.
sp.save_npz("data/text_index.npz", T.tocsr(), compressed=True)


# Save text IDF values.
np.save("data/txt_idf.npy", v_txt.idf_.astype(np.float32))


# SAVE METADATA
# Store similarity reference, keyword IDF and text vocabulary.
json.dump(
    {
        "ref": ref,
        "kw_idf": {k: float(v) for k, v in kw_idf.items()},
        "txt_vocab": {k: int(v) for k, v in v_txt.vocabulary_.items()}
    },
    open("data/meta.json", "w")
)


# FINAL OUTPUT
# Display dataset size, feature dimensions and reference similarity.
print(
    "movies:", len(lite),
    "| features:", X.shape,
    "| ref cosine:", round(ref, 3)
)

