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
# If a file path is provided through the terminal, use it.
# Otherwise, load movies.pkl by default.

SRC = sys.argv[1] if len(sys.argv) > 1 else "movies.pkl"


# GENRE LIST
# These are the genres that the program knows how to identify.
# Multi-word genres are placed first to help avoid partial matching.

KNOWN_GENRES = [
    "Science Fiction", "TV Movie", "Action", "Adventure", "Animation", "Comedy",
    "Crime", "Documentary", "Drama", "Family", "Fantasy", "Foreign", "History",
    "Horror", "Music", "Mystery", "Romance", "Thriller", "War", "Western",
]


# FEATURE WEIGHTS
# These weights decide how much importance each movie attribute receives
# when calculating similarity between two movies.

W_OVERVIEW, W_KEYWORDS, W_GENRES, W_DIRECTOR, W_CAST = 0.90, 1.00, 0.90, 0.50, 0.45


# FUNCTION 1: GENRE EXTRACTION

def parse_genres(s: str):
    # Create an empty list to store extracted genres.
    # 'rest' contains the original genre string.

    out, rest = [], s or ""

    for g in KNOWN_GENRES:

        # Check whether the current known genre exists in the string.
        if g in rest:

            out.append(g)          # Add the matched genre to the result
            rest = rest.replace(g, " ")  # Remove the matched genre to avoid matching it again

    return out  # Return the extracted genres as a list


# FUNCTION 2: CAST NAME BIGRAMS

def name_bigrams(s: str):

    # Convert text into lowercase and split it into individual tokens.
    # Regular expression [^\s]+ matches groups of characters separated by whitespace.

    toks = re.findall(r"[^\s]+", (s or "").lower())

    # Create pairs of consecutive tokens.
    # Example: ["tom", "cruise", "brad", "pitt"]
    # Output: ["tom_cruise", "cruise_brad", "brad_pitt"]

    return [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


# LOADING THE MOVIE DATASET

# Open the pickle file in binary read mode and load its contents.
# reset_index() creates a fresh index starting from zero.
# drop=True prevents the old index from becoming an extra column.

movies = pickle.load(open(SRC, "rb")).reset_index(drop=True)


# HANDLING MISSING VALUES

# Replace missing values with empty strings.
# astype(str) ensures that every value is stored as text.
# This prevents errors during text processing.

for c in ["genres", "keywords", "overview", "cast", "crew"]:
    movies[c] = movies[c].fillna("").astype(str)


# APPLY GENRE EXTRACTION TO EVERY MOVIE

# The apply() function calls parse_genres() on every row
# and stores the extracted genres in a new column.

movies["genre_list"] = movies["genres"].apply(parse_genres)


# DIRECTOR INFORMATION

# Current implementation copies the entire crew string into director.
# IMPORTANT: This does NOT actually extract only the director.
# Proper crew parsing would be needed to identify the person
# whose job is specifically "Director".

movies["director"] = movies["crew"].str.strip()


# FEATURE ENGINEERING: CONVERTING MOVIE INFORMATION INTO FEATURES

# Each vectorizer converts a different type of movie information
# into numerical vectors using TF-IDF or token-based processing.


# 1. OVERVIEW VECTORIZER

v_over = TfidfVectorizer(
    stop_words="english",  # Remove common words such as "the", "is", "and"
    ngram_range=(1, 2),    # Consider single words and pairs of consecutive words
    min_df=2,              # Ignore words appearing in fewer than 2 documents
    sublinear_tf=True,     # Apply logarithmic scaling to term frequency
    max_features=40000     # Keep at most 40,000 vocabulary features
)


# 2. KEYWORD VECTORIZER

v_kw = TfidfVectorizer(
    stop_words="english",  # Remove common English words
    min_df=1,              # Keep words appearing at least once
    sublinear_tf=True      # Reduce the influence of frequently repeated terms
)


# 3. GENRE VECTORIZER

v_gen = TfidfVectorizer(
    analyzer=lambda g: g  # Treat each genre in the list as a separate token
)


# 4. DIRECTOR VECTORIZER

v_dir = TfidfVectorizer(
    analyzer=lambda d: [d] if d else []  # Treat the complete director string as one token
)


# 5. CAST VECTORIZER

v_cast = TfidfVectorizer(
    analyzer=name_bigrams  # Use the custom function to generate cast bigrams
)


# BUILDING THE FEATURE BLOCKS

# fit_transform() performs two operations:
# 1. fit: learns the vocabulary and IDF values from the dataset.
# 2. transform: converts the text into numerical vectors.
#
# Each feature block is paired with its corresponding weight.

blocks = [
    (v_over.fit_transform(movies["overview"]), W_OVERVIEW),   # Movie plot features
    (v_kw.fit_transform(movies["keywords"]), W_KEYWORDS),    # Movie keyword features
    (v_gen.fit_transform(movies["genre_list"]), W_GENRES),   # Genre features
    (v_dir.fit_transform(movies["director"].str.lower()), W_DIRECTOR),  # Director features
    (v_cast.fit_transform(movies["cast"]), W_CAST),          # Cast features
]


# COMBINING ALL FEATURE BLOCKS INTO ONE MATRIX

# normalize(b): Gives each feature block unit-length rows.
# * w: Applies the importance weight of that feature block.
# sp.hstack(): Combines all five feature matrices horizontally.
# tocsr(): Converts the result into CSR sparse matrix format.
#
# The result contains all movie attributes in one combined representation.

X = sp.hstack([normalize(b) * w for b, w in blocks]).tocsr()


# Normalize the final combined vectors again.
# float32 uses less memory than float64 while retaining useful precision.

X = normalize(X).astype(np.float32)


# QUALITY SCORE: WEIGHTED MOVIE RATINGS

# R = Average movie rating.
# v = Number of votes received by each movie.

R, v = movies["vote_average"].astype(float), movies["vote_count"].astype(float)


# C = Overall average rating of all movies.
# m = 60th percentile of movie vote counts.
# These values are used to calculate a Bayesian-style weighted rating.

C, m = R.mean(), v.quantile(0.60)


# Weighted rating formula:
# Movies with fewer votes are pulled closer to the overall average rating.
# Movies with more votes receive greater importance for their own rating.

wr = v / (v + m) * R + m / (v + m) * C


# Save the calculated weighted rating in the dataset.

movies["wr"] = wr


# MIN-MAX NORMALIZATION
# Converts weighted ratings into a scale between 0 and 1.
# Higher quality scores indicate better weighted ratings.

movies["quality"] = ((wr - wr.min()) / (wr.max() - wr.min())).astype(np.float32)


# FREE-TEXT SEARCH MATRIX
# This is separate from the movie-to-movie recommendation matrix.
# It helps users search using descriptions or movie vibes.

# Combine the title, genres, keywords and overview into one document per movie.

docs = (
    movies["title"] + " " +
    movies["genres"] + " " +
    movies["keywords"] + " " +
    movies["overview"]
)


# Create another TF-IDF vectorizer for free-text searching.

v_txt = TfidfVectorizer(
    stop_words="english",  # Remove common English words
    ngram_range=(1, 2),    # Consider single words and two-word phrases
    min_df=2,              # Ignore terms appearing in fewer than 2 documents
    sublinear_tf=True,     # Apply logarithmic term-frequency scaling
    max_features=40000,    # Limit vocabulary size to 40,000
    dtype=np.float32       # Store values using 32-bit floating-point numbers
)


# Convert all movie documents into TF-IDF numerical vectors.
# T is used for searching movies based on user-written descriptions.

T = v_txt.fit_transform(docs)


# SIMILARITY CALIBRATION
# Establish a reference value for displaying match percentages.


# X contains normalized movie feature vectors.
# Multiplying X by its transpose calculates cosine similarities
# between every pair of movies.

# NOTE: toarray() converts the entire similarity matrix into dense format.
# This can consume a lot of RAM when the dataset becomes large.

S = (X @ X.T).toarray()


# Set diagonal values to zero.
# Each movie has a similarity of 1 with itself.
# We don't want self-similarity to affect calibration.

np.fill_diagonal(S, 0)


# Find the highest similarity score for each movie.
# This represents the similarity of its closest matching movie.

top1 = S.max(axis=1)


# Calculate the 95th percentile of the highest similarity scores.
# This becomes the reference value used for match percentage calibration.

ref = float(np.percentile(top1, 95))


# KEYWORD IDF
# Store the IDF value of every keyword.
# This information can later help explain shared themes between movies.

kw_idf = dict(
    zip(
        v_kw.get_feature_names_out(),  # All keywords learned by the vectorizer
        v_kw.idf_                      # Corresponding IDF values
    )
)


# CREATING A LIGHTWEIGHT MOVIE DATASET

# Keep only the columns required by the recommendation application.
# This avoids storing unnecessary information.

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
# Example: ["Action", "Comedy"] becomes "Action|Comedy".
# This makes the data easier to store in CSV format.

lite["genre_list"] = lite["genre_list"].apply(lambda g: "|".join(g))


# SAVING THE GENERATED MODEL FILES

# Save the lightweight movie dataset as a compressed CSV file.
# index=False prevents pandas from adding an unnecessary index column.

lite.to_csv("data/movies_lite.csv.gz", index=False)


# Save the combined movie feature matrix.
# NPZ is a compressed format that supports sparse matrices.

sp.save_npz("data/features.npz", X, compressed=True)


# Save the free-text search matrix.
# T.tocsr() ensures it is stored in CSR sparse matrix format.

sp.save_npz("data/text_index.npz", T.tocsr(), compressed=True)


# Save the IDF values learned by the free-text vectorizer.
# These values can be reused by the search engine.

np.save("data/txt_idf.npy", v_txt.idf_.astype(np.float32))


# SAVE METADATA IN JSON FORMAT

# ref: Reference similarity used for match percentage calibration.
# kw_idf: Keyword IDF values used for recommendation explanations.
# txt_vocab: Maps text vocabulary terms to their numerical feature indices.

json.dump(
    {
        "ref": ref,
        "kw_idf": {k: float(v) for k, v in kw_idf.items()},
        "txt_vocab": {k: int(v) for k, v in v_txt.vocabulary_.items()}
    },
    open("data/meta.json", "w")
)


# FINAL OUTPUT

# Print the number of movies processed.
# X.shape shows the dimensions of the combined feature matrix.
# ref cosine displays the reference similarity rounded to 3 decimal places.

print(
    "movies:", len(lite),
    "| features:", X.shape,
    "| ref cosine:", round(ref, 3)
)

json.dump({"ref": ref, "kw_idf": {k: float(v) for k, v in kw_idf.items()},
           "txt_vocab": {k: int(v) for k, v in v_txt.vocabulary_.items()}},
          open("data/meta.json", "w"))
print("movies:", len(lite), "| features:", X.shape, "| ref cosine:", round(ref, 3))
