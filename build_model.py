"""
One-time script: converts the old movies.pkl (+ 184 MB similarity.pkl) into a
small feature matrix (a few MB). Run it only if you change the dataset/weights:

    python build_model.py path/to/movies.pkl

The app itself never needs the old similarity.pkl any more.
"""
import json
import pickle
import re
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SRC = sys.argv[1] if len(sys.argv) > 1 else "movies.pkl"

KNOWN_GENRES = [
    "Science Fiction", "TV Movie", "Action", "Adventure", "Animation", "Comedy",
    "Crime", "Documentary", "Drama", "Family", "Fantasy", "Foreign", "History",
    "Horror", "Music", "Mystery", "Romance", "Thriller", "War", "Western",
]

# block weights: how much each kind of information matters for "similar"
W_OVERVIEW, W_KEYWORDS, W_GENRES, W_DIRECTOR, W_CAST = 0.90, 1.00, 0.90, 0.50, 0.45


def parse_genres(s: str):
    out, rest = [], s or ""
    for g in KNOWN_GENRES:          # multi-word genres are listed first
        if g in rest:
            out.append(g)
            rest = rest.replace(g, " ")
    return out


def name_bigrams(s: str):
    toks = re.findall(r"[^\s]+", (s or "").lower())
    return [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


movies = pickle.load(open(SRC, "rb")).reset_index(drop=True)
for c in ["genres", "keywords", "overview", "cast", "crew"]:
    movies[c] = movies[c].fillna("").astype(str)

movies["genre_list"] = movies["genres"].apply(parse_genres)
movies["director"] = movies["crew"].str.strip()
#feature block
v_over = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2,
                         sublinear_tf=True, max_features=40000)
v_kw = TfidfVectorizer(stop_words="english", min_df=1, sublinear_tf=True)
v_gen = TfidfVectorizer(analyzer=lambda g: g)
v_dir = TfidfVectorizer(analyzer=lambda d: [d] if d else [])
v_cast = TfidfVectorizer(analyzer=name_bigrams)

blocks = [
    (v_over.fit_transform(movies["overview"]), W_OVERVIEW),
    (v_kw.fit_transform(movies["keywords"]), W_KEYWORDS),
    (v_gen.fit_transform(movies["genre_list"]), W_GENRES),
    (v_dir.fit_transform(movies["director"].str.lower()), W_DIRECTOR),
    (v_cast.fit_transform(movies["cast"]), W_CAST),
]
X = sp.hstack([normalize(b) * w for b, w in blocks]).tocsr()
X = normalize(X).astype(np.float32)

# quality score (IMDB-style weighted rating, scaled to 0..1) 
R, v = movies["vote_average"].astype(float), movies["vote_count"].astype(float)
C, m = R.mean(), v.quantile(0.60)
wr = v / (v + m) * R + m / (v + m) * C
movies["wr"] = wr
movies["quality"] = ((wr - wr.min()) / (wr.max() - wr.min())).astype(np.float32)

# free-text search matrix (for "describe a vibe") 
docs = (movies["title"] + " " + movies["genres"] + " " + movies["keywords"] + " "
        + movies["overview"])
v_txt = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2,
                        sublinear_tf=True, max_features=40000, dtype=np.float32)
T = v_txt.fit_transform(docs)

# calibration: what does a "100% match" look like? 
S = (X @ X.T).toarray()
np.fill_diagonal(S, 0)
top1 = S.max(axis=1)
ref = float(np.percentile(top1, 95))

#  keyword idf (used for "shared themes" explanations) 
kw_idf = dict(zip(v_kw.get_feature_names_out(), v_kw.idf_))

lite = movies[["title", "genre_list", "keywords", "overview", "director", "cast",
               "vote_average", "vote_count", "popularity", "quality"]].copy()
lite["genre_list"] = lite["genre_list"].apply(lambda g: "|".join(g))

# portable formats only (CSV / JSON / NPY / NPZ) -> work with any pandas/numpy version
lite.to_csv("data/movies_lite.csv.gz", index=False)
sp.save_npz("data/features.npz", X, compressed=True)
sp.save_npz("data/text_index.npz", T.tocsr(), compressed=True)
np.save("data/txt_idf.npy", v_txt.idf_.astype(np.float32))
json.dump({"ref": ref, "kw_idf": {k: float(v) for k, v in kw_idf.items()},
           "txt_vocab": {k: int(v) for k, v in v_txt.vocabulary_.items()}},
          open("data/meta.json", "w"))
print("movies:", len(lite), "| features:", X.shape, "| ref cosine:", round(ref, 3))
