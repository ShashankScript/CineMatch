
"""Recommendation engine (pure Python – no Streamlit, easy to test)."""

from __future__ import annotations  # Supports modern type hints and forward references

import json  # Reads metadata stored in JSON format
import re  # Performs text processing and pattern matching
from dataclasses import dataclass, field  # Creates structured data classes
from pathlib import Path  # Handles file and folder paths

import numpy as np  # Performs numerical calculations
import pandas as pd  # Loads and processes movie datasets
import scipy.sparse as sp  # Stores large matrices efficiently
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # Common English words to ignore
from sklearn.preprocessing import normalize  # Normalizes feature vectors


DATA = Path(__file__).parent / "data"  # Locates the data folder beside this Python file


# Converts common user words into related words available in the dataset
SYNONYMS = {
    "sci-fi": "science fiction",
    "scifi": "science fiction",
    "sci fi": "science fiction",
    "scary": "horror terrifying",
    "spooky": "horror",
    "funny": "comedy humor",
    "hilarious": "comedy humor",
    "romantic": "romance love",
    "rom-com": "romance comedy",
    "cartoon": "animation",
    "kids": "family animation",
    "feel-good": "comedy family friendship",
    "twist": "twist mystery",
    "heist": "heist robbery",
    "space": "space astronaut",
    "zombies": "zombie",
    "superhero": "superhero marvel dc comic",
    "true story": "based on true story biography",
    "detective": "detective murder investigation",
    "mind-bending": "dream reality mind psychological",
    "epic": "epic war battle adventure",
    "emotional": "drama loss family love",
    "cozy": "family friendship comedy",
}


@dataclass  # Automatically creates methods like __init__ for storing recommendation data
class Rec:
    idx: int  # Stores the movie's row index
    title: str  # Stores the movie title
    cosine: float  # Stores the cosine similarity score
    match: int  # Stores the match percentage (0-100)
    reasons: list[str] = field(default_factory=list)  # Stores explanations for recommendations


class Engine:

    def __init__(self):  # Constructor: runs automatically when an Engine object is created

        # Loads the movie dataset from a compressed CSV file
        self.movies: pd.DataFrame = pd.read_csv(
            DATA / "movies_lite.csv.gz",
            keep_default_na=False
        )

        # Splits genres separated by | into individual Python list elements
        self.movies["genre_list"] = self.movies["genre_list"].apply(
            lambda s: [g for g in str(s).split("|") if g]
        )

        # Loads the precomputed movie feature matrix
        self.X = sp.load_npz(DATA / "features.npz").tocsr()

        # Loads the precomputed text feature matrix used for text search
        self.T = sp.load_npz(DATA / "text_index.npz").tocsr()

        # Reads metadata containing reference values, vocabulary and IDF information
        meta = json.load(open(DATA / "meta.json", encoding="utf-8"))

        self.ref: float = meta["ref"]  # Reference similarity value for match percentage
        self.kw_idf: dict = meta["kw_idf"]  # Stores keyword importance weights
        self.txt_vocab: dict = meta["txt_vocab"]  # Maps text terms to matrix column indices

        # Loads precomputed IDF weights for text search
        self.txt_idf = np.load(DATA / "txt_idf.npy")

        m = self.movies  # Creates a shorter reference to the movie DataFrame

        self.titles = m["title"].tolist()  # Converts all movie titles into a Python list

        self._title_to_idx = {}  # Creates a dictionary to map titles to row indices

        for i, t in enumerate(self.titles):
            self._title_to_idx.setdefault(t, i)  # Stores the first index of each movie title

        # Converts quality scores into a NumPy array
        self.quality = m["quality"].to_numpy(dtype=np.float32)

        # Stores movie ratings in a NumPy array
        self.rating = m["vote_average"].to_numpy(dtype=np.float32)

        # Stores the total number of votes received by each movie
        self.votes = m["vote_count"].to_numpy()

        # Converts movie genres into sets for faster comparison
        self.genre_sets = [set(g) for g in m["genre_list"]]

        # Collects all unique genres available in the dataset
        self.all_genres = sorted({g for s in self.genre_sets for g in s})

        # Converts movie keywords into sets for quick comparison
        self._kw_tokens = [set(k.split()) for k in m["keywords"]]



    def index_of(self, title: str) -> int:
        # Returns the row index of a movie using its title
        return self._title_to_idx[title]


    def _match_pct(self, cos: float) -> int:
        # Converts cosine similarity into a calibrated percentage between 0 and 99
        return int(round(min(0.99, max(0.0, cos / self.ref)) * 100))


    def _mask(self, genres, min_rating, exclude):
        # Initially selects movies having ratings greater than or equal to the minimum rating
        ok = self.rating >= float(min_rating)

        # Removes movies having fewer than 20 votes
        ok &= self.votes >= 20

        if genres:
            wanted = set(genres)  # Converts selected genres into a set

            # Keeps movies having at least one matching genre
            ok &= np.array([bool(wanted & s) for s in self.genre_sets])

        for i in exclude:
            ok[i] = False  # Excludes movies already selected by the user

        return ok  # Returns a Boolean array of eligible movies


    @staticmethod
    def _boost(quality, weight):
        # Adjusts scores according to movie quality
        # Higher quality increases the score, while lower quality reduces it
        return 1.0 + weight * (quality - 0.5) * 2.0



    def recommend(self, titles: list[str], n: int = 10, genres=None,
                  min_rating: float = 0.0, quality_weight: float = 0.15,
                  pool: int | None = None) -> list[Rec]:

        # Converts selected movie titles into their corresponding row indices
        idxs = [self.index_of(t) for t in titles]

        # Returns an empty list if the user has not selected any movies
        if not idxs:
            return []

        # Calculates the average feature vector of selected movies
        # This represents the user's overall movie preference
        q = normalize(sp.csr_matrix(self.X[idxs].mean(axis=0)))

        # Calculates cosine similarity between the preference vector and every movie
        sims = (self.X @ q.T).toarray().ravel()

        # Adjusts similarity scores using the movie quality factor
        score = sims * self._boost(self.quality, quality_weight)

        # Removes movies that do not satisfy genre, rating and exclusion filters
        score[~self._mask(genres, min_rating, idxs)] = -1

        # Determines the number of top recommendation candidates
        k = pool or n

        # Finds the indices of the highest-scoring movies without sorting everything
        top = np.argpartition(-score, min(k, len(score) - 1))[:k]

        # Sorts the selected movies in descending order of recommendation score
        top = top[np.argsort(-score[top])]

        out = []  # Stores the final recommendation objects

        for i in top:

            # Skips movies having zero or negative scores
            if score[i] <= 0:
                continue

            # Generates explanations for why this movie was recommended
            reasons = self._explain(idxs, int(i))

            # Creates a recommendation object containing movie details and scores
            out.append(Rec(
                int(i),
                self.titles[i],
                float(sims[i]),
                self._match_pct(float(sims[i])),
                reasons
            ))

        return out  # Returns the final ranked movie recommendations



    def _explain(self, src_idxs: list[int], j: int) -> list[str]:

        # Finds the selected movie that is most similar to the recommended movie
        best = max(
            src_idxs,
            key=lambda s: float(self.X[s].multiply(self.X[j]).sum())
        )

        # Retrieves information about the selected and recommended movies
        a, b = self.movies.iloc[best], self.movies.iloc[j]

        reasons = []  # Stores explanation messages

        # Finds genres common to both movies
        shared_g = [g for g in a["genre_list"] if g in self.genre_sets[j]]

        if shared_g:
            reasons.append("Genres: " + ", ".join(shared_g[:3]))

        # Checks whether both movies have the same director
        if a["director"] and a["director"] == b["director"]:
            reasons.append(f"Same director: {a['director']}")

        # Finds common actors in both movies
        cast = self._shared_cast(a["cast"], b["cast"])

        if cast:
            reasons.append("Cast: " + ", ".join(cast[:2]))

        # Finds common keywords or themes between both movies
        shared_kw = self._kw_tokens[best] & self._kw_tokens[j]

        # Removes short keywords and sorts the remaining ones by IDF importance
        shared_kw = sorted(
            (t for t in shared_kw if len(t) > 3),
            key=lambda t: -self.kw_idf.get(t, 0)
        )[:3]

        if shared_kw:
            reasons.append("Themes: " + ", ".join(shared_kw))

        # If multiple movies were selected, identifies the closest matching movie
        if len(src_idxs) > 1:
            reasons.append(f"Closest to: {self.titles[best]}")

        return reasons  # Returns all explanation reasons



    @staticmethod
    def _shared_cast(a: str, b: str) -> list[str]:

        # Splits both cast strings into individual words
        ta, tb = a.split(), b.split()

        # Creates pairs of consecutive words from the second movie's cast
        bigrams_b = {(x, y) for x, y in zip(tb, tb[1:])}

        # Finds matching consecutive word pairs in both movie casts
        hits = [
            i for i, p in enumerate(zip(ta, ta[1:]))
            if p in bigrams_b
        ]

        names, run = [], []

        # Combines overlapping matching pairs into complete names
        for i in hits:
            if run and i == run[-1] + 1:
                run.append(i)
            else:
                if run:
                    names.append(run)
                run = [i]

        if run:
            names.append(run)

        out = []

        # Converts matched word groups back into actor names
        for r in names:
            words = ta[r[0]: r[-1] + 2]

            # Handles multiple two-word names joined together
            if len(words) in (4, 6, 8) and len(words) > 3:
                out += [
                    " ".join(words[i:i + 2])
                    for i in range(0, len(words), 2)
                ]
            else:
                out.append(" ".join(words))

        return out  # Returns common cast names


    def _vectorise_query(self, text: str):
        """Converts a text query into a TF-IDF sparse vector."""

        # Extracts words containing at least two characters
        # Converts text into lowercase and removes common English stop words
        toks = [
            t for t in re.findall(r"(?u)\b\w\w+\b", text.lower())
            if t not in ENGLISH_STOP_WORDS
        ]

        # Creates bigrams (two consecutive words) along with individual words
        terms = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]

        counts: dict[int, int] = {}  # Stores the frequency of each recognized term

        for t in terms:

            # Finds the column index of the term in the saved vocabulary
            j = self.txt_vocab.get(t)

            if j is not None:
                counts[j] = counts.get(j, 0) + 1  # Counts term frequency

        # Returns None if no query terms exist in the vocabulary
        if not counts:
            return None

        # Extracts vocabulary column indices
        cols = np.array(list(counts))

        # Calculates sublinear TF-IDF values using term frequency and IDF
        vals = (
            (1 + np.log(np.array(list(counts.values()))))
            * self.txt_idf[cols]
        )

        # Normalizes the TF-IDF vector to unit length
        vals = vals / np.linalg.norm(vals)

        # Creates a sparse matrix containing the query's TF-IDF values
        return sp.csr_matrix(
            (vals, (np.zeros(len(cols), dtype=int), cols)),
            shape=(1, self.T.shape[1]),
            dtype=np.float32
        )


    def search_text(self, query: str, n: int = 10, genres=None,
                    min_rating=0.0, quality_weight: float = 0.15) -> list[Rec]:

        # Converts the user's query into lowercase
        q = query.lower()

        # Expands the query using related words from the synonym dictionary
        for k, v in SYNONYMS.items():
            if k in q:
                q = q.replace(k, f"{k} {v}")

        # Converts the expanded query into a TF-IDF vector
        qv = self._vectorise_query(q)

        # Returns no results if the query has no recognized vocabulary terms
        if qv is None:
            return []

        # Calculates cosine similarity between the query and every movie's text features
        sims = (self.T @ qv.T).toarray().ravel()

        # Finds dataset genres explicitly mentioned in the user's query
        wanted = {g for g in self.all_genres if g.lower() in q}

        if wanted:

            # Calculates the fraction of requested genres present in each movie
            frac = np.array([
                len(wanted & s) / len(wanted)
                for s in self.genre_sets
            ])

            # Rewards movies that contain the requested genres
            sims = sims * (0.6 + 0.8 * frac)

        # Adjusts similarity scores according to movie quality
        score = sims * self._boost(self.quality, max(quality_weight, 0.3))

        # Removes movies that do not satisfy the selected filters
        score[~self._mask(genres, min_rating, [])] = -1

        # Selects the top N movies based on their recommendation scores
        top = np.argsort(-score)[:n]

        # Finds the highest similarity score for match percentage calculation
        peak = max(float(sims[top[0]]), 1e-9) if len(top) else 1

        out = []

        for i in top:

            # Skips movies with invalid or zero recommendation scores
            if score[i] <= 0:
                continue

            # Calculates match percentage relative to the highest similarity result
            pct = int(round(min(0.99, float(sims[i]) / peak) * 100))

            # Creates a recommendation object with movie details and genre explanation
            out.append(Rec(
                int(i),
                self.titles[i],
                float(sims[i]),
                pct,
                ["Genres: " + ", ".join(
                    self.movies.iloc[i]["genre_list"][:3]
                )] if self.movies.iloc[i]["genre_list"] else []
            ))

        return out  # Returns the final text-based recommendations



    def top_rated(self, n: int = 10, genres=None) -> list[int]:

        # Copies movie quality scores
        score = self.quality.copy()

        # Applies genre, rating and vote-count filters
        score[~self._mask(genres, 0, [])] = -1

        # Returns the indices of the top N highest-quality movies
        return list(np.argsort(-score)[:n])



    def row(self, idx: int):

        # Returns the complete dataset row for the given movie index
        return self.movies.iloc[idx]
