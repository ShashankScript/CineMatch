
"""Recommendation engine (pure Python – easy to test)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize


DATA = Path(__file__).parent / "data"


# RELATED SEARCH TERMS
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


# RECOMMENDATION DATA
@dataclass
class Rec:
    idx: int
    title: str
    cosine: float
    match: int
    reasons: list[str] = field(default_factory=list)


class Engine:

    def __init__(self):

        # Load movie dataset.
        self.movies: pd.DataFrame = pd.read_csv(
            DATA / "movies_lite.csv.gz",
            keep_default_na=False
        )

        # Convert genres into lists.
        self.movies["genre_list"] = self.movies["genre_list"].apply(
            lambda s: [g for g in str(s).split("|") if g]
        )

        # Load precomputed feature matrices.
        self.X = sp.load_npz(DATA / "features.npz").tocsr()
        self.T = sp.load_npz(DATA / "text_index.npz").tocsr()

        # Load model metadata.
        meta = json.load(open(DATA / "meta.json", encoding="utf-8"))

        self.ref: float = meta["ref"]
        self.kw_idf: dict = meta["kw_idf"]
        self.txt_vocab: dict = meta["txt_vocab"]

        # Load text IDF weights.
        self.txt_idf = np.load(DATA / "txt_idf.npy")

        m = self.movies

        self.titles = m["title"].tolist()

        # Map movie titles to their first index.
        self._title_to_idx = {}

        for i, t in enumerate(self.titles):
            self._title_to_idx.setdefault(t, i)

        # Store movie scores and ratings.
        self.quality = m["quality"].to_numpy(dtype=np.float32)
        self.rating = m["vote_average"].to_numpy(dtype=np.float32)
        self.votes = m["vote_count"].to_numpy()

        # Prepare genres and keywords for comparison.
        self.genre_sets = [set(g) for g in m["genre_list"]]
        self.all_genres = sorted({g for s in self.genre_sets for g in s})
        self._kw_tokens = [set(k.split()) for k in m["keywords"]]


    def index_of(self, title: str) -> int:
        # Get movie index from its title.
        return self._title_to_idx[title]


    def _match_pct(self, cos: float) -> int:
        # Convert similarity into a percentage.
        return int(round(min(0.99, max(0.0, cos / self.ref)) * 100))


    def _mask(self, genres, min_rating, exclude):
        # Filter by rating and vote count.
        ok = self.rating >= float(min_rating)
        ok &= self.votes >= 20

        if genres:
            wanted = set(genres)

            # Keep movies with matching genres.
            ok &= np.array([bool(wanted & s) for s in self.genre_sets])

        for i in exclude:
            ok[i] = False

        return ok


    @staticmethod
    def _boost(quality, weight):
        # Adjust scores using movie quality.
        return 1.0 + weight * (quality - 0.5) * 2.0


    def recommend(self, titles: list[str], n: int = 10, genres=None,
                  min_rating: float = 0.0, quality_weight: float = 0.15,
                  pool: int | None = None) -> list[Rec]:

        # Get indices of selected movies.
        idxs = [self.index_of(t) for t in titles]

        if not idxs:
            return []

        # Build the user's preference vector.
        q = normalize(sp.csr_matrix(self.X[idxs].mean(axis=0)))

        # Calculate similarity with all movies.
        sims = (self.X @ q.T).toarray().ravel()

        # Apply quality adjustment.
        score = sims * self._boost(self.quality, quality_weight)

        # Apply filters and exclude selected movies.
        score[~self._mask(genres, min_rating, idxs)] = -1

        # Select and rank top candidates.
        k = pool or n
        top = np.argpartition(-score, min(k, len(score) - 1))[:k]
        top = top[np.argsort(-score[top])]

        out = []

        for i in top:

            if score[i] <= 0:
                continue

            # Generate recommendation explanations.
            reasons = self._explain(idxs, int(i))

            out.append(Rec(
                int(i),
                self.titles[i],
                float(sims[i]),
                self._match_pct(float(sims[i])),
                reasons
            ))

        return out


    def _explain(self, src_idxs: list[int], j: int) -> list[str]:

        # Find the closest selected movie.
        best = max(
            src_idxs,
            key=lambda s: float(self.X[s].multiply(self.X[j]).sum())
        )

        a, b = self.movies.iloc[best], self.movies.iloc[j]

        reasons = []

        # Find shared genres.
        shared_g = [g for g in a["genre_list"] if g in self.genre_sets[j]]

        if shared_g:
            reasons.append("Genres: " + ", ".join(shared_g[:3]))

        # Check for the same director.
        if a["director"] and a["director"] == b["director"]:
            reasons.append(f"Same director: {a['director']}")

        # Find common actors.
        cast = self._shared_cast(a["cast"], b["cast"])

        if cast:
            reasons.append("Cast: " + ", ".join(cast[:2]))

        # Find shared keywords.
        shared_kw = self._kw_tokens[best] & self._kw_tokens[j]

        # Rank keywords by IDF importance.
        shared_kw = sorted(
            (t for t in shared_kw if len(t) > 3),
            key=lambda t: -self.kw_idf.get(t, 0)
        )[:3]

        if shared_kw:
            reasons.append("Themes: " + ", ".join(shared_kw))

        # Identify the closest selected movie.
        if len(src_idxs) > 1:
            reasons.append(f"Closest to: {self.titles[best]}")

        return reasons


    @staticmethod
    def _shared_cast(a: str, b: str) -> list[str]:

        # Split cast names into words.
        ta, tb = a.split(), b.split()

        # Create bigrams from the second cast.
        bigrams_b = {(x, y) for x, y in zip(tb, tb[1:])}

        # Find matching bigrams.
        hits = [
            i for i, p in enumerate(zip(ta, ta[1:]))
            if p in bigrams_b
        ]

        names, run = [], []

        # Combine consecutive matching pairs.
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

        # Convert matching words into actor names.
        for r in names:
            words = ta[r[0]: r[-1] + 2]

            if len(words) in (4, 6, 8) and len(words) > 3:
                out += [
                    " ".join(words[i:i + 2])
                    for i in range(0, len(words), 2)
                ]
            else:
                out.append(" ".join(words))

        return out


    def _vectorise_query(self, text: str):
        """Convert text into a TF-IDF vector."""

        # Tokenize text and remove stop words.
        toks = [
            t for t in re.findall(r"(?u)\b\w\w+\b", text.lower())
            if t not in ENGLISH_STOP_WORDS
        ]

        # Generate unigrams and bigrams.
        terms = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]

        counts: dict[int, int] = {}

        for t in terms:

            # Find term in saved vocabulary.
            j = self.txt_vocab.get(t)

            if j is not None:
                counts[j] = counts.get(j, 0) + 1

        if not counts:
            return None

        # Get vocabulary indices.
        cols = np.array(list(counts))

        # Calculate TF-IDF values.
        vals = (
            (1 + np.log(np.array(list(counts.values()))))
            * self.txt_idf[cols]
        )

        # Normalize the query vector.
        vals = vals / np.linalg.norm(vals)

        # Create sparse query matrix.
        return sp.csr_matrix(
            (vals, (np.zeros(len(cols), dtype=int), cols)),
            shape=(1, self.T.shape[1]),
            dtype=np.float32
        )


    def search_text(self, query: str, n: int = 10, genres=None,
                    min_rating=0.0, quality_weight: float = 0.15) -> list[Rec]:

        # Expand query using related terms.
        q = query.lower()

        for k, v in SYNONYMS.items():
            if k in q:
                q = q.replace(k, f"{k} {v}")

        # Convert query into a TF-IDF vector.
        qv = self._vectorise_query(q)

        if qv is None:
            return []

        # Calculate text similarity with all movies.
        sims = (self.T @ qv.T).toarray().ravel()

        # Find genres mentioned in the query.
        wanted = {g for g in self.all_genres if g.lower() in q}

        if wanted:

            # Boost movies matching requested genres.
            frac = np.array([
                len(wanted & s) / len(wanted)
                for s in self.genre_sets
            ])

            sims = sims * (0.6 + 0.8 * frac)

        # Apply quality adjustment and filters.
        score = sims * self._boost(self.quality, max(quality_weight, 0.3))
        score[~self._mask(genres, min_rating, [])] = -1

        # Select top recommendations.
        top = np.argsort(-score)[:n]

        # Get highest similarity for percentage calculation.
        peak = max(float(sims[top[0]]), 1e-9) if len(top) else 1

        out = []

        for i in top:

            if score[i] <= 0:
                continue

            # Calculate relative match percentage.
            pct = int(round(min(0.99, float(sims[i]) / peak) * 100))

            out.append(Rec(
                int(i),
                self.titles[i],
                float(sims[i]),
                pct,
                ["Genres: " + ", ".join(
                    self.movies.iloc[i]["genre_list"][:3]
                )] if self.movies.iloc[i]["genre_list"] else []
            ))

        return out


    def top_rated(self, n: int = 10, genres=None) -> list[int]:

        # Rank movies by quality.
        score = self.quality.copy()

        # Apply genre and rating filters.
        score[~self._mask(genres, 0, [])] = -1

        return list(np.argsort(-score)[:n])


    def row(self, idx: int):

        # Get complete movie details by index.
        return self.movies.iloc[idx]
