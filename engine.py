"""Recommendation engine (pure Python – no Streamlit, easy to test)."""
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

# plain-English words people use -> words that exist in the data
SYNONYMS = {
    "sci-fi": "science fiction", "scifi": "science fiction", "sci fi": "science fiction",
    "scary": "horror terrifying", "spooky": "horror", "funny": "comedy humor",
    "hilarious": "comedy humor", "romantic": "romance love", "rom-com": "romance comedy",
    "cartoon": "animation", "kids": "family animation", "feel-good": "comedy family friendship",
    "twist": "twist mystery", "heist": "heist robbery",
    "space": "space astronaut", "zombies": "zombie", "superhero": "superhero marvel dc comic",
    "true story": "based on true story biography", "detective": "detective murder investigation",
    "mind-bending": "dream reality mind psychological", "epic": "epic war battle adventure",
    "emotional": "drama loss family love", "cozy": "family friendship comedy",
}


@dataclass
class Rec:
    idx: int
    title: str
    cosine: float
    match: int                      # 0-100, calibrated
    reasons: list[str] = field(default_factory=list)


class Engine:
    def __init__(self):
        self.movies: pd.DataFrame = pd.read_csv(DATA / "movies_lite.csv.gz", keep_default_na=False)
        self.movies["genre_list"] = self.movies["genre_list"].apply(
            lambda s: [g for g in str(s).split("|") if g])
        self.X = sp.load_npz(DATA / "features.npz").tocsr()
        self.T = sp.load_npz(DATA / "text_index.npz").tocsr()
        meta = json.load(open(DATA / "meta.json", encoding="utf-8"))
        self.ref: float = meta["ref"]
        self.kw_idf: dict = meta["kw_idf"]
        self.txt_vocab: dict = meta["txt_vocab"]
        self.txt_idf = np.load(DATA / "txt_idf.npy") # load data

        m = self.movies
        self.titles = m["title"].tolist()
        self._title_to_idx = {}
        for i, t in enumerate(self.titles):
            self._title_to_idx.setdefault(t, i)
        self.quality = m["quality"].to_numpy(dtype=np.float32)
        self.rating = m["vote_average"].to_numpy(dtype=np.float32)
        self.votes = m["vote_count"].to_numpy()
        self.genre_sets = [set(g) for g in m["genre_list"]]
        self.all_genres = sorted({g for s in self.genre_sets for g in s})
        self._kw_tokens = [set(k.split()) for k in m["keywords"]]  # builds quick lookups

    # utils
    def index_of(self, title: str) -> int:
        return self._title_to_idx[title]

    def _match_pct(self, cos: float) -> int:
        return int(round(min(0.99, max(0.0, cos / self.ref)) * 100)) # computes match %

    def _mask(self, genres, min_rating, exclude): # _mask is the filter, keeps movie ratings
        ok = self.rating >= float(min_rating)
        ok &= self.votes >= 20
        if genres:
            wanted = set(genres)
            ok &= np.array([bool(wanted & s) for s in self.genre_sets])
        for i in exclude:
            ok[i] = False
        return ok

    @staticmethod
    def _boost(quality, weight):
        # multiplicative: similarity decides, quality only nudges (+/- weight)
        return 1.0 + weight * (quality - 0.5) * 2.0

    # recommendations
    def recommend(self, titles: list[str], n: int = 10, genres=None,
                  min_rating: float = 0.0, quality_weight: float = 0.15,
                  pool: int | None = None) -> list[Rec]:
        idxs = [self.index_of(t) for t in titles]
        if not idxs:
            return []
        q = normalize(sp.csr_matrix(self.X[idxs].mean(axis=0)))
        sims = (self.X @ q.T).toarray().ravel()
        score = sims * self._boost(self.quality, quality_weight)
        score[~self._mask(genres, min_rating, idxs)] = -1
        k = pool or n
        top = np.argpartition(-score, min(k, len(score) - 1))[:k]
        top = top[np.argsort(-score[top])]
        out = []
        for i in top:
            if score[i] <= 0:
                continue
            reasons = self._explain(idxs, int(i))
            out.append(Rec(int(i), self.titles[i], float(sims[i]),
                           self._match_pct(float(sims[i])), reasons))
        return out

    def _explain(self, src_idxs: list[int], j: int) -> list[str]:
        # explain against the selected movie that is most similar to candidate j
        best = max(src_idxs, key=lambda s: float(self.X[s].multiply(self.X[j]).sum()))
        a, b = self.movies.iloc[best], self.movies.iloc[j]
        reasons = []
        shared_g = [g for g in a["genre_list"] if g in self.genre_sets[j]]
        if shared_g:
            reasons.append("Genres: " + ", ".join(shared_g[:3]))
        if a["director"] and a["director"] == b["director"]:
            reasons.append(f"Same director: {a['director']}")
        cast = self._shared_cast(a["cast"], b["cast"])
        if cast:
            reasons.append("Cast: " + ", ".join(cast[:2]))
        shared_kw = self._kw_tokens[best] & self._kw_tokens[j]
        shared_kw = sorted((t for t in shared_kw if len(t) > 3),
                           key=lambda t: -self.kw_idf.get(t, 0))[:3]
        if shared_kw:
            reasons.append("Themes: " + ", ".join(shared_kw))
        if len(src_idxs) > 1:
            reasons.append(f"Closest to: {self.titles[best]}")
        return reasons

    @staticmethod
    def _shared_cast(a: str, b: str) -> list[str]:
        ta, tb = a.split(), b.split()
        bigrams_b = {(x, y) for x, y in zip(tb, tb[1:])}
        hits = [i for i, p in enumerate(zip(ta, ta[1:])) if p in bigrams_b]
        names, run = [], []
        for i in hits:                       # merge overlapping bigrams: "Samuel L." + "L. Jackson"
            if run and i == run[-1] + 1:
                run.append(i)
            else:
                if run:
                    names.append(run)
                run = [i]
        if run:
            names.append(run)
        out = []
        for r in names:
            words = ta[r[0]: r[-1] + 2]
            if len(words) in (4, 6, 8) and len(words) > 3:   # several two-word names glued together
                out += [" ".join(words[i:i + 2]) for i in range(0, len(words), 2)]
            else:
                out.append(" ".join(words))
        return out

    #  vibe search
    def _vectorise_query(self, text: str):
        """Same maths as TfidfVectorizer(stop_words='english', ngram_range=(1,2), sublinear_tf=True)."""
        toks = [t for t in re.findall(r"(?u)\b\w\w+\b", text.lower()) if t not in ENGLISH_STOP_WORDS]
        terms = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
        counts: dict[int, int] = {}
        for t in terms:
            j = self.txt_vocab.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        if not counts:
            return None
        cols = np.array(list(counts))
        vals = (1 + np.log(np.array(list(counts.values())))) * self.txt_idf[cols]
        vals = vals / np.linalg.norm(vals)
        return sp.csr_matrix((vals, (np.zeros(len(cols), dtype=int), cols)),
                             shape=(1, self.T.shape[1]), dtype=np.float32)

    def search_text(self, query: str, n: int = 10, genres=None, min_rating=0.0,
                    quality_weight: float = 0.15) -> list[Rec]:
        q = query.lower()
        for k, v in SYNONYMS.items():
            if k in q:
                q = q.replace(k, f"{k} {v}")
        qv = self._vectorise_query(q)
        if qv is None:
            return []
        sims = (self.T @ qv.T).toarray().ravel()
        wanted = {g for g in self.all_genres if g.lower() in q}
        if wanted:                                   # reward movies that actually carry the genre
            frac = np.array([len(wanted & s) / len(wanted) for s in self.genre_sets])
            sims = sims * (0.6 + 0.8 * frac)
        score = sims * self._boost(self.quality, max(quality_weight, 0.3))
        score[~self._mask(genres, min_rating, [])] = -1
        top = np.argsort(-score)[:n]
        peak = max(float(sims[top[0]]), 1e-9) if len(top) else 1
        out = []
        for i in top:
            if score[i] <= 0:
                continue
            pct = int(round(min(0.99, float(sims[i]) / peak) * 100))
            out.append(Rec(int(i), self.titles[i], float(sims[i]), pct,
                           ["Genres: " + ", ".join(self.movies.iloc[i]["genre_list"][:3])]
                           if self.movies.iloc[i]["genre_list"] else []))
        return out

    #  top rated
    def top_rated(self, n: int = 10, genres=None) -> list[int]:
        score = self.quality.copy()
        score[~self._mask(genres, 0, [])] = -1
        return list(np.argsort(-score)[:n])

    def row(self, idx: int):
        return self.movies.iloc[idx]
