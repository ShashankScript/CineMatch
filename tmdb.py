"""Small TMDB client. All functions fail soft: they return None / [] on error."""

from __future__ import annotations

import math
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://api.themoviedb.org/3"
IMG = "https://image.tmdb.org/t/p"

_session = None


# Creates a reusable HTTP session with automatic retries
def session() -> requests.Session:
    global _session

    if _session is None:
        s = requests.Session()

        retry = Retry(
            total=2,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        s.mount(
            "https://",
            HTTPAdapter(max_retries=retry, pool_maxsize=20)
        )

        _session = s

    return _session


# Sends GET requests to TMDB API and handles errors
def _get(path: str, api_key: str, **params):
    try:
        r = session().get(
            f"{API}{path}",
            params={"api_key": api_key, **params},
            timeout=10
        )

        return r.json() if r.status_code == 200 else None

    except Exception:
        return None


# Extracts unique words from movie descriptions
def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 3
    }


def find_movie(title: str, overview: str, vote_count: int, api_key: str):
    """Finds the correct movie by comparing title, overview and vote count."""

    data = _get(
        "/search/movie",
        api_key,
        query=title,
        include_adult="false"
    )

    if not data or not data.get("results"):
        return None

    ours = _tokens(overview)
    best, best_score = None, -1.0

    for r in data["results"][:10]:

        # Gives extra weight to exact title matches
        score = 3.0 if (
            r.get("title", "").lower().strip()
            == title.lower().strip()
        ) else 0.0

        theirs = _tokens(r.get("overview", ""))

        if ours and theirs:
            # Uses Jaccard similarity to compare movie descriptions
            score += 4.0 * len(ours & theirs) / len(ours | theirs)

        score -= 0.15 * abs(
            math.log1p(r.get("vote_count", 0))
            - math.log1p(vote_count)
        )

        if score > best_score:
            best, best_score = r, score

    return best


# Generates the complete URL of a movie poster
def poster(path: str | None, size: str = "w500"):
    return f"{IMG}/{size}{path}" if path else None


def details(tmdb_id: int, api_key: str, region: str = "IN"):

    # Fetches movie details, videos, credits and watch providers
    d = _get(
        f"/movie/{tmdb_id}",
        api_key,
        append_to_response="videos,credits,watch/providers"
    )

    if not d:
        return None

    trailer = None

    # Finds the official YouTube trailer
    for v in (d.get("videos") or {}).get("results", []):
        if v.get("site") == "YouTube" and v.get("type") == "Trailer":
            trailer = f"https://www.youtube.com/watch?v={v['key']}"

            if v.get("official"):
                break

    # Extracts the first 8 actors
    cast = [
        {
            "name": c["name"],
            "character": c.get("character", ""),
            "photo": poster(c.get("profile_path"), "w185")
        }
        for c in (d.get("credits") or {}).get("cast", [])[:8]
    ]

    # Finds the movie director
    director = next(
        (
            c["name"]
            for c in (d.get("credits") or {}).get("crew", [])
            if c.get("job") == "Director"
        ),
        None
    )

    # Retrieves streaming information for the selected region
    prov = (
        (d.get("watch/providers") or {}).get("results") or {}
    ).get(region) or {}

    where = {
        "stream": [
            p["provider_name"]
            for p in prov.get("flatrate", [])
        ],
        "rent": [
            p["provider_name"]
            for p in prov.get("rent", [])
        ],
        "buy": [
            p["provider_name"]
            for p in prov.get("buy", [])
        ],
        "link": prov.get("link"),
    }

    return {
        "id": d["id"],
        "title": d.get("title"),
        "tagline": d.get("tagline") or "",
        "overview": d.get("overview") or "",
        "year": (d.get("release_date") or "")[:4] or "N/A",
        "runtime": d.get("runtime") or 0,
        "rating": d.get("vote_average") or 0,
        "votes": d.get("vote_count") or 0,
        "genres": [g["name"] for g in d.get("genres", [])],
        "poster": poster(d.get("poster_path")),
        "backdrop": poster(d.get("backdrop_path"), "w1280"),
        "trailer": trailer,
        "cast": cast,
        "director": director,
        "where": where,
        "imdb": (
            f"https://www.imdb.com/title/{d['imdb_id']}"
            if d.get("imdb_id") else None
        ),
    }


def card_data(title: str, overview: str, vote_count: int,
              api_key: str, region: str = "IN"):
    """Fetches all information required for one movie card."""

    hit = find_movie(title, overview, vote_count, api_key)

    return details(hit["id"], api_key, region) if hit else None


# Fetches weekly trending movies from TMDB
def trending(api_key: str, n: int = 10):
    data = _get("/trending/movie/week", api_key)

    if not data:
        return []

    return [
        {
            "id": m["id"],
            "title": m["title"],
            "poster": poster(m.get("poster_path")),
            "rating": m.get("vote_average") or 0,
            "year": (m.get("release_date") or "")[:4]
        }
        for m in data.get("results", [])[:n]
    ]
