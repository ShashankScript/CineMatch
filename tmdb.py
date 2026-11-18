
"""Small TMDB client (no Streamlit imports).
All functions fail soft: they return None / [] on error.
"""

from __future__ import annotations  # Supports modern type hints

import math  # Used for logarithmic calculations
import re  # Used for extracting words from text

import requests  # Sends HTTP requests to the TMDB API
from requests.adapters import HTTPAdapter  # Configures connection handling
from urllib3.util.retry import Retry  # Automatically retries failed API requests


API = "https://api.themoviedb.org/3"  # Base URL of TMDB API
IMG = "https://image.tmdb.org/t/p"  # Base URL for TMDB movie images

_session = None  # Stores a reusable HTTP session


# ---------------- SESSION MANAGEMENT ----------------

def session() -> requests.Session:
    global _session  # Allows this function to access the global session variable

    if _session is None:  # Creates a session only if one does not already exist

        s = requests.Session()  # Creates a reusable HTTP session

        # Configures automatic retry for temporary API failures
        retry = Retry(
            total=2,  # Retries a failed request up to 2 times
            backoff_factor=0.6,  # Adds a delay between retry attempts
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes that trigger retries
            allowed_methods=["GET"]  # Retries only GET requests
        )

        # Mounts the retry adapter for HTTPS connections
        s.mount(
            "https://",
            HTTPAdapter(max_retries=retry, pool_maxsize=20)
        )

        _session = s  # Saves the session for future API calls

    return _session  # Returns the reusable session


# ---------------- API REQUEST FUNCTION ----------------

def _get(path: str, api_key: str, **params):

    try:
        # Sends a GET request to the TMDB API
        # Automatically adds the API key and other parameters
        r = session().get(
            f"{API}{path}",
            params={"api_key": api_key, **params},
            timeout=10  # Stops waiting after 10 seconds
        )

        # Converts the response into JSON if the HTTP status is 200
        # Otherwise returns None
        return r.json() if r.status_code == 200 else None

    except Exception:
        # Handles network errors, timeouts and other request failures
        return None


# ---------------- TEXT TOKENIZATION ----------------

def _tokens(text: str) -> set[str]:

    # Converts text into lowercase words and removes words shorter than 4 characters
    # Returns a set to remove duplicate words
    return {
        t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 3
    }


# ---------------- FIND THE CORRECT MOVIE ----------------

def find_movie(title: str, overview: str, vote_count: int, api_key: str):
    """Searches TMDB by title and identifies the most suitable matching movie.

    It handles remakes and movies having identical titles by comparing
    their plot descriptions and vote counts.
    """

    # Searches TMDB for movies matching the given title
    data = _get(
        "/search/movie",
        api_key,
        query=title,
        include_adult="false"
    )

    # Returns None if the API response is empty or contains no results
    if not data or not data.get("results"):
        return None

    # Converts our dataset's movie overview into a set of unique words
    ours = _tokens(overview)

    best, best_score = None, -1.0  # Stores the best matching movie and its score

    # Checks the first 10 search results to identify the correct movie
    for r in data["results"][:10]:

        # Gives 3 extra points if the movie title matches exactly
        score = 3.0 if (
            r.get("title", "").lower().strip()
            == title.lower().strip()
        ) else 0.0

        # Extracts important words from the TMDB movie overview
        theirs = _tokens(r.get("overview", ""))

        if ours and theirs:

            # Calculates Jaccard similarity between the two movie overviews
            # Common words / Total unique words
            score += 4.0 * len(ours & theirs) / len(ours | theirs)

        # Compares vote counts using logarithms
        # A large difference in vote counts reduces the matching score
        score -= 0.15 * abs(
            math.log1p(r.get("vote_count", 0))
            - math.log1p(vote_count)
        )

        # Updates the best matching movie if the current score is higher
        if score > best_score:
            best, best_score = r, score

    return best  # Returns the most suitable TMDB movie result


# ---------------- POSTER AND IMAGE URL ----------------

def poster(path: str | None, size: str = "w500"):

    # Builds the complete TMDB image URL using the requested image size
    # Returns None if the movie does not have an image path
    return f"{IMG}/{size}{path}" if path else None


# ---------------- COMPLETE MOVIE DETAILS ----------------

def details(tmdb_id: int, api_key: str, region: str = "IN"):

    # Fetches movie details along with videos, credits and watch providers
    d = _get(
        f"/movie/{tmdb_id}",
        api_key,
        append_to_response="videos,credits,watch/providers"
    )

    # Returns None if TMDB does not provide movie details
    if not d:
        return None

    # ---------------- TRAILER EXTRACTION ----------------

    trailer = None  # Initially assumes no trailer is available

    # Checks all available movie videos
    for v in (d.get("videos") or {}).get("results", []):

        # Selects YouTube videos classified as trailers
        if v.get("site") == "YouTube" and v.get("type") == "Trailer":

            # Creates the YouTube trailer URL using its video key
            trailer = f"https://www.youtube.com/watch?v={v['key']}"

            # Stops searching if an official trailer is found
            if v.get("official"):
                break

    # ---------------- CAST EXTRACTION ----------------

    # Extracts the first 8 actors with their character names and profile photos
    cast = [
        {
            "name": c["name"],  # Actor's real name
            "character": c.get("character", ""),  # Character played in the movie
            "photo": poster(c.get("profile_path"), "w185")  # Actor's profile image
        }
        for c in (d.get("credits") or {}).get("cast", [])[:8]
    ]

    # Finds the director from the movie's crew information
    director = next(
        (
            c["name"]
            for c in (d.get("credits") or {}).get("crew", [])
            if c.get("job") == "Director"
        ),
        None  # Returns None if no director is found
    )

    # ---------------- STREAMING PROVIDERS ----------------

    # Gets the watch provider information for the selected country
    prov = (
        (d.get("watch/providers") or {}).get("results") or {}
    ).get(region) or {}

    # Organizes available streaming, rental and purchase platforms
    where = {
        # Platforms where the movie is available for streaming
        "stream": [
            p["provider_name"]
            for p in prov.get("flatrate", [])
        ],

        # Platforms where the movie can be rented
        "rent": [
            p["provider_name"]
            for p in prov.get("rent", [])
        ],

        # Platforms where the movie can be purchased
        "buy": [
            p["provider_name"]
            for p in prov.get("buy", [])
        ],

        # Link to TMDB's watch-provider information
        "link": prov.get("link"),
    }

    # ---------------- FINAL MOVIE INFORMATION ----------------

    # Returns all collected movie information in a Python dictionary
    return {

        "id": d["id"],  # TMDB's unique movie ID

        "title": d.get("title"),  # Movie title

        "tagline": d.get("tagline") or "",  # Short promotional tagline

        "overview": d.get("overview") or "",  # Movie plot description

        # Extracts the release year from the release date
        # Returns N/A if the release date is unavailable
        "year": (d.get("release_date") or "")[:4] or "N/A",

        "runtime": d.get("runtime") or 0,  # Movie duration in minutes

        "rating": d.get("vote_average") or 0,  # Average user rating

        "votes": d.get("vote_count") or 0,  # Total number of votes

        # Extracts all movie genres
        "genres": [
            g["name"]
            for g in d.get("genres", [])
        ],

        # Generates the movie poster URL
        "poster": poster(d.get("poster_path")),

        # Generates a high-resolution backdrop image URL
        "backdrop": poster(d.get("backdrop_path"), "w1280"),

        "trailer": trailer,  # YouTube trailer URL

        "cast": cast,  # List of actors and their characters

        "director": director,  # Movie director's name

        "where": where,  # Streaming, rental and purchase information

        # Creates the IMDb movie URL if an IMDb ID is available
        "imdb": (
            f"https://www.imdb.com/title/{d['imdb_id']}"
            if d.get("imdb_id") else None
        ),
    }


# ---------------- COMPLETE MOVIE CARD DATA ----------------

def card_data(
    title: str,
    overview: str,
    vote_count: int,
    api_key: str,
    region: str = "IN"
):
    """Fetches everything required to display one movie card."""

    # First searches TMDB to identify the correct movie
    hit = find_movie(title, overview, vote_count, api_key)

    # If a matching movie is found, fetches its complete details
    # Otherwise returns None
    return details(hit["id"], api_key, region) if hit else None


# ---------------- TRENDING MOVIES ----------------

def trending(api_key: str, n: int = 10):

    # Fetches movies trending worldwide over the current week
    data = _get("/trending/movie/week", api_key)

    # Returns an empty list if the API request fails
    if not data:
        return []

    # Extracts the required information from the first N trending movies
    return [
        {
            "id": m["id"],  # Unique TMDB movie ID

            "title": m["title"],  # Movie title

            # Generates the movie poster URL
            "poster": poster(m.get("poster_path")),

            # Extracts the movie's average rating
            "rating": m.get("vote_average") or 0,

            # Extracts the release year
            "year": (m.get("release_date") or "")[:4]
        }
        for m in data.get("results", [])[:n]
    ]

