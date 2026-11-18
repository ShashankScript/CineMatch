
"""CineMatch – content-based movie recommender (Streamlit)."""
from __future__ import annotations

import html
import os
import re
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

import tmdb
from engine import Engine

# Configure the Streamlit page
st.set_page_config(page_title="CineMatch", layout="wide",
                   initial_sidebar_state="expanded")

# Navigation pages
NAV = ["For You", "Vibe Search", "Discover", "Watchlist"]
NAV_ITEMS = [("For You", ":material/movie_filter:"),
             ("Vibe Search", ":material/search:"),
             ("Discover", ":material/explore:"),
             ("Watchlist", ":material/bookmark:")]

YEAR_MIN, YEAR_MAX = 1920, 2026

# Supported streaming regions and their country codes
REGIONS = {"India": "IN", "United States": "US", "United Kingdom": "GB", "Canada": "CA",
           "Australia": "AU", "Germany": "DE", "France": "FR", "Japan": "JP"}

# Example searches and starter movies
VIBES = ["dark sci-fi with a twist ending", "funny movie about friends on a road trip",
         "scary haunted house", "heartwarming animated family adventure",
         "tense heist with a clever plan", "epic war drama"]
STARTERS = ["Inception", "The Dark Knight", "Toy Story", "Avatar", "The Godfather"]


# RESOURCES AND CACHING

@st.cache_resource(show_spinner="Loading movie intelligence…")
def get_engine() -> Engine:
    return Engine()  # Load the recommendation engine once


@st.cache_resource
def get_store() -> dict:
    return {"cards": {}, "det": {}, "trend": (0.0, [])}  # Store cached API responses


def get_api_key() -> str:
    try:
        return str(st.secrets["TMDB_API_KEY"]).strip()  # Read key from Streamlit secrets
    except Exception:
        return os.environ.get("TMDB_API_KEY", "").strip()  # Fallback to environment variable


engine = get_engine()
STORE = get_store()
KEY = get_api_key()
HAS_KEY = bool(KEY)
DAY = 86400  # Cache duration: 24 hours


# TMDB API FUNCTIONS

def fetch_card(title: str, region: str):
    if not HAS_KEY:
        return None

    k = (title, region)
    hit = STORE["cards"].get(k)

    if hit and time.time() - hit[0] < DAY:
        return hit[1]  # Return cached movie details

    row = engine.row(engine.index_of(title))
    d = tmdb.card_data(title, row["overview"], int(row["vote_count"]), KEY, region)

    if d:                                   # Cache only successful responses
        STORE["cards"][k] = (time.time(), d)

    return d


def fetch_details(tmdb_id: int, region: str):
    k = (tmdb_id, region)
    hit = STORE["det"].get(k)

    if hit and time.time() - hit[0] < DAY:
        return hit[1]  # Avoid another API request if data is cached

    d = tmdb.details(tmdb_id, KEY, region)

    if d:
        STORE["det"][k] = (time.time(), d)

    return d


def fetch_many(titles: list[str], region: str) -> dict:
    if not HAS_KEY or not titles:
        return {}

    # Fetch multiple movie details concurrently to save time
    with ThreadPoolExecutor(max_workers=8) as ex:
        return dict(zip(titles, ex.map(lambda t: fetch_card(t, region), titles)))


def get_trending():
    ts, items = STORE["trend"]

    if items and time.time() - ts < 3600:
        return items  # Reuse trending results for one hour

    items = tmdb.trending(KEY, 10) if HAS_KEY else []

    if items:
        STORE["trend"] = (time.time(), items)

    return items


# CUSTOM STYLING

CSS = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")


def md(s: str):
    """Render custom HTML through Streamlit markdown."""
    st.markdown(re.sub(r"\s*\n\s*", " ", s).strip(), unsafe_allow_html=True)


md(f"<style>{CSS}</style>")
esc = html.escape  # Escape HTML special characters to prevent HTML injection


# SVG ICONS

BRAND_ICON = (
    '<svg viewBox="0 0 32 32" aria-hidden="true">'
    '<rect x="5" y="4" width="22" height="24" rx="5" />'
    '<path d="M10 4v24M22 4v24M5 11h5M5 21h5M22 11h5M22 21h5" />'
    '<path d="M14 12.5 20 16l-6 3.5z" class="brand-mark-play" />'
    '</svg>'
)
FILM_ICON = (
    '<svg class="placeholder-icon" viewBox="0 0 48 48" aria-hidden="true">'
    '<rect x="7" y="8" width="34" height="32" rx="5" />'
    '<path d="M7 17h34M7 31h34M15 8v9M25 8v9M35 8v9M15 31v9M25 31v9M35 31v9" />'
    '</svg>'
)
PERSON_ICON = (
    '<svg class="placeholder-icon" viewBox="0 0 48 48" aria-hidden="true">'
    '<circle cx="24" cy="16" r="8" />'
    '<path d="M8 41c1.5-9 7.1-14 16-14s14.5 5 16 14" />'
    '</svg>'
)
STAR_ICON = (
    '<svg class="inline-icon star-icon" viewBox="0 0 20 20" aria-hidden="true">'
    '<path d="m10 2.3 2.35 4.76 5.25.76-3.8 3.71.9 5.23L10 14.29l-4.7 2.47.9-5.23L2.4 7.82l5.25-.76z" />'
    '</svg>'
)


def poster_placeholder(title: str = "", person: bool = False) -> str:
    icon = PERSON_ICON if person else FILM_ICON  # Select the appropriate placeholder
    label = f'<span>{esc(title)}</span>' if title else ""
    return f'<div class="noposter">{icon}{label}</div>'


# SESSION STATE AND CALLBACKS

ss = st.session_state  # Stores user selections between Streamlit reruns

ss.setdefault("view", NAV[0])
if ss["view"] not in NAV:
    ss["view"] = NAV[0]

# Initialize filters and watchlist with default values
ss.setdefault("watch", [])
ss.setdefault("n_results", 10)
ss.setdefault("genre_f", [])
ss.setdefault("min_rating", 0.0)
ss.setdefault("year_f", (YEAR_MIN, YEAR_MAX))
ss.setdefault("qboost", 0.15)
ss.setdefault("region_name", "India")


def cb_set_sel(titles):
    ss["sel"] = list(titles)
    ss["_sel"] = list(titles)
    ss["view"] = NAV[0]  # Navigate to For You


def cb_vibe(q):
    ss["vibe_q"] = q
    ss["_vibe"] = q
    ss["view"] = NAV[1]  # Navigate to Vibe Search


def cb_view(name):
    ss["view"] = name  # Change the active page


def cb_toggle_watch(title):
    if title in ss["watch"]:
        ss["watch"].remove(title)  # Remove saved movie
    else:
        ss["watch"].append(title)  # Add movie to watchlist


def cb_reset_filters():
    # Restore all filters to their default values
    ss["n_results"] = 10
    ss["genre_f"] = []
    ss["min_rating"] = 0.0
    ss["year_f"] = (YEAR_MIN, YEAR_MAX)
    ss["qboost"] = 0.15
    ss["region_name"] = "India"


def cb_detail(kind, ref, title):
    ss["detail"] = {"kind": kind, "ref": ref, "title": title}  # Store selected movie details


# SIDEBAR

with st.sidebar:
    md(f'<div class="brand-lockup"><span class="brand-mark">{BRAND_ICON}</span>'
       '<span class="brand-word">Cine<span>Match</span></span></div>'
       '<div class="tagline" style="margin-bottom:14px">Thoughtful matches for your next movie night.</div>')

    st.markdown("**Filters**")
    md('<div class="filter-group-label">RESULT COUNT</div>')

    n_results = st.select_slider("Number of results", options=[5, 10, 15],
                                 key="n_results")

    md('<div class="filter-group-label">MATCH CRITERIA</div>')

    genre_f = st.multiselect("Genres", engine.all_genres, placeholder="Any genre",
                             key="genre_f")

    min_rating = st.slider("Minimum rating", 0.0, 9.0, step=0.5,
                           key="min_rating")

    # Year filtering requires TMDB data
    year_f = st.slider("Release year", YEAR_MIN, YEAR_MAX,
                       disabled=not HAS_KEY,
                       key="year_f",
                       help="Needs a TMDB API key (release years come from TMDB).")

    qboost = st.slider("Favour well-rated movies", 0.0, 0.5, step=0.05,
                       key="qboost",
                       help="0 = purely similar. Higher = nudges highly-rated, popular movies upward.")

    md('<div class="filter-group-label">AVAILABILITY</div>')

    region_name = st.selectbox("Where-to-watch region", list(REGIONS),
                               key="region_name")
    REGION = REGIONS[region_name]  # Convert region name into country code

    # Count how many filters differ from their defaults
    filters_active = sum((
        n_results != 10,
        bool(genre_f),
        min_rating > 0,
        HAS_KEY and tuple(year_f) != (YEAR_MIN, YEAR_MAX),
        qboost != 0.15,
        region_name != "India",
    ))

    st.button("Reset filters", key="reset_filters", icon=":material/filter_alt_off:",
              on_click=cb_reset_filters, disabled=filters_active == 0,
              use_container_width=True)

    st.caption(f"{filters_active} filter{'s' if filters_active != 1 else ''} adjusted"
               if filters_active else "Using default filters")

    st.divider()
    st.caption(f"Watchlist · {len(ss['watch'])} saved")

    if not HAS_KEY:
        st.warning("No TMDB key found – showing recommendations without posters. "
                   "Add `TMDB_API_KEY` to `.streamlit/secrets.toml`.")


# MOVIE CARD DATA

def build_item(idx: int, d: dict | None, match=None, reasons=None):
    row = engine.row(idx)

    return {
        "idx": idx,
        "title": row["title"],
        "d": d,
        "match": match,
        "reasons": reasons or [],
        "poster": d["poster"] if d else None,
        "rating": (d["rating"] if d else float(row["vote_average"])),
        "year": (d["year"] if d and d["year"] != "N/A" else ""),
        "runtime": d["runtime"] if d else 0,
        "genres": d["genres"] if d and d["genres"] else list(row["genre_list"]),
    }


# GENERATE HTML FOR A MOVIE CARD

def card_html(it: dict) -> str:
    img = (f'<img src="{esc(it["poster"])}" alt="{esc(it["title"])}" loading="lazy">'
           if it["poster"] else poster_placeholder(it["title"]))

    badge = f'<div class="badge">{it["match"]}% match</div>' if it.get("match") is not None else ""

    rate = (f'<div class="rate">{STAR_ICON}<span>{it["rating"]:.1f}</span></div>'
            if it["rating"] else "")

    bits = [b for b in [str(it["year"]) if it["year"] else "",
                        f'{it["runtime"] // 60}h {it["runtime"] % 60}m' if it["runtime"] else ""] if b]

    chips = "".join(f'<span class="chip">{esc(g)}</span>' for g in it["genres"][:3])

    bar = f'<div class="bar"><i style="width:{it["match"]}%"></i></div>' if it.get("match") is not None else ""

    why = ""
    if it["reasons"]:
        why = '<div class="why">' + "<br>".join(
            "<b>" + esc(r.split(":")[0]) + ":</b>" + esc(":".join(r.split(":")[1:])) if ":" in r else esc(r)
            for r in it["reasons"][:3]) + "</div>"

    return (f'<div class="card"><div class="poster">{img}{badge}{rate}</div>'
            f'<div class="ctitle">{esc(it["title"])}</div>'
            f'<div class="meta">{" · ".join(bits) if bits else "&nbsp;"}</div>'
            f'<div class="chips">{chips}</div>{bar}{why}</div>')


# DISPLAY MOVIE CARDS IN A GRID

def grid(items: list[dict], prefix: str, mode: str = "rec", per_row: int = 5):

    # Create rows containing the specified number of movie cards
    for r in range(0, len(items), per_row):
        cols = st.columns(per_row)

        for col, it in zip(cols, items[r: r + per_row]):
            with col:
                md(card_html(it))

                b1, b2 = st.columns(2)
                key = f'{prefix}_{it["idx"]}'

                # Identify whether the movie belongs to our dataset or TMDB
                kind, ref = ("ds", it["title"]) if it["idx"] is not None else ("tmdb", it["tmdb_id"])

                b1.button("Details", key=f"{key}_d", icon=":material/info:",
                          on_click=cb_detail, args=(kind, ref, it["title"]))

                if mode == "rec":
                    saved = it["title"] in ss["watch"]
                    b2.button("Saved" if saved else "Save", key=f"{key}_w",
                              icon=":material/bookmark:" if saved else ":material/bookmark_add:",
                              on_click=cb_toggle_watch, args=(it["title"],))

                elif mode == "discover":
                    b2.button("Similar", key=f"{key}_s", icon=":material/compare_arrows:",
                              on_click=cb_set_sel, args=([it["title"]],))

                elif mode == "watch":
                    b2.button("Remove", key=f"{key}_x", icon=":material/bookmark_remove:",
                              on_click=cb_toggle_watch, args=(it["title"],))


# LOADING PLACEHOLDER

def skeleton(n: int):
    ph = st.empty()  # Create a temporary placeholder

    ph.markdown('<div class="skgrid">' + '<div class="skel"></div>' * min(n, 10) + "</div>",
                unsafe_allow_html=True)

    return ph


# SECTION HEADING

def section(title: str, sub: str = ""):
    md(f'<div class="sec">{esc(title)}<small>{esc(sub)}</small></div>')


# MOVIE DETAILS DIALOG

def open_dialog(info: dict):

    # Fetch details based on the movie source
    if info["kind"] == "ds":
        d = fetch_card(info["title"], REGION)
    else:
        d = fetch_details(info["ref"], REGION)

    ds_idx = engine._title_to_idx.get(info["title"])
    row = engine.row(ds_idx) if ds_idx is not None else None

    @st.dialog(info["title"], width="large")
    def _show():
        left, right = st.columns([1, 2])

        poster = d["poster"] if d else None

        with left:
            inner = f'<img src="{esc(poster)}" alt="{esc(info["title"])}">' if poster else poster_placeholder(info["title"])
            md(f'<div class="poster">{inner}</div>')

        with right:
            if d and d["tagline"]:
                md(f'<div class="detail-tag">“{esc(d["tagline"])}”</div>')

            rating = d["rating"] if d else float(row["vote_average"])
            meta = [f'{STAR_ICON} {rating:.1f}/10']

            if d:
                meta += [x for x in [d["year"], f'{d["runtime"] // 60}h {d["runtime"] % 60}m' if d["runtime"] else ""] if x]

            md(f'<div class="meta" style="font-size:14px;margin-bottom:8px">{" · ".join(meta)}</div>')

            genres = d["genres"] if d else (list(row["genre_list"]) if row is not None else [])

            md('<div class="chips" style="margin-bottom:10px">' + "".join(f'<span class="chip">{esc(g)}</span>' for g in genres) + "</div>")

            overview = (d["overview"] if d and d["overview"] else (row["overview"] if row is not None else ""))
            st.write(overview)

            director = (d["director"] if d else None) or (row["director"] if row is not None else "")

            if director:
                st.caption(f"Directed by **{director}**")

            c1, c2, c3 = st.columns(3)

            if row is not None:
                if c1.button("Find similar", key="dlg_sim", icon=":material/auto_awesome:"):
                    cb_set_sel([info["title"]])
                    st.rerun()

                saved = info["title"] in ss["watch"]

                c2.button("Saved" if saved else "Save", key="dlg_w",
                          icon=":material/bookmark:" if saved else ":material/bookmark_add:",
                          on_click=cb_toggle_watch, args=(info["title"],))

            if d and d.get("imdb"):
                c3.link_button("IMDb ↗", d["imdb"])

        # Display trailer if available
        if d and d["trailer"]:
            st.markdown("##### Trailer")
            st.video(d["trailer"])

        # Display regional streaming availability
        if d and d["where"]:
            w = d["where"]
            st.markdown(f"##### Where to watch · {region_name}")

            if not (w["stream"] or w["rent"] or w["buy"]):
                st.caption("No watch-provider data for this region.")

            for label, names in (("Stream", w["stream"]), ("Rent", w["rent"]), ("Buy", w["buy"])):
                if names:
                    md(f'<div><b style="color:#c4b5fd">{label}:</b> ' + "".join(f'<span class="pill">{esc(n)}</span>' for n in names[:8]) + "</div>")

        # Display top cast members
        if d and d["cast"]:
            st.markdown("##### Top cast")
            cols = st.columns(4)

            for i, c in enumerate(d["cast"][:8]):
                with cols[i % 4]:
                    photo = (f'<img src="{esc(c["photo"])}" alt="{esc(c["name"])}" style="width:100%;border-radius:12px;aspect-ratio:2/3;object-fit:cover">'
                             if c["photo"] else f'<div class="poster">{poster_placeholder(person=True)}</div>')

                    md(f'{photo}<div class="meta" style="margin:6px 0 14px"><b style="color:#fff">{esc(c["name"])}</b><br>{esc(c["character"][:28])}</div>')

        if not d:
            st.caption("Live TMDB details are unavailable right now.")

    _show()


# PAGE HEADER AND NAVIGATION

nav_cols = st.columns(4)

for nav_col, (label, icon) in zip(nav_cols, NAV_ITEMS):
    with nav_col:
        st.button(label, key=f"nav_{label.lower().replace(' ', '_')}", icon=icon,
                  type="primary" if ss["view"] == label else "secondary",
                  use_container_width=True, on_click=cb_view, args=(label,))

view = ss["view"]


# FILTER MOVIES BY RELEASE YEAR

def apply_year(items_titles_data, lo, hi):
    out = []

    for t, d in items_titles_data:
        try:
            y = int(d["year"]) if d else None
        except (ValueError, TypeError):
            y = None

        if y is not None and lo <= y <= hi:
            out.append((t, d))

    return out


# VIEW 1: FOR YOU

if view == NAV[0]:

    if "sel" not in ss:
        ss["sel"] = ss.get("_sel", [])

    # Allow users to select up to three favourite movies
    st.multiselect("Movies you love", engine.titles, key="sel", max_selections=3,
                   placeholder="Search and pick 1–3 movies you love…", label_visibility="collapsed")

    ss["_sel"] = list(ss["sel"])
    titles = ss["sel"]

    if not titles:

        # Show the welcome screen when no movie is selected
        md('<div class="hero"><div class="hero-copy"><div class="hero-eyebrow">A better way to pick your next film</div>'
           '<h2>Good films.<br><span class="hero-accent">Better matches.</span></h2>'
           '<p>Start with a film you love. We’ll follow the story, themes, cast and creative DNA to find your next watch — with a reason for every match.</p>'
           '<div class="steps"><div class="step"><b>1</b>Choose a favourite</div>'
           '<div class="step"><b>2</b>Explore tailored matches</div>'
           '<div class="step"><b>3</b>Pick your next watch</div></div></div>'
           '<div class="hero-art" aria-hidden="true"><div class="hero-orbit"></div>'
           '<div class="film-card card-a"></div><div class="film-card card-b"></div><div class="film-card card-c"></div></div></div>')

        section("Start with a favourite", "one tap to begin")
        cols = st.columns(len(STARTERS))

        for c, t in zip(cols, STARTERS):
            c.button(t, key=f"st_{t}", on_click=cb_set_sel, args=([t],))

    else:
        year_active = HAS_KEY and tuple(year_f) != (YEAR_MIN, YEAR_MAX)

        # Fetch extra results when year filtering is active
        pool = min(n_results * 4, 40) if year_active else n_results

        # Generate recommendations using the ML engine
        recs = engine.recommend(titles, n=n_results, genres=genre_f or None, min_rating=min_rating,
                                quality_weight=qboost, pool=pool)

        ph = skeleton(n_results)
        sel_cards = fetch_many(titles, REGION)
        rec_cards = fetch_many([r.title for r in recs], REGION)
        ph.empty()

        # Use the first selected movie's backdrop as the banner
        first = sel_cards.get(titles[0])

        bg = (f'style="background-image:url({esc(first["backdrop"])})"' if first and first["backdrop"] else
              'style="background:linear-gradient(135deg,#1b1233,#0d0d14)"')

        sub = (first["tagline"] or first["overview"][:170] + "…") if first and first["overview"] else ""

        md(f'<div class="banner" {bg}><div class="shade"></div><div class="txt">'
           f'<div class="kick">Because you liked</div>'
           f'<div class="ttl">{esc(" + ".join(titles))}</div><div class="sub">{esc(sub)}</div></div></div>')

        # Apply release-year filter to recommendations
        if year_active:
            kept = {t for t, _ in apply_year([(r.title, rec_cards.get(r.title)) for r in recs], *year_f)}
            recs = [r for r in recs if r.title in kept]

        recs = recs[:n_results]

        if not recs:
            md('<div class="empty">No movies match these filters. Try relaxing genre, rating or year.</div>')

        else:
            section("Your next watch",
                    f"Because you liked {' + '.join(titles)} · {len(recs)} matches")

            items = [build_item(r.idx, rec_cards.get(r.title), r.match, r.reasons) for r in recs]

            for it in items:
                it["tmdb_id"] = None

            grid(items, "rec", "rec")

        # Explain the recommendation system to users
        with st.expander("How does CineMatch work?"):
            st.markdown(
                "- **Content-based filtering.** Every movie becomes a vector built from its plot (TF-IDF), "
                "keywords, genres, director and top cast – each weighted by importance.\n"
                "- **Similarity** is the cosine between vectors (for several picks, between the average of them).\n"
                "- **Quality nudge.** A small, adjustable boost for well-rated movies (IMDB-style weighted rating) "
                "stops obscure low-rated films from topping the list.\n"
                "- **Match %** is calibrated: 100% ≈ the strongest typical match in the whole dataset.\n"
                "- **Posters, trailers, cast & streaming info** come live from TMDB.")


# VIEW 2: VIBE SEARCH

elif view == NAV[1]:

    md('<div class="page-kicker">Mood-led discovery</div><div class="page-title">What are you in the mood for?</div>'
       '<div class="page-lede">Describe a feeling, a setting, or a story. We’ll find films that live in that world.</div>')

    if "vibe_q" not in ss:
        ss["vibe_q"] = ss.get("_vibe", "")

    st.text_input("Vibe", key="vibe_q", placeholder="e.g. dark sci-fi with a twist ending",
                  label_visibility="collapsed")

    ss["_vibe"] = ss["vibe_q"]

    st.caption("Try one:")
    cols = st.columns(3)

    for i, v in enumerate(VIBES):
        cols[i % 3].button(v, key=f"vibe_{i}", on_click=cb_vibe, args=(v,))

    q = ss["vibe_q"].strip()

    if q:
        year_active = HAS_KEY and tuple(year_f) != (YEAR_MIN, YEAR_MAX)

        # Search movies based on the user's text description
        recs = engine.search_text(q, n=min(n_results * 4, 40) if year_active else n_results,
                                  genres=genre_f or None, min_rating=min_rating, quality_weight=qboost)

        if not recs:
            md('<div class="empty">Nothing matched – try different words (genres, themes, settings).</div>')

        else:
            ph = skeleton(n_results)
            cards = fetch_many([r.title for r in recs], REGION)
            ph.empty()

            if year_active:
                kept = {t for t, _ in apply_year([(r.title, cards.get(r.title)) for r in recs], *year_f)}
                recs = [r for r in recs if r.title in kept]

            recs = recs[:n_results]

            section(f"Results for “{q}”", "ranked by relevance")

            items = [build_item(r.idx, cards.get(r.title), r.match, r.reasons) for r in recs]

            for it in items:
                it["tmdb_id"] = None

            grid(items, "vibe", "rec")


# VIEW 3: DISCOVER

elif view == NAV[2]:

    md('<div class="page-kicker">The wider world of film</div><div class="page-title">Discover something good.</div>'
       '<div class="page-lede">Trending now, enduring favourites, and a few paths you may not have found on your own.</div>')

    section("Trending this week", "live from TMDB")
    trend = get_trending()

    if not trend:
        md('<div class="empty">Trending needs a TMDB API key and an internet connection.</div>')

    else:
        # Convert TMDB trending results into movie card format
        items = [{"idx": None, "tmdb_id": m["id"], "title": m["title"], "poster": m["poster"],
                  "rating": m["rating"], "year": m["year"], "runtime": 0, "genres": [],
                  "reasons": [], "match": None} for m in trend]

        # Trending movies are outside our dataset, so only show details
        for r in range(0, len(items), 5):
            cols = st.columns(5)

            for col, it in zip(cols, items[r: r + 5]):
                with col:
                    md(card_html(it))
                    st.button("Details", key=f'tr_{it["tmdb_id"]}', icon=":material/info:",
                              on_click=cb_detail, args=("tmdb", it["tmdb_id"], it["title"]))

    # Show the highest-rated movies in our dataset
    section("All-time top rated", "from the CineMatch library")

    top = engine.top_rated(10, genre_f or None)
    cards = fetch_many([engine.titles[i] for i in top], REGION)

    items = [build_item(i, cards.get(engine.titles[i])) for i in top]

    for it in items:
        it["tmdb_id"] = None

    grid(items, "top", "discover")


# VIEW 4: WATCHLIST

else:

    md('<div class="page-kicker">For another movie night</div><div class="page-title">Your watchlist.</div>'
       '<div class="page-lede">A personal shortlist of films worth coming back to.</div>')

    section("Saved films", f'{len(ss["watch"])} saved')

    if not ss["watch"]:
        md('<div class="empty">Nothing saved yet. Use <b>Save</b> on a film to add it to your list.</div>')

    else:
        cards = fetch_many(list(ss["watch"]), REGION)

        items = [build_item(engine.index_of(t), cards.get(t)) for t in ss["watch"]]

        for it in items:
            it["tmdb_id"] = None

        grid(items, "wl", "watch")

        # Convert watchlist data into a downloadable CSV file
        df = pd.DataFrame([{"Title": it["title"], "Rating": round(float(it["rating"]), 1),
                            "Genres": ", ".join(it["genres"]),
                            "Director": engine.row(it["idx"])["director"]} for it in items])

        st.download_button("Download watchlist (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "cinematch_watchlist.csv", "text/csv")

    st.caption("The watchlist lasts for this browser session.")


# OPEN MOVIE DETAILS WHEN A DETAILS BUTTON IS CLICKED

_detail = ss.pop("detail", None)  # Retrieve and remove pending detail request

if _detail:
    open_dialog(_detail)


# FOOTER

md('<div class="footer">CineMatch · Thoughtful recommendations for your next movie night · '
   'Movie data and images from TMDB (not endorsed or certified by TMDB)</div>')
