"""
hn_collector.py — Hacker News Pain Point Collector (v2)
Uses the official Algolia HN Search API (free, no auth needed)

Fixes vs v1:
  - Removed the strict post-fetch phrase filter (was killing 90%+ of results)
  - Wider date range: Jan 2022 → March 2025 (HN has lower volume, need more history)
  - Replaced Reddit-brained phrases with HN-native language
  - Added Ask HN specific targeting (highest signal post type on HN)
  - Targets stories more heavily (Ask HN posts are stories, not comments)

Output: raw_posts_hn.json
        progress_hn.json (resumable checkpoint)

Dependencies: pip install requests
"""

import requests
import json
import time
import os
from datetime import datetime, timezone

# ─── CONFIG ────────────────────────────────────────────────────────────────────

OUTPUT_FILE   = "raw_posts_hn.json"
PROGRESS_FILE = "progress_hn.json"

# Wider date range — HN has lower volume per phrase than Reddit
# Going back to Jan 2022 gives ~3 years of signal
DATE_FROM = int(datetime(2022, 1, 1,  tzinfo=timezone.utc).timestamp())
DATE_TO   = int(datetime(2025, 3, 31, tzinfo=timezone.utc).timestamp())

BASE_URL      = "https://hn.algolia.com/api/v1/search_by_date"
HITS_PER_PAGE = 50
MAX_PAGES     = 20   # Algolia caps at 1000 results (20 x 50) per query
REQUEST_DELAY = 0.4

# ─── PHRASES ───────────────────────────────────────────────────────────────────
# Split into two groups:
#   A) Ask HN style — searched against stories only
#   B) General resignation/unmet-need — searched against comments + stories

ASK_HN_PHRASES = [
    ("Ask HN: Is there a tool",                 "tool_search"),
    ("Ask HN: Does anyone know of a tool",      "tool_search"),
    ("Ask HN: Is there software that",          "tool_search"),
    ("Ask HN: Does something exist that",       "tool_search"),
    ("Ask HN: Has anyone built",                "tool_search"),
    ("Ask HN: Looking for a tool",              "tool_search"),
    ("Ask HN: Recommendations for a tool",      "tool_search"),
    ("Ask HN: Any tools for",                   "tool_search"),
    ("Ask HN: What do you use for",             "tool_search"),
    ("Ask HN: How do you handle",               "tool_search"),
    ("Ask HN: How do you manage",               "tool_search"),
    ("Ask HN: Alternatives to",                 "tool_search"),
    ("Ask HN: I can't find a good tool",        "unsolved"),
    ("Ask HN: frustrated with lack of",         "unsolved"),
    ("Ask HN: I built this because nothing",    "built_own"),
]

GENERAL_PHRASES = [
    # Built own because nothing existed
    ("I built this because nothing",            "built_own"),
    ("I wrote this because I couldn't find",    "built_own"),
    ("I made this because no existing",         "built_own"),
    ("I rolled my own because",                 "built_own"),
    ("I hacked together because",               "built_own"),
    ("I wrote a script because nothing",        "built_own"),
    ("couldn't find anything that did",         "built_own"),
    ("nothing existed that could",              "built_own"),
    ("no tool seemed to",                       "built_own"),

    # Explicit unmet need / still unsolved
    ("I couldn't find a good solution",         "unsolved"),
    ("I haven't found anything that",           "unsolved"),
    ("I can't find anything that",              "unsolved"),
    ("does anything exist that",                "unsolved"),
    ("is there anything that can",              "unsolved"),
    ("anyone know of something that",           "unsolved"),
    ("still waiting for someone to build",      "unsolved"),
    ("no good solution exists for",             "unsolved"),
    ("the tooling for this is terrible",        "unsolved"),
    ("surprised there isn't a tool",            "unsolved"),
    ("surprised no one has built",              "unsolved"),
    ("gap in the market for",                   "unsolved"),

    # Manual workaround / resignation
    ("I just use a spreadsheet",                "resignation"),
    ("we use a spreadsheet for",                "resignation"),
    ("I ended up just using",                   "resignation"),
    ("I ended up writing my own",               "resignation"),
    ("I gave up and just",                      "resignation"),
    ("we just do it manually",                  "resignation"),
    ("still doing it manually",                 "resignation"),
    ("I've accepted that I have to",            "resignation"),
    ("I've been doing this manually",           "resignation"),
    ("just copy paste it",                      "resignation"),

    # Show HN: built own (confirms gap was real)
    ("Show HN: I built this because",           "built_own"),
    ("Show HN: I made this because",            "built_own"),
    ("Show HN: couldn't find a good",           "built_own"),
]

GENERAL_TAGS = ["comment", "story"]

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": []}

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

def load_posts():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []

def save_posts(posts):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)

def dedupe(posts):
    seen, out = set(), []
    for p in posts:
        if p["id"] not in seen:
            seen.add(p["id"])
            out.append(p)
    return out

def normalise_hit(hit, phrase, label, tag):
    if tag == "comment":
        body  = hit.get("comment_text", "") or ""
        title = ""
    else:
        title = hit.get("title", "") or ""
        body  = hit.get("story_text", "") or ""

    return {
        "id":          hit.get("objectID", ""),
        "title":       title,
        "body":        body,
        "subreddit":   "hackernews",
        "score":       hit.get("points", 0) or 0,
        "url":         f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
        "created_utc": hit.get("created_at_i", 0),
        "phrase":      phrase,
        "label":       label,
        "source":      "hackernews",
        "type":        tag,
        "author":      hit.get("author", ""),
    }

def has_minimum_content(post):
    combined = (post["title"] + " " + post["body"]).strip()
    return len(combined) > 30

# ─── COLLECTOR ─────────────────────────────────────────────────────────────────

def collect(phrase, label, tag, existing_ids):
    collected = []
    page = 0

    while page < MAX_PAGES:
        params = {
            "query":          phrase,
            "tags":           tag,
            "numericFilters": f"created_at_i>{DATE_FROM},created_at_i<{DATE_TO}",
            "hitsPerPage":    HITS_PER_PAGE,
            "page":           page,
        }

        try:
            r = requests.get(BASE_URL, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            print(f"\n    warning: Request error page {page}: {e} — retrying in 5s")
            time.sleep(5)
            continue
        except json.JSONDecodeError:
            print(f"\n    warning: JSON error page {page} — skipping")
            break

        hits     = data.get("hits", [])
        nb_pages = data.get("nbPages", 1)

        if not hits:
            break

        for hit in hits:
            oid = hit.get("objectID", "")
            if not oid or oid in existing_ids:
                continue
            post = normalise_hit(hit, phrase, label, tag)
            if has_minimum_content(post):
                collected.append(post)
                existing_ids.add(oid)

        page += 1
        if page >= nb_pages:
            break

        time.sleep(REQUEST_DELAY)

    return collected

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HN Collector v2 — Unmet Need Mining")
    print(f"Date range  : Jan 2022 to Mar 2025")
    print(f"Ask HN phrases : {len(ASK_HN_PHRASES)}")
    print(f"General phrases: {len(GENERAL_PHRASES)}")
    print("=" * 60)

    progress     = load_progress()
    all_posts    = load_posts()
    existing_ids = {p["id"] for p in all_posts}

    print(f"Resuming — {len(all_posts)} posts already, "
          f"{len(progress['completed'])} combos done\n")

    total_new = 0

    # Ask HN phrases (stories only)
    print("── Ask HN phrases (stories) ──────────────────")
    for phrase, label in ASK_HN_PHRASES:
        key = f"{phrase}|story"
        if key in progress["completed"]:
            print(f"  skip  \"{phrase[:55]}\"")
            continue
        print(f"  -> \"{phrase[:55]}\" ... ", end="", flush=True)
        new = collect(phrase, label, "story", existing_ids)
        all_posts.extend(new)
        total_new += len(new)
        print(f"{len(new)} posts  (total: {len(all_posts)})")
        progress["completed"].append(key)
        save_progress(progress)
        save_posts(all_posts)
        time.sleep(REQUEST_DELAY)

    # General phrases (comments + stories)
    print("\n── General phrases (comments + stories) ──────")
    for phrase, label in GENERAL_PHRASES:
        for tag in GENERAL_TAGS:
            key = f"{phrase}|{tag}"
            if key in progress["completed"]:
                print(f"  skip  [{tag}] \"{phrase[:50]}\"")
                continue
            print(f"  -> [{tag}] \"{phrase[:50]}\" ... ", end="", flush=True)
            new = collect(phrase, label, tag, existing_ids)
            all_posts.extend(new)
            total_new += len(new)
            print(f"{len(new)} posts  (total: {len(all_posts)})")
            progress["completed"].append(key)
            save_progress(progress)
            save_posts(all_posts)
            time.sleep(REQUEST_DELAY)

    # Final dedup
    before    = len(all_posts)
    all_posts = dedupe(all_posts)
    save_posts(all_posts)

    print("\n" + "=" * 60)
    print(f"New this run       : {total_new}")
    print(f"Duplicates removed : {before - len(all_posts)}")
    print(f"Total saved        : {len(all_posts)}")
    print(f"Output             : {OUTPUT_FILE}")
    print("=" * 60)
    print("Next: delete embeddings_hn.npy + embedding_ids_hn.json, then run cluster_hn.py")

if __name__ == "__main__":
    main()
