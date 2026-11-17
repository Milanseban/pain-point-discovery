import requests
import json
import time
import os
from datetime import datetime, timezone

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_URL = "https://arctic-shift.photon-reddit.com"

# 9 months back from today
AFTER  = "2025-06-01"
BEFORE = "2026-03-25"

# Pain signal phrases — each one is a separate query pass
PAIN_PHRASES = [
    "wish there was",
    "why is there no",
    "why isnt there",
    "i have to manually",
    "does anyone else hate",
    "so annoying that",
    "no good tool for",
    "anyone else frustrated",
    "hate having to",
    "every single time i",
    "sick of manually",
    "there should be an app",
    "need a better way",
    "takes so long to",
    "waste of time every",
]

SUBREDDITS = [
    # Trades & home services
    "plumbing", "electricians", "HVAC", "HomeImprovement", "Plumbers",
    "construction", "landscaping", "pressurewashing", "handyman",
    # Freelancers
    "freelance", "copywriting", "graphic_design", "forhire", "web_design",
    "SEO", "socialmedia",
    # Small business & startups
    "smallbusiness", "Entrepreneur", "startups", "sweatystartup",
    "SideProject", "ecommerce",
    # Professional services
    "Accounting", "realestate", "legaladvice", "InsuranceAgent",
    "financialplanning", "bookkeeping",
    # Health & medical
    "nursing", "medicine", "pharmacy", "physicaltherapy", "Dentistry",
    "mentalhealth",
    # Creative & media
    "photography", "videography", "WeAreTheMusicMakers", "Filmmakers",
    "editors",
    # Tech & IT
    "sysadmin", "devops", "ITCareerQuestions", "msp", "techsupport", "aws",
    # Retail & ecommerce
    "Etsy", "FulfillmentByAmazon", "dropship", "shopify", "flipping",
    # Food & hospitality
    "KitchenConfidential", "restaurantowners", "catering", "bartenders",
    "barista",
    # Education & coaching
    "Teachers", "tutors", "personaltraining", "onlineteaching", "academia",
    # Property & landlords
    "Landlord", "PropertyManagement", "realestateinvesting", "airbnb",
    # Niche hobbyist (money involved)
    "woodworking", "3Dprinting", "crafts", "needlepoint", "leathercraft",
    "jewelry",
]

OUTPUT_FILE = "raw_posts.json"
PROGRESS_FILE = "progress.json"

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

def load_existing_posts():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_posts(posts_dict):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(posts_dict, f, ensure_ascii=False, indent=2)

def fetch_posts(subreddit, phrase, before=None):
    """
    Fetches one page of posts from Arctic Shift.
    Returns list of posts or empty list on failure.
    """
    params = {
        "subreddit": subreddit,
        "query": phrase,
        "after": AFTER,
        "before": before if before else BEFORE,
        "limit": 100,
        "sort": "desc",
        "fields": "id,title,selftext,created_utc,subreddit,score,num_comments",
    }

    try:
        resp = requests.get(
            f"{BASE_URL}/api/posts/search",
            params=params,
            timeout=30,
            headers={"User-Agent": "pain-point-miner/1.0"}
        )

        if resp.status_code == 429:
            print("    Rate limited — waiting 10s...")
            time.sleep(10)
            return None  # caller will retry

        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code} for r/{subreddit} — skipping")
            return []

        data = resp.json()
        return data.get("data", [])

    except requests.exceptions.Timeout:
        print(f"    Timeout for r/{subreddit} — skipping")
        return []
    except Exception as e:
        print(f"    Error: {e}")
        return []

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    posts_dict = load_existing_posts()  # keyed by post ID for dedup
    progress = load_progress()

    total_combos = len(SUBREDDITS) * len(PAIN_PHRASES)
    done = 0

    print(f"Starting collection — {len(SUBREDDITS)} subreddits × {len(PAIN_PHRASES)} phrases")
    print(f"Date range: {AFTER} → {BEFORE}")
    print(f"Output: {OUTPUT_FILE}\n")

    for subreddit in SUBREDDITS:
        for phrase in PAIN_PHRASES:
            key = f"{subreddit}|{phrase}"
            done += 1

            if progress.get(key):
                print(f"[{done}/{total_combos}] r/{subreddit} + '{phrase}' — already done, skipping")
                continue

            print(f"[{done}/{total_combos}] r/{subreddit} + '{phrase}'", end="", flush=True)

            # Paginate through all results
            before_cursor = BEFORE
            page = 0
            new_posts = 0

            while True:
                results = fetch_posts(subreddit, phrase, before=before_cursor)

                if results is None:
                    # Rate limited, retry once
                    results = fetch_posts(subreddit, phrase, before=before_cursor)
                    if results is None:
                        break

                if not results:
                    break

                for post in results:
                    pid = post.get("id")
                    if pid and pid not in posts_dict:
                        posts_dict[pid] = post
                        new_posts += 1

                # If we got fewer than 100, we've hit the end
                if len(results) < 100:
                    break

                # Move cursor back to earliest post in this batch
                earliest = min(results, key=lambda p: p.get("created_utc", 0))
                before_cursor = earliest.get("created_utc")

                if not before_cursor:
                    break

                page += 1
                time.sleep(0.5)  # be polite between pages

            print(f" → {new_posts} new posts (total: {len(posts_dict)})")

            # Mark done and save periodically
            progress[key] = True
            save_progress(progress)

            if done % 10 == 0:
                save_posts(posts_dict)
                print(f"  💾 Saved checkpoint ({len(posts_dict)} posts total)")

            time.sleep(0.3)  # polite pause between queries

    save_posts(posts_dict)
    print(f"\nDone. {len(posts_dict)} unique posts saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
