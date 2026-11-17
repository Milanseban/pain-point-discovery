import json
import os
import time
import numpy as np
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  # set this in your environment, never hard-code it
INPUT_FILE     = "raw_posts.json"
EMBEDDINGS_FILE = "embeddings.npy"    # cached so you don't re-embed if you rerun
IDS_FILE        = "embedding_ids.json"
OUTPUT_CLUSTERS = "clusters.json"
OUTPUT_REPORT   = "report.html"

EMBED_MODEL     = "text-embedding-3-large"
BATCH_SIZE      = 100                 # OpenAI allows up to 2048 per request but 100 is safe

# HDBSCAN settings
MIN_CLUSTER_SIZE = 3    # minimum posts to form a cluster
MIN_SAMPLES      = 2    # controls how conservative clustering is

# ── Setup ──────────────────────────────────────────────────────────────────────

client = OpenAI(api_key=OPENAI_API_KEY)

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_posts():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    posts = list(data.values())
    print(f"Loaded {len(posts)} posts from {INPUT_FILE}")
    return posts

def make_text(post):
    """Combine title + body into one string for embedding."""
    title = post.get("title", "").strip()
    body  = post.get("selftext", "").strip()
    if body in ("[deleted]", "[removed]", ""):
        return title
    # Truncate body to 500 chars to keep tokens manageable
    return f"{title}. {body[:500]}"

def embed_posts(posts):
    texts = [make_text(p) for p in posts]
    ids   = [p.get("id") for p in posts]

    # Check if cached embeddings exist
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(IDS_FILE):
        print("Found cached embeddings — loading...")
        embeddings = np.load(EMBEDDINGS_FILE)
        with open(IDS_FILE) as f:
            cached_ids = json.load(f)
        if len(embeddings) == len(posts):
            print(f"Cache matches ({len(embeddings)} embeddings). Skipping API calls.")
            return embeddings, cached_ids
        else:
            print("Cache size mismatch — re-embedding...")

    print(f"Embedding {len(texts)} posts in batches of {BATCH_SIZE}...")
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches}...", end="", flush=True)

        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch
            )
            batch_embeddings = [item.embedding for item in resp.data]
            all_embeddings.extend(batch_embeddings)
            print(f" done ({len(batch_embeddings)} embeddings)")
        except Exception as e:
            print(f" ERROR: {e}")
            raise

        time.sleep(0.2)  # small pause between batches

    embeddings = np.array(all_embeddings)

    # Cache to disk
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(IDS_FILE, "w") as f:
        json.dump(ids, f)
    print(f"Embeddings saved to {EMBEDDINGS_FILE}")

    return embeddings, ids

def cluster_embeddings(embeddings):
    try:
        import hdbscan
    except ImportError:
        print("Installing hdbscan...")
        os.system("pip install hdbscan")
        import hdbscan

    print(f"Clustering {len(embeddings)} embeddings...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = list(labels).count(-1)
    print(f"Found {n_clusters} clusters, {n_noise} unclustered posts")

    return labels

def build_clusters(posts, ids, labels):
    """Group posts by cluster label."""
    id_to_post = {p.get("id"): p for p in posts}
    clusters = {}

    for idx, label in enumerate(labels):
        if label == -1:
            continue  # skip noise
        label = int(label)
        pid = ids[idx]
        post = id_to_post.get(pid, {})
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({
            "id": pid,
            "title": post.get("title", ""),
            "body": post.get("selftext", "")[:300],
            "subreddit": post.get("subreddit", ""),
            "score": post.get("score", 0),
            "num_comments": post.get("num_comments", 0),
            "created_utc": post.get("created_utc", 0),
        })

    # Sort clusters by size descending
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    return sorted_clusters

def get_cluster_label(posts_in_cluster):
    """Ask GPT to summarise what the cluster is complaining about."""
    samples = posts_in_cluster[:5]
    titles = "\n".join([f"- {p['title']}" for p in samples])

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"These Reddit posts are all expressing the same frustration or unmet need. In one short sentence (max 12 words), what is the core problem they share?\n\n{titles}"
            }],
            max_tokens=40,
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except:
        return posts_in_cluster[0]["title"]

def generate_report(sorted_clusters):
    """Generate a readable HTML report."""
    print("Generating cluster labels via GPT...")
    cluster_data = []

    for i, (label, posts) in enumerate(sorted_clusters):
        print(f"  Labelling cluster {i+1}/{len(sorted_clusters)}...", end="", flush=True)
        summary = get_cluster_label(posts)
        subreddits = list(set(p["subreddit"] for p in posts))
        cluster_data.append({
            "rank": i + 1,
            "label": label,
            "summary": summary,
            "count": len(posts),
            "subreddits": subreddits,
            "posts": posts
        })
        print(f" '{summary}'")
        time.sleep(0.3)

    # Save clusters JSON
    with open(OUTPUT_CLUSTERS, "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, ensure_ascii=False, indent=2)
    print(f"Clusters saved to {OUTPUT_CLUSTERS}")

    # Build HTML report
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reddit Pain Point Clusters</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #1a1a1a; padding: 2rem; }
  h1 { font-size: 1.6rem; font-weight: 600; margin-bottom: 0.25rem; }
  .meta { color: #666; font-size: 0.9rem; margin-bottom: 2rem; }
  .cluster { background: white; border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; border: 1px solid #e5e5e5; }
  .cluster-header { display: flex; align-items: baseline; gap: 1rem; margin-bottom: 0.75rem; }
  .rank { font-size: 0.8rem; font-weight: 600; color: #999; min-width: 28px; }
  .summary { font-size: 1.05rem; font-weight: 600; flex: 1; }
  .count { font-size: 0.85rem; background: #f0f0f0; padding: 2px 10px; border-radius: 20px; color: #555; }
  .subs { font-size: 0.8rem; color: #888; margin-bottom: 0.75rem; }
  .sub-tag { display: inline-block; background: #eef2ff; color: #4f46e5; padding: 2px 8px; border-radius: 4px; margin: 2px; font-size: 0.75rem; }
  details summary { font-size: 0.85rem; color: #4f46e5; cursor: pointer; user-select: none; }
  details summary:hover { text-decoration: underline; }
  .post { margin-top: 0.75rem; padding: 0.75rem; background: #fafafa; border-radius: 6px; border-left: 3px solid #e5e5e5; }
  .post-title { font-size: 0.9rem; font-weight: 500; margin-bottom: 0.25rem; }
  .post-meta { font-size: 0.75rem; color: #999; }
  .post-body { font-size: 0.82rem; color: #555; margin-top: 0.4rem; line-height: 1.5; }
</style>
</head>
<body>
<h1>Reddit Pain Point Report</h1>
<p class="meta">""" + f"{len(sorted_clusters)} clusters found across {sum(len(p[1]) for p in sorted_clusters)} posts" + """</p>
"""

    for c in cluster_data:
        sub_tags = "".join([f'<span class="sub-tag">r/{s}</span>' for s in c["subreddits"]])
        posts_html = ""
        for p in c["posts"][:8]:
            body_snippet = p["body"].strip()
            body_line = f'<div class="post-body">{body_snippet}</div>' if body_snippet and body_snippet not in ("[deleted]", "[removed]") else ""
            posts_html += f"""
            <div class="post">
              <div class="post-title">{p['title']}</div>
              <div class="post-meta">r/{p['subreddit']} · {p['score']} upvotes · {p['num_comments']} comments</div>
              {body_line}
            </div>"""

        html += f"""
<div class="cluster">
  <div class="cluster-header">
    <span class="rank">#{c['rank']}</span>
    <span class="summary">{c['summary']}</span>
    <span class="count">{c['count']} posts</span>
  </div>
  <div class="subs">{sub_tags}</div>
  <details>
    <summary>Show example posts</summary>
    {posts_html}
  </details>
</div>"""

    html += "\n</body>\n</html>"

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved to {OUTPUT_REPORT} — open this in your browser")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    posts = load_posts()

    if not posts:
        print("No posts found. Make sure raw_posts.json exists and isn't empty.")
        return

    embeddings, ids = embed_posts(posts)
    labels = cluster_embeddings(embeddings)
    sorted_clusters = build_clusters(posts, ids, labels)
    generate_report(sorted_clusters)

    print(f"\nDone. Open {OUTPUT_REPORT} in your browser to see the results.")

if __name__ == "__main__":
    main()
