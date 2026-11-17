import json
import os
import time
import numpy as np
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  # set this in your environment, never hard-code it
INPUT_FILE      = "raw_posts_hn.json"
EMBEDDINGS_FILE = "embeddings_hn.npy"
IDS_FILE        = "embedding_ids_hn.json"
OUTPUT_CLUSTERS = "clusters_hn.json"
OUTPUT_REPORT   = "report_hn.html"

EMBED_MODEL      = "text-embedding-3-large"
BATCH_SIZE       = 100
MIN_CLUSTER_SIZE = 3
MIN_SAMPLES      = 2

# ── Setup ──────────────────────────────────────────────────────────────────────

client = OpenAI(api_key=OPENAI_API_KEY)

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_posts():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Handle both list output (new collectors) and dict output (old Reddit collector)
    posts = list(data.values()) if isinstance(data, dict) else data
    print(f"Loaded {len(posts)} posts from {INPUT_FILE}")
    return posts

def make_text(post):
    """Combine title + body into one string for embedding.
    Handles both 'body' (new collectors) and 'selftext' (old Reddit collector).
    """
    title = post.get("title", "").strip()
    body  = post.get("body", post.get("selftext", "")).strip()
    if body in ("[deleted]", "[removed]", ""):
        return title if title else "no content"
    return f"{title}. {body[:500]}"

def embed_posts(posts):
    texts = [make_text(p) for p in posts]
    ids   = [p.get("id", str(i)) for i, p in enumerate(posts)]

    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(IDS_FILE):
        print("Found cached embeddings — loading...")
        embeddings = np.load(EMBEDDINGS_FILE)
        with open(IDS_FILE) as f:
            cached_ids = json.load(f)
        if len(embeddings) == len(posts):
            print(f"Cache matches ({len(embeddings)} embeddings). Skipping API calls.")
            return embeddings, cached_ids
        else:
            print(f"Cache size mismatch ({len(embeddings)} cached vs {len(posts)} posts) — re-embedding...")

    print(f"Embedding {len(texts)} posts in batches of {BATCH_SIZE}...")
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch       = texts[i:i + BATCH_SIZE]
        batch_num   = (i // BATCH_SIZE) + 1
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches}...", end="", flush=True)

        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            batch_embeddings = [item.embedding for item in resp.data]
            all_embeddings.extend(batch_embeddings)
            print(f" done ({len(batch_embeddings)} embeddings)")
        except Exception as e:
            print(f" ERROR: {e}")
            raise

        time.sleep(0.2)

    embeddings = np.array(all_embeddings)
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
    labels     = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = list(labels).count(-1)
    print(f"Found {n_clusters} clusters, {n_noise} unclustered posts")
    return labels

def build_clusters(posts, ids, labels):
    id_to_post = {p.get("id", str(i)): p for i, p in enumerate(posts)}
    clusters   = {}

    for idx, label in enumerate(labels):
        if label == -1:
            continue
        label = int(label)
        pid   = ids[idx]
        post  = id_to_post.get(pid, {})

        body = post.get("body", post.get("selftext", ""))
        if body in ("[deleted]", "[removed]", None):
            body = ""

        if label not in clusters:
            clusters[label] = []

        clusters[label].append({
            "id":          pid,
            "title":       post.get("title", ""),
            "body":        (body or "")[:300],
            "subreddit":   post.get("subreddit", "hackernews"),
            "score":       post.get("score", 0),
            "url":         post.get("url", ""),
            "source":      post.get("source", "hackernews"),
            "type":        post.get("type", ""),
            "phrase":      post.get("phrase", ""),
            "created_utc": post.get("created_utc", 0),
        })

    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    return sorted_clusters

def get_cluster_label(posts_in_cluster):
    samples = posts_in_cluster[:5]
    snippets = []
    for p in samples:
        text = p["title"] or p["body"]
        snippets.append(f"- {text[:150]}")
    content = "\n".join(snippets)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    "These Hacker News posts all express the same unmet need or missing tool. "
                    "In one short sentence (max 12 words), what is the core problem or gap they share?\n\n"
                    f"{content}"
                )
            }],
            max_tokens=40,
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return posts_in_cluster[0]["title"][:80]

def generate_report(sorted_clusters):
    print("Generating cluster labels via GPT...")
    cluster_data = []

    for i, (label, posts) in enumerate(sorted_clusters):
        print(f"  Labelling cluster {i+1}/{len(sorted_clusters)}...", end="", flush=True)
        summary = get_cluster_label(posts)
        sources  = list(set(p["source"] for p in posts))
        phrases  = list(set(p["phrase"] for p in posts if p.get("phrase")))
        cluster_data.append({
            "rank":    i + 1,
            "label":   label,
            "summary": summary,
            "count":   len(posts),
            "sources": sources,
            "phrases": phrases,
            "posts":   posts,
        })
        print(f" '{summary}'")
        time.sleep(0.3)

    with open(OUTPUT_CLUSTERS, "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, ensure_ascii=False, indent=2)
    print(f"Clusters saved to {OUTPUT_CLUSTERS}")

    # ── HTML Report ────────────────────────────────────────────────────────────

    total_posts = sum(len(p[1]) for p in sorted_clusters)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HN Pain Point Clusters</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f5f5f5; color: #1a1a1a; padding: 2rem; max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem; }}
  .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 2rem; }}
  .cluster {{ background: white; border-radius: 10px; padding: 1.25rem 1.5rem;
              margin-bottom: 1rem; border: 1px solid #e5e5e5; }}
  .cluster-header {{ display: flex; align-items: baseline; gap: 1rem; margin-bottom: 0.6rem; }}
  .rank {{ font-size: 0.8rem; font-weight: 700; color: #999; min-width: 28px; }}
  .summary {{ font-size: 1.05rem; font-weight: 600; flex: 1; }}
  .count {{ font-size: 0.85rem; background: #f0f0f0; padding: 2px 10px;
             border-radius: 20px; color: #555; white-space: nowrap; }}
  .phrases {{ margin-bottom: 0.6rem; }}
  .phrase-tag {{ display: inline-block; background: #fff7ed; color: #c2410c;
                  padding: 2px 8px; border-radius: 4px; margin: 2px; font-size: 0.72rem; }}
  details summary {{ font-size: 0.85rem; color: #4f46e5; cursor: pointer; user-select: none; }}
  details summary:hover {{ text-decoration: underline; }}
  .post {{ margin-top: 0.75rem; padding: 0.75rem; background: #fafafa;
            border-radius: 6px; border-left: 3px solid #e5e5e5; }}
  .post-title {{ font-size: 0.9rem; font-weight: 600; margin-bottom: 0.25rem; }}
  .post-title a {{ color: #1a1a1a; text-decoration: none; }}
  .post-title a:hover {{ text-decoration: underline; color: #4f46e5; }}
  .post-meta {{ font-size: 0.75rem; color: #999; margin-bottom: 0.3rem; }}
  .post-body {{ font-size: 0.82rem; color: #555; line-height: 1.55; }}
  .type-comment {{ border-left-color: #6366f1; }}
  .type-story   {{ border-left-color: #f59e0b; }}
</style>
</head>
<body>
<h1>Hacker News — Unmet Need Clusters</h1>
<p class="meta">{len(sorted_clusters)} clusters · {total_posts} posts · source: Algolia HN API</p>
"""

    for c in cluster_data:
        phrase_tags = "".join(
            f'<span class="phrase-tag">{ph}</span>'
            for ph in c["phrases"][:6]
        )

        posts_html = ""
        for p in c["posts"][:8]:
            body_snippet = (p["body"] or "").strip()
            body_line    = f'<div class="post-body">{body_snippet}</div>' if body_snippet else ""
            url          = p.get("url", "#")
            title_text   = p["title"] or p["body"][:80] or "(no title)"
            post_type    = p.get("type", "")
            score        = p.get("score", 0)
            phrase       = p.get("phrase", "")
            type_class   = f"type-{post_type}" if post_type else ""

            posts_html += f"""
            <div class="post {type_class}">
              <div class="post-title"><a href="{url}" target="_blank">{title_text}</a></div>
              <div class="post-meta">
                {post_type} · score: {score}
                {f'· phrase: <em>{phrase}</em>' if phrase else ''}
              </div>
              {body_line}
            </div>"""

        html += f"""
<div class="cluster">
  <div class="cluster-header">
    <span class="rank">#{c['rank']}</span>
    <span class="summary">{c['summary']}</span>
    <span class="count">{c['count']} posts</span>
  </div>
  <div class="phrases">{phrase_tags}</div>
  <details>
    <summary>Show {min(len(c['posts']), 8)} example posts</summary>
    {posts_html}
  </details>
</div>"""

    html += "\n</body>\n</html>"

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved to {OUTPUT_REPORT} — open in your browser")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    posts = load_posts()
    if not posts:
        print(f"No posts found. Check that {INPUT_FILE} exists and isn't empty.")
        return

    embeddings, ids = embed_posts(posts)
    labels          = cluster_embeddings(embeddings)
    sorted_clusters = build_clusters(posts, ids, labels)
    generate_report(sorted_clusters)

    print(f"\nDone. Open {OUTPUT_REPORT} in your browser.")

if __name__ == "__main__":
    main()
