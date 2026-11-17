# Pain-Point Discovery Pipeline

A Python pipeline that mines online discussions for problems people keep complaining about, then groups the posts by meaning so recurring needs stand out.

## How it works

1. **Collect** (`collector.py`): pulls Reddit posts from June 2025 to March 2026 across 70 subreddits in 12 professional categories (trades, freelancers, small business, health, property and more) using the free Arctic Shift API. Posts are filtered by 15 "pain" phrases such as "I have to manually" and "wish there was". Progress is checkpointed so long runs can resume.
2. **Embed** (`cluster.py`): converts each post into a 3,072-dimension vector with OpenAI's `text-embedding-3-large`. Embeddings are cached to disk so reruns don't repeat API calls.
3. **Cluster**: groups similar posts with HDBSCAN, which works out the number of clusters itself.
4. **Label and report**: GPT-4o-mini writes a short label for each cluster, and the results are written to a ranked HTML report.

## Results

**Reddit run:** 39,264 posts embedded and grouped into 1,419 clusters.

Every opportunity the report surfaced was then checked against existing products, and none were worth building. That led to the main finding: complaint language mostly points to problems that already have solutions people haven't found yet. The stronger signal is resignation language ("I use a spreadsheet for this", "I haven't found anything that actually does"), which is where the next version is aimed.

**Hacker News extension** (`hn_collector.py`, `cluster_hn.py`): uses the Algolia HN Search API and targets Ask HN posts. The run embedded 27,197 posts, but HDBSCAN produced only 3 clusters with the same settings as Reddit, so the parameters need retuning for HN data.

## Running it

```
pip install -r requirements.txt
```

Set your OpenAI API key as an environment variable (never put it in the code):

```
# Windows PowerShell
$env:OPENAI_API_KEY="your-key"

# macOS / Linux
export OPENAI_API_KEY="your-key"
```

Then:

```
python collector.py
python cluster.py
```

Data files (raw posts, embeddings and reports) are not included. They are large and contain other people's posts.

## Tech

Python, Arctic Shift API, Algolia HN Search API, OpenAI embeddings (`text-embedding-3-large`), GPT-4o-mini, HDBSCAN, NumPy
