#!/usr/bin/env python3
"""LLM-based sentiment classifier for Amazon Gift Card reviews.

Classifies the sentiment expressed in a review's title + text as
positive / neutral / negative, using an OpenAI-compatible LLM endpoint.

Ground truth is derived from the star rating:  1-2 = negative, 3 = neutral, 4-5 = positive.

Usage:
    .venv/bin/python sentiment_classifier.py [--samples 150] [--seed 42] [--workers 8]
"""

import argparse
import csv
import gzip
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Gift_Cards.jsonl.gz"
RESULTS_DIR = BASE_DIR / "results"

POSITIVE = "positive"
NEUTRAL = "neutral"
NEGATIVE = "negative"
LABEL_BY_RATING = {1: NEGATIVE, 2: NEGATIVE, 3: NEUTRAL, 4: POSITIVE, 5: POSITIVE}

SYSTEM_PROMPT = (
    "You are an expert sentiment classifier for Amazon product reviews. "
    "Your only job is to classify the sentiment expressed in a review as exactly "
    "one of: positive, neutral, or negative. Respond with a single word and nothing else."
)

USER_PROMPT = """Classify the sentiment of this Amazon review.

Title: {title}

Review text: {text}

Sentiment (reply with exactly one word: positive, neutral, or negative):"""

# ----------------------------------------------------------------------------
# Endpoint config: env vars first, then the Hermes config as fallback
# ----------------------------------------------------------------------------
def load_endpoint():
    """Return (base_url, api_key). Falls back to ~/.hermes/config.yaml."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if base_url and api_key:
        return base_url, api_key

    try:
        import yaml
        cfg = yaml.safe_load((Path.home() / ".hermes" / "config.yaml").read_text())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Could not load Hermes config: {exc}") from exc

    candidates = []

    def walk(node):
        if isinstance(node, dict):
            if "base_url" in node and "api_key" in node:
                candidates.append(node)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(cfg)
    if not candidates:
        raise SystemExit("No OpenAI-compatible endpoint (base_url + api_key) found in config.")
    # Prefer the one whose base_url mentions 'dobolyi', else the first found.
    chosen = next((c for c in candidates if "dobolyi" in str(c["base_url"])), candidates[0])
    return str(chosen["base_url"]), str(chosen["api_key"])


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_reviews():
    reviews = []
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                reviews.append(json.loads(line))
    return reviews


def sample_reviews(reviews, n, seed):
    rng = random.Random(seed)
    return rng.sample(reviews, n)


# ----------------------------------------------------------------------------
# LLM classification
# ----------------------------------------------------------------------------
def parse_sentiment(raw):
    """Normalize a raw model response into positive/neutral/negative (or None)."""
    text = (raw or "").strip().lower()
    # Handle JSON-ish responses like {"sentiment": "positive"}
    if text.startswith("{"):
        try:
            text = json.loads(text).get("sentiment", text)
        except json.JSONDecodeError:
            pass
    # Whole-word match anywhere in the response (robust to
    # "The sentiment is positive." style answers)
    for word, label in (("negative", NEGATIVE), ("neutral", NEUTRAL), ("positive", POSITIVE)):
        if re.search(rf"\b{word}s?\b", text):
            return label
    # Abbreviations (exact token)
    mapping = {
        "pos": POSITIVE, "neut": NEUTRAL, "nuetral": NEUTRAL, "neg": NEGATIVE,
    }
    for token in re.findall(r"[a-z]+", text):
        if token in mapping:
            return mapping[token]
    return None


def classify_one(client, model, review):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(
            title=review.get("title") or "(no title)",
            text=(review.get("text") or "").strip(),
        )},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=512,  # deepseek-v4-flash reasons first; the answer lands in content
    )
    raw = resp.choices[0].message.content
    label = parse_sentiment(raw)
    if label is None:  # one retry with a stricter nudge
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": "That was not a valid answer. Reply with exactly one word: "
                       "positive, neutral, or negative.",
        })
        raw2 = client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=512,
        ).choices[0].message.content
        label = parse_sentiment(raw2)
        raw = f"{raw} -> {raw2}" if label is None else raw2
    return label, raw


# ----------------------------------------------------------------------------
# Metrics (pure Python, no sklearn)
# ----------------------------------------------------------------------------
def metrics(true_labels, pred_labels, classes):
    n = len(true_labels)
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    accuracy = correct / n if n else 0.0
    per_class = {}
    for cls in classes:
        tp = sum(t == cls and p == cls for t, p in zip(true_labels, pred_labels))
        fp = sum(t != cls and p == cls for t, p in zip(true_labels, pred_labels))
        fn = sum(t == cls and p != cls for t, p in zip(true_labels, pred_labels))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1, "n": sum(t == cls for t in true_labels)}
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)
    return accuracy, per_class, macro_f1


def confusion(true_labels, pred_labels, classes):
    return [[sum(t == a and p == b for t, p in zip(true_labels, pred_labels)) for b in classes] for a in classes]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="LLM sentiment classifier (3 classes)")
    ap.add_argument("--samples", type=int, default=150, help="number of random reviews to classify")
    ap.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    ap.add_argument("--workers", type=int, default=8, help="concurrent LLM calls")
    ap.add_argument("--model", default="DeepSeek-V4-Flash-0731")
    args = ap.parse_args()

    base_url, api_key = load_endpoint()
    print(f"Endpoint : {base_url}")
    print(f"Model    : {args.model}")
    print(f"Key      : {api_key[:6]}...{api_key[-4:]} (masked)")
    print(f"Sample   : {args.samples} random reviews (seed={args.seed}), {args.workers} workers")
    print()

    reviews = load_reviews()
    sampled = sample_reviews(reviews, args.samples, args.seed)
    print(f"Loaded {len(reviews):,} reviews; sampled {len(sampled)}")
    print("True label distribution:", dict(Counter(LABEL_BY_RATING[r["rating"]] for r in sampled)))
    print()

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0)
    classes = [POSITIVE, NEUTRAL, NEGATIVE]
    results = {}  # id(hashable index) -> (label, raw)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(classify_one, client, args.model, r): i
                   for i, r in enumerate(sampled)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[i] = (None, f"ERROR: {exc}")
            done += 1
            if done % 25 == 0 or done == len(futures):
                print(f"  classified {done}/{len(futures)} ({time.time()-t0:.0f}s)")
    elapsed = time.time() - t0

    true_labels = [LABEL_BY_RATING[r["rating"]] for r in sampled]
    pred_labels, raws = [], []
    unparsed = 0
    for i, _ in enumerate(sampled):
        label, raw = results[i]
        if label is None:
            unparsed += 1
            label = "unparsed"
        pred_labels.append(label)
        raws.append(raw)

    print(f"\nDone in {elapsed:.0f}s. Unparsed/unfailed responses: {unparsed}")
    print()
    print("Predicted label distribution:", dict(Counter(pred_labels)))
    print()

    accuracy, per_class, macro_f1 = metrics(true_labels, pred_labels, classes)
    print(f"Accuracy : {accuracy:.1%}   Macro-F1: {macro_f1:.3f}")
    print()
    print(f"{'class':<10} {'n':>4} {'precision':>10} {'recall':>9} {'f1':>7}")
    for cls in classes:
        m = per_class[cls]
        print(f"{cls:<10} {m['n']:>4} {m['precision']:>10.3f} {m['recall']:>9.3f} {m['f1']:>7.3f}")
    print()
    print("Confusion matrix  (rows = true, cols = predicted)")
    header = " " * 8 + "".join(f"{c[:8]:>10}" for c in classes)
    print(header)
    for row_cls, row in zip(classes, confusion(true_labels, pred_labels, classes)):
        print(f"{row_cls[:8]:<8}" + "".join(f"{v:>10}" for v in row))

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "sampled_reviews.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["idx", "user_id", "asin", "rating", "true_label",
                         "predicted_label", "raw_response", "title", "text"])
        for i, r in enumerate(sampled):
            writer.writerow([i, r.get("user_id"), r.get("asin"), r.get("rating"),
                             true_labels[i], pred_labels[i], raws[i],
                             r.get("title", ""), r.get("text", "")])
    print(f"\nSaved per-review results -> {csv_path}")


if __name__ == "__main__":
    main()
