"""Stream and inspect the original Kaggle TWCS CSV without materializing it.

Run from the repository root:
    .\\venv\\Scripts\\python.exe src\\inspect_twcs.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from collections import Counter
from pathlib import Path


DEFAULT_CSV = Path("data/raw/twcs/twcs.csv")


def profile(csv_path: Path) -> None:
    """Print one-pass file-level counts. `response_tweet_id` is comma-delimited."""
    missing: Counter[str] = Counter()
    inbound: Counter[str] = Counter()
    outbound_by_brand: Counter[str] = Counter()
    authors: set[str] = set()
    rows = parent_edges = response_edges = 0

    with csv_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        print(f"file bytes: {csv_path.stat().st_size:,}")
        print(f"columns ({len(reader.fieldnames or [])}): {reader.fieldnames}")
        print(
            "logical types: tweet_id=int; author_id=str; inbound=bool; "
            "created_at=timestamp string; text=str; response_tweet_id="
            "comma-delimited tweet-id string; in_response_to_tweet_id=tweet-id string"
        )
        for row in reader:
            rows += 1
            authors.add(row["author_id"])
            inbound[row["inbound"]] += 1
            if row["inbound"] == "False":
                outbound_by_brand[row["author_id"]] += 1
            for column, value in row.items():
                if not value:
                    missing[column] += 1
            parent_edges += len([x for x in row["in_response_to_tweet_id"].split(",") if x])
            response_edges += len([x for x in row["response_tweet_id"].split(",") if x])

    print(f"rows: {rows:,}; unique authors: {len(authors):,}")
    print(f"inbound/outbound: {dict(inbound)}")
    print(f"missing values: {dict(missing)}")
    print(f"outbound brand accounts: {len(outbound_by_brand)}")
    print(f"direct parent/child edges: {parent_edges:,}/{response_edges:,}")
    print("brand distribution (outbound tweets):")
    for brand, count in outbound_by_brand.most_common():
        print(f"  {brand}: {count:,}")


def show_chain(csv_path: Path, root_id: str, scan_rows: int) -> None:
    """Show the response-descendant tree for a root from an intentionally bounded scan."""
    records: dict[str, dict[str, str]] = {}
    with csv_path.open(encoding="utf-8", newline="") as source:
        for index, row in enumerate(csv.DictReader(source)):
            records[row["tweet_id"]] = row
            if index + 1 >= scan_rows:
                break

    def visit(tweet_id: str, depth: int, seen: set[str]) -> None:
        if tweet_id in seen or tweet_id not in records:
            return
        seen.add(tweet_id)
        row = records[tweet_id]
        prefix = "  " * depth
        print(f"{prefix}tweet_id: {tweet_id}")
        for column in (
            "author_id",
            "inbound",
            "text",
            "in_response_to_tweet_id",
            "response_tweet_id",
            "created_at",
        ):
            print(f"{prefix}{column}: {row[column]}")
        for child_id in filter(None, row["response_tweet_id"].split(",")):
            visit(child_id, depth + 1, seen)

    visit(root_id, 0, set())


def validate(csv_path: Path) -> None:
    """Perform file-wide integrity checks in one streaming pass.

    Three compact bitmaps track observed IDs and the two reference directions;
    this avoids loading CSV rows or millions of Python objects into memory.
    """
    ids = bytearray()
    parent_targets = bytearray()
    response_targets = bytearray()

    def mark(bitmap: bytearray, value: int) -> None:
        if value >= len(bitmap):
            bitmap.extend(b"\0" * (value + 1 - len(bitmap)))
        bitmap[value] = 1

    rows = duplicate_ids = 0
    inbound: Counter[str] = Counter()
    outbound_by_brand: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    parent_values = response_values = response_multi_value_rows = 0
    parent_tokens = response_tokens = 0
    dangling_parent_tokens = dangling_response_tokens = 0
    invalid_timestamps = 0
    multi_response_examples: list[tuple[str, str]] = []

    with csv_path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            rows += 1
            tweet_id = int(row["tweet_id"])
            duplicate_ids += bool(tweet_id < len(ids) and ids[tweet_id])
            mark(ids, tweet_id)
            inbound[row["inbound"]] += 1
            if row["inbound"] == "False":
                outbound_by_brand[row["author_id"]] += 1
            for column, value in row.items():
                if not value:
                    missing[column] += 1
            try:
                datetime.strptime(row["created_at"], "%a %b %d %H:%M:%S %z %Y")
            except ValueError:
                invalid_timestamps += 1

            parent_ids = [x for x in row["in_response_to_tweet_id"].split(",") if x]
            response_ids = [x for x in row["response_tweet_id"].split(",") if x]
            parent_values += bool(parent_ids)
            response_values += bool(response_ids)
            parent_tokens += len(parent_ids)
            response_tokens += len(response_ids)
            for value in parent_ids:
                mark(parent_targets, int(value))
            for value in response_ids:
                mark(response_targets, int(value))
            if len(response_ids) > 1:
                response_multi_value_rows += 1
                if len(multi_response_examples) < 10:
                    multi_response_examples.append((row["tweet_id"], row["response_tweet_id"]))

    dangling_parent_tokens = sum(
        bool(value) and (index >= len(ids) or not ids[index])
        for index, value in enumerate(parent_targets)
    )
    dangling_response_tokens = sum(
        bool(value) and (index >= len(ids) or not ids[index])
        for index, value in enumerate(response_targets)
    )

    print(f"rows: {rows:,}")
    print(f"unique tweet_ids: {sum(ids):,}; duplicate tweet_id rows: {duplicate_ids:,}")
    print(f"inbound/outbound: {dict(inbound)}; sum: {sum(inbound.values()):,}")
    print(f"brand accounts: {len(outbound_by_brand)}")
    for brand, count in outbound_by_brand.most_common():
        print(f"  {brand}: {count:,}")
    print(f"missing values: {dict(missing)}")
    print(
        "relationship values/tokens: "
        f"parent={parent_values:,}/{parent_tokens:,}; "
        f"response={response_values:,}/{response_tokens:,}"
    )
    print(
        "distinct missing relationship targets: "
        f"parent={dangling_parent_tokens:,}; response={dangling_response_tokens:,}"
    )
    print(
        f"multi-ID response rows: {response_multi_value_rows:,}; "
        f"examples: {multi_response_examples}"
    )
    print(f"invalid created_at values: {invalid_timestamps:,}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--chain", help="tweet_id to reconstruct")
    parser.add_argument("--validate", action="store_true", help="run two-pass integrity validation")
    parser.add_argument("--scan-rows", type=int, default=150_000)
    args = parser.parse_args()
    if args.validate:
        validate(args.csv)
    elif args.chain:
        show_chain(args.csv, args.chain, args.scan_rows)
    else:
        profile(args.csv)
