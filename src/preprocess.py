"""Build a reproducible, component-preserving AmazonHelp development dataset.

The source CSV is read sequentially and is never modified.  A temporary SQLite
edge index makes traversing branched conversations practical without loading the
2.8M-row source file into memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


RAW_COLUMNS = [
    "tweet_id", "author_id", "inbound", "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id",
]
TIMESTAMP_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def split_ids(value: str) -> list[int]:
    """Preserve raw values elsewhere; split only for graph traversal/auditing."""
    return [int(part) for part in value.split(",") if part]


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_FORMAT)


def reconstruct_components(
    rows: Iterable[dict[str, str]],
) -> dict[int, set[int]]:
    """Small in-memory reference implementation used by tests.

    `in_response_to_tweet_id` is canonical. Missing parents are not invented;
    they simply leave a node as an observed component root.
    """
    parent: dict[int, int] = {}
    children: dict[int, set[int]] = defaultdict(set)
    node_ids: set[int] = set()
    for row in rows:
        child = int(row["tweet_id"])
        node_ids.add(child)
        for parent_id in split_ids(row.get("in_response_to_tweet_id", "")):
            parent[child] = parent_id
            children[parent_id].add(child)

    components: dict[int, set[int]] = {}
    visited: set[int] = set()
    for node in sorted(node_ids):
        if node in visited:
            continue
        root = node
        seen_ancestors: set[int] = set()
        while root in parent and parent[root] in node_ids and root not in seen_ancestors:
            seen_ancestors.add(root)
            root = parent[root]
        stack = [root]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component or current not in node_ids:
                continue
            component.add(current)
            stack.extend(children.get(current, ()))
        visited.update(component)
        components[root] = component
    return components


def initialise_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE edges (child_id INTEGER NOT NULL, parent_id INTEGER NOT NULL);
        CREATE INDEX edges_parent_idx ON edges(parent_id);
        CREATE TABLE roots (tweet_id INTEGER PRIMARY KEY, author_id TEXT NOT NULL);
        CREATE TABLE amazon_direct_parents (tweet_id INTEGER PRIMARY KEY);
        """
    )
    return connection


def build_graph_index(raw_csv: Path, database: sqlite3.Connection, brand: str) -> dict[str, int]:
    """Stream the raw CSV into a compact canonical parent→child index."""
    stats: Counter[str] = Counter()
    edge_batch: list[tuple[int, int]] = []
    root_batch: list[tuple[int, str]] = []
    amazon_batch: list[tuple[int]] = []
    batch_size = 25_000

    with raw_csv.open(encoding="utf-8", newline="") as source, database:
        for row in csv.DictReader(source):
            stats["raw_rows"] += 1
            tweet_id = int(row["tweet_id"])
            parent_ids = split_ids(row["in_response_to_tweet_id"])
            if row["inbound"] == "True" and not parent_ids:
                root_batch.append((tweet_id, row["author_id"]))
            for parent_id in parent_ids:
                edge_batch.append((tweet_id, parent_id))
            if row["inbound"] == "False" and row["author_id"] == brand:
                stats["brand_messages"] += 1
                amazon_batch.extend((parent_id,) for parent_id in parent_ids)
            if len(edge_batch) >= batch_size:
                database.executemany("INSERT INTO edges VALUES (?, ?)", edge_batch)
                edge_batch.clear()
            if len(root_batch) >= batch_size:
                database.executemany("INSERT OR IGNORE INTO roots VALUES (?, ?)", root_batch)
                root_batch.clear()
            if len(amazon_batch) >= batch_size:
                database.executemany(
                    "INSERT OR IGNORE INTO amazon_direct_parents VALUES (?)", amazon_batch
                )
                amazon_batch.clear()
        database.executemany("INSERT INTO edges VALUES (?, ?)", edge_batch)
        database.executemany("INSERT OR IGNORE INTO roots VALUES (?, ?)", root_batch)
        database.executemany(
            "INSERT OR IGNORE INTO amazon_direct_parents VALUES (?)", amazon_batch
        )
    return dict(stats)


def choose_roots(database: sqlite3.Connection, target: int, brand: str) -> list[tuple[int, str, str]]:
    candidates = database.execute(
        """
        SELECT roots.tweet_id, roots.author_id
        FROM roots INNER JOIN amazon_direct_parents
        ON roots.tweet_id = amazon_direct_parents.tweet_id
        ORDER BY roots.tweet_id
        """
    ).fetchall()
    selected = [
        (tweet_id, author_id, f"{brand.lower()}_{tweet_id}")
        for tweet_id, author_id in candidates[:target]
    ]
    database.execute("CREATE TABLE selected_roots (tweet_id INTEGER PRIMARY KEY, customer_author_id TEXT, conversation_id TEXT)")
    database.executemany("INSERT INTO selected_roots VALUES (?, ?, ?)", selected)
    database.commit()
    return selected, len(candidates)


def materialize_component_nodes(database: sqlite3.Connection) -> dict[int, str]:
    """Traverse all descendants, retaining branches, from each selected root."""
    database.execute(
        "CREATE TABLE component_nodes (tweet_id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL)"
    )
    query = """
        WITH RECURSIVE tree(conversation_id, tweet_id) AS (
            SELECT conversation_id, tweet_id FROM selected_roots
            UNION
            SELECT tree.conversation_id, edges.child_id
            FROM tree JOIN edges ON edges.parent_id = tree.tweet_id
        )
        SELECT conversation_id, tweet_id FROM tree
    """
    batch: list[tuple[int, str]] = []
    with database:
        for conversation_id, tweet_id in database.execute(query):
            batch.append((tweet_id, conversation_id))
            if len(batch) >= 25_000:
                database.executemany("INSERT OR IGNORE INTO component_nodes VALUES (?, ?)", batch)
                batch.clear()
        database.executemany("INSERT OR IGNORE INTO component_nodes VALUES (?, ?)", batch)
    return {tweet_id: conversation_id for tweet_id, conversation_id in database.execute("SELECT tweet_id, conversation_id FROM component_nodes")}


def write_outputs(
    raw_csv: Path,
    output_dir: Path,
    brand: str,
    selected_roots: list[tuple[int, str, str]],
    node_to_conversation: dict[int, str],
    candidate_count: int,
    index_stats: dict[str, int],
) -> dict[str, object]:
    """Stream selected rows, then write chronologically ordered component files."""
    messages: dict[str, list[dict[str, str]]] = defaultdict(list)
    response_target_counts: Counter[int] = Counter()
    selected_parent_targets: Counter[int] = Counter()
    with raw_csv.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            conversation_id = node_to_conversation.get(int(row["tweet_id"]))
            if conversation_id is None:
                continue
            row = dict(row)
            row["conversation_id"] = conversation_id
            row["created_at_utc"] = parse_timestamp(row["created_at"]).isoformat()
            row["message_role"] = "customer" if row["inbound"] == "True" else "support"
            messages[conversation_id].append(row)
            response_target_counts.update(split_ids(row["response_tweet_id"]))
            selected_parent_targets.update(split_ids(row["in_response_to_tweet_id"]))

    # A final streaming ID pass determines whether selected response references
    # point outside the source file, without retaining all source IDs in memory.
    unresolved = set(response_target_counts) | set(selected_parent_targets)
    with raw_csv.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            unresolved.discard(int(row["tweet_id"]))
            if not unresolved:
                break

    message_path = output_dir / "amazonhelp_dev_messages.csv"
    conversation_path = output_dir / "amazonhelp_dev_conversations.csv"
    report_path = output_dir / "amazonhelp_preprocessing_report.json"
    message_fields = ["conversation_id", "created_at_utc", "message_role", *RAW_COLUMNS]
    conversation_fields = [
        "conversation_id", "brand", "customer_author_id", "start_time", "end_time",
        "number_of_messages", "number_of_customer_messages", "number_of_brand_messages",
    ]
    root_authors = {conversation_id: author for _, author, conversation_id in selected_roots}
    conversation_rows: list[dict[str, object]] = []
    total_customer = total_brand = 0
    lengths: list[int] = []
    with message_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=message_fields)
        writer.writeheader()
        for conversation_id in sorted(messages):
            rows = sorted(messages[conversation_id], key=lambda row: (row["created_at_utc"], int(row["tweet_id"])))
            writer.writerows(rows)
            customer_count = sum(row["inbound"] == "True" for row in rows)
            brand_count = sum(row["inbound"] == "False" for row in rows)
            total_customer += customer_count
            total_brand += brand_count
            lengths.append(len(rows))
            conversation_rows.append({
                "conversation_id": conversation_id,
                "brand": brand,
                "customer_author_id": root_authors[conversation_id],
                "start_time": rows[0]["created_at_utc"],
                "end_time": rows[-1]["created_at_utc"],
                "number_of_messages": len(rows),
                "number_of_customer_messages": customer_count,
                "number_of_brand_messages": brand_count,
            })
    with conversation_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=conversation_fields)
        writer.writeheader()
        writer.writerows(conversation_rows)

    selected_node_ids = set(node_to_conversation)
    canonical_parent_outside_component = sum(
        count for target, count in selected_parent_targets.items() if target not in selected_node_ids
    )
    report: dict[str, object] = {
        "source": str(raw_csv), "brand": brand, "selection_method": "lowest canonical customer-root tweet_ids with a direct AmazonHelp reply",
        "raw_rows_indexed": index_stats["raw_rows"], "amazonhelp_outbound_messages": index_stats["brand_messages"],
        "amazonhelp_related_customer_root_components_found": candidate_count,
        "components_selected": len(conversation_rows),
        "components_rejected": candidate_count - len(conversation_rows),
        "rejection_reasons": {"target_cap_after_deterministic_ordering": candidate_count - len(conversation_rows)},
        "selected_component_messages": sum(lengths), "average_messages_per_selected_conversation": sum(lengths) / len(lengths),
        "minimum_messages": min(lengths), "maximum_messages": max(lengths),
        "customer_messages": total_customer, "brand_messages": total_brand,
        "time_range": {"start": min(row["start_time"] for row in conversation_rows), "end": max(row["end_time"] for row in conversation_rows)},
        "remaining_dangling_relationships": {
            "canonical_parent_tokens_outside_selected_component": canonical_parent_outside_component,
            "distinct_response_targets_missing_from_source": len(unresolved),
            "response_reference_tokens_missing_from_source": sum(response_target_counts[target] for target in unresolved),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run(raw_csv: Path, output_dir: Path, brand: str, target: int, overwrite: bool) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "amazonhelp_dev_messages.csv",
        output_dir / "amazonhelp_dev_conversations.csv",
        output_dir / "amazonhelp_preprocessing_report.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}. Use --overwrite.")
    for path in existing:
        path.unlink()

    database_path = output_dir / ".amazonhelp_graph.sqlite"
    if database_path.exists():
        database_path.unlink()
    try:
        database = initialise_database(database_path)
        index_stats = build_graph_index(raw_csv, database, brand)
        selected_roots, candidate_count = choose_roots(database, target, brand)
        node_to_conversation = materialize_component_nodes(database)
        report = write_outputs(raw_csv, output_dir, brand, selected_roots, node_to_conversation, candidate_count, index_stats)
        database.close()
        return report
    finally:
        if database_path.exists():
            database_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=Path, default=Path("data/raw/twcs/twcs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--brand", default="AmazonHelp")
    parser.add_argument("--target", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.raw_csv, arguments.output_dir, arguments.brand, arguments.target, arguments.overwrite)
    print(json.dumps(result, indent=2))
