import pandas as pd
from pathlib import Path

GOLDEN = Path("data/evaluation/golden_set_clean.csv")
WEAK = Path("data/processed/amazonhelp_holdout_weak_labels.csv")

df = pd.read_csv(GOLDEN)
weak = pd.read_csv(WEAK)

# Keep only the columns needed from the weak-label file.
weak = weak[
    ["tweet_id", "weak_intent", "weak_label_confidence"]
].copy()

# Match using tweet_id.
df["tweet_id"] = df["tweet_id"].astype(str)
weak["tweet_id"] = weak["tweet_id"].astype(str)

df = df.drop(columns=["weak_intent", "weak_label_confidence"], errors="ignore")

df = df.merge(
    weak,
    on="tweet_id",
    how="left",
    validate="one_to_one",
)

if len(df) != 200:
    raise ValueError(f"Expected 200 rows, found {len(df)}")

missing = df["weak_intent"].isna().sum()

if missing:
    raise ValueError(
        f"{missing} Golden Set examples did not receive weak labels."
    )

# Put columns in the expected order.
columns = [
    "tweet_id",
    "conversation_id",
    "created_at",
    "text",
    "weak_intent",
    "weak_label_confidence",
    "human_intent",
    "human_notes",
    "is_ambiguous",
    "needs_context",
]

df = df[columns]

df.to_csv(GOLDEN, index=False)

print("=" * 60)
print("CLEAN GOLDEN SET PREPARED")
print("=" * 60)
print(f"Rows: {len(df)}")
print(f"Missing weak labels: {missing}")
print(f"Output: {GOLDEN}")