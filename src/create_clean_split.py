import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

INPUT = Path("data/processed/amazonhelp_train_conversations.csv")
OUTPUT_DIR = Path("data/processed")

df = pd.read_csv(INPUT)

# Split by complete conversations, never individual messages.
conversation_ids = df["conversation_id"].drop_duplicates()

train_ids, holdout_ids = train_test_split(
    conversation_ids,
    test_size=0.20,
    random_state=42,
)

train = df[df["conversation_id"].isin(train_ids)].copy()
holdout = df[df["conversation_id"].isin(holdout_ids)].copy()

train.to_csv(
    OUTPUT_DIR / "amazonhelp_train_conversations.csv",
    index=False,
)

holdout.to_csv(
    OUTPUT_DIR / "amazonhelp_holdout_conversations.csv",
    index=False,
)

print("=" * 60)
print("CLEAN CONVERSATION SPLIT")
print("=" * 60)

print(f"Total conversations : {len(conversation_ids):,}")
print(f"Train conversations : {len(train_ids):,}")
print(f"Holdout conversations: {len(holdout_ids):,}")

print(f"Train messages      : {len(train):,}")
print(f"Holdout messages    : {len(holdout):,}")

print()
print("Files created:")
print(" ", OUTPUT_DIR / "amazonhelp_train_conversations.csv")
print(" ", OUTPUT_DIR / "amazonhelp_holdout_conversations.csv")