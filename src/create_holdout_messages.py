import pandas as pd
from pathlib import Path

MESSAGES = Path("data/processed/amazonhelp_dev_messages.csv")
HOLDOUT = Path("data/processed/amazonhelp_holdout_conversations.csv")
OUTPUT = Path("data/processed/amazonhelp_holdout_messages.csv")

print("Loading holdout conversation IDs...")

holdout_ids = set(
    pd.read_csv(HOLDOUT)["conversation_id"].astype(str)
)

print(f"Holdout conversations: {len(holdout_ids):,}")

print("Loading messages...")

messages = pd.read_csv(MESSAGES)
messages["conversation_id"] = messages["conversation_id"].astype(str)

holdout_messages = messages[
    messages["conversation_id"].isin(holdout_ids)
].copy()

holdout_messages.to_csv(OUTPUT, index=False)

print()
print("=" * 60)
print("HOLDOUT MESSAGE FILE CREATED")
print("=" * 60)
print(f"Messages: {len(holdout_messages):,}")
print(f"Conversations: {holdout_messages['conversation_id'].nunique():,}")
print(f"Output: {OUTPUT}")