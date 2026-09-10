import pandas as pd
from pathlib import Path

MESSAGES = Path("data/processed/amazonhelp_dev_messages.csv")
TRAIN_CONVERSATIONS = Path("data/processed/amazonhelp_train_conversations.csv")
OUTPUT = Path("data/processed/amazonhelp_train_messages.csv")

print("Loading conversation IDs...")
train_ids = set(
    pd.read_csv(TRAIN_CONVERSATIONS)["conversation_id"].astype(str)
)

print(f"Training conversations: {len(train_ids):,}")

print("Loading messages...")
messages = pd.read_csv(MESSAGES)

messages["conversation_id"] = messages["conversation_id"].astype(str)

train_messages = messages[
    messages["conversation_id"].isin(train_ids)
].copy()

train_messages.to_csv(OUTPUT, index=False)

print()
print("=" * 60)
print("TRAINING MESSAGE FILE CREATED")
print("=" * 60)
print(f"Messages: {len(train_messages):,}")
print(f"Conversations: {train_messages['conversation_id'].nunique():,}")
print(f"Output: {OUTPUT}")