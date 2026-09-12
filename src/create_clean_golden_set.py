import pandas as pd
from pathlib import Path

HOLDOUT = Path("data/processed/amazonhelp_holdout_conversations.csv")
MESSAGES = Path("data/processed/amazonhelp_dev_messages.csv")
OUTPUT = Path("data/evaluation/golden_set_clean.csv")

SEED = 42
N = 200

print("Loading holdout conversations...")
holdout = pd.read_csv(HOLDOUT)
holdout_ids = set(holdout["conversation_id"].astype(str))

print(f"Holdout conversations: {len(holdout_ids):,}")

print("Loading messages...")
messages = pd.read_csv(MESSAGES)
messages["conversation_id"] = messages["conversation_id"].astype(str)

# Only customer messages from holdout conversations.
customer = messages[
    (messages["conversation_id"].isin(holdout_ids)) &
    (messages["inbound"] == True)
].copy()

# Prefer conversation-start messages when available.
customer["created_at"] = pd.to_datetime(
    customer["created_at"], errors="coerce"
)

customer = customer.sort_values(
    ["conversation_id", "created_at", "tweet_id"]
)

roots = customer.groupby("conversation_id", as_index=False).first()

# Sample 200 different conversations.
golden = roots.sample(
    n=N,
    random_state=SEED
).copy()

golden = golden.sort_values("conversation_id").reset_index(drop=True)

result = golden[
    [
        "tweet_id",
        "conversation_id",
        "created_at",
        "text",
    ]
].copy()

# Columns to be filled manually.
result["human_intent"] = ""
result["human_notes"] = ""
result["is_ambiguous"] = ""
result["needs_context"] = ""

result.to_csv(OUTPUT, index=False)

print()
print("=" * 60)
print("CLEAN GOLDEN SET CREATED")
print("=" * 60)
print(f"Examples: {len(result):,}")
print(f"Unique conversations: {result['conversation_id'].nunique():,}")
print(f"Unique tweets: {result['tweet_id'].nunique():,}")
print(f"Output: {OUTPUT}")