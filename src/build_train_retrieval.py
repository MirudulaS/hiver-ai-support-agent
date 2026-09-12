import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/amazonhelp_train_messages.csv")
OUTPUT = Path("data/processed/amazonhelp_train_customer_messages.csv")

df = pd.read_csv(INPUT)

customer = df[df["inbound"] == True].copy()

customer.to_csv(OUTPUT, index=False)

print("=" * 60)
print("TRAINING RETRIEVAL DATASET CREATED")
print("=" * 60)
print(f"Training conversations: {customer['conversation_id'].nunique():,}")
print(f"Customer messages: {len(customer):,}")
print(f"Output: {OUTPUT}")