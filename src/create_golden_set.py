from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "amazonhelp_weak_labels.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "golden_set.csv"


INTENTS = [
    "ACCOUNT_ACCESS_SECURITY",
    "DELIVERY_DELAY",
    "DELIVERY_PROOF_FAILURE",
    "DIGITAL_TECHNICAL",
    "ORDER_MANAGEMENT",
    "PAYMENT_BILLING_GIFTCARD",
    "PRIME_MEMBERSHIP",
    "PRODUCT_SELLER_ISSUE",
    "RETURN_REFUND_REPLACEMENT",
    "SUPPORT_ESCALATION",
]


def main():
    df = pd.read_csv(INPUT_FILE)

    df = df[df["inbound"] == True].copy()

    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]

    selected_parts = []

    # ---------------------------------------------------------
    # 1. 10 examples from every substantive intent = 100
    # ---------------------------------------------------------

    for intent in INTENTS:
        subset = df[df["weak_intent"] == intent]

        n = min(10, len(subset))

        if n > 0:
            selected_parts.append(
                subset.sample(
                    n=n,
                    random_state=100 + INTENTS.index(intent),
                )
            )

    # ---------------------------------------------------------
    # 2. 50 difficult examples
    # ---------------------------------------------------------

    difficult = df[
        (df["weak_intent"] != "CONTEXT_NEEDED")
        & (
            df["weak_label_confidence"].isin(
                ["low", "medium"]
            )
        )
    ]

    difficult = difficult.sample(
        n=min(50, len(difficult)),
        random_state=200,
    )

    selected_parts.append(difficult)

    # ---------------------------------------------------------
    # 3. 40 context-dependent examples
    # ---------------------------------------------------------

    context = df[
        df["weak_intent"] == "CONTEXT_NEEDED"
    ]

    context = context.sample(
        n=min(40, len(context)),
        random_state=300,
    )

    selected_parts.append(context)

    # ---------------------------------------------------------
    # Combine and remove duplicates
    # ---------------------------------------------------------

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    selected = selected.drop_duplicates(
        subset=["tweet_id"]
    )

    # If duplicates reduced the set below 190,
    # fill from remaining customer messages.
    if len(selected) < 190:
        remaining = df[
            ~df["tweet_id"].isin(selected["tweet_id"])
        ]

        extra = remaining.sample(
            n=min(200 - len(selected), len(remaining)),
            random_state=400,
        )

        selected = pd.concat(
            [selected, extra],
            ignore_index=True,
        )

    # Cap at 200 without destroying class coverage.
    if len(selected) > 200:
        selected = selected.sample(
            n=200,
            random_state=500,
        )

    selected = selected.sample(
        frac=1,
        random_state=600,
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Human annotation fields
    # ---------------------------------------------------------

    output = selected[
        [
            "tweet_id",
            "conversation_id",
            "created_at",
            "text",
            "weak_intent",
            "weak_label_confidence",
        ]
    ].copy()

    output["human_intent"] = ""
    output["human_notes"] = ""
    output["is_ambiguous"] = ""
    output["needs_context"] = ""

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Created: {OUTPUT_FILE}")
    print(f"Examples: {len(output)}")

    print("\nWeak-label distribution:")
    print(
        output["weak_intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()