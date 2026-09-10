from pathlib import Path
import pandas as pd


INPUT = Path("data/evaluation/golden_set.csv")
OUTPUT = Path("data/evaluation/golden_set.csv")


# Human-reviewed labels for rows 1-200.
# Format:
# row_number: (human_intent, is_ambiguous, needs_context)

labels = {
    1: ("ORDER_MANAGEMENT", "yes", "no"),
    2: ("DELIVERY_DELAY", "no", "no"),
    3: ("CONTEXT_NEEDED", "yes", "yes"),
    4: ("DIGITAL_TECHNICAL", "no", "no"),
    5: ("PRIME_MEMBERSHIP", "no", "no"),
    6: ("SUPPORT_ESCALATION", "no", "no"),
    7: ("PRIME_MEMBERSHIP", "no", "no"),
    8: ("CONTEXT_NEEDED", "no", "no"),
    9: ("SUPPORT_ESCALATION", "no", "no"),
    10: ("DELIVERY_PROOF_FAILURE", "yes", "no"),
    11: ("DELIVERY_DELAY", "yes", "no"),
    12: ("SUPPORT_ESCALATION", "no", "no"),
    13: ("DELIVERY_DELAY", "yes", "yes"),
    14: ("ORDER_MANAGEMENT", "no", "no"),
    15: ("DELIVERY_PROOF_FAILURE", "no", "no"),
    16: ("CONTEXT_NEEDED", "yes", "yes"),
    17: ("CONTEXT_NEEDED", "yes", "yes"),
    18: ("DELIVERY_DELAY", "no", "no"),
    19: ("ORDER_MANAGEMENT", "yes", "no"),
    20: ("DELIVERY_PROOF_FAILURE", "yes", "no"),
    21: ("RETURN_REFUND_REPLACEMENT", "no", "no"),
    22: ("DELIVERY_PROOF_FAILURE", "no", "no"),
    23: ("ACCOUNT_ACCESS_SECURITY", "yes", "no"),
    24: ("ORDER_MANAGEMENT", "yes", "no"),
    25: ("PRODUCT_SELLER_ISSUE", "no", "no"),

    26: ("ORDER_MANAGEMENT", "yes", "no"),
    27: ("RETURN_REFUND_REPLACEMENT", "no", "no"),
    28: ("RETURN_REFUND_REPLACEMENT", "no", "no"),
    29: ("SUPPORT_ESCALATION", "yes", "no"),
    30: ("DIGITAL_TECHNICAL", "yes", "no"),
    31: ("ACCOUNT_ACCESS_SECURITY", "no", "no"),
    32: ("PRODUCT_SELLER_ISSUE", "yes", "no"),
    33: ("ACCOUNT_ACCESS_SECURITY", "no", "no"),
    34: ("SUPPORT_ESCALATION", "no", "no"),
    35: ("PAYMENT_BILLING_GIFTCARD", "no", "no"),
    36: ("PRIME_MEMBERSHIP", "yes", "no"),
    37: ("SUPPORT_ESCALATION", "no", "no"),
    38: ("PRIME_MEMBERSHIP", "no", "no"),
    39: ("SUPPORT_ESCALATION", "yes", "yes"),
    40: ("DELIVERY_PROOF_FAILURE", "yes", "no"),
    41: ("DELIVERY_DELAY", "no", "no"),
    42: ("CONTEXT_NEEDED", "yes", "yes"),
    43: ("CONTEXT_NEEDED", "yes", "yes"),
    44: ("DIGITAL_TECHNICAL", "no", "no"),
    45: ("PRODUCT_SELLER_ISSUE", "no", "no"),
    46: ("DELIVERY_DELAY", "no", "no"),
    47: ("SUPPORT_ESCALATION", "no", "no"),
    48: ("CONTEXT_NEEDED", "no", "no"),
    49: ("CONTEXT_NEEDED", "yes", "yes"),
    50: ("DELIVERY_PROOF_FAILURE", "no", "no"),

    51: ("DELIVERY_PROOF_FAILURE", "yes", "no"),
    52: ("DIGITAL_TECHNICAL", "no", "no"),
    53: ("PAYMENT_BILLING_GIFTCARD", "no", "no"),
    54: ("DELIVERY_PROOF_FAILURE", "no", "no"),
    55: ("PAYMENT_BILLING_GIFTCARD", "yes", "no"),
    56: ("ORDER_MANAGEMENT", "yes", "no"),
    57: ("DIGITAL_TECHNICAL", "yes", "no"),
    58: ("CONTEXT_NEEDED", "yes", "yes"),
    59: ("PRIME_MEMBERSHIP", "no", "no"),
    60: ("CONTEXT_NEEDED", "yes", "yes"),
    61: ("CONTEXT_NEEDED", "yes", "no"),
    62: ("DELIVERY_PROOF_FAILURE", "no", "no"),
    63: ("RETURN_REFUND_REPLACEMENT", "no", "no"),
    64: ("PAYMENT_BILLING_GIFTCARD", "yes", "no"),
    65: ("DELIVERY_PROOF_FAILURE", "no", "no"),
    66: ("ORDER_MANAGEMENT", "no", "no"),
    67: ("DELIVERY_PROOF_FAILURE", "yes", "no"),
    68: ("DELIVERY_DELAY", "no", "no"),
    69: ("CONTEXT_NEEDED", "yes", "yes"),
    70: ("SUPPORT_ESCALATION", "no", "no"),
    71: ("DELIVERY_PROOF_FAILURE", "yes", "no"),
    72: ("SUPPORT_ESCALATION", "yes", "yes"),
    73: ("DELIVERY_DELAY", "no", "no"),
    74: ("DELIVERY_DELAY", "no", "no"),
    75: ("DELIVERY_PROOF_FAILURE", "no", "no"),

    76: ("DIGITAL_TECHNICAL", "no", "no"),
    77: ("SUPPORT_ESCALATION", "no", "no"),
    78: ("PAYMENT_BILLING_GIFTCARD", "no", "no"),
    79: ("PAYMENT_BILLING_GIFTCARD", "yes", "yes"),
    80: ("CONTEXT_NEEDED", "yes", "yes"),
    81: ("DELIVERY_PROOF_FAILURE", "yes", "no"),
    82: ("PRIME_MEMBERSHIP", "no", "no"),
    83: ("SUPPORT_ESCALATION", "no", "no"),
    84: ("DELIVERY_DELAY", "yes", "no"),
    85: ("PAYMENT_BILLING_GIFTCARD", "no", "no"),
    86: ("DELIVERY_PROOF_FAILURE", "no", "no"),
    87: ("CONTEXT_NEEDED", "yes", "no"),
    88: ("DELIVERY_DELAY", "no", "no"),
    89: ("CONTEXT_NEEDED", "yes", "yes"),
    90: ("SUPPORT_ESCALATION", "yes", "yes"),
    91: ("CONTEXT_NEEDED", "no", "no"),
    92: ("CONTEXT_NEEDED", "yes", "yes"),
    93: ("PRODUCT_SELLER_ISSUE", "yes", "no"),
    94: ("DELIVERY_DELAY", "no", "no"),
    95: ("DELIVERY_DELAY", "no", "no"),
    96: ("DELIVERY_DELAY", "yes", "no"),
    97: ("ACCOUNT_ACCESS_SECURITY", "no", "no"),
    98: ("ORDER_MANAGEMENT", "yes", "no"),
    99: ("SUPPORT_ESCALATION", "yes", "no"),
    100: ("CONTEXT_NEEDED", "yes", "yes"),

    101: ("DELIVERY_PROOF_FAILURE", "no", "no"),
    102: ("CONTEXT_NEEDED", "yes", "yes"),
    103: ("DELIVERY_PROOF_FAILURE", "no", "no"),
    104: ("CONTEXT_NEEDED", "yes", "yes"),
    105: ("DIGITAL_TECHNICAL", "no", "no"),
    106: ("DELIVERY_DELAY", "no", "no"),
    107: ("ACCOUNT_ACCESS_SECURITY", "no", "no"),
    108: ("CONTEXT_NEEDED", "yes", "yes"),
    109: ("SUPPORT_ESCALATION", "no", "no"),
    110: ("DELIVERY_DELAY", "no", "no"),
    111: ("ACCOUNT_ACCESS_SECURITY", "no", "no"),
    112: ("DELIVERY_DELAY", "no", "no"),
    113: ("SUPPORT_ESCALATION", "no", "no"),
    114: ("CONTEXT_NEEDED", "yes", "yes"),
    115: ("RETURN_REFUND_REPLACEMENT", "no", "no"),
    116: ("CONTEXT_NEEDED", "yes", "yes"),
    117: ("ACCOUNT_ACCESS_SECURITY", "yes", "no"),
    118: ("PAYMENT_BILLING_GIFTCARD", "yes", "no"),
    119: ("DELIVERY_DELAY", "no", "no"),
    120: ("PRIME_MEMBERSHIP", "no", "no"),
    121: ("DELIVERY_DELAY", "no", "no"),
    122: ("CONTEXT_NEEDED", "yes", "yes"),
    123: ("PRIME_MEMBERSHIP", "yes", "no"),
    124: ("PAYMENT_BILLING_GIFTCARD", "no", "no"),
    125: ("SUPPORT_ESCALATION", "no", "no"),

    126: ("RETURN_REFUND_REPLACEMENT", "no", "no"),
    127: ("DELIVERY_PROOF_FAILURE", "yes", "yes"),
    128: ("DELIVERY_DELAY", "no", "no"),
    129: ("SUPPORT_ESCALATION", "yes", "yes"),
    130: ("PAYMENT_BILLING_GIFTCARD", "no", "no"),
    131: ("PAYMENT_BILLING_GIFTCARD", "no", "no"),
    132: ("DELIVERY_DELAY", "no", "no"),
    133: ("PRIME_MEMBERSHIP", "no", "no"),
    134: ("ORDER_MANAGEMENT", "no", "no"),
    135: ("DELIVERY_DELAY", "no", "no"),
    136: ("DELIVERY_DELAY", "no", "no"),
    137: ("ORDER_MANAGEMENT", "no", "no"),
    138: ("RETURN_REFUND_REPLACEMENT", "no", "no"),
    139: ("SUPPORT_ESCALATION", "yes", "yes"),
    140: ("ACCOUNT_ACCESS_SECURITY", "no", "no"),
    141: ("CONTEXT_NEEDED", "yes", "yes"),
    142: ("PRODUCT_SELLER_ISSUE", "no", "no"),
    143: ("DIGITAL_TECHNICAL", "no", "no"),
    144: ("DIGITAL_TECHNICAL", "yes", "no"),
    145: ("PRODUCT_SELLER_ISSUE", "yes", "yes"),
    146: ("RETURN_REFUND_REPLACEMENT", "no", "no"),
    147: ("RETURN_REFUND_REPLACEMENT", "no", "no"),
    148: ("CONTEXT_NEEDED", "yes", "yes"),
    149: ("PRIME_MEMBERSHIP", "no", "no"),
    150: ("PAYMENT_BILLING_GIFTCARD", "yes", "no"),

    151: ("CONTEXT_NEEDED", "yes", "yes"),
    152: ("PRIME_MEMBERSHIP", "no", "no"),
    153: ("SUPPORT_ESCALATION", "no", "no"),
    154: ("DELIVERY_PROOF_FAILURE", "no", "no"),
    155: ("PRIME_MEMBERSHIP", "no", "no"),
    156: ("CONTEXT_NEEDED", "yes", "yes"),
    157: ("CONTEXT_NEEDED", "yes", "yes"),
    158: ("RETURN_REFUND_REPLACEMENT", "no", "no"),
    159: ("ACCOUNT_ACCESS_SECURITY", "no", "no"),
    160: ("CONTEXT_NEEDED", "yes", "yes"),
    161: ("SUPPORT_ESCALATION", "no", "no"),
    162: ("CONTEXT_NEEDED", "yes", "yes"),
    163: ("ORDER_MANAGEMENT", "no", "no"),
    164: ("CONTEXT_NEEDED", "yes", "yes"),
    165: ("CONTEXT_NEEDED", "yes", "yes"),
    166: ("DELIVERY_DELAY", "no", "no"),
    167: ("PRODUCT_SELLER_ISSUE", "no", "no"),
    168: ("SUPPORT_ESCALATION", "yes", "yes"),
    169: ("PAYMENT_BILLING_GIFTCARD", "yes", "yes"),
    170: ("CONTEXT_NEEDED", "yes", "yes"),
    171: ("DELIVERY_PROOF_FAILURE", "yes", "yes"),
    172: ("DELIVERY_DELAY", "no", "no"),
    173: ("CONTEXT_NEEDED", "no", "no"),
    174: ("RETURN_REFUND_REPLACEMENT", "no", "no"),
    175: ("PRIME_MEMBERSHIP", "no", "no"),

    176: ("SUPPORT_ESCALATION", "no", "no"),
    177: ("SUPPORT_ESCALATION", "no", "no"),
    178: ("PRODUCT_SELLER_ISSUE", "no", "no"),
    179: ("CONTEXT_NEEDED", "yes", "yes"),
    180: ("CONTEXT_NEEDED", "no", "no"),
    181: ("DIGITAL_TECHNICAL", "no", "no"),
    182: ("DELIVERY_PROOF_FAILURE", "no", "no"),
    183: ("CONTEXT_NEEDED", "no", "no"),
    184: ("DIGITAL_TECHNICAL", "no", "no"),
    185: ("CONTEXT_NEEDED", "yes", "yes"),
    186: ("ORDER_MANAGEMENT", "yes", "yes"),
    187: ("SUPPORT_ESCALATION", "no", "no"),
    188: ("CONTEXT_NEEDED", "yes", "yes"),
    189: ("CONTEXT_NEEDED", "yes", "yes"),
    190: ("DIGITAL_TECHNICAL", "no", "no"),
    191: ("PRIME_MEMBERSHIP", "no", "no"),
    192: ("DIGITAL_TECHNICAL", "no", "no"),
    193: ("CONTEXT_NEEDED", "no", "no"),
    194: ("CONTEXT_NEEDED", "yes", "yes"),
    195: ("DELIVERY_DELAY", "no", "no"),
    196: ("SUPPORT_ESCALATION", "no", "no"),
    197: ("CONTEXT_NEEDED", "yes", "yes"),
    198: ("SUPPORT_ESCALATION", "no", "no"),
    199: ("DELIVERY_DELAY", "yes", "no"),
    200: ("ORDER_MANAGEMENT", "yes", "no"),
}


def main():
    # Check input file exists
    if not INPUT.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT}"
        )

    # Read the 200-row golden set
    df = pd.read_csv(INPUT)

    # Make sure we really have 200 examples
    if len(df) != 200:
        raise ValueError(
            f"Expected exactly 200 rows, but found {len(df)} rows."
        )

    # Make sure all required columns exist
    required_columns = [
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

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # IMPORTANT:
    # Empty CSV columns are automatically interpreted by pandas
    # as float64 because they contain only NaN values.
    #
    # Convert the columns we are going to fill into strings first.
    df["human_intent"] = df["human_intent"].fillna("").astype(str)
    df["human_notes"] = df["human_notes"].fillna("").astype(str)
    df["is_ambiguous"] = df["is_ambiguous"].fillna("").astype(str)
    df["needs_context"] = df["needs_context"].fillna("").astype(str)

    # Check that we have exactly 200 labels
    if len(labels) != 200:
        raise ValueError(
            f"Expected 200 labels, but found {len(labels)}."
        )

    # Fill the human annotation columns
    for row_number, (intent, ambiguous, context) in labels.items():

        # Python/DataFrame index starts at 0,
        # while our row numbers start at 1.
        idx = row_number - 1

        df.loc[idx, "human_intent"] = intent
        df.loc[idx, "human_notes"] = ""
        df.loc[idx, "is_ambiguous"] = ambiguous
        df.loc[idx, "needs_context"] = context

    # Save the completed golden set
    df.to_csv(OUTPUT, index=False)

    # Print confirmation
    print()
    print("=" * 60)
    print("GOLDEN SET UPDATED SUCCESSFULLY")
    print("=" * 60)
    print(f"File: {OUTPUT}")
    print(f"Rows: {len(df)}")
    print()

    print("Human intent distribution:")
    print(df["human_intent"].value_counts().sort_index())

    print()
    print("Ambiguous examples:")
    print(df["is_ambiguous"].value_counts().sort_index())

    print()
    print("Needs context:")
    print(df["needs_context"].value_counts().sort_index())

    print()
    print("First 10 rows:")
    print(
        df[
            [
                "tweet_id",
                "human_intent",
                "is_ambiguous",
                "needs_context",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("Done!")


if __name__ == "__main__":
    main()