from pathlib import Path
import argparse
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# FILES
# ============================================================

MESSAGES_FILE = Path(
    "data/processed/amazonhelp_dev_messages.csv"
)

CONVERSATIONS_FILE = Path(
    "data/processed/amazonhelp_dev_conversations.csv"
)

OUTPUT_FILE = Path(
    "data/evaluation/retrieval_examples.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 5

MIN_SIMILARITY = 0.10


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Basic text normalization used for retrieval.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().lower()


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the processed AmazonHelp messages and conversations.
    """

    if not MESSAGES_FILE.exists():
        raise FileNotFoundError(
            f"Messages file not found: {MESSAGES_FILE}"
        )

    if not CONVERSATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Conversations file not found: {CONVERSATIONS_FILE}"
        )

    messages = pd.read_csv(
        MESSAGES_FILE
    )

    conversations = pd.read_csv(
        CONVERSATIONS_FILE
    )

    return messages, conversations


# ============================================================
# PREPARE RETRIEVAL CORPUS
# ============================================================

def prepare_corpus(messages):
    """
    Create one searchable document for each customer message.

    We retrieve customer messages because they describe the
    actual support problem. Once a similar message is found,
    we use the conversation_id to recover the historical
    support responses from the same conversation.
    """

    required_columns = [
        "tweet_id",
        "conversation_id",
        "inbound",
        "text",
    ]

    missing = [
        column
        for column in required_columns
        if column not in messages.columns
    ]

    if missing:
        raise ValueError(
            f"Messages file is missing columns: {missing}"
        )

    # Customer messages only
    customer_messages = messages[
        messages["inbound"] == True
    ].copy()

    customer_messages["search_text"] = (
        customer_messages["text"]
        .fillna("")
        .map(clean_text)
    )

    # Remove empty messages
    customer_messages = customer_messages[
        customer_messages["search_text"].str.len() > 0
    ].copy()

    customer_messages = customer_messages.reset_index(
        drop=True
    )

    return customer_messages


# ============================================================
# BUILD TF-IDF INDEX
# ============================================================

def build_index(customer_messages):
    """
    Build a simple TF-IDF retrieval index.

    This is intentionally lightweight and reproducible.
    """

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        customer_messages["search_text"]
    )

    return vectorizer, matrix


# ============================================================
# GET SUPPORT REPLIES
# ============================================================

def get_support_replies(
    messages,
    conversation_id,
):
    """
    Return the support/brand messages from a conversation,
    ordered chronologically.

    inbound=True  -> customer
    inbound=False -> support/brand
    """

    conversation = messages[
        messages["conversation_id"] == conversation_id
    ].copy()

    if conversation.empty:
        return []

    if "created_at" in conversation.columns:
        conversation["created_at_parsed"] = pd.to_datetime(
            conversation["created_at"],
            errors="coerce",
            utc=True,
        )

        conversation = conversation.sort_values(
            [
                "created_at_parsed",
                "tweet_id",
            ]
        )

    support_messages = conversation[
        conversation["inbound"] == False
    ]

    replies = []

    for _, row in support_messages.iterrows():

        text = str(row.get("text", "")).strip()

        if not text:
            continue

        replies.append(
            {
                "tweet_id": row["tweet_id"],
                "text": text,
                "created_at": row.get(
                    "created_at",
                    "",
                ),
            }
        )

    return replies


# ============================================================
# RETRIEVE SIMILAR CASES
# ============================================================

def retrieve_similar_cases(
    query,
    customer_messages,
    vectorizer,
    matrix,
    messages,
    top_k=TOP_K,
):
    """
    Retrieve the most similar historical customer messages.

    Only return cases that have at least one historical
    support reply. This makes them useful as evidence for
    response drafting.
    """

    query_clean = clean_text(query)

    if not query_clean:
        return []

    query_vector = vectorizer.transform(
        [query_clean]
    )

    similarities = cosine_similarity(
        query_vector,
        matrix,
    )[0]

    # Highest similarity first
    ranked_indices = similarities.argsort()[::-1]

    results = []

    seen_conversations = set()

    for index in ranked_indices:

        similarity = float(
            similarities[index]
        )

        if similarity < MIN_SIMILARITY:
            break

        row = customer_messages.iloc[index]

        conversation_id = row[
            "conversation_id"
        ]

        # Avoid returning multiple nearly-identical
        # messages from the same conversation.
        if conversation_id in seen_conversations:
            continue

        support_replies = get_support_replies(
            messages,
            conversation_id,
        )

        if not support_replies:
            continue

        seen_conversations.add(
            conversation_id
        )

        results.append(
            {
                "tweet_id": row["tweet_id"],
                "conversation_id": conversation_id,
                "similarity": similarity,
                "customer_message": str(
                    row["text"]
                ),
                "support_replies": support_replies,
            }
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_results(
    query,
    results,
):
    """
    Print retrieved historical cases in a readable format.
    """

    print()
    print("=" * 70)
    print("CUSTOMER MESSAGE")
    print("=" * 70)
    print(query)

    print()
    print("=" * 70)
    print("HISTORICAL EVIDENCE")
    print("=" * 70)

    if not results:
        print("No sufficiently similar historical cases found.")
        return

    for number, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(
            f"[{number}] Similarity: "
            f"{result['similarity']:.3f}"
        )

        print(
            f"Conversation ID: "
            f"{result['conversation_id']}"
        )

        print(
            f"Customer: "
            f"{result['customer_message']}"
        )

        print("Historical support response(s):")

        for reply in result[
            "support_replies"
        ]:

            print(
                f"  - {reply['text']}"
            )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    query,
    results,
):
    """
    Save retrieved evidence to CSV.

    The complete support reply list is stored as text so
    the evidence can later be inspected or passed into a
    response-generation component.
    """

    rows = []

    for rank, result in enumerate(
        results,
        start=1,
    ):

        replies = "\n".join(
            [
                reply["text"]
                for reply in result[
                    "support_replies"
                ]
            ]
        )

        rows.append(
            {
                "query": query,
                "rank": rank,
                "similarity": result[
                    "similarity"
                ],
                "tweet_id": result[
                    "tweet_id"
                ],
                "conversation_id": result[
                    "conversation_id"
                ],
                "customer_message": result[
                    "customer_message"
                ],
                "historical_support_replies": replies,
            }
        )

    output_df = pd.DataFrame(
        rows
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved retrieval results to: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Retrieve similar historical "
            "Amazon customer-support cases."
        )
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=(
            "My package says it was delivered "
            "but I never received it."
        ),
        help="Customer message to search for.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of historical cases to retrieve.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("Loading processed dataset...")

    messages, conversations = load_data()

    print(
        f"Messages loaded: {len(messages):,}"
    )

    print(
        f"Conversations loaded: "
        f"{len(conversations):,}"
    )

    # --------------------------------------------------------
    # Prepare corpus
    # --------------------------------------------------------

    print()
    print("Preparing customer-message retrieval corpus...")

    customer_messages = prepare_corpus(
        messages
    )

    print(
        f"Customer messages indexed: "
        f"{len(customer_messages):,}"
    )

    # --------------------------------------------------------
    # Build index
    # --------------------------------------------------------

    print()
    print("Building TF-IDF retrieval index...")

    vectorizer, matrix = build_index(
        customer_messages
    )

    print(
        f"TF-IDF matrix shape: "
        f"{matrix.shape}"
    )

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    print()
    print("Searching historical conversations...")

    results = retrieve_similar_cases(
        query=args.query,
        customer_messages=customer_messages,
        vectorizer=vectorizer,
        matrix=matrix,
        messages=messages,
        top_k=args.top_k,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_results(
        args.query,
        results,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        args.query,
        results,
    )


if __name__ == "__main__":
    main()