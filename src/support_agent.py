from pathlib import Path
import argparse
import re

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# FILES
# ============================================================

MESSAGES_FILE = Path(
    "data/processed/amazonhelp_dev_messages.csv"
)

MODEL_FILE = Path(
    "models/intent_classifier.joblib"
)

VECTORIZER_FILE = Path(
    "models/tfidf_vectorizer.joblib"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 3

MIN_SIMILARITY = 0.10


# ============================================================
# INTENT DEFINITIONS
# ============================================================

INTENT_DESCRIPTIONS = {
    "ACCOUNT_ACCESS_SECURITY":
        "Account login, password, hacked account, or security issue.",

    "CONTEXT_NEEDED":
        "The message does not contain enough information to identify the issue.",

    "DELIVERY_DELAY":
        "Package or order is delayed, late, or delivery date was missed.",

    "DELIVERY_PROOF_FAILURE":
        "Package is marked delivered, attempted, missing, or delivery handoff failed.",

    "DIGITAL_TECHNICAL":
        "Technical issue involving Kindle, Fire TV, Alexa, apps, website, or digital services.",

    "ORDER_MANAGEMENT":
        "Order cancellation, address change, preorder, order status, or order management.",

    "PAYMENT_BILLING_GIFTCARD":
        "Payment, card, charge, billing, gift card, or payment method issue.",

    "PRIME_MEMBERSHIP":
        "Prime membership, Prime benefits, subscription, renewal, or Prime delivery benefit.",

    "PRODUCT_SELLER_ISSUE":
        "Defective, damaged, fake, wrong, or seller-related product issue.",

    "RETURN_REFUND_REPLACEMENT":
        "Return, refund, replacement, or exchange request.",

    "SUPPORT_ESCALATION":
        "Customer cannot get useful support, needs escalation, callback, or repeated support failed.",
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Basic text normalization for retrieval.
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
# LOAD SYSTEM
# ============================================================

def load_system():

    if not MESSAGES_FILE.exists():
        raise FileNotFoundError(
            f"Messages file not found: {MESSAGES_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Classifier model not found: {MODEL_FILE}"
        )

    if not VECTORIZER_FILE.exists():
        raise FileNotFoundError(
            f"Vectorizer not found: {VECTORIZER_FILE}"
        )

    messages = pd.read_csv(
        MESSAGES_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    return messages, model, vectorizer


# ============================================================
# PREPARE RETRIEVAL CORPUS
# ============================================================

def prepare_retrieval_corpus(messages):

    customer_messages = messages[
        messages["inbound"] == True
    ].copy()

    customer_messages["search_text"] = (
        customer_messages["text"]
        .fillna("")
        .map(clean_text)
    )

    customer_messages = customer_messages[
        customer_messages["search_text"].str.len() > 0
    ].copy()

    customer_messages = customer_messages.reset_index(
        drop=True
    )

    return customer_messages


# ============================================================
# BUILD RETRIEVAL INDEX
# ============================================================

def build_retrieval_index(customer_messages):

    # Reuse the same TF-IDF vectorizer settings
    # as the baseline classifier.

    from sklearn.feature_extraction.text import TfidfVectorizer

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

    conversation = messages[
        messages["conversation_id"] == conversation_id
    ].copy()

    if conversation.empty:
        return []

    if "created_at" in conversation.columns:

        conversation["created_at_parsed"] = pd.to_datetime(
            conversation["created_at"],
            errors="coerce",
            format="mixed",
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

        text = str(
            row.get("text", "")
        ).strip()

        if not text:
            continue

        replies.append(text)

    return replies


# ============================================================
# CLASSIFY INTENT
# ============================================================

def classify_intent(
    message,
    model,
    vectorizer,
):

    cleaned = clean_text(
        message
    )

    vector = vectorizer.transform(
        [cleaned]
    )

    prediction = model.predict(
        vector
    )[0]

    # Probability if available
    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            vector
        )[0]

        confidence = float(
            probabilities.max()
        )

    return prediction, confidence


# ============================================================
# RETRIEVE HISTORICAL EVIDENCE
# ============================================================

def retrieve_evidence(
    message,
    messages,
    customer_messages,
    vectorizer,
    matrix,
    top_k=TOP_K,
):

    cleaned = clean_text(
        message
    )

    if not cleaned:
        return []

    query_vector = vectorizer.transform(
        [cleaned]
    )

    similarities = cosine_similarity(
        query_vector,
        matrix,
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    seen_conversations = set()

    for index in ranked_indices:

        similarity = float(
            similarities[index]
        )

        if similarity < MIN_SIMILARITY:
            break

        row = customer_messages.iloc[
            index
        ]

        conversation_id = row[
            "conversation_id"
        ]

        # Don't return multiple messages
        # from the same conversation.
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
                "tweet_id": row[
                    "tweet_id"
                ],
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
# DECISION: AUTO-HANDLE OR ESCALATE
# ============================================================

def make_decision(
    intent,
    classifier_confidence,
    evidence,
):
    """
    Decide whether the agent should automatically handle
    the customer message or escalate it to a human.

    Decision uses three signals:

    1. Intent confidence
    2. Strength of historical evidence
    3. Whether the intent normally requires human review
    """

    # --------------------------------------------------------
    # No historical evidence
    # --------------------------------------------------------

    if not evidence:

        return (
            "ESCALATE",
            "No sufficiently similar historical support case was found."
        )

    # --------------------------------------------------------
    # Context-needed cases
    # --------------------------------------------------------

    if intent == "CONTEXT_NEEDED":

        return (
            "ESCALATE",
            "The customer message does not contain enough information to identify the issue."
        )

    # --------------------------------------------------------
    # Classifier confidence
    # --------------------------------------------------------

    if classifier_confidence is None:

        return (
            "ESCALATE",
            "The classifier did not provide a confidence score."
        )

    # --------------------------------------------------------
    # Best historical similarity
    # --------------------------------------------------------

    best_similarity = max(
        item["similarity"]
        for item in evidence
    )

    # --------------------------------------------------------
    # Very low classifier confidence
    # --------------------------------------------------------

    if classifier_confidence < 0.50:

        return (
            "ESCALATE",
            f"Intent confidence is too low ({classifier_confidence:.2f})."
        )

    # --------------------------------------------------------
    # Weak historical evidence
    # --------------------------------------------------------

    if best_similarity < 0.20:

        return (
            "ESCALATE",
            f"Historical evidence is weak (best similarity: {best_similarity:.2f})."
        )

    # --------------------------------------------------------
    # Moderate confidence requires stronger evidence
    # --------------------------------------------------------

    if classifier_confidence < 0.65:

        if best_similarity < 0.35:

            return (
                "ESCALATE",
                "Intent confidence is moderate and the strongest historical case is not similar enough."
            )

    # --------------------------------------------------------
    # Sensitive / higher-risk intents
    # --------------------------------------------------------

    if intent == "ACCOUNT_ACCESS_SECURITY":

        if classifier_confidence < 0.80:

            return (
                "ESCALATE",
                "Account-security issues require higher classification confidence before automatic handling."
            )

    # --------------------------------------------------------
    # Support escalation
    # --------------------------------------------------------

    if intent == "SUPPORT_ESCALATION":

        return (
            "ESCALATE",
            "The customer is already reporting unsuccessful support interactions, so human review is appropriate."
        )

    # --------------------------------------------------------
    # Strong case
    # --------------------------------------------------------

    return (
        "AUTO-HANDLE",
        (
            f"Intent confidence is {classifier_confidence:.2f} "
            f"and the strongest historical case has "
            f"{best_similarity:.2f} similarity."
        )
    )

def draft_response(
    message,
    intent,
    evidence,
    decision,
):
    """
    Create a grounded response using historical support behavior.

    This version intentionally does NOT call an external LLM.
    It uses simple templates so the project remains reproducible
    and inexpensive.
    """

    if decision == "ESCALATE":

        return (
            "Thanks for reaching out. "
            "I’m sorry we couldn’t confidently determine the issue "
            "from the information provided. "
            "A support representative should review this and assist further."
        )

    # --------------------------------------------------------
    # Get the strongest historical support reply.
    # --------------------------------------------------------

    best_reply = ""

    if evidence:

        replies = evidence[0][
            "support_replies"
        ]

        if replies:
            best_reply = replies[0]

    # --------------------------------------------------------
    # Intent-specific grounded templates.
    # --------------------------------------------------------

    if intent == "DELIVERY_PROOF_FAILURE":

        response = (
            "Sorry about that. If your package is showing as delivered "
            "but you haven't received it, please check the delivery "
            "details and carrier information first. "
            "If it still cannot be located, support can investigate "
            "the delivery further."
        )

    elif intent == "DELIVERY_DELAY":

        response = (
            "Sorry that your delivery is taking longer than expected. "
            "Please check the latest tracking information for the order. "
            "If the promised delivery date has passed, support can "
            "review the delivery status and available options."
        )

    elif intent == "RETURN_REFUND_REPLACEMENT":

        response = (
            "Sorry about the trouble with your order. "
            "Please check the order's return or refund options. "
            "If the refund or replacement is already in progress, "
            "support can review its current status."
        )

    elif intent == "PAYMENT_BILLING_GIFTCARD":

        response = (
            "Sorry about the payment issue. "
            "Please check the payment method and the transaction "
            "details associated with the order. "
            "If the charge still looks incorrect, support can "
            "review the transaction."
        )

    elif intent == "ACCOUNT_ACCESS_SECURITY":

        response = (
            "For account access or security issues, please use the "
            "official account recovery or security process. "
            "If you still cannot access the account or believe it "
            "has been compromised, support should review it."
        )

    elif intent == "PRIME_MEMBERSHIP":

        response = (
            "Sorry about the trouble with your Prime service. "
            "Please check your Prime membership and the benefit "
            "associated with the order. "
            "If the expected Prime benefit was not applied, "
            "support can review the membership and order details."
        )

    elif intent == "DIGITAL_TECHNICAL":

        response = (
            "Sorry you're having trouble with the digital service. "
            "Please check the device or application settings and "
            "try the relevant troubleshooting steps. "
            "If the problem continues, support can investigate further."
        )

    elif intent == "PRODUCT_SELLER_ISSUE":

        response = (
            "Sorry about the issue with the product. "
            "Please check the order and product details to review "
            "the available support, return, or replacement options."
        )

    elif intent == "ORDER_MANAGEMENT":

        response = (
            "I can help with the order issue. "
            "Please check the order details for the available "
            "cancellation, address, or order-management options."
        )

    elif intent == "SUPPORT_ESCALATION":

        response = (
            "I'm sorry you've had trouble getting this resolved. "
            "Based on similar support cases, this should be reviewed "
            "by a support representative so they can investigate "
            "the issue and provide the appropriate next step."
        )

    else:

        response = (
            "Thanks for reaching out. "
            "A support representative should review the details "
            "and help with the issue."
        )

    return response


# ============================================================
# PRINT AGENT RESULT
# ============================================================

def print_agent_result(
    message,
    intent,
    confidence,
    decision,
    reason,
    response,
    evidence,
):

    print()
    print("=" * 70)
    print("AI SUPPORT AGENT")
    print("=" * 70)

    print()
    print("CUSTOMER MESSAGE")
    print("----------------")
    print(message)

    print()
    print("INTENT")
    print("------")
    print(intent)

    if confidence is not None:

        print(
            f"Classifier confidence: "
            f"{confidence:.3f}"
        )

    print()
    print("DECISION")
    print("--------")
    print(decision)

    print()
    print("REASON")
    print("------")
    print(reason)

    print()
    print("DRAFT RESPONSE")
    print("--------------")
    print(response)

    print()
    print("HISTORICAL EVIDENCE")
    print("-------------------")

    if not evidence:

        print("No historical evidence found.")

    else:

        for number, item in enumerate(
            evidence,
            start=1,
        ):

            print()
            print(
                f"[{number}] "
                f"Similarity: "
                f"{item['similarity']:.3f}"
            )

            print(
                f"Conversation: "
                f"{item['conversation_id']}"
            )

            print(
                f"Customer case: "
                f"{item['customer_message']}"
            )

            print(
                "Historical support:"
            )

            for reply in item[
                "support_replies"
            ]:

                print(
                    f"  - {reply}"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the AI customer-support agent."
        )
    )

    parser.add_argument(
        "message",
        nargs="?",
        default=(
            "My package says it was delivered "
            "but I never received it."
        ),
        help="Customer message.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        "Loading support-agent components..."
    )

    messages, model, classifier_vectorizer = (
        load_system()
    )

    # --------------------------------------------------------
    # Prepare retrieval corpus
    # --------------------------------------------------------

    print(
        "Preparing historical evidence..."
    )

    customer_messages = (
        prepare_retrieval_corpus(
            messages
        )
    )

    # --------------------------------------------------------
    # Build retrieval index
    # --------------------------------------------------------

    retrieval_vectorizer, retrieval_matrix = (
        build_retrieval_index(
            customer_messages
        )
    )

    # --------------------------------------------------------
    # Classify
    # --------------------------------------------------------

    print(
        "Classifying customer message..."
    )

    intent, confidence = classify_intent(
        args.message,
        model,
        classifier_vectorizer,
    )

    # --------------------------------------------------------
    # Retrieve evidence
    # --------------------------------------------------------

    print(
        "Retrieving historical support cases..."
    )

    evidence = retrieve_evidence(
        message=args.message,
        messages=messages,
        customer_messages=customer_messages,
        vectorizer=retrieval_vectorizer,
        matrix=retrieval_matrix,
        top_k=TOP_K,
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    decision, reason = make_decision(
        intent=intent,
        classifier_confidence=confidence,
        evidence=evidence,
    )

    # --------------------------------------------------------
    # Draft
    # --------------------------------------------------------

    response = draft_response(
        message=args.message,
        intent=intent,
        evidence=evidence,
        decision=decision,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_agent_result(
        message=args.message,
        intent=intent,
        confidence=confidence,
        decision=decision,
        reason=reason,
        response=response,
        evidence=evidence,
    )


if __name__ == "__main__":
    main()