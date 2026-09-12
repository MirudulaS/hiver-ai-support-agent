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
    Basic text normalization for retrieval and classification.
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
    cleaned = clean_text(message)

    # --------------------------------------------------------
    # Explicit high-confidence rules
    # --------------------------------------------------------

    # Account / security
    if any(
        phrase in cleaned
        for phrase in [
            "hacked",
            "account hacked",
            "someone accessed my account",
            "unauthorized access",
            "unauthorized activity",
            "fraudulently",
            "fraudulent",
            "account blocked",
            "account is blocked",
            "locked out",
            "can't log in",
            "cannot log in",
            "can't login",
            "cannot login",
            "password reset",
            "forgot my password",
            "change my password",
            "changed my email",
            "changed email",
            "phone number changed",
        ]
    ):
        return "ACCOUNT_ACCESS_SECURITY", 0.99

    # --------------------------------------------------------
    # Support escalation
    # --------------------------------------------------------

    # These indicate that the customer has already tried
    # getting help and the issue remains unresolved.
    if any(
        phrase in cleaned
        for phrase in [
            "no one helped",
            "nobody helped",
            "no help",
            "not resolved",
            "still not resolved",
            "not been resolved",
            "no resolution",
            "didn't get any resolution",
            "did not get any resolution",
            "already contacted support",
            "contacted support",
            "contacted customer service",
            "talked to support",
            "spoke to support",
            "many agents",
            "multiple agents",
            "15 agents",
            "several agents",
            "two escalations",
            "escalate",
            "still waiting for your reply",
            "still waiting for a response",
            "waiting for your response",
            "waiting for a response",
            "no satisfactory response",
            "customer service are useless",
        ]
    ):
        return "SUPPORT_ESCALATION", 0.99

    # --------------------------------------------------------
    # Delivery proof / failed handoff
    # --------------------------------------------------------

    # Explicit delivered-but-not-received cases.
    if (
        any(
            phrase in cleaned
            for phrase in [
                "marked as delivered",
                "shows as delivered",
                "says it was delivered",
                "says it has been delivered",
                "website says i received",
                "website says i got",
                "delivered but",
                "delivered and i haven't",
                "delivered and i have not",
                "delivered but i never",
                "delivered to wrong address",
                "wrong address",
                "wrongly delivered",
                "wrong order was delivered",
                "incorrect order was delivered",
                "left at the door",
                "left in dirt",
                "left at my door",
                "attempted delivery",
                "undeliverable",
                "package was returned",
                "package returned",
            ]
        )
    ):
        return "DELIVERY_PROOF_FAILURE", 0.99

    # Missing/empty/wrong contents are product/package issues,
    # but should be escalated later by make_decision().
    if any(
        phrase in cleaned
        for phrase in [
            "item missing",
            "missing item",
            "item is missing",
            "missing from the box",
            "missing in the box",
            "something missing",
            "part missing",
            "empty package",
            "empty pkg",
            "wrong item",
            "wrong product",
            "wrong order",
            "received the wrong",
            "damaged",
            "broken",
            "ruined",
            "smashed",
            "warped",
            "ripped",
            "poor packaging",
            "bad packaging",
            "no bubblewrap",
            "no bubble wrap",
            "fake product",
            "counterfeit",
        ]
    ):
        return "PRODUCT_SELLER_ISSUE", 0.99

    # --------------------------------------------------------
    # Return / refund / replacement
    # --------------------------------------------------------

    if any(
        phrase in cleaned
        for phrase in [
            "want a refund",
            "need a refund",
            "refund",
            "return this",
            "return my",
            "return an item",
            "return item",
            "replacement",
            "replace this",
            "exchange this",
            "exchange it",
            "exchange available",
        ]
    ):
        return "RETURN_REFUND_REPLACEMENT", 0.99

    # --------------------------------------------------------
    # Payment / billing
    # --------------------------------------------------------

    if any(
        phrase in cleaned
        for phrase in [
            "charged",
            "charge",
            "billing",
            "payment",
            "paid for",
            "payment page",
            "payment number",
            "payment method",
            "credit card",
            "debit card",
            "cashback",
            "cash back",
            "gift card",
            "wallet",
            "cod",
            "cash on delivery",
            "gst invoice",
            "invoice",
        ]
    ):
        return "PAYMENT_BILLING_GIFTCARD", 0.99

    # --------------------------------------------------------
    # Digital technical
    # --------------------------------------------------------

    if any(
        phrase in cleaned
        for phrase in [
            "kindle",
            "fire tv",
            "alexa",
            "echo",
            "chromecast",
            "amazon app",
            "app not working",
            "app doesn't work",
            "app does not work",
            "website not working",
            "website doesn't work",
            "website does not work",
            "technical issue",
            "system error",
            "error message",
            "button on my app",
            "update your app",
            "app issue",
        ]
    ):
        return "DIGITAL_TECHNICAL", 0.99

    # --------------------------------------------------------
    # Prime membership
    # --------------------------------------------------------

    if any(
        phrase in cleaned
        for phrase in [
            "prime membership",
            "prime subscription",
            "prime renewal",
            "prime member",
            "prime membership charged",
            "prime subscription charged",
            "amazon prime",
            "amazon prime student",
            "prime student",
            "charged for prime",
            "pay for prime",
            "prime benefit",
            "prime benefits",
        ]
    ):
        return "PRIME_MEMBERSHIP", 0.99

    # --------------------------------------------------------
    # Delivery delay
    # --------------------------------------------------------

    if any(
        phrase in cleaned
        for phrase in [
            "delivery delayed",
            "delivery delay",
            "delivery is late",
            "delivery late",
            "package is late",
            "package was late",
            "order is late",
            "order was late",
            "still hasn't arrived",
            "still has not arrived",
            "hasn't arrived",
            "has not arrived",
            "not arrived",
            "not yet delivered",
            "not delivered yet",
            "not delivered",
            "waiting for my order",
            "waiting for my package",
            "waiting for delivery",
            "overdue",
            "days late",
            "week late",
            "next day delivery",
            "one day delivery",
            "promised delivery date",
            "delivery date has passed",
            "no delivery",
            "no update to tracking",
        ]
    ):
        return "DELIVERY_DELAY", 0.99

    # --------------------------------------------------------
    # Order management
    # --------------------------------------------------------

    if any(
        phrase in cleaned
        for phrase in [
            "cancel my order",
            "cancelled my order",
            "order cancelled",
            "order cancellation",
            "change my order",
            "change the order",
            "change delivery address",
            "change my address",
            "delivery address",
            "order status",
            "track my order",
            "preorder",
            "pre-ordered",
            "preordered",
            "cannot buy",
            "can't buy",
            "can't purchase",
            "cannot purchase",
            "option to buy",
            "not available to buy",
            "out of stock",
            "sold out",
            "cod available",
            "exchange available",
        ]
    ):
        return "ORDER_MANAGEMENT", 0.99

    # --------------------------------------------------------
    # Fallback to trained classifier
    # --------------------------------------------------------

    if not cleaned:
        return "CONTEXT_NEEDED", 0.99

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        confidence = float(probabilities.max())

    return prediction, confidence


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
    """

    # No historical evidence
    if not evidence:

        return (
            "ESCALATE",
            "No sufficiently similar historical support case was found."
        )

    # Context-needed cases
    if intent == "CONTEXT_NEEDED":

        return (
            "ESCALATE",
            "The customer message does not contain enough information to identify the issue."
        )

    # No confidence
    if classifier_confidence is None:

        return (
            "ESCALATE",
            "The classifier did not provide a confidence score."
        )

    # Best historical similarity
    best_similarity = max(
        item["similarity"]
        for item in evidence
    )

    # Very low classifier confidence
    if classifier_confidence < 0.50:

        return (
            "ESCALATE",
            f"Intent confidence is too low ({classifier_confidence:.2f})."
        )

    # Weak historical evidence
    if best_similarity < 0.20:

        return (
            "ESCALATE",
            f"Historical evidence is weak (best similarity: {best_similarity:.2f})."
        )

    # Moderate confidence requires stronger evidence
    if classifier_confidence < 0.65:

        if best_similarity < 0.35:

            return (
                "ESCALATE",
                "Intent confidence is moderate and the strongest historical case is not similar enough."
            )

    # Sensitive intent
    if intent == "ACCOUNT_ACCESS_SECURITY":

        if classifier_confidence < 0.80:

            return (
                "ESCALATE",
                "Account-security issues require higher classification confidence before automatic handling."
            )
    # --------------------------------------------------------
    # Product / missing-item cases
    # --------------------------------------------------------

    if intent == "PRODUCT_SELLER_ISSUE":

        return (
            "ESCALATE",
            "Product or missing-item issues may require order-level review by a support representative."
        )
    
    # Support escalation
    if intent == "SUPPORT_ESCALATION":

        return (
            "ESCALATE",
            "The customer is already reporting unsuccessful support interactions, so human review is appropriate."
        )

    # Strong case
    return (
        "AUTO-HANDLE",
        (
            f"Intent confidence is {classifier_confidence:.2f} "
            f"and the strongest historical case has "
            f"{best_similarity:.2f} similarity."
        )
    )


# ============================================================
# DRAFT RESPONSE
# ============================================================

def draft_response(
    message,
    intent,
    evidence,
    decision,
):
    """
    Create a concise support response using the predicted intent
    and the customer's actual message.

    The response avoids inventing order-specific facts.
    """

    text = clean_text(message)

    # --------------------------------------------------------
    # Detect specific scenarios from actual customer message
    # --------------------------------------------------------

    delivered_not_received = (
        any(
            phrase in text
            for phrase in [
                "says it was delivered",
                "says it has been delivered",
                "marked as delivered",
                "shows as delivered",
                "states it has been delivered",
                "delivered but",
                "delivered and i haven't",
                "delivered and i have not",
            ]
        )
        and any(
            phrase in text
            for phrase in [
                "never received",
                "haven't received",
                "have not received",
                "not received",
                "didn't receive",
                "did not receive",
            ]
        )
    )

    missing_item = any(
        phrase in text
        for phrase in [
            "item missing",
            "missing item",
            "item is missing",
            "missing from the box",
            "missing in the box",
            "something missing",
            "part missing",
        ]
    )

    # --------------------------------------------------------
    # ESCALATE
    # --------------------------------------------------------

    if decision == "ESCALATE":

        if delivered_not_received:

            return (
                "I'm sorry that your package shows as delivered "
                "but you have not received it. Please check the "
                "delivery details and the location where the carrier "
                "may have left it. If it still cannot be found, "
                "a support representative should investigate the delivery."
            )

        if missing_item:

            return (
                "I'm sorry that an item is missing from your delivery. "
                "Please check the order and package details. A support "
                "representative should review the order and help resolve "
                "the missing-item issue."
            )

        if intent == "DELIVERY_DELAY":

            return (
                "Sorry that your delivery is delayed. Please check the "
                "latest tracking information and the promised delivery "
                "date. Since the delivery is overdue, a support "
                "representative should review the order and available options."
            )

        if intent == "DELIVERY_PROOF_FAILURE":

            return (
                "I'm sorry that there is a problem with your delivery. "
                "Please check the delivery details and carrier information. "
                "A support representative should investigate the delivery "
                "and help with the next step."
            )

        if intent == "RETURN_REFUND_REPLACEMENT":

            return (
                "Sorry about the trouble with your return or refund. "
                "Please check the current order and refund or replacement "
                "details. A support representative should review the case "
                "and help with the next step."
            )

        if intent == "PAYMENT_BILLING_GIFTCARD":

            return (
                "Sorry about the payment or billing issue. Please check "
                "the transaction and payment details. A support "
                "representative should review the charge and help resolve "
                "the issue."
            )

        if intent == "ACCOUNT_ACCESS_SECURITY":

            return (
                "I'm sorry you're having trouble with your account. "
                "For security reasons, please use the official account "
                "recovery or security process. A support representative "
                "should review the case if access cannot be restored "
                "or you suspect unauthorized activity."
            )

        if intent == "PRIME_MEMBERSHIP":

            return (
                "Sorry about the trouble with your Prime membership "
                "or benefits. Please check your membership details. "
                "A support representative should review the case and "
                "help with the appropriate next step."
            )

        if intent == "DIGITAL_TECHNICAL":

            return (
                "Sorry you're having trouble with the digital service. "
                "Please check the device or application settings and "
                "try the relevant troubleshooting steps. If the issue "
                "continues, a support representative should investigate it."
            )

        if intent == "PRODUCT_SELLER_ISSUE":

            return (
                "Sorry about the problem with the product or seller. "
                "Please check the order and product details. A support "
                "representative should review the case and help with "
                "the appropriate resolution."
            )

        if intent == "ORDER_MANAGEMENT":

            return (
                "Sorry about the order issue. Please check the current "
                "order details and available order-management options. "
                "A support representative should review the order "
                "and help with the appropriate next step."
            )

        if intent == "SUPPORT_ESCALATION":

            return (
                "I'm sorry you've had trouble getting this resolved. "
                "Since previous support interactions have not resolved "
                "the issue, a support representative should review the "
                "case and provide the appropriate next step."
            )

        return (
            "Thanks for reaching out. There isn't enough information "
            "in the message to confidently determine the issue. "
            "A support representative should review the case and assist."
        )

    # --------------------------------------------------------
    # AUTO-HANDLE
    # --------------------------------------------------------

    if delivered_not_received:

        return (
            "Sorry about that. If your package is showing as delivered "
            "but you haven't received it, please check the delivery "
            "details and carrier information first. If it still cannot "
            "be located, support can investigate the delivery further."
        )

    if missing_item:

        return (
            "Sorry about the missing item. Please check the order and "
            "package details and the items listed in your order. If the "
            "item is still missing, support can review the order and "
            "help with the appropriate resolution."
        )

    if intent == "DELIVERY_PROOF_FAILURE":

        return (
            "Sorry about the delivery issue. Please check the delivery "
            "details and carrier information first. If the package or "
            "item still cannot be located, support can investigate further."
        )

    if intent == "DELIVERY_DELAY":

        return (
            "Sorry that your delivery is taking longer than expected. "
            "Please check the latest tracking information for the order. "
            "If the promised delivery date has passed, support can "
            "review the delivery status and available options."
        )

    if intent == "RETURN_REFUND_REPLACEMENT":

        return (
            "Sorry about the trouble with your order. Please check the "
            "order's return or refund options. If the refund or replacement "
            "is already in progress, support can review its current status."
        )

    if intent == "PAYMENT_BILLING_GIFTCARD":

        return (
            "Sorry about the payment issue. Please check the payment "
            "method and transaction details associated with the order. "
            "If the charge still looks incorrect, support can review "
            "the transaction."
        )

    if intent == "ACCOUNT_ACCESS_SECURITY":

        return (
            "For account access or security issues, please use the "
            "official account recovery or security process. If you still "
            "cannot access the account or believe it has been compromised, "
            "support should review it."
        )

    if intent == "PRIME_MEMBERSHIP":

        return (
            "Sorry about the trouble with your Prime service. Please "
            "check your Prime membership and the benefit associated "
            "with the order. If the expected Prime benefit was not "
            "applied, support can review the membership and order details."
        )

    if intent == "DIGITAL_TECHNICAL":

        return (
            "Sorry you're having trouble with the digital service. "
            "Please check the device or application settings and try "
            "the relevant troubleshooting steps. If the problem continues, "
            "support can investigate further."
        )

    if intent == "PRODUCT_SELLER_ISSUE":

        return (
            "Sorry about the issue with the product. Please check the "
            "order and product details to review the available support, "
            "return, or replacement options."
        )

    if intent == "ORDER_MANAGEMENT":

        return (
            "I can help with the order issue. Please check the order "
            "details for the available cancellation, address, or "
            "order-management options."
        )

    if intent == "SUPPORT_ESCALATION":

        return (
            "I'm sorry you've had trouble getting this resolved. "
            "A support representative should review the case and "
            "provide the appropriate next step."
        )

    return (
        "Thanks for reaching out. A support representative should "
        "review the details and help with the issue."
    )


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