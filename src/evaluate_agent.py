from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


GOLDEN_FILE = Path(
    "data/evaluation/golden_set.csv"
)

MESSAGES_FILE = Path(
    "data/processed/amazonhelp_dev_messages.csv"
)

MODEL_FILE = Path(
    "models/intent_classifier.joblib"
)

VECTORIZER_FILE = Path(
    "models/tfidf_vectorizer.joblib"
)

OUTPUT_FILE = Path(
    "data/evaluation/agent_evaluation.csv"
)

REPORT_FILE = Path(
    "data/evaluation/agent_evaluation_report.txt"
)


# ============================================================
# LOAD
# ============================================================

def load_data():

    golden = pd.read_csv(
        GOLDEN_FILE
    )

    messages = pd.read_csv(
        MESSAGES_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    classifier_vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    return (
        golden,
        messages,
        model,
        classifier_vectorizer,
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    return str(text).strip().lower()


# ============================================================
# BUILD SIMPLE RETRIEVAL INDEX
# ============================================================

def build_retrieval_index(messages):

    from sklearn.feature_extraction.text import (
        TfidfVectorizer
    )

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
    ].reset_index(drop=True)

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

    return (
        customer_messages,
        vectorizer,
        matrix,
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_evidence(
    message,
    messages,
    customer_messages,
    vectorizer,
    matrix,
):

    from sklearn.metrics.pairwise import cosine_similarity

    query = clean_text(message)

    if not query:
        return []

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        matrix,
    )[0]

    ranked = similarities.argsort()[::-1]

    results = []

    seen_conversations = set()

    for index in ranked:

        similarity = float(
            similarities[index]
        )

        if similarity < 0.10:
            break

        row = customer_messages.iloc[
            index
        ]

        conversation_id = row[
            "conversation_id"
        ]

        if conversation_id in seen_conversations:
            continue

        conversation = messages[
            messages["conversation_id"]
            == conversation_id
        ]

        support_messages = conversation[
            conversation["inbound"] == False
        ]

        if len(support_messages) == 0:
            continue

        seen_conversations.add(
            conversation_id
        )

        results.append(
            {
                "similarity": similarity,
                "conversation_id": conversation_id,
            }
        )

        if len(results) >= 3:
            break

    return results


# ============================================================
# DECISION LOGIC
# ============================================================

def make_decision(
    intent,
    confidence,
    evidence,
):

    if not evidence:

        return (
            "ESCALATE",
            "No sufficiently similar historical support case was found."
        )

    if intent == "CONTEXT_NEEDED":

        return (
            "ESCALATE",
            "The customer message does not contain enough information to identify the issue."
        )

    if confidence is None:

        return (
            "ESCALATE",
            "The classifier did not provide a confidence score."
        )

    best_similarity = max(
        item["similarity"]
        for item in evidence
    )

    if confidence < 0.50:

        return (
            "ESCALATE",
            "Intent confidence is too low."
        )

    if best_similarity < 0.20:

        return (
            "ESCALATE",
            "Historical evidence is weak."
        )

    if confidence < 0.65 and best_similarity < 0.35:

        return (
            "ESCALATE",
            "Intent confidence and historical evidence are both moderate."
        )

    if intent == "ACCOUNT_ACCESS_SECURITY":

        if confidence < 0.80:

            return (
                "ESCALATE",
                "Account-security issues require higher confidence."
            )

    if intent == "SUPPORT_ESCALATION":

        return (
            "ESCALATE",
            "The customer reports unsuccessful support interactions."
        )

    return (
        "AUTO-HANDLE",
        "Intent confidence and historical evidence passed the automatic-handling thresholds."
    )


# ============================================================
# EVALUATION
# ============================================================

def main():

    print()
    print("=" * 70)
    print("COMPLETE SUPPORT AGENT EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        golden,
        messages,
        model,
        classifier_vectorizer,
    ) = load_data()

    if len(golden) != 200:

        raise ValueError(
            f"Expected 200 golden examples, found {len(golden)}."
        )

    # --------------------------------------------------------
    # Build retrieval index ONCE
    # --------------------------------------------------------

    print()
    print("Building retrieval index...")

    (
        customer_messages,
        retrieval_vectorizer,
        retrieval_matrix,
    ) = build_retrieval_index(
        messages
    )

    print(
        f"Indexed customer messages: "
        f"{len(customer_messages):,}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = []

    print()
    print("Evaluating 200 examples...")

    for number, row in golden.iterrows():

        message = str(
            row["text"]
        )

        # -----------------------------------------------
        # Intent
        # -----------------------------------------------

        X = classifier_vectorizer.transform(
            [message]
        )

        intent = model.predict(
            X
        )[0]

        confidence = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                X
            )[0]

            confidence = float(
                probabilities.max()
            )

        # -----------------------------------------------
        # Evidence
        # -----------------------------------------------

        evidence = retrieve_evidence(
            message=message,
            messages=messages,
            customer_messages=customer_messages,
            vectorizer=retrieval_vectorizer,
            matrix=retrieval_matrix,
        )

        best_similarity = 0.0

        if evidence:

            best_similarity = max(
                item["similarity"]
                for item in evidence
            )

        # -----------------------------------------------
        # Decision
        # -----------------------------------------------

        decision, reason = make_decision(
            intent=intent,
            confidence=confidence,
            evidence=evidence,
        )

        # -----------------------------------------------
        # Save row
        # -----------------------------------------------

        results.append(
            {
                "tweet_id": row["tweet_id"],
                "conversation_id": row["conversation_id"],
                "text": message,

                "human_intent": row[
                    "human_intent"
                ],

                "predicted_intent": intent,

                "classifier_confidence": confidence,

                "is_ambiguous": row[
                    "is_ambiguous"
                ],

                "needs_context": row[
                    "needs_context"
                ],

                "decision": decision,

                "decision_reason": reason,

                "num_evidence_cases": len(
                    evidence
                ),

                "best_similarity": best_similarity,
            }
        )

    result_df = pd.DataFrame(
        results
    )

    # ========================================================
    # METRICS
    # ========================================================

    y_true = result_df[
        "human_intent"
    ]

    y_pred = result_df[
        "predicted_intent"
    ]

    intent_accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    intent_macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Decision metrics
    # --------------------------------------------------------

    context_rows = result_df[
        result_df["needs_context"].astype(str).str.lower()
        == "yes"
    ]

    if len(context_rows) > 0:

        context_escalation_rate = (
            context_rows["decision"]
            .eq("ESCALATE")
            .mean()
        )

    else:

        context_escalation_rate = 0.0

    auto_handle_rate = (
        result_df["decision"]
        .eq("AUTO-HANDLE")
        .mean()
    )

    escalate_rate = (
        result_df["decision"]
        .eq("ESCALATE")
        .mean()
    )

    evidence_rate = (
        result_df["num_evidence_cases"]
        .gt(0)
        .mean()
    )

    average_similarity = (
        result_df["best_similarity"]
        .mean()
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    classification = classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    report = f"""
COMPLETE SUPPORT AGENT EVALUATION
=================================

Golden examples: {len(result_df)}

Intent Accuracy:
{intent_accuracy:.4f}

Intent Macro F1:
{intent_macro_f1:.4f}

Auto-handle rate:
{auto_handle_rate:.4f}

Escalation rate:
{escalate_rate:.4f}

Context-needed escalation rate:
{context_escalation_rate:.4f}

Historical evidence found:
{evidence_rate:.4f}

Average best historical similarity:
{average_similarity:.4f}


CLASSIFICATION REPORT
=====================

{classification}
"""

    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Intent accuracy:              "
        f"{intent_accuracy:.4f}"
    )

    print(
        f"Intent macro F1:              "
        f"{intent_macro_f1:.4f}"
    )

    print(
        f"Auto-handle rate:             "
        f"{auto_handle_rate:.4f}"
    )

    print(
        f"Escalation rate:              "
        f"{escalate_rate:.4f}"
    )

    print(
        f"Context-needed escalation:    "
        f"{context_escalation_rate:.4f}"
    )

    print(
        f"Evidence found:               "
        f"{evidence_rate:.4f}"
    )

    print(
        f"Average best similarity:      "
        f"{average_similarity:.4f}"
    )

    print()
    print("Classification report:")
    print(
        classification
    )

    print()
    print("Files created:")
    print(
        f"  {OUTPUT_FILE}"
    )
    print(
        f"  {REPORT_FILE}"
    )

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()