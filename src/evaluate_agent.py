from pathlib import Path
import pandas as pd
import joblib

from support_agent import (
    classify_intent,
    make_decision,
    draft_response,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


# ============================================================
# FILES
# ============================================================

GOLDEN_FILE = Path(
    "data/evaluation/golden_set_clean.csv"
)

MESSAGES_FILE = Path(
    "data/processed/amazonhelp_train_messages.csv"
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
# BUILD RETRIEVAL INDEX
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
                "tweet_id": row["tweet_id"],
                "conversation_id": conversation_id,
                "similarity": similarity,
                "customer_message": str(
                    row["text"]
                ),
                "support_replies": (
                    support_messages["text"]
                    .fillna("")
                    .astype(str)
                    .tolist()
                ),
            }
        )

        if len(results) >= 3:
            break

    return results


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

        intent, confidence = classify_intent(
            message=message,
            model=model,
            vectorizer=classifier_vectorizer,
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
            classifier_confidence=confidence,
            evidence=evidence,
        )

        # -----------------------------------------------
        # Draft response
        # -----------------------------------------------

        response = draft_response(
            message=message,
            intent=intent,
            evidence=evidence,
            decision=decision,
        )

        # -----------------------------------------------
        # Save result
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

                "confidence": confidence,

                "decision": decision,

                "decision_reason": reason,

                "best_similarity": best_similarity,

                "evidence_found": bool(
                    evidence
                ),

                "agent_reply": response,
            }
        )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    y_true = results_df[
        "human_intent"
    ]

    y_pred = results_df[
        "predicted_intent"
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Decision metrics
    # --------------------------------------------------------

    auto_handle_rate = (
        results_df["decision"]
        == "AUTO-HANDLE"
    ).mean()

    escalation_rate = (
        results_df["decision"]
        == "ESCALATE"
    ).mean()

    context_mask = (
        results_df["human_intent"]
        == "CONTEXT_NEEDED"
    )

    context_escalation = 0.0

    if context_mask.sum() > 0:

        context_escalation = (
            results_df.loc[
                context_mask,
                "decision"
            ]
            == "ESCALATE"
        ).mean()

    evidence_found_rate = (
        results_df["evidence_found"]
        .mean()
    )

    average_similarity = (
        results_df["best_similarity"]
        .mean()
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Intent accuracy:              "
        f"{accuracy:.4f}"
    )

    print(
        f"Intent macro F1:              "
        f"{macro_f1:.4f}"
    )

    print(
        f"Auto-handle rate:             "
        f"{auto_handle_rate:.4f}"
    )

    print(
        f"Escalation rate:              "
        f"{escalation_rate:.4f}"
    )

    print(
        f"Context-needed escalation:    "
        f"{context_escalation:.4f}"
    )

    print(
        f"Evidence found:               "
        f"{evidence_found_rate:.4f}"
    )

    print(
        f"Average best similarity:      "
        f"{average_similarity:.4f}"
    )

    print()
    print("Classification report:")
    print(report)

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    report_text = (
        "COMPLETE SUPPORT AGENT EVALUATION\n"
        "=================================\n\n"
        f"Examples: {len(results_df)}\n"
        f"Intent accuracy: {accuracy:.4f}\n"
        f"Intent macro F1: {macro_f1:.4f}\n"
        f"Auto-handle rate: {auto_handle_rate:.4f}\n"
        f"Escalation rate: {escalation_rate:.4f}\n"
        f"Context-needed escalation: {context_escalation:.4f}\n"
        f"Evidence found: {evidence_found_rate:.4f}\n"
        f"Average best similarity: {average_similarity:.4f}\n\n"
        "Classification report:\n"
        f"{report}\n"
    )

    REPORT_FILE.write_text(
        report_text,
        encoding="utf-8",
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