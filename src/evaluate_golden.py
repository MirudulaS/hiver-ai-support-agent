from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


GOLDEN_FILE = Path("data/evaluation/golden_set.csv")
MODEL_FILE = Path("models/intent_classifier.joblib")
VECTORIZER_FILE = Path("models/tfidf_vectorizer.joblib")

OUTPUT_FILE = Path("data/evaluation/golden_predictions.csv")
REPORT_FILE = Path("data/evaluation/golden_evaluation_report.txt")
CONFUSION_FILE = Path("data/evaluation/golden_confusion_matrix.csv")


def main():
    # ---------------------------------------------------------
    # 1. Check files
    # ---------------------------------------------------------

    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(
            f"Golden set not found: {GOLDEN_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    if not VECTORIZER_FILE.exists():
        raise FileNotFoundError(
            f"Vectorizer not found: {VECTORIZER_FILE}"
        )

    # ---------------------------------------------------------
    # 2. Load data and model
    # ---------------------------------------------------------

    df = pd.read_csv(GOLDEN_FILE)

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    # ---------------------------------------------------------
    # 3. Basic validation
    # ---------------------------------------------------------

    if len(df) != 200:
        raise ValueError(
            f"Expected 200 golden examples, found {len(df)}."
        )

    required_columns = [
        "tweet_id",
        "text",
        "human_intent",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Remove rows without human labels
    df = df[df["human_intent"].notna()].copy()

    if len(df) != 200:
        raise ValueError(
            f"Expected 200 human-labelled examples, found {len(df)}."
        )

    # ---------------------------------------------------------
    # 4. Prepare text
    # ---------------------------------------------------------

    texts = df["text"].fillna("").astype(str)

    X = vectorizer.transform(texts)

    # ---------------------------------------------------------
    # 5. Predict
    # ---------------------------------------------------------

    predictions = model.predict(X)

    df["predicted_intent"] = predictions

    # ---------------------------------------------------------
    # 6. Calculate metrics
    # ---------------------------------------------------------

    y_true = df["human_intent"]
    y_pred = df["predicted_intent"]

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # ---------------------------------------------------------
    # 7. Classification report
    # ---------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )

    # ---------------------------------------------------------
    # 8. Confusion matrix
    # ---------------------------------------------------------

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    confusion_df.index.name = "actual"
    confusion_df.columns.name = "predicted"

    # ---------------------------------------------------------
    # 9. Save predictions
    # ---------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # 10. Save confusion matrix
    # ---------------------------------------------------------

    confusion_df.to_csv(
        CONFUSION_FILE
    )

    # ---------------------------------------------------------
    # 11. Save report
    # ---------------------------------------------------------

    report_text = f"""
GOLDEN EVALUATION
=================

Number of examples: {len(df)}

Accuracy: {accuracy:.4f}
Macro F1: {macro_f1:.4f}
Weighted F1: {weighted_f1:.4f}

Classification Report
---------------------

{report}

Confusion Matrix
----------------

{confusion_df.to_string()}
"""

    REPORT_FILE.write_text(
        report_text,
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 12. Print results
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("GOLDEN SET EVALUATION")
    print("=" * 60)

    print(f"Examples:     {len(df)}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")

    print()
    print("Classification Report")
    print("----------------------")
    print(report)

    print()
    print("Files created:")
    print(f"  Predictions:      {OUTPUT_FILE}")
    print(f"  Report:           {REPORT_FILE}")
    print(f"  Confusion matrix: {CONFUSION_FILE}")

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()