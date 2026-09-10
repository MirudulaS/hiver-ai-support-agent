from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "amazonhelp_weak_labels.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "intent_classifier.joblib"
VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
SPLIT_FILE = MODEL_DIR / "classifier_split.json"


def load_data():
    df = pd.read_csv(INPUT_FILE)

    df = df[df["inbound"] == True].copy()

    df["text"] = df["text"].fillna("").astype(str)
    df["weak_intent"] = df["weak_intent"].fillna("CONTEXT_NEEDED")

    return df


def split_by_conversation(df):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            groups=df["conversation_id"],
        )
    )

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    return train, test


def train():
    df = load_data()

    print(f"Total customer messages: {len(df):,}")
    print(f"Total conversations: {df['conversation_id'].nunique():,}")

    train_df, test_df = split_by_conversation(df)

    print(f"Training messages: {len(train_df):,}")
    print(f"Test messages: {len(test_df):,}")

    print(
        f"Training conversations: "
        f"{train_df['conversation_id'].nunique():,}"
    )

    print(
        f"Test conversations: "
        f"{test_df['conversation_id'].nunique():,}"
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    y_train = train_df["weak_intent"]
    y_test = test_df["weak_intent"]

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    probabilities = classifier.predict_proba(X_test)

    confidence = probabilities.max(axis=1)

    print("\n=== Overall Results ===")

    print(
        f"Accuracy: "
        f"{accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"Macro F1: "
        f"{f1_score(y_test, predictions, average='macro'):.4f}"
    )

    print(
        f"Weighted F1: "
        f"{f1_score(y_test, predictions, average='weighted'):.4f}"
    )

    print("\n=== Per Intent ===")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("\n=== Confusion Matrix ===")

    labels = classifier.classes_

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    print(matrix_df)

    # Evaluate without CONTEXT_NEEDED
    mask = y_test != "CONTEXT_NEEDED"

    if mask.any():
        print("\n=== Results excluding CONTEXT_NEEDED ===")

        print(
            f"Accuracy: "
            f"{accuracy_score(y_test[mask], predictions[mask]):.4f}"
        )

        print(
            f"Macro F1: "
            f"{f1_score(
                y_test[mask],
                predictions[mask],
                average='macro'
            ):.4f}"
        )

        print(
            f"Weighted F1: "
            f"{f1_score(
                y_test[mask],
                predictions[mask],
                average='weighted'
            ):.4f}"
        )

    # Save model
    joblib.dump(classifier, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    split_info = {
        "random_state": 42,
        "test_size": 0.20,
        "train_messages": len(train_df),
        "test_messages": len(test_df),
        "train_conversations": int(
            train_df["conversation_id"].nunique()
        ),
        "test_conversations": int(
            test_df["conversation_id"].nunique()
        ),
    }

    with open(SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    # Save test predictions for analysis
    results = test_df[
        ["tweet_id", "conversation_id", "text", "weak_intent"]
    ].copy()

    results["predicted_intent"] = predictions
    results["confidence"] = confidence

    results.to_csv(
        BASE_DIR / "data" / "processed" /
        "classifier_test_predictions.csv",
        index=False,
    )

    print("\nModels saved:")
    print(MODEL_FILE)
    print(VECTORIZER_FILE)

    print("\nTest predictions saved.")


if __name__ == "__main__":
    train()