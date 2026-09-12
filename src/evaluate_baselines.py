from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


GOLDEN_FILE = Path("data/evaluation/golden_set_clean.csv")
OUTPUT_FILE = Path("data/evaluation/baseline_results.csv")


def main():
    df = pd.read_csv(GOLDEN_FILE)

    y_true = df["human_intent"]

    # ---------------------------------------------------------
    # Baseline 1: Majority class
    # ---------------------------------------------------------
    majority_class = y_true.mode()[0]
    majority_pred = [majority_class] * len(df)

    majority_accuracy = accuracy_score(y_true, majority_pred)
    majority_macro_f1 = f1_score(
        y_true,
        majority_pred,
        average="macro",
        zero_division=0,
    )

    # ---------------------------------------------------------
    # Baseline 2: Existing weak-label rules
    # ---------------------------------------------------------
    weak_pred = df["weak_intent"]

    weak_accuracy = accuracy_score(y_true, weak_pred)
    weak_macro_f1 = f1_score(
        y_true,
        weak_pred,
        average="macro",
        zero_division=0,
    )

    # ---------------------------------------------------------
    # Our classifier
    # ---------------------------------------------------------
    agent_results = pd.read_csv(
        "data/evaluation/agent_evaluation.csv"
    )

    agent_accuracy = accuracy_score(
        agent_results["human_intent"],
        agent_results["predicted_intent"],
    )

    agent_macro_f1 = f1_score(
        agent_results["human_intent"],
        agent_results["predicted_intent"],
        average="macro",
        zero_division=0,
    )

    results = pd.DataFrame([
        {
            "system": "Majority baseline",
            "accuracy": majority_accuracy,
            "macro_f1": majority_macro_f1,
        },
        {
            "system": "Weak-label rules",
            "accuracy": weak_accuracy,
            "macro_f1": weak_macro_f1,
        },
        {
            "system": "TF-IDF + Logistic Regression",
            "accuracy": agent_accuracy,
            "macro_f1": agent_macro_f1,
        },
    ])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)
    print(results.to_string(index=False))

    print("\nMajority baseline classification report:")
    print(
        classification_report(
            y_true,
            majority_pred,
            zero_division=0,
        )
    )

    print("\nWeak-label baseline classification report:")
    print(
        classification_report(
            y_true,
            weak_pred,
            zero_division=0,
        )
    )

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()