import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI


INPUT_FILE = Path("data/evaluation/human_reply_review.csv")
OUTPUT_FILE = Path("data/evaluation/llm_judge_results.csv")
SUMMARY_FILE = Path("data/evaluation/reply_quality_report.json")

MODEL = "gpt-5-mini"


def judge_reply(client, customer_message, agent_reply):
    prompt = f"""
You are evaluating an AI customer-support reply.

Customer message:
{customer_message}

AI support reply:
{agent_reply}

Score the AI reply from 1 to 5 on:

1. quality:
Does the response appropriately address the customer's issue?

2. grounding:
Does the response avoid inventing facts and stay grounded in the information available?

3. helpfulness:
Would this response be useful to the customer as a next step?

Scoring:
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Return ONLY valid JSON:
{{
  "quality": <integer 1-5>,
  "grounding": <integer 1-5>,
  "helpfulness": <integer 1-5>
}}
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    text = response.output_text.strip()
    return json.loads(text)


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in PowerShell first."
        )

    df = pd.read_csv(INPUT_FILE)

    client = OpenAI()

    results = []

    for i, row in df.iterrows():
        print(f"Judging {i + 1}/{len(df)}...")

        scores = judge_reply(
            client,
            str(row["text"]),
            str(row["agent_reply"]),
        )

        results.append(
            {
                "tweet_id": row["tweet_id"],
                "human_quality": row["human_reply_quality"],
                "human_grounding": row["human_grounding"],
                "human_helpfulness": row["human_helpfulness"],
                "llm_quality": scores["quality"],
                "llm_grounding": scores["grounding"],
                "llm_helpfulness": scores["helpfulness"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    metrics = {}

    for dimension in ["quality", "grounding", "helpfulness"]:
        human = results_df[f"human_{dimension}"]
        llm = results_df[f"llm_{dimension}"]

        exact_agreement = (human == llm).mean()
        within_one = ((human - llm).abs() <= 1).mean()
        correlation = human.corr(llm)

        metrics[dimension] = {
            "human_mean": round(float(human.mean()), 2),
            "llm_mean": round(float(llm.mean()), 2),
            "exact_agreement": round(float(exact_agreement), 3),
            "within_one_agreement": round(float(within_one), 3),
            "pearson_correlation": round(float(correlation), 3),
        }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("LLM-AS-JUDGE RESULTS")
    print("=" * 50)

    for dimension, values in metrics.items():
        print(f"\n{dimension.upper()}")
        print(f"Human mean: {values['human_mean']}/5")
        print(f"LLM mean: {values['llm_mean']}/5")
        print(f"Exact agreement: {values['exact_agreement']:.1%}")
        print(f"Within 1 point: {values['within_one_agreement']:.1%}")
        print(f"Pearson correlation: {values['pearson_correlation']:.3f}")

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()