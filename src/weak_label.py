"""Deterministically weak-label inbound AmazonHelp messages from processed data.

No model, LLM, embedding, or raw TWCS file is used. Rules mirror the observed
AmazonHelp language in the intent-discovery report and retain an audit reason.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


TAXONOMY = {
    "ACCOUNT_ACCESS_SECURITY", "PAYMENT_BILLING_GIFTCARD", "RETURN_REFUND_REPLACEMENT",
    "DELIVERY_PROOF_FAILURE", "DELIVERY_DELAY", "ORDER_MANAGEMENT", "PRIME_MEMBERSHIP",
    "DIGITAL_TECHNICAL", "PRODUCT_SELLER_ISSUE", "SUPPORT_ESCALATION", "CONTEXT_NEEDED",
}
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
HANDLE_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Analysis-only normalization; the original text is written unchanged."""
    text = URL_RE.sub(" ", text.lower())
    text = HANDLE_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def matches(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def short_or_acknowledgement(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    acknowledgement = {
        "done", "thanks", "thank you", "thx", "ok", "okay", "yes", "no",
        "already did", "sent it", "dm sent", "i did", "will do", "got it",
    }
    return len(tokens) <= 3 or text.strip().lower() in acknowledgement


def classify_self_contained(text: str) -> tuple[str, str, str]:
    """Return (intent, confidence, explanation) without conversation inheritance.

    Explicit security/remedy/payment requests take priority. A deliberate
    exception to the supplied default order is that product-defect language
    outranks delivery wording when delivery succeeded (e.g. "delivered fake").
    Prime-delivery complaints likewise route to delivery rather than membership.
    """
    value = normalize(text)
    if not value:
        return "CONTEXT_NEEDED", "low", "empty_after_analysis_normalization"

    if matches(value, r"\b(hacked|hack|password|log ?in|sign ?in|unauthori[sz]ed|account (?:locked|access|security)|security|verification)\b"):
        return "ACCOUNT_ACCESS_SECURITY", "high", "explicit_account_access_or_security_signal"
    if matches(value, r"\b(gift ?card|giftcard|credit card|debit card|charged|charge|payment|billing|voucher|balance)\b|\bpay (?:via|by|with)\b|\bno option to pay\b"):
        return "PAYMENT_BILLING_GIFTCARD", "high", "explicit_payment_billing_or_giftcard_signal"
    if matches(value, r"\b(return|refund|money back|replacement|replace|exchange)\b"):
        return "RETURN_REFUND_REPLACEMENT", "high", "explicit_return_refund_or_replacement_signal"

    # Product complaint wins over proof-of-delivery only when the object arrived.
    if matches(value, r"\b(damaged|defective|faulty|fake|counterfeit|wrong item|wrong product|seller|quality|price drop|out of stock|in stock|unavailable)\b|\b(?:item|product|title|stock)\b.{0,30}\bavailable\b"):
        return "PRODUCT_SELLER_ISSUE", "high", "explicit_product_condition_seller_or_availability_signal"
    if matches(value, r"\b(says|said|marked|shows|showed) (?:it )?(?:was |has been )?delivered\b|\b(delivered|delivery)\b.*\b(not|never|haven.t|didn.t|missing)\b|\b(left|threw) (?:it |the package|package)\b|\bdelivery (?:driver|person).*(?:not|didn.t|never)\b"):
        return "DELIVERY_PROOF_FAILURE", "high", "delivery_proof_nonreceipt_or_driver_failure_signal"
    if matches(value, r"\b(delivery|deliver|package|parcel|shipment|shipping|courier|arriv|late|delay|still waiting|haven.t received|not received|not arrived)\b"):
        return "DELIVERY_DELAY", "medium", "delivery_eta_delay_or_nonarrival_signal"
    if matches(value, r"\b(cancel|cancelled|cancellation|track|tracking|order status|change (?:my )?(?:address|order)|order (?:number|id|page)|pre[ -]?order)\b"):
        return "ORDER_MANAGEMENT", "high", "explicit_tracking_cancellation_address_or_preorder_signal"

    # Named devices/services are technical; generic site/app/link needs an error
    # cue so that a delivery-tracking link is not mislabeled as a tech issue.
    if matches(value, r"\b(kindle|alexa|echo|fire ?(?:tv|stick)|prime video|amazon video|audible|amazon music)\b") or matches(value, r"\b(app|website|site|webpage|browser|link)\b.{0,40}\b(error|broken|not working|doesn.t work|can.t|unable|won.t|problem)\b"):
        return "DIGITAL_TECHNICAL", "medium", "named_device_digital_service_or_site_signal"
    # Prime is membership only if shipment language and Prime Video did not match above.
    if matches(value, r"\bprime\b"):
        return "PRIME_MEMBERSHIP", "medium", "prime_membership_benefit_or_subscription_signal"
    if matches(value, r"\b(customer (?:service|care|support)|no response|no reply|call(?:ed|ing| back)?|representative|agent|speak to|contact)\b"):
        return "SUPPORT_ESCALATION", "medium", "explicit_contact_or_prior_support_failure_signal"
    return "CONTEXT_NEEDED", "low", "no_precise_observed_intent_signal"


def label_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Label in chronological component order and cautiously inherit prior context."""
    by_conversation: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_conversation[row["conversation_id"]].append(row)

    output: list[dict[str, str]] = []
    for conversation_id in sorted(by_conversation):
        prior_intent: tuple[str, str] | None = None  # (tweet_id, intent)
        ordered = sorted(
            by_conversation[conversation_id],
            key=lambda item: (item.get("created_at_utc", item["created_at"]), int(item["tweet_id"])),
        )
        for row in ordered:
            intent, confidence, reason = classify_self_contained(row["text"])
            normalized = normalize(row["text"])
            if intent == "CONTEXT_NEEDED" and short_or_acknowledgement(normalized) and prior_intent:
                prior_tweet_id, prior_label = prior_intent
                intent = prior_label
                confidence = "low"
                reason = f"context_inherited_from_prior_customer_tweet:{prior_tweet_id}"
            # Only explicit, self-contained issue statements become inherited context.
            elif intent != "CONTEXT_NEEDED" and confidence in {"high", "medium"}:
                prior_intent = (row["tweet_id"], intent)
            row = dict(row)
            row["weak_intent"] = intent
            row["weak_label_confidence"] = confidence
            row["weak_label_reason"] = reason
            output.append(row)
    return output


def run(input_path: Path, output_path: Path, overwrite: bool) -> dict[str, Counter[str]]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite to replace it.")
    with input_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        source_fields = reader.fieldnames or []
        inbound_rows = [dict(row) for row in reader if row["inbound"] == "True"]
    labelled = label_rows(inbound_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fields = [*source_fields, "weak_intent", "weak_label_confidence", "weak_label_reason"]
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(labelled)

    intents = Counter(row["weak_intent"] for row in labelled)
    confidence = Counter(row["weak_label_confidence"] for row in labelled)
    if len(labelled) != len(inbound_rows):
        raise AssertionError("output row count differs from inbound source count")
    if set(intents) - TAXONOMY:
        raise AssertionError("output contains an intent outside the approved taxonomy")
    if any(not row["weak_label_reason"] for row in labelled):
        raise AssertionError("an audit reason is missing")
    return {"intents": intents, "confidence": confidence, "rows": Counter({"inbound_rows": len(labelled)})}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/amazonhelp_dev_messages.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/amazonhelp_weak_labels.csv"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run(args.input, args.output, args.overwrite)
    print("weak intent counts:", dict(result["intents"]))
    print("confidence counts:", dict(result["confidence"]))
    print("sanity check: PASS", dict(result["rows"]))
