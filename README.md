# AI Support Agent for Customer Conversations

An AI support agent built from the Kaggle Customer Support on Twitter dataset.

The system:
1. classifies an incoming customer message into a compact support-intent taxonomy,
2. retrieves historically similar customer/support interactions,
3. drafts a grounded response,
4. decides whether the message can be auto-handled or should be escalated.

## 1. Dataset

Dataset: Customer Support on Twitter by ThoughtVector.

The raw dataset contains approximately 2.8M tweets across multiple brands.

For development, the project focuses on Amazon customer-support conversations (`AmazonHelp`).

The raw dataset is intentionally excluded from Git because of its size.

Expected location:

`data/raw/twcs/twcs.csv`

## 2. Intent taxonomy

The system uses 11 intents:

- `DELIVERY_DELAY`
- `DELIVERY_PROOF_FAILURE`
- `ORDER_MANAGEMENT`
- `RETURN_REFUND_REPLACEMENT`
- `PAYMENT_BILLING_GIFTCARD`
- `ACCOUNT_ACCESS_SECURITY`
- `PRIME_MEMBERSHIP`
- `PRODUCT_SELLER_ISSUE`
- `DIGITAL_TECHNICAL`
- `SUPPORT_ESCALATION`
- `CONTEXT_NEEDED`

`CONTEXT_NEEDED` is used when the message is too short, ambiguous, multilingual, or dependent on missing conversation context.

## 3. Pipeline

```text
Raw Twitter conversations
          |
          v
Conversation reconstruction
          |
          v
Intent discovery / weak labeling
          |
          v
TF-IDF + Logistic Regression
          |
          v
Intent classification
          |
          v
Historical evidence retrieval
          |
          v
Response drafting
          |
          v
Auto-handle / Escalate decision