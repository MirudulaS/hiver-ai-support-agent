# Decision Log

## 1. Chose a small intent taxonomy
**Decision:** Use 11 support intents instead of trying to reproduce every fine-grained topic in the dataset.

**Why:** The assignment asks for a small intent set. A compact taxonomy makes classification and evaluation easier to interpret.

---

## 2. Added CONTEXT_NEEDED
**Decision:** Include `CONTEXT_NEEDED` as an explicit intent.

**Why:** Many Twitter messages are extremely short or depend on previous conversation turns. Forcing these into a concrete intent would create false confidence.

---

## 3. Reconstructed conversations from reply IDs
**Decision:** Build conversation components using `in_response_to_tweet_id` and `response_tweet_id`.

**Why:** Twitter support interactions are multi-turn. Treating every tweet independently would lose important context.

---

## 4. Split by conversation, not individual tweets
**Decision:** Keep complete conversations within either train or holdout.

**Why:** Randomly splitting tweets could leak nearly identical conversation context into both training and evaluation and produce an unrealistically high score.

---

## 5. Used a chronological message order
**Decision:** Order messages using `created_at` within reconstructed conversations.

**Why:** The support agent should only use information that would have been available at the time of the customer message.

---

## 6. Used weak labels for initial training
**Decision:** Generate initial intent labels using transparent keyword/rule-based labeling.

**Why:** The raw dataset does not contain a clean intent field. Weak labeling provides a reproducible starting point without manually labeling thousands of messages.

---

## 7. Used TF-IDF + Logistic Regression as the learned baseline
**Decision:** Use TF-IDF features with Logistic Regression.

**Why:** It is fast, interpretable, inexpensive to run, and provides a strong classical baseline for short customer-support messages.

---

## 8. Added rule-based intent overrides
**Decision:** Use explicit high-signal rules before the learned classifier for several intents.

**Why:** Certain phrases such as account hacking, previous unresolved support contacts, or explicit refund requests provide stronger evidence than generic lexical similarity.

---

## 9. Separate intent classification from escalation
**Decision:** Do not make the predicted intent alone determine whether a message is auto-handled.

**Why:** A message can have a clear intent but still be unsafe or inappropriate to answer automatically.

---

## 10. Escalate low-confidence cases
**Decision:** Escalate when confidence or retrieval similarity is low.

**Why:** The system should prefer a human handoff over an unsupported answer when evidence is weak.

---

## 11. Always escalate product/seller issues
**Decision:** Product/seller issues are currently routed to human support.

**Why:** These cases can require order-specific investigation, replacement decisions, or evidence that is not available in the public conversation data.

---

## 12. Retrieval is used as evidence, not as proof of resolution
**Decision:** Historical support replies are treated as examples of how the brand responded, not proof that the original issue was successfully resolved.

**Why:** The dataset does not contain a reliable explicit resolution/success field.

---

## 13. Evaluate on a held-out conversation set
**Decision:** Use 200 customer-root examples from conversations excluded from classifier training.

**Why:** This gives a clean evaluation boundary and avoids evaluating on the same conversations used for weak-label training.

---

## 14. Added human reply evaluation
**Decision:** Manually review 30 generated replies for quality, grounding, and helpfulness.

**Why:** Automated intent accuracy does not tell us whether the actual customer-facing response is useful.

---

## 15. LLM-as-judge is implemented but not executed
**Decision:** Include an LLM-as-judge evaluator without fabricating its results.

**Why:** The OpenAI API account had no available API credits. Reporting invented judge scores would make the evaluation invalid, so the limitation is documented instead.