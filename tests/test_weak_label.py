import unittest

from src.weak_label import classify_self_contained, label_rows


class WeakLabelTests(unittest.TestCase):
    def test_security_outranks_payment(self):
        label, confidence, _ = classify_self_contained("My account was hacked and charged")
        self.assertEqual((label, confidence), ("ACCOUNT_ACCESS_SECURITY", "high"))

    def test_return_outranks_product_condition(self):
        label, confidence, _ = classify_self_contained("The damaged item needs a refund")
        self.assertEqual((label, confidence), ("RETURN_REFUND_REPLACEMENT", "high"))

    def test_product_outranks_delivery_after_arrival(self):
        label, _, _ = classify_self_contained("My delivered item is counterfeit")
        self.assertEqual(label, "PRODUCT_SELLER_ISSUE")

    def test_pay_via_card_is_payment_not_product_availability(self):
        label, _, _ = classify_self_contained("There was no option to pay via card")
        self.assertEqual(label, "PAYMENT_BILLING_GIFTCARD")

    def test_generic_tracking_site_is_not_automatically_technical(self):
        label, _, _ = classify_self_contained("Where is the tracking site for my order?")
        self.assertNotEqual(label, "DIGITAL_TECHNICAL")

    def test_short_followup_inherits_prior_customer_issue(self):
        rows = [
            {"tweet_id": "1", "conversation_id": "c", "created_at": "a", "text": "My package is late", "inbound": "True"},
            {"tweet_id": "2", "conversation_id": "c", "created_at": "b", "text": "Done", "inbound": "True"},
        ]
        labelled = label_rows(rows)
        self.assertEqual(labelled[1]["weak_intent"], "DELIVERY_DELAY")
        self.assertEqual(labelled[1]["weak_label_confidence"], "low")


if __name__ == "__main__":
    unittest.main()
