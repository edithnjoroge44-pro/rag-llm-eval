"""Synthetic knowledge corpus + query/answer pairs for the RAG eval project."""
import random

DOCS = [
    {"id": "d1", "title": "Refund policy", "text": "Customers may request a full refund within 30 days of purchase provided the item is unused and in its original packaging. Refunds are processed to the original payment method within 5-7 business days."},
    {"id": "d2", "title": "Shipping times", "text": "Standard shipping takes 3-5 business days within the country and 7-14 business days internationally. Express shipping is available for orders placed before 2pm."},
    {"id": "d3", "title": "Password reset", "text": "To reset a password, click 'Forgot password' on the login screen. A reset link is emailed within 2 minutes and expires after 30 minutes. Contact support if the email does not arrive."},
    {"id": "d4", "title": "Account deletion", "text": "Users may delete their account from Settings -> Privacy. Deletion is irreversible and removes all personal data within 24 hours."},
    {"id": "d5", "title": "Subscription billing", "text": "Subscriptions renew automatically monthly or annually. You can cancel any time and keep access until the end of the current billing period. Refunds for unused periods are prorated."},
    {"id": "d6", "title": "Return eligibility", "text": "Items eligible for return include clothing, electronics and accessories. Perishable goods, opened software and custom items are non-returnable."},
]

# query -> relevant doc id(s) for evaluation
QUERIES = [
    {"q": "How long do refunds take to process?", "relevant": ["d1"]},
    {"q": "What are international shipping times?", "relevant": ["d2"]},
    {"q": "How do I reset my password?", "relevant": ["d3"]},
    {"q": "Can I delete my account and is it permanent?", "relevant": ["d4"]},
    {"q": "When does my subscription renew?", "relevant": ["d5"]},
    {"q": "Are electronics returnable?", "relevant": ["d6"]},
    {"q": "What is the refund window?", "relevant": ["d1"]},
    {"q": "How fast is express shipping?", "relevant": ["d2"]},
]


def get_corpus():
    return DOCS


def get_queries():
    return QUERIES
