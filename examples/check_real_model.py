"""Run labeled emails through a real System 1 and calibrate it. Configure the model in .env like the API does:

    HUNCH_SYSTEM1_MODEL=gpt-4o-mini
    HUNCH_SYSTEM1_URL=https://api.openai.com/v1
    HUNCH_SYSTEM1_API_KEY=...
    HUNCH_SYSTEM1_EXTRA={"reasoning_effort": "none"}   # optional extra request fields

then `python examples/check_real_model.py`. Shows how confident the model is, whether that confidence is earned, and
the thresholds calibration would pick."""
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from hunch import Router, Store, calibrate
from hunch.config import system1_from_env

QUESTIONS = {
    "needs_reply": {"type": "noul", "instructions": "Does this email need a personal reply?"},
    "category": {"type": "choice", "instructions": "What kind of email is this?",
                 "criteria": {"work": "From a real person about work", "promo": "Marketing or a newsletter", "bill": "A receipt, invoice or payment notice"}},
    "urgency": {"type": "score", "instructions": "How soon does this need attention?", "criteria": ["Never", "This week", "Today"]},
}

# (email, needs_reply, category, urgency)
EMAILS = [
    ("Hi, can you send me the Q3 numbers before our 3pm call today?", True, "work", 2),
    ("50% OFF everything this weekend only! Unsubscribe below.", False, "promo", 0),
    ("Your receipt from Uber: $14.20", False, "bill", 0),
    ("Hey, are you free for coffee next week to talk about the project?", True, "work", 1),
    ("Your invoice #4821 for $2,300 is due Friday.", False, "bill", 1),
    ("This week in tech: 10 stories you missed", False, "promo", 0),
    ("The server is down and customers are complaining. Need you now.", True, "work", 2),
    ("Thanks for your order! It ships in 3-5 days.", False, "bill", 0),
    ("Can you review my PR when you get a chance?", True, "work", 1),
    ("Last chance: your free trial ends tonight", False, "promo", 2),
    ("Payment failed for your subscription. Update your card to avoid cancellation.", False, "bill", 2),
    ("Quick question about the lease agreement, call me when you can", True, "work", 1),
    ("New arrivals just for you", False, "promo", 0),
    ("Reminder: team offsite agenda attached, please confirm attendance by Thursday", True, "work", 1),
    ("Your statement is ready to view", False, "bill", 0),
    ("lol did you see the game last night", True, "work", 0),
    ("Meeting moved to 10am tomorrow, ok?", True, "work", 1),
    ("Congratulations, you've been selected for a free cruise!", False, "promo", 0),
    ("URGENT: wire transfer approval needed within the hour", True, "work", 2),
    ("Welcome to our newsletter! Here is what to expect.", False, "promo", 0),
    ("Can you cover my shift on Saturday? Let me know today if possible.", True, "work", 2),
    ("Your parking ticket of $65 must be paid within 30 days", False, "bill", 1),
    ("FYI, the design doc is updated. No action needed.", False, "work", 0),
    ("What do you think about the new logo options? Would love your feedback.", True, "work", 1),
    ("Flash sale: 24 hours only!", False, "promo", 1),
    ("Rent is due tomorrow, please transfer to the usual account", True, "bill", 2),
    ("Following up on my last email, any update?", True, "work", 1),
    ("Your electricity bill for September: $88.40", False, "bill", 0),
    ("Join us for our webinar on cloud cost savings", False, "promo", 0),
    ("I need the signed contract back by end of day or we lose the deal.", True, "work", 2),
    ("Please find the minutes from yesterday's meeting attached.", False, "work", 0),
    ("Do you still want the extra ticket for Friday?", True, "work", 1),
    ("Your package was delivered to the front desk", False, "bill", 0),
    ("We miss you! Come back for 20% off.", False, "promo", 0),
    ("Can you confirm the budget numbers by Wednesday? Finance is waiting.", True, "work", 1),
    ("Security alert: new sign-in from an unknown device", False, "work", 2),
    ("Happy birthday! Hope you have a great one", True, "work", 0),
    ("Invoice overdue by 30 days. Final notice before collections.", False, "bill", 2),
    ("Can we push the deadline to next month? Let me know.", True, "work", 1),
    ("Top 5 productivity apps you need in 2026", False, "promo", 0),
]


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    store = Store(Path(tempfile.mkdtemp()) / "check.db")
    router = Router(system1_from_env(), log=store)
    for text, *truth in EMAILS:
        result = router.decide(text, QUESTIONS)
        for key, answer in zip(QUESTIONS, truth):
            store.label(result["id"], key, answer)
    print(f"{os.environ['HUNCH_SYSTEM1_MODEL']}: {len(EMAILS)} emails graded against your labels\n")
    rows = store.labeled()
    for key, report in calibrate(store, target_accuracy=0.9, min_samples=20, save=False).items():
        confidences = sorted(r["confidence"] for r in rows if r["key"] == key)
        wrong = [r for r in rows if r["key"] == key and r["answer"] != r["truth"]]
        print(f"{key}: accuracy {report['accuracy']:.0%}, median confidence {confidences[len(confidences) // 2]:.2f}, "
              f"wrong answers had confidence {[round(r['confidence'], 2) for r in wrong]}")
        print(f"   calibrated threshold for 90%: {report['threshold']}, escalates {report['escalation_rate']}")


if __name__ == "__main__":
    main()
