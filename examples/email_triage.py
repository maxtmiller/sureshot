"""Triage an IMAP inbox with sureshot, the personal case: flag mail that needs a reply today and move promotions out
of the inbox. Configure the models as for the API, plus the mailbox:

    IMAP_HOST=imap.gmail.com IMAP_USER=you@gmail.com IMAP_PASSWORD=<an app password> \\
    SURESHOT_SYSTEM1_MODEL=gpt-4o-mini SURESHOT_SYSTEM1_PROVIDER=openai OPENAI_API_KEY=... \\
    python examples/email_triage.py --dry-run

Drop --dry-run once its choices look right, and run it from cron every 15 minutes. It reads unseen mail without
marking it read, asks the email_triage questions in examples/questions.yaml, and logs every decision to triage.db
(SURESHOT_DB), so `sureshot review --db triage.db` lists what to label and `sureshot calibrate --db triage.db` learns
thresholds from your labels. PROMOTIONS_FOLDER (default "Promotions") must exist; on Gmail it is a label."""
import email
import imaplib
import os
import re
import sys
from email.policy import default
from pathlib import Path
from sureshot import router_from_env

os.environ.setdefault("SURESHOT_QUESTIONS", str(Path(__file__).with_name("questions.yaml")))
os.environ.setdefault("SURESHOT_DB", "triage.db")


def text_of(raw):
    message = email.message_from_bytes(raw, policy=default)
    body = message.get_body(preferencelist=("plain", "html"))
    content = body.get_content() if body else ""
    if body and body.get_content_type() == "text/html":
        content = re.sub(r"<[^>]+>", " ", content)
    content = re.sub(r"\s+", " ", content).strip()[:4000]
    return f"From: {message['from']}\nSubject: {message['subject']}\n\n{content}"


def main(dry_run):
    router = router_from_env()
    imap = imaplib.IMAP4_SSL(os.environ["IMAP_HOST"])
    imap.login(os.environ["IMAP_USER"], os.environ["IMAP_PASSWORD"])
    imap.select("INBOX")
    for uid in imap.uid("SEARCH", None, "UNSEEN")[1][0].split():
        text = text_of(imap.uid("FETCH", uid, "(BODY.PEEK[])")[1][0][1])
        answers = {key: a["answer"] for key, a in router.decide(text, "email_triage")["answers"].items()}
        flag = answers["needs_reply"] and answers["urgency"] == 2
        move = answers["category"] == "promotion"
        print(f"{text.splitlines()[1][:70]:70}  {answers}  {'FLAG' if flag else ''}{'MOVE' if move else ''}")
        if dry_run:
            continue
        if flag:
            imap.uid("STORE", uid, "+FLAGS", "(\\Flagged)")
        if move:
            imap.uid("MOVE", uid, os.environ.get("PROMOTIONS_FOLDER", "Promotions"))
    imap.logout()


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
