"""Evaluate the policy assistant on eval/questions.csv.

Cheap by design: no LLM judge. Retrieval is checked against document/page
metadata (embedding calls only), and answers with keyword checks.

Usage: set GOOGLE_API_KEY, then `python eval/eval.py`.
Writes eval/results.csv and prints a summary. `--regrade` re-scores the saved
answers in results.csv without calling the model.
"""

import csv
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app import respond, retriever  # noqa: E402  (loads the index; no model calls)

# Matches the refusal the prompts ask for, not incidental phrases like "not covered"
REFUSAL = re.compile(r"could not be found|not found in|no information (?:is |was )?(?:available|found|provided)|not mentioned in", re.I)


def norm(text):
    return text.lower().replace(",", "").replace("_", " ")


def contains_all(answer, spec):
    """spec: groups separated by ';', alternatives within a group by '|'."""
    return all(any(alt in answer for alt in group.split("|")) for group in spec.split(";"))


def cites(answer, doc, page):
    return doc.lower() in answer and re.search(rf"(?:pages?|p\.?)\s*:?\s*(?:\d+\s*(?:-|–|and|,)\s*)*{page}\b", answer) is not None


REGRADE = "--regrade" in sys.argv
rows = list(csv.DictReader(open(os.path.join(ROOT, "eval", "questions.csv"), encoding="utf-8")))
if REGRADE:
    saved = {r["id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "eval", "results.csv"), encoding="utf-8"))}
results = []
for row in rows:
    in_scope = row["category"] != "refusal"
    answer = saved[row["id"]]["answer"] if REGRADE else respond(row["question"])
    a = norm(answer)
    refused = bool(REFUSAL.search(a))
    r = {"id": row["id"], "category": row["category"], "question": row["question"]}
    if in_scope:
        if REGRADE:
            r["retrieval_hit"] = saved[row["id"]]["retrieval_hit"] == "True"
        else:
            docs = retriever.invoke(row["question"])
            r["retrieval_hit"] = any(
                row["expected_doc"].lower() in d.metadata["document_name"].lower()
                and d.metadata.get("page", 0) + 1 == int(row["expected_page"])
                for d in docs
            )
        r["correct"] = contains_all(a, row["must_contain"]) and not refused
        r["cited"] = cites(a, row["expected_doc"], row["expected_page"])
        r["refused_wrongly"] = refused
    else:
        r["correct"] = refused
    r["answer"] = " ".join(answer.split())
    results.append(r)
    print(f"[{row['id']}] {'PASS' if r['correct'] else 'FAIL'}  {row['question']}", flush=True)
    if not REGRADE:
        time.sleep(1)

fields = ["id", "category", "question", "retrieval_hit", "correct", "cited", "refused_wrongly", "answer"]
with open(os.path.join(ROOT, "eval", "results.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(results)

ins = [r for r in results if r["category"] != "refusal"]
out = [r for r in results if r["category"] == "refusal"]
pct = lambda xs, k: f"{sum(r[k] for r in xs)}/{len(xs)} ({100 * sum(r[k] for r in xs) / len(xs):.0f}%)"
print("\nIn-scope questions:", len(ins))
print("Retrieval hit@5:   ", pct(ins, "retrieval_hit"))
print("Answer accuracy:   ", pct(ins, "correct"))
print("Citation accuracy: ", pct(ins, "cited"))
print("False refusals:    ", sum(r["refused_wrongly"] for r in ins))
print("Out-of-scope refused:", pct(out, "correct"))
for cat in ["fact", "multi", "calc", "holiday"]:
    print(f"  {cat:8s} accuracy:", pct([r for r in ins if r["category"] == cat], "correct"))
