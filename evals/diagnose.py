"""diagnose.py — run judge.py's scoring logic per-case, with reasons shown."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from judge import make_client, score_case

MODEL_ID = os.environ.get("MODEL_ID", "gpt-5.4-mini")

with open("evals/cases.jsonl") as fh:
    cases = [json.loads(line) for line in fh if line.strip()]
client = make_client()

for c in cases:
    verdict = score_case(client, MODEL_ID, c["context"], c["answer"])
    status = "PASS" if verdict["grounded"] else "FAIL"
    print(f"[{status}] {c['id']}: {c['question']}")
    print(f"       reason: {verdict['reason']}")
    print(f"       answer: {c['answer'][:150]}")
    print()
