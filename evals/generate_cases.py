#!/usr/bin/env python3
"""generate_cases.py — runs each question in eval_design.json through the
real RAG service, capturing {context, answer} pairs in the exact JSONL shape
judge.py expects. Run this whenever the service or corpus changes, before
running judge.py."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag_service import answer


def main():
    design_path = os.path.join(os.path.dirname(__file__), "eval_design.json")
    out_path = os.path.join(os.path.dirname(__file__), "cases.jsonl")

    with open(design_path, encoding="utf-8") as fh:
        design = json.load(fh)

    with open(out_path, "w", encoding="utf-8") as out:
        for case in design:
            context, ans = answer(case["question"])
            out.write(json.dumps({
                "id": case["id"],
                "question": case["question"],
                "context": context,
                "answer": ans,
            }) + "\n")
            print(f"[{case['id']}] generated")

    print(f"\nWrote {len(design)} cases to {out_path}")


if __name__ == "__main__":
    main()