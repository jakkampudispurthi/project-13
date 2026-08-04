#!/usr/bin/env python3
"""judge.py — a minimal LLM-as-a-judge groundedness evaluator.

This is the gate ci.yml calls directly (`python evals/judge.py ...`). It scores whether each
answer is GROUNDED in the retrieved context (RAG faithfulness) -- not whether it
"sounds good." The judge is itself a model, so it gets the same suspicion we give
any confident partner:
  * the rubric is EXPLICIT and example-grounded (not "rate 1-10, vibes");
  * we validate the judge against human labels before trusting it;
  * we watch for "judge drift" -- the judge's own scores moving over time.

Provider note (as of 2026): this uses the `openai` SDK, but it is NOT tied to
OpenAI. The OpenAI SDK is just the de-facto wire format; point `base_url` at any
OpenAI-compatible endpoint and pass that vendor's key. Set two env vars:
  * OPENAI_BASE_URL -- e.g. https://api.openai.com/v1, a cloud vendor's /v1 URL,
    or a local server like http://localhost:11434/v1 (Ollama) or a vLLM box.
  * OPENAI_API_KEY  -- the key for whatever endpoint you pointed at.
  * MODEL_ID        -- pin the exact model ID; never "latest".
This is the SAME file the CI gate calls. In CI it must point at a CLOUD endpoint:
a GitHub-hosted runner cannot reach Ollama on your box (project-13.md, App. A.12.5).

Usage (matches ci.yml's eval gate):
    python judge.py evals/cases.jsonl --min-groundedness 0.85 --min-pass-rate 0.90
Exit 0 if the groundedness rate >= --min-groundedness, else exit 1 (fails the build).
"""
import argparse
import json
import os
import sys

from openai import OpenAI

RUBRIC = """You are a strict groundedness judge. Given CONTEXT and an ANSWER,
return JSON {"grounded": true|false, "reason": "<one sentence>"}.
Rule: grounded=true ONLY if every factual claim in ANSWER is supported by CONTEXT.
A fluent answer that adds facts not in CONTEXT is grounded=false (a hallucination).
Do not reward confidence. Do not reward style. Support only.
Return ONLY the JSON object, nothing else."""


def make_client():
    """An OpenAI-compatible client. base_url + api_key come from the environment,
    so the same code talks to OpenAI, a cloud vendor, vLLM, or local Ollama."""
    return OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),  # None => OpenAI's default /v1
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed-for-local"),
    )


def score_case(client, model_id, context, answer):
    resp = client.chat.completions.create(
        model=model_id,
        temperature=0,  # a judge should be as deterministic as the API allows
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"},
        ],
    )
    raw = resp.choices[0].message.content
    return json.loads(raw)


def run(eval_file, client, model_id):
    cases = [json.loads(line) for line in open(eval_file) if line.strip()]
    if not cases:
        raise SystemExit(f"no eval cases found in {eval_file}")
    grounded = 0
    for c in cases:
        verdict = score_case(client, model_id, c["context"], c["answer"])
        grounded += 1 if verdict["grounded"] else 0
    rate = grounded / len(cases)
    print(f"groundedness = {rate:.3f} over {len(cases)} cases")
    return rate


def main():
    ap = argparse.ArgumentParser(description="LLM-as-a-judge groundedness gate.")
    ap.add_argument("eval_file", help="JSONL of {context, answer} cases")
    ap.add_argument("--min-groundedness", type=float, default=0.85,
                    help="fail the build below this groundedness rate")
    ap.add_argument("--min-pass-rate", type=float, default=None,
                    help="alias accepted for ci.yml compatibility")
    ap.add_argument("--model-id", default=os.environ.get("MODEL_ID"),
                    help="pinned model ID (or set MODEL_ID env var)")
    args = ap.parse_args()

    if not args.model_id:
        raise SystemExit("set --model-id or the MODEL_ID env var (pin it; no 'latest')")

    threshold = args.min_groundedness
    if args.min_pass_rate is not None:
        threshold = max(threshold, args.min_pass_rate)

    rate = run(args.eval_file, make_client(), args.model_id)
    if rate < threshold:
        print(f"FAIL: groundedness {rate:.3f} < threshold {threshold:.3f}")
        sys.exit(1)
    print(f"PASS: groundedness {rate:.3f} >= threshold {threshold:.3f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
