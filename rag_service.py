#!/usr/bin/env python3
"""rag_service.py — the school help-desk RAG bot.

Simple, fixed-corpus retrieval: the policy doc is small enough that we do
straightforward section-based keyword retrieval rather than embeddings.
Retrieval and generation are separate functions so tests can exercise
retrieval alone (deterministic, no API calls, fast) while the eval gate
exercises the full pipeline (retrieval + generation, judged for grounding).

Model ID is PINNED here, not "latest" -- see deploy-notes discipline from
Ch.12. Reused verified-current model from Project 12: gpt-5.4-mini.
"""
import os
import re

from openai import OpenAI

MODEL_ID = os.environ.get("MODEL_ID", "gpt-5.4-mini")  # pinned; override via env in CI
POLICY_DOC_PATH = os.path.join(os.path.dirname(__file__), "policy_docs", "school_policies.txt")

SYSTEM_PROMPT = """You are the Grace Christian School staff help desk assistant.
Answer ONLY using the provided CONTEXT. If the context does not address the
question, say so explicitly rather than guessing or inventing an answer.
Never state a policy that is not written in the context, even if it sounds
plausible. Be concise."""


def load_corpus():
    with open(POLICY_DOC_PATH, encoding="utf-8") as fh:
        return fh.read()


def split_sections(corpus):
    """Split the corpus into (heading, body) chunks on numbered sub-headings
    like '1.1 Resetting a Student's Password'."""
    pattern = re.compile(r"\n(\d\.\d [A-Z][^\n]+)\n")
    parts = pattern.split(corpus)
    sections = []
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((heading, body))
    return sections


def retrieve(question, corpus=None, top_k=2):
    """Naive keyword-overlap retrieval: score each section by shared words
    with the question, return the top_k highest-scoring sections joined as
    context. Deterministic, no API calls -- unit-testable on its own."""
    corpus = corpus or load_corpus()
    sections = split_sections(corpus)
    q_words = set(re.findall(r"[a-z]+", question.lower()))

    scored = []
    for heading, body in sections:
        text_words = set(re.findall(r"[a-z]+", (heading + " " + body).lower()))
        overlap = len(q_words & text_words)
        scored.append((overlap, heading, body))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for s in scored[:top_k] if s[0] > 0]
    if not top:
        return "(no matching policy section found)"
    return "\n\n".join(f"{h}\n{b}" for _, h, b in top)


def answer(question, client=None, model_id=None):
    """Full RAG call: retrieve context, generate an answer, return both."""
    context = retrieve(question)
    client = client or OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    resp = client.chat.completions.create(
        model=model_id or MODEL_ID,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
        ],
    )
    return context, resp.choices[0].message.content


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What should I do during a lockdown?"
    ctx, ans = answer(q)
    print("CONTEXT:\n", ctx)
    print("\nANSWER:\n", ans)