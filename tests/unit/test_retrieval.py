import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from rag_service import retrieve, split_sections, load_corpus


def test_corpus_loads():
    corpus = load_corpus()
    assert len(corpus) > 100
    assert "Lockdown Procedure" in corpus


def test_split_sections_finds_known_sections():
    sections = split_sections(load_corpus())
    headings = [h for h, _ in sections]
    assert any("Lockdown" in h for h in headings)
    assert any("Snow Day" in h for h in headings)
    assert len(sections) >= 8


def test_retrieve_lockdown_question_returns_lockdown_section():
    context = retrieve("What do I do during a lockdown?")
    assert "Lockdown Procedure" in context
    assert "remain silent" in context.lower()


def test_retrieve_unrelated_question_returns_no_match_marker():
    context = retrieve("xyzzyplughqwerty nonsense words")
    assert "no matching policy section" in context.lower()


def test_retrieve_password_question_returns_account_section():
    context = retrieve("Can I text the temporary password to a student?")
    assert "temporary password" in context.lower()