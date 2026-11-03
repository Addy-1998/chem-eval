"""Edge cases for answer extraction and the two scorers."""

import asyncio
from types import SimpleNamespace

import pytest
from inspect_ai.scorer import CORRECT, INCORRECT, Target

from chem_eval.scorers import (
    extract_answer,
    molar_mass_within,
    parse_number,
    smiles_match,
)


def run(scorer_fn, completion: str, target: str, stop_reason: str = "stop"):
    """Call an Inspect scorer on a fake TaskState holding only the model output."""
    state = SimpleNamespace(
        output=SimpleNamespace(completion=completion, stop_reason=stop_reason)
    )
    return asyncio.run(scorer_fn(state, Target(target)))


# --- extraction -----------------------------------------------------------

def test_extract_uses_last_answer_block():
    text = "Maybe <answer>CO</answer>... no, actually <answer>CCO</answer>"
    assert extract_answer(text) == "CCO"


def test_extract_strips_whitespace_and_backticks():
    assert extract_answer("<ANSWER>\n `CCO` \n</ANSWER>") == "CCO"


def test_extract_missing_tag():
    assert extract_answer("The SMILES is CCO") is None


@pytest.mark.parametrize(
    "text, expected",
    [("180.16", 180.16), ("180.16 g/mol", 180.16), ("1,234.5", 1234.5),
     ("~46.07", 46.07), ("1.8016e2", 180.16), ("g/mol", None)],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


# --- SMILES scorer --------------------------------------------------------

@pytest.mark.parametrize(
    "answer",
    ["CCO", "OCC", "C(O)C", "[CH3][CH2][OH]"],  # different spellings, same molecule
)
def test_equivalent_smiles_are_correct(answer):
    s = run(smiles_match(), f"<answer>{answer}</answer>", "CCO")
    assert s.value == CORRECT and s.metadata["status"] == "correct"


def test_kekule_benzene_matches_aromatic():
    s = run(smiles_match(), "<answer>C1=CC=CC=C1</answer>", "c1ccccc1")
    assert s.value == CORRECT


def test_wrong_molecule():
    s = run(smiles_match(), "<answer>CCCO</answer>", "CCO")
    assert s.value == INCORRECT and s.metadata["status"] == "wrong"


def test_missing_stereo_is_wrong_but_flagged():
    # L-alanine without the stereocentre
    s = run(smiles_match(), "<answer>CC(N)C(=O)O</answer>", "C[C@H](N)C(=O)O")
    assert s.value == INCORRECT
    assert s.metadata["match_ignore_stereo"] is True


def test_wrong_enantiomer_is_wrong():
    s = run(smiles_match(), "<answer>C[C@@H](N)C(=O)O</answer>", "C[C@H](N)C(=O)O")
    assert s.value == INCORRECT and s.metadata["match_ignore_stereo"] is True


def test_tautomer_is_wrong_but_flagged():
    # 2-pyridone vs 2-hydroxypyridine
    s = run(smiles_match(), "<answer>Oc1ccccn1</answer>", "O=c1cccc[nH]1")
    assert s.value == INCORRECT
    assert s.metadata["match_tautomer"] is True


def test_unparseable_smiles_tracked_separately():
    s = run(smiles_match(), "<answer>C1CC(</answer>", "CCO")
    assert s.value == INCORRECT and s.metadata["status"] == "unparseable"


def test_no_answer_tag_tracked_separately():
    s = run(smiles_match(), "CCO", "CCO")
    assert s.value == INCORRECT and s.metadata["status"] == "no_answer"


def test_boxed_fallback_is_diagnostic_only():
    # No <answer> tag: headline stays INCORRECT, but lenient_correct records the boxed answer
    s = run(smiles_match(), r"Final answer: $$\boxed{OCC}$$", "CCO")
    assert s.value == INCORRECT and s.metadata["status"] == "no_answer"
    assert s.metadata["lenient_correct"] is True


def test_boxed_fallback_still_applies_the_strict_check():
    s = run(molar_mass_within(), r"\boxed{130.18}", "128.1740")  # 1.6% off
    assert s.metadata["lenient_correct"] is False


def test_answer_tag_takes_priority_over_boxed():
    s = run(smiles_match(), r"\boxed{CCO} ... <answer>CCCO</answer>", "CCO")
    assert s.value == INCORRECT and s.metadata["lenient_correct"] is False


def test_truncated_output_tracked_separately():
    s = run(smiles_match(), "Let me think step by step about", "CCO", stop_reason="max_tokens")
    assert s.value == INCORRECT and s.metadata["status"] == "truncated"


# --- molar-mass scorer ----------------------------------------------------

@pytest.mark.parametrize(
    "answer, expected",
    [("180.16", CORRECT), ("180.158 g/mol", CORRECT),
     ("181.0", CORRECT),     # 0.47% off -> inside 0.5%
     ("181.1", INCORRECT),   # 0.52% off -> outside
     ("179.2", INCORRECT)],
)
def test_molar_mass_tolerance(answer, expected):
    s = run(molar_mass_within(), f"<answer>{answer}</answer>", "180.1590")
    assert s.value == expected


def test_molar_mass_unparseable():
    s = run(molar_mass_within(), "<answer>about two hundred</answer>", "180.1590")
    assert s.value == INCORRECT and s.metadata["status"] == "unparseable"
