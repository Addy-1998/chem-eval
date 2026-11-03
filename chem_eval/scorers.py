"""Deterministic RDKit-based scorers.

Answer extraction (the one rule used for the HEADLINE score):
    the text inside the LAST <answer>...</answer> block of the model output.
    No tag -> status "no_answer", or "truncated" if generation stopped because it
    hit max_tokens (e.g. a reasoning model that never finished thinking). Both are
    scored incorrect but counted separately, so a formatting or budget failure is
    never confused with a chemistry mistake.

Every Score carries metadata["status"], one of:
    correct | wrong | unparseable | no_answer | truncated

Diagnostic only (never changes the score): when there is no <answer> tag, the
last LaTeX \\boxed{...} in the output is judged with the SAME check and stored as
metadata["lenient_correct"]. qwen3:8b often ignored the requested tag and used
\\boxed{} instead; this lets the report show how much of a model gap is format
compliance rather than chemistry.
"""

import re

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")  # RDKit otherwise prints parse errors to stderr

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# --- extraction -------------------------------------------------------------

def _clean(text: str) -> str:
    # Tolerate markdown code formatting, e.g. <answer>`CCO`</answer>
    return text.strip().strip("`").strip()


def extract_answer(completion: str) -> str | None:
    """Return the stripped content of the last <answer> block, or None."""
    matches = ANSWER_RE.findall(completion)
    return _clean(matches[-1]) if matches else None


def extract_boxed(completion: str) -> str | None:
    """Return the content of the last \\boxed{...} (diagnostic fallback), or None."""
    matches = BOXED_RE.findall(completion)
    return _clean(matches[-1]) if matches else None


def parse_number(text: str) -> float | None:
    """First number in text (commas removed), or None. '180.16 g/mol' -> 180.16."""
    match = NUMBER_RE.search(text.replace(",", ""))
    return float(match.group()) if match else None


# --- chemistry helpers ------------------------------------------------------

def canonical_smiles(smiles: str, isomeric: bool = True) -> str | None:
    """RDKit canonical SMILES, or None if the string does not parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def canonical_tautomer_smiles(smiles: str) -> str | None:
    """Canonical SMILES of RDKit's canonical tautomer (diagnostic only)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(rdMolStandardize.TautomerEnumerator().Canonicalize(mol))


# --- judges: (extracted answer, reference) -> (status, explanation, extra metadata)

def judge_smiles(answer: str, reference: str) -> tuple[str, str, dict]:
    """Strict check: canonical isomeric SMILES must be identical."""
    predicted = canonical_smiles(answer)
    if predicted is None:
        return "unparseable", "answer does not parse as SMILES", {}
    ref = canonical_smiles(reference)
    extra = {
        "match_ignore_stereo": canonical_smiles(answer, isomeric=False)
        == canonical_smiles(reference, isomeric=False),
        "match_tautomer": canonical_tautomer_smiles(answer)
        == canonical_tautomer_smiles(reference),
    }
    status = "correct" if predicted == ref else "wrong"
    return status, f"predicted {predicted} vs reference {ref}", extra


def judge_molar_mass(answer: str, reference: str, rel_tol: float) -> tuple[str, str, dict]:
    """Relative-error check against the RDKit reference molar mass."""
    value = parse_number(answer)
    if value is None:
        return "unparseable", "no number found in answer", {}
    ref = float(reference)
    rel_error = abs(value - ref) / ref
    status = "correct" if rel_error <= rel_tol else "wrong"
    return status, f"{value} vs reference {ref:.4f} (rel. error {rel_error:.4%})", {
        "rel_error": rel_error
    }


def score_output(state: TaskState, target: Target, judge) -> Score:
    """Extract the answer, apply `judge`, and build a Score with full metadata."""
    completion = state.output.completion
    answer = extract_answer(completion)

    if answer is None:
        truncated = state.output.stop_reason == "max_tokens"
        metadata = {"status": "truncated" if truncated else "no_answer"}
        boxed = extract_boxed(completion)
        metadata["lenient_correct"] = (
            boxed is not None and judge(boxed, target.text)[0] == "correct"
        )
        if boxed is not None:
            metadata["boxed_answer"] = boxed
        return Score(
            value=INCORRECT,
            answer=None,
            explanation="hit max_tokens before answering" if truncated else "no <answer> tag",
            metadata=metadata,
        )

    status, explanation, extra = judge(answer, target.text)
    return Score(
        value=CORRECT if status == "correct" else INCORRECT,
        answer=answer,
        explanation=explanation,
        metadata={"status": status, "lenient_correct": status == "correct", **extra},
    )


# --- Inspect scorers --------------------------------------------------------

@scorer(metrics=[accuracy(), stderr()])
def smiles_match():
    """Correct iff canonical isomeric SMILES of answer == that of the target.

    Also records diagnostics that do NOT change the score:
      match_ignore_stereo - would it be correct if stereochemistry were ignored?
      match_tautomer      - would it be correct after tautomer canonicalisation?
      lenient_correct     - correct if a missing <answer> tag falls back to \\boxed{}
    """

    async def score(state: TaskState, target: Target) -> Score:
        return score_output(state, target, judge_smiles)

    return score


@scorer(metrics=[accuracy(), stderr()])
def molar_mass_within(rel_tol: float = 0.005):
    """Correct iff |answer - target| / target <= rel_tol (default 0.5%)."""

    async def score(state: TaskState, target: Target) -> Score:
        return score_output(
            state, target, lambda ans, ref: judge_molar_mass(ans, ref, rel_tol)
        )

    return score
