"""Inspect task definitions.

Run e.g.:  uv run inspect eval chem_eval/tasks.py --model ollama/gemma4:e2b
"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from chem_eval.dataset import molar_mass_dataset, name_to_smiles_dataset
from chem_eval.scorers import molar_mass_within, smiles_match

# Temperature 0 so reruns are as repeatable as the provider allows.
# max_tokens is generous so reasoning models are not cut off before <answer>
# (a smoke run at 4096 truncated gemma4:e2b mid-reasoning; see LEARNING.md).
CONFIG = GenerateConfig(temperature=0.0, max_tokens=8192)


@task
def name_to_smiles() -> Task:
    """Molecule name -> SMILES, scored by canonical SMILES equality."""
    return Task(
        dataset=name_to_smiles_dataset(),
        solver=generate(),
        scorer=smiles_match(),
        config=CONFIG,
    )


@task
def molar_mass() -> Task:
    """Molecule name -> molar mass in g/mol, scored within 0.5% relative tolerance."""
    return Task(
        dataset=molar_mass_dataset(),
        solver=generate(),
        scorer=molar_mass_within(rel_tol=0.005),
        config=CONFIG,
    )
