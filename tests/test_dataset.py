"""The golden dataset must be valid before any model is scored against it."""

import pytest
from rdkit import Chem

from chem_eval.dataset import (
    FORMAT_EXAMPLE_MASS,
    FORMAT_EXAMPLE_SMILES,
    load_molecules,
    molar_mass_dataset,
    name_to_smiles_dataset,
    reference_molar_mass,
)

ROWS = load_molecules()


def test_size_and_columns():
    assert len(ROWS) == 30
    assert set(ROWS[0]) == {"name", "smiles", "difficulty", "pubchem_cid", "pubchem_mw"}


def test_difficulty_labels_and_balance():
    labels = [r["difficulty"] for r in ROWS]
    assert set(labels) == {"easy", "medium", "hard"}
    assert all(labels.count(d) == 10 for d in ("easy", "medium", "hard"))


def test_names_and_molecules_unique():
    assert len({r["name"] for r in ROWS}) == len(ROWS)
    assert len({Chem.CanonSmiles(r["smiles"]) for r in ROWS}) == len(ROWS)


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["name"])
def test_reference_smiles_parses(row):
    assert Chem.MolFromSmiles(row["smiles"]) is not None


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["name"])
def test_rdkit_molar_mass_agrees_with_pubchem(row):
    # Independent cross-check of the molar-mass reference (PubChem's own value).
    rdkit_mw = reference_molar_mass(row["smiles"])
    assert rdkit_mw == pytest.approx(float(row["pubchem_mw"]), rel=1e-3)


def test_inspect_datasets_build():
    n2s, mm = name_to_smiles_dataset(), molar_mass_dataset()
    assert len(n2s) == len(mm) == 30
    for s in [*n2s, *mm]:
        assert s.metadata["name"] in s.input


def test_format_examples_do_not_leak_answers():
    # The <answer> example in the prompt must not be the answer to any sample.
    references = {Chem.CanonSmiles(r["smiles"]) for r in ROWS}
    assert Chem.CanonSmiles(FORMAT_EXAMPLE_SMILES) not in references
    example_mass = float(FORMAT_EXAMPLE_MASS)
    for r in ROWS:
        assert abs(example_mass - reference_molar_mass(r["smiles"])) / example_mass > 0.05
