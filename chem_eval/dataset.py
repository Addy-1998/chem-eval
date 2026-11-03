"""Turn data/molecules.csv into Inspect Samples for both task types."""

import csv
from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample
from rdkit import Chem
from rdkit.Chem import Descriptors

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "molecules.csv"

# Both prompts tell the model exactly how the answer will be extracted:
# the content of the LAST <answer>...</answer> block in the response.
# The format examples must NOT be answers to any sample (tests check this):
# CCC is propane (not in the dataset); 250.5 is far from every reference mass.
FORMAT_EXAMPLE_SMILES = "CCC"
FORMAT_EXAMPLE_MASS = "250.5"

NAME_TO_SMILES_PROMPT = (
    "Give the SMILES for {name}.\n\n"
    "Include full stereochemistry (@, @@, / and \\) wherever the molecule has "
    "defined stereocentres or double-bond geometry.\n"
    "You may reason first. End your response with the SMILES string alone inside "
    f"<answer></answer> tags, e.g. <answer>{FORMAT_EXAMPLE_SMILES}</answer>."
)

MOLAR_MASS_PROMPT = (
    "What is the molar mass of {name} in g/mol?\n\n"
    "Use standard atomic weights. You may reason first. End your response with "
    "the number alone (no units) inside <answer></answer> tags, "
    f"e.g. <answer>{FORMAT_EXAMPLE_MASS}</answer>."
)


def load_molecules(path: Path = DATA_PATH) -> list[dict[str, str]]:
    """Read the golden molecule CSV into a list of row dicts."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def reference_molar_mass(smiles: str) -> float:
    """Molar mass in g/mol from RDKit (average atomic weights, implicit Hs included)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"reference SMILES does not parse: {smiles}")
    return Descriptors.MolWt(mol)


def _metadata(row: dict[str, str]) -> dict[str, str]:
    return {
        "name": row["name"],
        "difficulty": row["difficulty"],
        "reference_smiles": row["smiles"],
        "pubchem_cid": row["pubchem_cid"],
    }


def name_to_smiles_dataset(path: Path = DATA_PATH) -> MemoryDataset:
    """One sample per molecule: name in, reference SMILES as target."""
    samples = [
        Sample(
            id=f"n2s-{i:02d}",
            input=NAME_TO_SMILES_PROMPT.format(name=row["name"]),
            target=row["smiles"],
            metadata=_metadata(row),
        )
        for i, row in enumerate(load_molecules(path))
    ]
    return MemoryDataset(samples, name="name_to_smiles")


def molar_mass_dataset(path: Path = DATA_PATH) -> MemoryDataset:
    """One sample per molecule: name in, RDKit molar mass (as a string) as target."""
    samples = [
        Sample(
            id=f"mm-{i:02d}",
            input=MOLAR_MASS_PROMPT.format(name=row["name"]),
            target=f"{reference_molar_mass(row['smiles']):.4f}",
            metadata=_metadata(row),
        )
        for i, row in enumerate(load_molecules(path))
    ]
    return MemoryDataset(samples, name="molar_mass")
