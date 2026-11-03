"""Build data/molecules.csv by looking each molecule up on PubChem.

Run once (needs network):  uv run python scripts/build_dataset.py

Why: hand-typed reference SMILES are a classic source of silent errors in
chemistry evals (wrong stereocentre, missing H, typo). Taking the reference
from PubChem and recording the CID makes every gold answer traceable.
The output CSV is committed, so the eval itself never needs the network.
"""

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from rdkit import Chem

# (name shown to the model, name used for the PubChem query, difficulty)
MOLECULES: list[tuple[str, str, str]] = [
    # easy: very common small molecules
    ("water", "water", "easy"),
    ("methanol", "methanol", "easy"),
    ("ethanol", "ethanol", "easy"),
    ("acetic acid", "acetic acid", "easy"),
    ("acetone", "acetone", "easy"),
    ("benzene", "benzene", "easy"),
    ("toluene", "toluene", "easy"),
    ("phenol", "phenol", "easy"),
    ("cyclohexane", "cyclohexane", "easy"),
    ("carbon dioxide", "carbon dioxide", "easy"),
    # medium: drugs / heterocycles / IUPAC-style names, no stereochemistry
    ("aspirin", "aspirin", "medium"),
    ("caffeine", "caffeine", "medium"),
    ("paracetamol", "acetaminophen", "medium"),
    ("ibuprofen", "ibuprofen", "medium"),
    ("naphthalene", "naphthalene", "medium"),
    ("pyridine", "pyridine", "medium"),
    ("benzoic acid", "benzoic acid", "medium"),
    ("2-methylpropan-2-ol", "tert-butanol", "medium"),
    ("ethyl ethanoate", "ethyl acetate", "medium"),
    ("indole", "indole", "medium"),
    # hard: larger molecules and/or defined stereochemistry
    ("L-alanine", "L-alanine", "hard"),
    ("(S)-nicotine", "nicotine", "hard"),
    ("L-menthol", "levomenthol", "hard"),
    ("L-ascorbic acid", "ascorbic acid", "hard"),
    ("testosterone", "testosterone", "hard"),
    ("cholesterol", "cholesterol", "hard"),
    ("morphine", "morphine", "hard"),
    ("quinine", "quinine", "hard"),
    ("capsaicin", "capsaicin", "hard"),
    ("sildenafil", "sildenafil", "hard"),
]

URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/SMILES,MolecularWeight/JSON"
OUT = Path(__file__).resolve().parents[1] / "data" / "molecules.csv"


def fetch(query: str) -> dict:
    """Return the first PubChem property record for a compound name."""
    url = URL.format(urllib.parse.quote(query))
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)["PropertyTable"]["Properties"][0]


def main() -> None:
    rows = []
    for name, query, difficulty in MOLECULES:
        rec = fetch(query)
        mol = Chem.MolFromSmiles(rec["SMILES"])
        assert mol is not None, f"PubChem SMILES for {name} does not parse"
        rows.append(
            {
                "name": name,
                "smiles": Chem.MolToSmiles(mol),  # RDKit canonical, isomeric
                "difficulty": difficulty,
                "pubchem_cid": rec["CID"],
                "pubchem_mw": rec["MolecularWeight"],
            }
        )
        print(f"{name:22s} CID {rec['CID']:>8}  {rows[-1]['smiles']}")
        time.sleep(0.25)  # PubChem asks for <= 5 requests/second

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
