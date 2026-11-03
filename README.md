# chem-eval

A small, rigorous evaluation of LLMs on chemistry questions whose answers can be
checked automatically. Built on [Inspect](https://inspect.aisi.org.uk/) and scored
deterministically with [RDKit](https://www.rdkit.org/), with no LLM judge.

**Status (2026-11-03):** v1 evaluates two **local** models (Ollama). No API
model was run for v1, so there are no frontier-model numbers here. The harness
supports one (see *Reproduce*), and this is a known gap. Every number in this
README is copied from [`results/report.md`](results/report.md), which
`analysis/report.py` generates from the Inspect logs in `logs/`.

## Tasks

| task | prompt | correct if |
|---|---|---|
| `name_to_smiles` | "Give the SMILES for *name*" (+ asks for full stereochemistry) | RDKit canonical isomeric SMILES of the answer == that of the reference |
| `molar_mass` | "What is the molar mass of *name* in g/mol?" | within 0.5% relative error of RDKit `Descriptors.MolWt(reference)` |

**Answer extraction:** the content of the last `<answer>...</answer>` block.
The prompt states this format. Failures are counted separately and never merged
into "wrong chemistry":

- `unparseable`: tag present, but not valid SMILES / no number
- `no_answer`: no tag
- `truncated`: hit `max_tokens` before answering

**Dataset:** 30 molecules ([`data/molecules.csv`](data/molecules.csv)), 10 easy /
10 medium / 10 hard. Reference SMILES come from PubChem by name (CID recorded),
via `scripts/build_dataset.py`, not typed by hand. Tests check that every SMILES
parses and that RDKit's molar mass agrees with PubChem's within 0.1%.

## Results

Models: `ollama/gemma4:e2b` and `ollama/qwen3:8b`, run locally (RTX 5070 Laptop,
8 GB), temperature 0, `max_tokens` 8192, one run each. Cells: accuracy (k/n) [Wilson 95% CI].

| model | task | overall | easy | medium | hard |
|---|---|---|---|---|---|
| gemma4:e2b | molar_mass | 90% (27/30) [74%, 97%] | 100% (10/10) | 100% (10/10) | 70% (7/10) |
| gemma4:e2b | name_to_smiles | 37% (11/30) [22%, 54%] | 90% (9/10) | 20% (2/10) | 0% (0/10) |
| qwen3:8b | molar_mass | 73% (22/30) [56%, 86%] | 100% (10/10) | 60% (6/10) | 60% (6/10) |
| qwen3:8b | name_to_smiles | 23% (7/30) [12%, 41%] | 70% (7/10) | 0% (0/10) | 0% (0/10) |

Per-difficulty CIs (n = 10) are very wide, e.g. 10/10 → [72%, 100%]; see the full report.

**Paired comparison (both tasks, 60 items):** gemma4:e2b − qwen3:8b = **+15.0%**,
paired bootstrap 95% CI [+3.3%, +26.7%].

### Read this before quoting the comparison

**Most of that gap is format compliance, not chemistry.** qwen3:8b often ignored
the requested `<answer>` tag and wrote LaTeX `\boxed{...}` instead: 7 of 30
molar_mass outputs had no tag. Applying the *same* correctness check to the last
`\boxed{}` (a diagnostic added after reading the outputs, not the headline):

| | strict (reported) | lenient `\boxed{}` fallback |
|---|---|---|
| qwen3:8b molar_mass | 73% (22/30) | 93% (28/30) |
| gemma4:e2b − qwen3:8b, both tasks | +15.0% [+3.3%, +26.7%] | +5.0% [−3.3%, +13.3%] |

Under lenient extraction the CI includes 0: **this run cannot distinguish the two
models on chemistry**. The strict number stays the headline because the format
rule was fixed before the runs. Instruction-following is a real part of
deployability, but it is a different skill from chemistry.

### Other caveats

- **Molar mass mostly measures recall.** gemma4:e2b gives testosterone's molar mass
  correctly (288.41) but cannot write its SMILES. These values are widely published,
  so high molar_mass accuracy does not show the model can compute masses from structure.
- **Contamination.** All 30 molecules are famous and appear on the public web.
  Scores are an upper bound on performance for unfamiliar molecules.
- **Truncation.** gemma4:e2b hit the 8192-token limit on 7/30 name_to_smiles items
  (reasoning loops), and qwen3:8b on 6/30. They are scored incorrect and reported separately.
  An early 4096-token smoke run truncated even *methanol*, so the limit was raised before the full runs.
- **Strict SMILES.** Missing or wrong stereochemistry is wrong. The report shows
  stereo-blind and tautomer-merged diagnostics; for these two models neither changes any count.
- **Small n, one seed, one prompt.** 30 items per task; no run-to-run variance estimate yet.
- **Rescoring.** The logs in `logs/` were rescored with `inspect score` after the
  `\boxed{}` diagnostic was added. Generations are byte-identical to the originals
  in `logs/original_scoring/`, and strict scores did not change.

**Cost/latency:** both models run locally, so cost is $0. Median per-sample latency is
8–14 s for gemma4:e2b and ~140 s for qwen3:8b on this laptop (concurrency 2).

## Reproduce

```bash
uv sync
uv run pytest
uv run inspect eval chem_eval/tasks.py --model ollama/gemma4:e2b --log-dir logs --max-connections 2
uv run inspect eval chem_eval/tasks.py --model ollama/qwen3:8b  --log-dir logs --max-connections 2
uv run inspect eval chem_eval/tasks.py --model anthropic/claude-sonnet-5-5 --log-dir logs   # optional, not run for v1; needs ANTHROPIC_API_KEY
uv run python analysis/report.py       # -> results/report.md
uv run inspect view                    # browse every prompt, output and score
```

API keys are read from environment variables only; none are stored in the repo.

## Layout

```
data/molecules.csv         golden dataset (PubChem-sourced, CID per row)
scripts/build_dataset.py   rebuilds the CSV from PubChem
chem_eval/dataset.py       CSV -> Inspect Samples + prompts
chem_eval/scorers.py       RDKit scorers + answer extraction
chem_eval/tasks.py         @task definitions
analysis/report.py         logs -> results/report.md (Wilson CIs, paired bootstrap)
tests/                     dataset validity, scorer edge cases, statistics
LEARNING.md                concepts and interview questions per component
```

## Roadmap (v2)

Run at least one API model (e.g. Claude) through the same tasks;
LLM-as-judge for free-text explanations, calibrated against hand labels (Cohen's κ);
DVC pipeline (data → eval → report); GitHub Actions running a 5-sample smoke eval on every PR.
