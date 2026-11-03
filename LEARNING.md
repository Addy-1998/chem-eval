# LEARNING.md

One entry per component, in the order they were built. For each: what it does,
the key concepts, where to look, and questions an interviewer might ask.

---

## 1. Golden dataset (`data/molecules.csv`, `scripts/build_dataset.py`)

**What it does.** 30 molecules (10 easy / 10 medium / 10 hard) with a reference
SMILES for each. The SMILES were not typed by hand: `build_dataset.py` looks each
name up on PubChem, parses the SMILES with RDKit, and stores RDKit's canonical
form plus the PubChem CID and PubChem's molecular weight. The CSV is committed,
so running the eval never needs the network.

**Key concepts**
- **SMILES**: a line notation for a molecule (`CCO` = ethanol). Many different
  strings describe the same molecule (`CCO`, `OCC`, `C(O)C`).
- **Canonical SMILES**: RDKit's algorithm picks *one* standard string per
  molecule, so two SMILES describe the same molecule iff their canonical forms
  are equal. *Isomeric* SMILES also encode stereochemistry (`@`, `@@`, `/`, `\`).
- **Provenance / traceable gold labels**: every reference can be checked
  against an external source (CID). Gold-label errors silently cap or distort
  every model's score.
- **Difficulty strata**: easy = tiny common molecules; medium = drugs and
  heterocycles, plus two IUPAC-style names, no stereo; hard = big and/or
  stereo-defined molecules.

**Study:** `scripts/build_dataset.py::main`, `tests/test_dataset.py` (every
SMILES parses, RDKit mass agrees with PubChem within 0.1%, no duplicates).

**Interviewer questions**
1. Why is "my 30 reference answers are correct" a claim you have to test, and how did you test it?
2. These molecules are all over the internet. What does that mean for interpreting the scores (memorisation vs. reasoning)?
3. With n = 10 per difficulty level, how wide is a 95% CI on per-difficulty accuracy? Is 30 enough?

---

## 2. Inspect dataset + prompts (`chem_eval/dataset.py`)

**What it does.** Converts each CSV row into an Inspect `Sample` for each task.
`input` is the prompt, `target` is the reference answer (SMILES, or the RDKit
molar mass as a string), and `metadata` carries difficulty/name/CID for the
analysis.

**Key concepts**
- **Inspect `Sample` / `MemoryDataset`**: Inspect's unit of evaluation; the
  scorer receives `target` and the model output.
- **Answer-format contract**: the prompt states exactly how the answer will be
  extracted (the last `<answer>...</answer>` block).
- **Prompt leakage**: the first draft used `<answer>CCO</answer>` and `46.07` as
  format examples. Those are ethanol's SMILES and molar mass, and ethanol is in
  the dataset, so the prompt contained the answer to one sample. Fixed (propane
  `CCC`, `250.5`), with a test that no format example matches any reference.
- **Reference molar mass**: `Descriptors.MolWt` sums *average* atomic weights
  including implicit hydrogens (not the monoisotopic "exact mass").

**Study:** `NAME_TO_SMILES_PROMPT`, `MOLAR_MASS_PROMPT`, `reference_molar_mass`,
`tests/test_dataset.py::test_format_examples_do_not_leak_answers`.

**Interviewer questions**
1. How can the format example in a prompt leak an answer, and how would you catch it automatically?
2. Why does the prompt explicitly ask for stereochemistry? What would be unfair if it didn't?
3. What is the difference between `MolWt` and `ExactMolWt`, and which one does "molar mass" mean?

---

## 3. Scorers (`chem_eval/scorers.py`)

**What it does.** Two deterministic scorers, no LLM judge.
- `smiles_match`: correct iff canonical isomeric SMILES(answer) == canonical
  isomeric SMILES(reference).
- `molar_mass_within`: correct iff relative error <= 0.5%.
Every score carries a `status`: `correct | wrong | unparseable | no_answer | truncated`.

**Key concepts**
- **Inspect scorers**: an `async def score(state, target) -> Score` wrapped in
  `@scorer(metrics=[accuracy(), stderr()])`. `Score.value` is `CORRECT`/`INCORRECT`;
  `Score.metadata` is free-form and ends up in the log.
- **Separating failure modes**: "wrong chemistry", "invalid SMILES", "didn't use
  the format", and "ran out of tokens" all count as incorrect, but are counted
  separately. The smoke run found a real case: gemma4:e2b spent all 4096 tokens
  reasoning about *methanol* and never answered. That is a budget artefact,
  not a chemistry error, so `max_tokens` was raised to 8192 and `truncated` became its own status.
- **Strictness diagnostics**: `match_ignore_stereo` and `match_tautomer` record
  whether the answer would pass under looser rules. They don't change the score,
  but they show how much the headline number depends on scoring choices
  (e.g. sildenafil's lactam vs hydroxy tautomer are the same drug, but their SMILES differ).
- **Relative tolerance**: `|x - ref| / ref <= 0.005`, so the allowed error
  scales with molecule size (~0.09 g/mol for water, ~1.9 g/mol for cholesterol).

**Study:** `extract_answer`, `missing_answer_score`, `canonical_smiles`,
`canonical_tautomer_smiles`, `parse_number`, and `tests/test_scorers.py` (each
edge case is one small test).

**Interviewer questions**
1. Give three different SMILES strings that your scorer must treat as equal, and one pair it must treat as different.
2. Why track "unparseable" and "truncated" separately instead of just reporting accuracy?
3. `parse_number` takes the *first* number in the answer. When could that silently mark a correct answer wrong?

---

## 4. Tasks (`chem_eval/tasks.py`)

**What it does.** Two `@task` functions that pair each dataset with
`generate()` (one model call, no tools) and its scorer.

**Key concepts**
- **Inspect `Task` = dataset + solver + scorer**. Run with `inspect eval chem_eval/tasks.py --model <provider/model>`.
- **`GenerateConfig`**: `temperature=0` for repeatability; a generous `max_tokens` for reasoning models.
- **Provider prefixes**: `ollama/qwen3:8b`, `anthropic/claude-sonnet-5-5`. Keys come from env vars (`ANTHROPIC_API_KEY`), never the repo.
- **Eval logs**: every run writes a `.eval` file in `logs/` with every prompt,
  output, score and token count. This is the source of truth for the README numbers.

**Study:** `tasks.py`, then open a log with `uv run inspect view`.

**Interviewer questions**
1. Temperature 0 isn't fully deterministic on most providers. Why, and how would you measure run-to-run variance?
2. Why is a single `generate()` call a fair baseline, and what would adding tools (e.g. RDKit as a tool) measure instead?
3. How would you make sure a model is not being cut off by `max_tokens` in a way that biases the comparison?

---

## 5. Analysis (`analysis/report.py`)

**What it does.** Reads the latest full, successful log for each (model, task)
pair and writes `results/report.md`: accuracy with Wilson 95% CIs (overall, per
task, per difficulty), an outcome breakdown, strictness diagnostics, a paired
bootstrap CI for the top-two difference, and tokens/latency.

**Key concepts**
- **Wilson score interval**: a CI for a proportion that behaves well at small n
  and near 0% / 100% (the naive "p ± 1.96·SE" interval can go below 0 or collapse
  to zero width at 30/30).
- **Paired bootstrap**: both models answer the *same* items, so you resample
  *items* (keeping both models' scores together), recompute the accuracy
  difference 10,000 times, and take the 2.5th/97.5th percentiles. Pairing
  removes item-difficulty variance from the comparison.
- **Traceability**: the report lists the exact log files it used, skips
  `--limit` smoke runs and failed runs, and every README number is copied from it.

- **Format compliance vs capability (found in the real run).** qwen3:8b often
  wrote `\boxed{206.31}` instead of `<answer>206.31</answer>`. Strict scoring gave
  Gemma a +15% lead with a CI excluding 0. Applying the same check to `\boxed{}`
  shrank that to +5% with a CI including 0. The strict rule stays the headline
  (it was fixed before the runs); the lenient number is a labelled diagnostic.
  This is the main "would this result mislead someone?" lesson of v1.
- **Rescoring without regenerating.** `inspect score <log> --scorer file.py@name`
  re-applies a scorer to saved outputs, so a scoring change never needs new model calls.

**Study:** `wilson_ci`, `paired_bootstrap_diff`, `load_results`, `lenient_table`,
`tests/test_stats.py`, `chem_eval/scorers.py::score_output`.

**Interviewer questions**
1. Why use Wilson instead of the normal-approximation interval here? What happens to each at 30/30?
2. Why is a *paired* bootstrap better than comparing two independent CIs for the difference between models?
3. If the 95% CI for the difference includes 0, what can and can't you conclude?
4. You added the lenient metric *after* seeing results. Why is that acceptable as a diagnostic but not as a replacement headline?
