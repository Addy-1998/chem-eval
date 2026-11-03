"""Read Inspect logs and write a markdown results report with confidence intervals.

Usage:  uv run python analysis/report.py [--log-dir logs] [--out results/report.md]

Rules (so every number is traceable to a log):
  * only logs with status "success" that cover the FULL dataset (no --limit);
  * if a model/task pair was run more than once, the most recent log is used;
  * the report lists the exact log files it read.
"""

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
from inspect_ai.log import list_eval_logs, read_eval_log

from chem_eval.dataset import load_molecules

DIFFICULTIES = ("easy", "medium", "hard")
N_BOOTSTRAP = 10_000
SEED = 0


@dataclass
class SampleResult:
    model: str
    task: str
    sample_id: str
    difficulty: str
    correct: bool
    status: str
    match_ignore_stereo: bool | None
    match_tautomer: bool | None
    lenient_correct: bool
    latency_s: float | None


# --- statistics -------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion k/n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED
) -> tuple[float, float, float]:
    """Mean(a) - mean(b) and its 95% percentile CI, resampling items in pairs.

    a[i] and b[i] are the two models' 0/1 scores on the SAME item i, so we
    resample item indices (not the two models independently). This keeps the
    pairing and gives a tighter, correct CI when the models share hard items.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


# --- loading ----------------------------------------------------------------

def load_results(log_dir: str) -> tuple[list[SampleResult], list[str], dict]:
    """Per-sample results from the latest full, successful log per (model, task)."""
    n_full = len(load_molecules())
    latest: dict[tuple[str, str], object] = {}
    for info in list_eval_logs(log_dir, recursive=False):
        header = read_eval_log(info, header_only=True)
        if header.status != "success" or header.results is None:
            continue
        if header.results.total_samples != n_full:  # skip --limit smoke runs
            continue
        key = (header.eval.model, header.eval.task.split("/")[-1])
        if key not in latest or header.eval.created > latest[key].eval.created:
            latest[key] = header

    results: list[SampleResult] = []
    usage: dict = {}
    used_logs = []
    for (model, task), header in sorted(latest.items()):
        log = read_eval_log(header.location)
        used_logs.append(Path(header.location).name)
        usage[(model, task)] = log.stats.model_usage.get(model)
        for s in log.samples:
            score = next(iter(s.scores.values()))
            meta = score.metadata or {}
            results.append(
                SampleResult(
                    model=model,
                    task=task,
                    sample_id=str(s.id),
                    difficulty=s.metadata["difficulty"],
                    correct=score.value == "C",
                    status=meta.get("status", "unknown"),
                    match_ignore_stereo=meta.get("match_ignore_stereo"),
                    match_tautomer=meta.get("match_tautomer"),
                    lenient_correct=bool(meta.get("lenient_correct", score.value == "C")),
                    latency_s=s.total_time,
                )
            )
    return results, used_logs, usage


# --- formatting -------------------------------------------------------------

def fmt_acc(rows: list[SampleResult]) -> str:
    k, n = sum(r.correct for r in rows), len(rows)
    if n == 0:
        return "-"
    lo, hi = wilson_ci(k, n)
    return f"{k/n:.0%} ({k}/{n}) [{lo:.0%}, {hi:.0%}]"


def accuracy_table(results: list[SampleResult], models: list[str], tasks: list[str]) -> list[str]:
    lines = ["| model | task | overall | easy | medium | hard |", "|---|---|---|---|---|---|"]
    for m in models:
        for t in [*tasks, "both"]:
            rows = [r for r in results if r.model == m and (t == "both" or r.task == t)]
            cells = [fmt_acc(rows)] + [
                fmt_acc([r for r in rows if r.difficulty == d]) for d in DIFFICULTIES
            ]
            label = f"**{t}**" if t == "both" else t
            lines.append(f"| {m} | {label} | " + " | ".join(cells) + " |")
    return lines


def status_table(results: list[SampleResult], models: list[str], tasks: list[str]) -> list[str]:
    statuses = ["correct", "wrong", "unparseable", "no_answer", "truncated"]
    lines = ["| model | task | " + " | ".join(statuses) + " |", "|---|---|" + "---|" * len(statuses)]
    for m in models:
        for t in tasks:
            c = Counter(r.status for r in results if r.model == m and r.task == t)
            lines.append(f"| {m} | {t} | " + " | ".join(str(c[s]) for s in statuses) + " |")
    return lines


def smiles_diagnostics(results: list[SampleResult], models: list[str]) -> list[str]:
    lines = [
        "| model | strict (reported) | if stereo ignored | if tautomers merged |",
        "|---|---|---|---|",
    ]
    for m in models:
        rows = [r for r in results if r.model == m and r.task == "name_to_smiles"]
        if not rows:
            continue
        n = len(rows)
        stereo = sum(bool(r.match_ignore_stereo) for r in rows)
        taut = sum(bool(r.match_tautomer) for r in rows)
        lines.append(
            f"| {m} | {sum(r.correct for r in rows)}/{n} | {stereo}/{n} | {taut}/{n} |"
        )
    return lines


def lenient_table(results: list[SampleResult], models: list[str], tasks: list[str]) -> list[str]:
    lines = [
        r"| model | task | strict `<answer>` (reported) | lenient: `\boxed{}` fallback |",
        "|---|---|---|---|",
    ]
    for m in models:
        for t in tasks:
            rows = [r for r in results if r.model == m and r.task == t]
            lenient = [r for r in rows if r.lenient_correct]
            lo, hi = wilson_ci(len(lenient), len(rows))
            lines.append(
                f"| {m} | {t} | {fmt_acc(rows)} | "
                f"{len(lenient)/len(rows):.0%} ({len(lenient)}/{len(rows)}) [{lo:.0%}, {hi:.0%}] |"
            )
    return lines


def comparison(
    results: list[SampleResult], models: list[str], lenient: bool = False
) -> list[str]:
    """Paired bootstrap for the top two models over all items they share."""
    by_model: dict[str, dict[tuple[str, str], bool]] = defaultdict(dict)
    for r in results:
        by_model[r.model][(r.task, r.sample_id)] = r.lenient_correct if lenient else r.correct
    ranked = sorted(models, key=lambda m: -np.mean(list(by_model[m].values())))
    if len(ranked) < 2:
        return ["Only one model has a complete run; no comparison possible."]
    top, second = ranked[:2]
    items = sorted(set(by_model[top]) & set(by_model[second]))
    a = np.array([by_model[top][i] for i in items], dtype=float)
    b = np.array([by_model[second][i] for i in items], dtype=float)
    diff, lo, hi = paired_bootstrap_diff(a, b)
    verdict = (
        "The 95% CI excludes 0, so the difference is unlikely to be noise at this sample size."
        if lo > 0 or hi < 0
        else "The 95% CI includes 0, so this run cannot distinguish the two models."
    )
    return [
        f"Top two models by overall accuracy: **{top}** vs **{second}**, "
        f"over {len(items)} paired items (both tasks).",
        "",
        f"Accuracy difference = **{diff:+.1%}**, paired bootstrap 95% CI "
        f"[{lo:+.1%}, {hi:+.1%}] ({N_BOOTSTRAP:,} resamples, seed {SEED}).",
        "",
        verdict,
    ]


def usage_table(results: list[SampleResult], usage: dict) -> list[str]:
    lines = [
        "| model | task | input tok | output tok | median latency/sample (s) | cost |",
        "|---|---|---|---|---|---|",
    ]
    for (m, t), u in sorted(usage.items()):
        lats = [r.latency_s for r in results if r.model == m and r.task == t and r.latency_s]
        cost = getattr(u, "total_cost", None) if u else None
        lines.append(
            f"| {m} | {t} | {u.input_tokens if u else '-'} | {u.output_tokens if u else '-'} | "
            f"{median(lats):.1f} | {f'${cost:.4f}' if cost else 'n/a'} |"
        )
    return lines


def build_report(log_dir: str) -> str:
    results, used_logs, usage = load_results(log_dir)
    if not results:
        return "No complete, successful eval logs found.\n"
    models = sorted({r.model for r in results})
    tasks = sorted({r.task for r in results})
    out = [
        "# chem-eval results",
        "",
        "Generated by `analysis/report.py` from Inspect logs. Cells are "
        "accuracy (correct/n) [Wilson 95% CI].",
        "",
        "## Accuracy",
        "",
        *accuracy_table(results, models, tasks),
        "",
        "## Outcome breakdown",
        "",
        "`unparseable` = answer tag present but not a valid SMILES / number; "
        "`no_answer` = no `<answer>` tag; `truncated` = hit max_tokens before answering.",
        "",
        *status_table(results, models, tasks),
        "",
        "## name_to_smiles: sensitivity to scoring strictness",
        "",
        "The reported score requires an exact canonical isomeric SMILES match. "
        "These columns show how many answers would match under looser rules "
        "(diagnostic only; they are not the headline numbers).",
        "",
        *smiles_diagnostics(results, models),
        "",
        "## Format-compliance sensitivity",
        "",
        "Headline scores only accept the requested `<answer>` tag. Some models "
        r"ignored it and wrote LaTeX `\boxed{...}` instead. The lenient column "
        r"applies the *same* correctness check to the last `\boxed{}` when the tag "
        "is missing. It is a diagnostic, added after inspecting the outputs, so it is "
        "not the headline number.",
        "",
        *lenient_table(results, models, tasks),
        "",
        "## Top-two comparison",
        "",
        "**Strict extraction (reported):**",
        "",
        *comparison(results, models),
        "",
        "**Lenient extraction (diagnostic):**",
        "",
        *comparison(results, models, lenient=True),
        "",
        "## Tokens, latency, cost",
        "",
        *usage_table(results, usage),
        "",
        "Latency is Inspect's per-sample wall-clock time and depends on hardware, "
        "concurrency and provider load. Cost is shown only if Inspect recorded it.",
        "",
        "## Logs used",
        "",
        *[f"- `{name}`" for name in used_logs],
        "",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--out", default="results/report.md")
    args = parser.parse_args()
    report = build_report(args.log_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
