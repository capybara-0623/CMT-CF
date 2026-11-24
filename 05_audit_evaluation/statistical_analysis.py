"""Paired, hierarchy-aware analysis for prediction and counterfactual experiments.

Analysis units are fixed test prefixes nested within process cases.  Bootstrap
resampling follows dataset family -> dataset -> training seed -> case -> prefix,
which avoids treating related logs or prefixes of one case as independent.
"""
import argparse
import glob
import os
from collections import namedtuple

import numpy as np
import pandas as pd


Comparison = namedtuple(
    "Comparison",
    [
        "name", "variant_a", "variant_b", "metric",
        "higher_is_better", "source",
    ],
)


PRIMARY_COMPARISONS = (
    Comparison(
        "Independent activated-rule audit compliance",
        "full",
        "no_dfa",
        "audit_score",
        True,
        "counterfactual",
    ),
    Comparison(
        "Multi-point probability shift",
        "multi_point_cf",
        "single_point_cf",
        "prob_shift",
        True,
        "counterfactual",
    ),
    Comparison(
        "Pareto validity",
        "pareto_ranking_cf",
        "score_ranking_cf",
        "cf_valid",
        True,
        "counterfactual",
    ),
    Comparison(
        "Pattern-feature next-activity accuracy",
        "full",
        "no_pattern",
        "next_act_correct",
        True,
        "prediction",
    ),
    Comparison(
        "Calibrated hard-constraint candidate coverage",
        "full",
        "fixed_constraint_cf",
        "candidate_returned",
        True,
        "counterfactual",
    ),
    Comparison(
        "Hard-DFA incremental audit compliance",
        "full",
        "soft_constraint_cf",
        "audit_score",
        True,
        "counterfactual",
    ),
    Comparison(
        "Soft-ranking probability shift",
        "soft_constraint_cf",
        "no_dfa",
        "prob_shift",
        True,
        "counterfactual",
    ),
    Comparison(
        "UWL next-time absolute error",
        "full",
        "no_uwl",
        "next_time_abs_error",
        False,
        "prediction",
    ),
)


def infer_dataset_family(dataset_name: str) -> str:
    name = str(dataset_name).lower()
    if name.startswith("sepsis"):
        return "Sepsis"
    if name.startswith("bpic15"):
        return "BPIC15"
    if name.startswith("bpic17"):
        return "BPIC17"
    return str(dataset_name).split("_")[0]


def load_cf_details(results_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(results_dir, "**", "cf_details_*.csv"), recursive=True))
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"skip unreadable {path}: {exc}")
            continue
        if {
            "dataset", "variant", "seed", "protocol_version",
            "case_id", "sample_idx",
        }.issubset(frame.columns):
            frame["source_file"] = path
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "No new-protocol cf_details CSV files were found. Re-run experiments "
            "after the held-out audit/manifest changes."
        )
    details = pd.concat(frames, ignore_index=True)
    if "dataset_family" not in details.columns:
        details["dataset_family"] = details["dataset"].map(infer_dataset_family)
    protocol = pd.to_numeric(details["protocol_version"], errors="coerce")
    if protocol.isna().any() or set(protocol.astype(int).unique()) != {2}:
        raise ValueError("Counterfactual inputs must contain only protocol-version 2 rows.")
    dedupe_keys = ["dataset", "variant", "seed", "case_id", "sample_idx"]
    if "cf_rank" in details.columns:
        dedupe_keys.append("cf_rank")
    duplicate = details.duplicated(subset=dedupe_keys, keep=False)
    if duplicate.any():
        examples = details.loc[duplicate, dedupe_keys + ["source_file"]].head(10)
        raise ValueError(
            "Duplicate counterfactual observation keys indicate mixed/stale inputs:\n{}"
            .format(examples.to_string(index=False))
        )
    return details


def validate_audit_provenance(details: pd.DataFrame) -> None:
    required = {
        "audit_rescore_schema_version", "audit_manifest_schema_version",
        "audit_manifest_sha256", "audit_rule_set_sha256",
        "audit_source_file_sha256", "audit_score",
    }
    missing = sorted(required.difference(details.columns))
    if missing:
        raise ValueError(
            "Independent audit comparison lacks provenance columns: {}. "
            "Run audit_rescore.py with the current code."
            .format(missing)
        )
    rescore_version = pd.to_numeric(
        details["audit_rescore_schema_version"], errors="coerce"
    )
    if rescore_version.isna().any() or set(rescore_version.astype(int).unique()) != {2}:
        raise ValueError("Independent audit rows do not all use rescore schema 2.")
    manifest_version = pd.to_numeric(
        details["audit_manifest_schema_version"], errors="coerce"
    )
    if manifest_version.isna().any() or set(manifest_version.astype(int).unique()) != {2}:
        raise ValueError("Independent audit rows do not all use manifest schema 2.")
    for dataset, group in details.groupby("dataset"):
        for field in ("audit_manifest_sha256", "audit_rule_set_sha256"):
            normalized = group[field].fillna("").astype(str).str.strip()
            values = set(normalized)
            if "" in values or len(values) != 1:
                raise ValueError(
                    "Dataset {!r} has {} distinct/non-empty {} values; results are mixed."
                    .format(dataset, len(values), field)
                )


def load_prediction_details(results_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(
        os.path.join(results_dir, "**", "predictions_*.csv"),
        recursive=True,
    ))
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"skip unreadable {path}: {exc}")
            continue
        if {
            "dataset", "variant", "seed", "protocol_version",
            "case_id", "sample_idx",
        }.issubset(frame.columns):
            frame["source_file"] = path
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            "No new-protocol prediction CSV files were found. Re-run experiments "
            "with seed-tagged prediction outputs."
        )
    details = pd.concat(frames, ignore_index=True)
    if "dataset_family" not in details.columns:
        details["dataset_family"] = details["dataset"].map(infer_dataset_family)
    protocol = pd.to_numeric(details["protocol_version"], errors="coerce")
    if protocol.isna().any() or set(protocol.astype(int).unique()) != {2}:
        raise ValueError("Prediction inputs must contain only protocol-version 2 rows.")
    keys = ["dataset", "variant", "seed", "case_id", "sample_idx"]
    duplicate = details.duplicated(subset=keys, keep=False)
    if duplicate.any():
        examples = details.loc[duplicate, keys + ["source_file"]].head(10)
        raise ValueError(
            "Duplicate prediction observation keys indicate mixed/stale inputs:\n{}"
            .format(examples.to_string(index=False))
        )
    return details


def aggregate_per_sample(details: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in details.columns:
        return pd.DataFrame()
    work = details[[
        "dataset_family", "dataset", "variant", "seed",
        "case_id", "sample_idx", metric,
    ]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    if work.empty:
        return work
    return (
        work.groupby(
            [
                "dataset_family", "dataset", "variant", "seed",
                "case_id", "sample_idx",
            ],
            as_index=False,
        )[metric]
        .mean()
    )


def paired_differences(details: pd.DataFrame, comparison: Comparison) -> pd.DataFrame:
    sample_metrics = aggregate_per_sample(details, comparison.metric)
    if sample_metrics.empty:
        return sample_metrics
    keys = ["dataset_family", "dataset", "seed", "case_id", "sample_idx"]
    pivot = sample_metrics.pivot_table(
        index=keys,
        columns="variant",
        values=comparison.metric,
        aggfunc="mean",
    ).reset_index()
    if comparison.variant_a not in pivot or comparison.variant_b not in pivot:
        return pd.DataFrame()
    paired = pivot.dropna(subset=[comparison.variant_a, comparison.variant_b]).copy()
    sign = 1.0 if comparison.higher_is_better else -1.0
    paired["difference"] = sign * (
        paired[comparison.variant_a] - paired[comparison.variant_b]
    )
    return paired


def eligible_pair_count(details: pd.DataFrame, comparison: Comparison) -> int:
    """Count fixed-manifest prefixes present for both compared variants."""
    keys = ["dataset_family", "dataset", "seed", "case_id", "sample_idx"]
    work = details[details["variant"].isin([
        comparison.variant_a, comparison.variant_b,
    ])][keys + ["variant"]].drop_duplicates()
    if work.empty:
        return 0
    present = work.groupby(keys)["variant"].nunique()
    return int((present == 2).sum())


def dataset_seed_pair_coverage(details: pd.DataFrame, comparison: Comparison,
                               paired: pd.DataFrame):
    """Return the minimum coverage and panel sizes across dataset-seed cells."""
    keys = ["dataset_family", "dataset", "seed", "case_id", "sample_idx"]
    work = details[details["variant"].isin([
        comparison.variant_a, comparison.variant_b,
    ])][keys + ["variant"]].drop_duplicates()
    present = work.groupby(keys)["variant"].nunique()
    eligible = present[present == 2].reset_index(name="variant_count")
    denominators = (
        eligible.groupby(["dataset", "seed"])
        .size()
        .reset_index(name="eligible")
    )
    numerators = (
        paired.groupby(["dataset", "seed"])
        .size()
        .reset_index(name="paired")
    )
    if denominators.empty:
        return float("nan"), 0, int(len(numerators))
    cells = denominators.merge(
        numerators, on=["dataset", "seed"], how="left"
    )
    cells["paired"] = cells["paired"].fillna(0)
    cells["coverage"] = cells["paired"] / cells["eligible"]
    return (
        float(cells["coverage"].min()),
        int(len(denominators)),
        int(len(numerators)),
    )


def hierarchical_bootstrap_mean(paired: pd.DataFrame, n_bootstrap=5000, seed=2026):
    """Bootstrap an equal-weight nested mean without flattening case prefixes."""
    if paired.empty:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(int(seed))
    families = paired["dataset_family"].drop_duplicates().tolist()
    estimates = np.empty(int(n_bootstrap), dtype=float)

    for draw in range(int(n_bootstrap)):
        family_estimates = []
        sampled_families = rng.choice(families, size=len(families), replace=True)
        for family in sampled_families:
            family_rows = paired[paired["dataset_family"] == family]
            datasets = family_rows["dataset"].drop_duplicates().tolist()
            sampled_datasets = rng.choice(datasets, size=len(datasets), replace=True)
            dataset_estimates = []
            for dataset in sampled_datasets:
                dataset_rows = family_rows[family_rows["dataset"] == dataset]
                seeds = dataset_rows["seed"].drop_duplicates().tolist()
                sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
                seed_estimates = []
                for run_seed in sampled_seeds:
                    run_rows = dataset_rows[dataset_rows["seed"] == run_seed]
                    cases = run_rows["case_id"].drop_duplicates().tolist()
                    sampled_cases = rng.choice(cases, size=len(cases), replace=True)
                    case_estimates = []
                    for case_id in sampled_cases:
                        case_rows = run_rows[run_rows["case_id"] == case_id]
                        values = case_rows["difference"].to_numpy(dtype=float)
                        if len(values):
                            sampled_prefixes = rng.choice(
                                values, size=len(values), replace=True
                            )
                            case_estimates.append(float(np.mean(sampled_prefixes)))
                    if case_estimates:
                        seed_estimates.append(float(np.mean(case_estimates)))
                if seed_estimates:
                    dataset_estimates.append(float(np.mean(seed_estimates)))
            if dataset_estimates:
                family_estimates.append(float(np.mean(dataset_estimates)))
        estimates[draw] = (
            float(np.mean(family_estimates))
            if family_estimates else float("nan")
        )
    return estimates[np.isfinite(estimates)]


def hierarchical_observed_mean(paired: pd.DataFrame) -> float:
    """Use the same equal-weight estimand as the hierarchical bootstrap."""
    level = paired.groupby(
        ["dataset_family", "dataset", "seed", "case_id"],
        as_index=False,
    )["difference"].mean()
    level = level.groupby(
        ["dataset_family", "dataset", "seed"], as_index=False,
    )["difference"].mean()
    level = level.groupby(
        ["dataset_family", "dataset"], as_index=False,
    )["difference"].mean()
    level = level.groupby(
        "dataset_family", as_index=False,
    )["difference"].mean()
    return float(level["difference"].mean())


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if not len(finite_indices):
        return adjusted
    order = finite_indices[np.argsort(p_values[finite_indices])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def analyse(results_dir, output_dir, n_bootstrap=5000, seed=2026,
            cf_results_dir=None, prediction_results_dir=None,
            min_pair_coverage=0.80, min_datasets=7,
            min_training_seeds=3):
    sources = {}
    cf_results_dir = cf_results_dir or results_dir
    prediction_results_dir = prediction_results_dir or results_dir
    try:
        sources["counterfactual"] = load_cf_details(cf_results_dir)
    except FileNotFoundError as exc:
        print(exc)
    try:
        sources["prediction"] = load_prediction_details(prediction_results_dir)
    except FileNotFoundError as exc:
        print(exc)

    rows = []
    for comparison in PRIMARY_COMPARISONS:
        details = sources.get(comparison.source)
        if details is None:
            print(f"skip {comparison.name}: {comparison.source} details unavailable")
            continue
        if comparison.metric == "audit_score":
            validate_audit_provenance(details)
        paired = paired_differences(details, comparison)
        if paired.empty:
            print(f"skip {comparison.name}: paired metric unavailable")
            continue
        bootstrap = hierarchical_bootstrap_mean(
            paired,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        if bootstrap.size == 0:
            continue
        eligible = eligible_pair_count(details, comparison)
        observed = hierarchical_observed_mean(paired)
        case_effects = paired.groupby(
            ["dataset_family", "dataset", "seed", "case_id"],
            as_index=False,
        )["difference"].mean()
        paired_cases = case_effects[["dataset", "case_id"]].drop_duplicates()
        ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
        lower_tail = (float(np.sum(bootstrap <= 0.0)) + 1.0) / (bootstrap.size + 1.0)
        upper_tail = (float(np.sum(bootstrap >= 0.0)) + 1.0) / (bootstrap.size + 1.0)
        p_two_sided = min(
            1.0,
            2.0 * min(lower_tail, upper_tail),
        )
        pair_coverage = (
            float(len(paired) / eligible) if eligible > 0 else float("nan")
        )
        (min_cell_pair_coverage, eligible_cell_count,
         paired_cell_count) = dataset_seed_pair_coverage(
            details, comparison, paired
        )
        dataset_count = int(paired["dataset"].nunique())
        training_seed_count = int(paired["seed"].nunique())
        quality_ready = bool(
            np.isfinite(pair_coverage)
            and pair_coverage >= float(min_pair_coverage)
            and np.isfinite(min_cell_pair_coverage)
            and min_cell_pair_coverage >= float(min_pair_coverage)
            and dataset_count >= int(min_datasets)
            and training_seed_count >= int(min_training_seeds)
            and eligible_cell_count >= int(min_datasets) * int(min_training_seeds)
            and paired_cell_count >= int(min_datasets) * int(min_training_seeds)
        )
        rows.append({
            "comparison": comparison.name,
            "variant_a": comparison.variant_a,
            "variant_b": comparison.variant_b,
            "metric": comparison.metric,
            "source": comparison.source,
            "better_direction": "A-B" if comparison.higher_is_better else "B-A",
            "mean_effect": observed,
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "bootstrap_p": p_two_sided,
            "bootstrap_replicates": int(n_bootstrap),
            "bootstrap_seed": int(seed),
            "paired_prefixes": int(len(paired)),
            "paired_case_runs": int(len(case_effects)),
            "paired_cases": int(len(paired_cases)),
            "eligible_prefixes": eligible,
            "pair_coverage": pair_coverage,
            "min_dataset_seed_pair_coverage": min_cell_pair_coverage,
            "eligible_dataset_seed_cells": eligible_cell_count,
            "paired_dataset_seed_cells": paired_cell_count,
            "datasets": dataset_count,
            "dataset_families": int(paired["dataset_family"].nunique()),
            "training_seeds": training_seed_count,
            "case_win_rate": float(np.mean(case_effects["difference"] > 0.0)),
            "quality_status": "primary_ready" if quality_ready else "descriptive_only",
            "quality_min_pair_coverage": float(min_pair_coverage),
            "quality_min_datasets": int(min_datasets),
            "quality_min_training_seeds": int(min_training_seeds),
        })

    if not rows:
        raise RuntimeError("No primary comparison had paired new-protocol observations.")
    result = pd.DataFrame(rows)
    eligible_p = result["bootstrap_p"].where(
        result["quality_status"] == "primary_ready", np.nan
    )
    result["holm_p"] = holm_adjust(eligible_p.to_numpy())
    result["significant_0_05"] = (
        result["quality_status"].eq("primary_ready") & result["holm_p"].lt(0.05)
    )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "paired_hierarchical_bootstrap.csv")
    result.to_csv(csv_path, index=False)

    latex_path = os.path.join(output_dir, "paired_hierarchical_bootstrap.tex")
    latex = result[
        [
            "comparison", "mean_effect", "ci95_low", "ci95_high",
            "paired_prefixes", "paired_cases", "pair_coverage",
            "dataset_families", "training_seeds", "holm_p",
            "quality_status",
        ]
    ].to_latex(index=False, float_format=lambda value: f"{value:.4f}")
    with open(latex_path, "w", encoding="utf-8") as handle:
        handle.write(latex)
    print(result.to_string(index=False))
    print(f"saved {csv_path}")
    print(f"saved {latex_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--output_dir", default="./paper_tables")
    parser.add_argument(
        "--cf_results_dir", default=None,
        help="Optional directory containing independently re-scored CF details.",
    )
    parser.add_argument(
        "--prediction_results_dir", default=None,
        help="Optional directory containing prediction detail files.",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--min_pair_coverage", type=float, default=0.80,
        help="Minimum paired-prefix coverage required for primary inference.",
    )
    parser.add_argument(
        "--min_datasets", type=int, default=7,
        help="Minimum datasets required for a primary paper comparison.",
    )
    parser.add_argument(
        "--min_training_seeds", type=int, default=3,
        help="Minimum distinct training seeds required for primary inference.",
    )
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    output_path = os.path.abspath(args.output_dir)
    if os.path.isdir(output_path) and os.listdir(output_path):
        parser.error(
            "--output_dir must be new or empty to prevent stale-table mixing: {}"
            .format(output_path)
        )
    if not 0.0 < args.min_pair_coverage <= 1.0:
        parser.error("--min_pair_coverage must be in (0, 1]")
    if args.min_datasets < 1 or args.min_training_seeds < 1:
        parser.error("minimum dataset/seed counts must be positive")
    analyse(
        args.results_dir, args.output_dir, args.bootstrap, args.seed,
        cf_results_dir=args.cf_results_dir,
        prediction_results_dir=args.prediction_results_dir,
        min_pair_coverage=args.min_pair_coverage,
        min_datasets=args.min_datasets,
        min_training_seeds=args.min_training_seeds,
    )


if __name__ == "__main__":
    main()
