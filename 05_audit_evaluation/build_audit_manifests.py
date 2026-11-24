"""Build independent Audit-DFA manifests without training predictive models."""

import argparse
import os

import numpy as np

from audit_constraints import (
    SUPPORTED_TEMPLATES,
    attach_external_rule_statistics,
    complete_case_trace_fingerprint,
    compile_audit_rules,
    discover_audit_rules,
    load_external_audit_rules,
    merge_audit_rules,
    persist_audit_manifest,
)
from data_loader import load_public_dataset


PAPER_DATASETS = [
    "sepsis_cases_1",
    "sepsis_cases_2",
    "sepsis_cases_4",
    "BPIC15_2_f2",
    "BPIC15_3_f2",
    "BPIC15_4_f2",
    "BPIC15_5_f2",
]


def _case_key(sample, fallback_index):
    for key in ("source_case_id", "case_id", "Case ID", "case:concept:name"):
        if sample.get(key) is not None:
            return str(sample[key])
    return "case-{}".format(fallback_index)


def split_train_audit_cases(samples, split_seed=42, audit_seed=2026,
                            audit_fraction=0.25):
    all_keys = [
        _case_key(sample, index) for index, sample in enumerate(samples)
    ]
    if len(set(all_keys)) != len(all_keys):
        raise ValueError("Dataset contains duplicate complete-case identifiers.")
    split_rng = np.random.default_rng(int(split_seed))
    shuffled = [samples[index] for index in split_rng.permutation(len(samples))]
    train_end = int(0.7 * len(shuffled))
    train_cases = shuffled[:train_end]
    keys = sorted(_case_key(sample, index) for index, sample in enumerate(train_cases))
    if len(keys) < 2:
        return [], set()
    audit_rng = np.random.default_rng(int(audit_seed))
    permuted = list(np.asarray(keys, dtype=object)[audit_rng.permutation(len(keys))])
    count = int(round(len(permuted) * float(audit_fraction)))
    count = min(max(count, 1), len(permuted) - 1)
    audit_ids = set(str(value) for value in permuted[:count])
    audit_cases = [
        sample for index, sample in enumerate(train_cases)
        if _case_key(sample, index) in audit_ids
    ]
    return audit_cases, audit_ids


def build_dataset_manifest(dataset, args):
    expected_paths = [
        os.path.join(args.output_dir, dataset + "_audit_rules.json"),
        os.path.join(args.output_dir, dataset + "_audit_rules.csv"),
    ]
    existing = [path for path in expected_paths if os.path.exists(path)]
    if existing and not args.overwrite_manifests:
        raise FileExistsError(
            "Refusing to overwrite existing audit manifest(s): {}. "
            "Use a new output directory or pass --overwrite_manifests."
            .format(existing)
        )
    samples, _ = load_public_dataset(
        dataset,
        data_path=args.data_path,
        bp17_anti_leak=args.bp17_anti_leak,
    )
    audit_cases, audit_ids = split_train_audit_cases(
        samples,
        split_seed=args.split_seed,
        audit_seed=args.audit_seed,
        audit_fraction=args.audit_fraction,
    )
    discovered = []
    discovery_diag = {}
    if args.audit_rule_source in ("heldout", "hybrid"):
        discovered, discovery_diag = discover_audit_rules(
            audit_cases,
            templates=args.audit_templates,
            min_activation_cases=args.audit_min_activation_cases,
            min_activation_fraction=args.audit_min_activation_fraction,
            min_confidence=args.audit_min_confidence,
            max_activities=args.audit_max_activities,
            max_constraints=args.audit_max_constraints,
            # Search uses existential positive subsequences; these pre-specified
            # global templates are a disjoint rule family by construction.
            search_formulas=[],
        )

    external = []
    if args.audit_rule_source in ("external", "hybrid"):
        external = load_external_audit_rules(args.audit_rule_file, dataset)
        external = attach_external_rule_statistics(external, audit_cases)
    selected, overlaps = merge_audit_rules(
        discovered,
        external,
        search_formulas=[],
        max_constraints=args.audit_max_constraints,
    )
    compiled, compile_diag = compile_audit_rules(
        selected, dfa_max_states=args.audit_dfa_max_states
    )
    if not selected:
        raise RuntimeError(
            "No independent audit rule met the pre-specified thresholds for {}."
            .format(dataset)
        )
    if compile_diag.get("failed", 0) or len(compiled) != len(selected):
        raise RuntimeError(
            "Audit-DFA compilation must be all-or-nothing for {}: {}".format(
                dataset, compile_diag.get("failures", [])
            )
        )
    diagnostics = dict(discovery_diag)
    diagnostics.update({
        "dataset": dataset,
        "dataset_case_count": len(samples),
        "complete_dataset_trace_sha256": complete_case_trace_fingerprint(samples),
        "source_mode": args.audit_rule_source,
        "split_seed": args.split_seed,
        "audit_seed": args.audit_seed,
        "audit_fraction": args.audit_fraction,
        "audit_case_count": len(audit_cases),
        "audit_case_ids": sorted(audit_ids),
        "case_grain": "complete_trace",
        "selection_uses_outcome_label": False,
        "search_audit_template_families_disjoint": True,
        "external_rule_count": len(external),
        "selected_rule_count": len(selected),
        "compiled_rule_count": len(compiled),
        "formula_overlap_count": int(
            discovery_diag.get("formula_overlap_count", 0)
        ) + int(overlaps),
        "compile": compile_diag,
        "builder": "build_audit_manifests.py",
    })
    paths = persist_audit_manifest(
        compiled, diagnostics, args.output_dir, dataset
    )
    print(
        "{}: audit cases={}, selected={}, compiled={}, manifest={}".format(
            dataset, len(audit_cases), len(selected), len(compiled), paths["json"]
        )
    )
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Build complete-case independent Audit-DFA manifests only."
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--audit_seed", type=int, default=2026)
    parser.add_argument("--audit_fraction", type=float, default=0.25)
    parser.add_argument(
        "--audit_rule_source",
        choices=["heldout", "external", "hybrid"],
        default="heldout",
    )
    parser.add_argument("--audit_rule_file", default=None)
    parser.add_argument(
        "--audit_templates",
        nargs="+",
        choices=list(SUPPORTED_TEMPLATES),
        default=list(SUPPORTED_TEMPLATES),
    )
    parser.add_argument("--audit_min_activation_cases", type=int, default=10)
    parser.add_argument("--audit_min_activation_fraction", type=float, default=0.02)
    parser.add_argument("--audit_min_confidence", type=float, default=0.80)
    parser.add_argument("--audit_max_activities", type=int, default=40)
    parser.add_argument("--audit_max_constraints", type=int, default=20)
    parser.add_argument("--audit_dfa_max_states", type=int, default=500)
    parser.add_argument("--output_dir", default="./audit_manifests")
    parser.add_argument(
        "--overwrite_manifests", action="store_true",
        help="Explicitly allow replacement of same-dataset manifest files.",
    )
    parser.add_argument(
        "--bp17_anti_leak", dest="bp17_anti_leak",
        action="store_true", default=True,
    )
    parser.add_argument(
        "--no_bp17_anti_leak", dest="bp17_anti_leak",
        action="store_false",
    )
    args = parser.parse_args()
    if not 0.0 < args.audit_fraction < 1.0:
        parser.error("--audit_fraction must be strictly between 0 and 1")
    if not 0.0 <= args.audit_min_confidence <= 1.0:
        parser.error("--audit_min_confidence must be in [0, 1]")
    if not 0.0 <= args.audit_min_activation_fraction <= 1.0:
        parser.error("--audit_min_activation_fraction must be in [0, 1]")
    if args.audit_min_activation_cases < 1 or args.audit_max_constraints < 1:
        parser.error("audit activation and constraint counts must be positive")
    if args.audit_rule_source in ("external", "hybrid") and not args.audit_rule_file:
        parser.error("--audit_rule_file is required for external or hybrid source")
    datasets = args.datasets or PAPER_DATASETS
    if args.data_path and len(datasets) != 1:
        parser.error("--data_path can only be used with exactly one dataset")
    os.makedirs(args.output_dir, exist_ok=True)
    for dataset in datasets:
        build_dataset_manifest(dataset, args)


if __name__ == "__main__":
    main()
