"""Fail-fast verification of paper-ready independent Audit-DFA outputs."""

import argparse
import csv
import glob
import hashlib
import json
import os

from audit_constraints import stable_rule_set_hash


PAPER_DATASETS = [
    "sepsis_cases_1",
    "sepsis_cases_2",
    "sepsis_cases_4",
    "BPIC15_2_f2",
    "BPIC15_3_f2",
    "BPIC15_4_f2",
    "BPIC15_5_f2",
]
RQ3_NAME = "Independent activated-rule audit compliance"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_seed(value):
    return str(int(float(value)))


def verify_manifests(manifest_dir, expected_datasets):
    paths = sorted(glob.glob(
        os.path.join(manifest_dir, "**", "*_audit_rules.json"),
        recursive=True,
    ))
    payloads = {}
    file_hashes = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        dataset = str(payload.get("dataset", "")).strip()
        if not dataset or dataset in payloads:
            raise ValueError(
                "Each dataset must have exactly one audit manifest; conflict at {}"
                .format(path)
            )
        if int(payload.get("schema_version", 0)) != 2:
            raise ValueError("Manifest schema must be 2: {}".format(path))
        rules = payload.get("rules", [])
        if not rules:
            raise ValueError("Manifest has no compiled rules: {}".format(path))
        if stable_rule_set_hash(rules) != payload.get("rule_set_sha256"):
            raise ValueError("Manifest rule-set hash mismatch: {}".format(path))
        diagnostics = payload.get("diagnostics", {}) or {}
        compile_diag = diagnostics.get("compile", {}) or {}
        if int(compile_diag.get("failed", -1)) != 0:
            raise ValueError("Manifest contains a failed rule compilation: {}".format(path))
        if int(compile_diag.get("compiled", -1)) != len(rules):
            raise ValueError("Manifest compiled-rule count mismatch: {}".format(path))
        if diagnostics.get("selection_uses_outcome_label") is not False:
            raise ValueError("Manifest does not prove label-free selection: {}".format(path))
        if diagnostics.get("case_grain") != "complete_trace":
            raise ValueError("Manifest is not complete-case based: {}".format(path))
        if int(diagnostics.get("formula_overlap_count", -1)) != 0:
            raise ValueError("Search/Audit formula overlap is not zero: {}".format(path))
        audit_ids = [str(value) for value in diagnostics.get("audit_case_ids", [])]
        if not audit_ids or len(audit_ids) != len(set(audit_ids)):
            raise ValueError("Manifest audit case IDs are missing/duplicated: {}".format(path))
        payloads[dataset] = payload
        file_hashes[dataset] = sha256_file(path)

    observed = set(payloads)
    expected = set(expected_datasets)
    if observed != expected:
        raise ValueError(
            "Manifest dataset panel mismatch; missing={}, extra={}"
            .format(sorted(expected - observed), sorted(observed - expected))
        )
    return payloads, file_hashes


def verify_rescore_summary(rescored_dir, expected_datasets, expected_seeds,
                           required_variants):
    path = os.path.join(rescored_dir, "audit_rescore_summary.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError("Missing audit_rescore_summary.csv: {}".format(path))
    rows = read_csv(path)
    indexed = {}
    for row in rows:
        key = (
            str(row.get("dataset", "")).strip(),
            str(row.get("variant", "")).strip(),
            canonical_seed(row.get("seed", "")),
        )
        if key in indexed:
            raise ValueError("Duplicate audit summary key: {}".format(key))
        indexed[key] = row
    for dataset in expected_datasets:
        for variant in required_variants:
            for seed in expected_seeds:
                key = (dataset, variant, str(seed))
                if key not in indexed:
                    raise ValueError("Missing audit summary cell: {}".format(key))
                row = indexed[key]
                if int(float(row.get("audit_rule_count", 0) or 0)) <= 0:
                    raise ValueError("No audit rule for summary cell: {}".format(key))
                if int(float(row.get("returned_query_count", 0) or 0)) <= 0:
                    raise ValueError("No returned CF query for summary cell: {}".format(key))
                if int(float(row.get("audit_applicable_query_count", 0) or 0)) <= 0:
                    raise ValueError("No applicable Audit-DFA query for summary cell: {}".format(key))
    return path


def verify_rescored_details(rescored_dir, expected_datasets, expected_seeds,
                            required_variants, manifest_hashes):
    paths = sorted(glob.glob(
        os.path.join(rescored_dir, "**", "cf_details_*.csv"), recursive=True
    ))
    if not paths:
        raise FileNotFoundError("No re-scored cf_details files in {}".format(rescored_dir))
    observed = set()
    for path in paths:
        rows = read_csv(path)
        if not rows:
            raise ValueError("Empty re-scored details file: {}".format(path))
        datasets = {str(row.get("dataset", "")).strip() for row in rows}
        variants = {str(row.get("variant", "")).strip() for row in rows}
        seeds = {canonical_seed(row.get("seed", "")) for row in rows}
        if len(datasets) != 1 or len(variants) != 1 or len(seeds) != 1:
            raise ValueError("Mixed dataset/variant/seed details file: {}".format(path))
        dataset = next(iter(datasets))
        variant = next(iter(variants))
        seed = next(iter(seeds))
        if dataset in expected_datasets and variant in required_variants and int(seed) in expected_seeds:
            observed.add((dataset, variant, seed))
        for row in rows:
            if str(row.get("audit_rescore_schema_version", "")).strip() != "2":
                raise ValueError("Wrong audit rescore schema in {}".format(path))
            if str(row.get("audit_manifest_schema_version", "")).strip() != "2":
                raise ValueError("Wrong audit manifest schema in {}".format(path))
            if row.get("audit_manifest_sha256") != manifest_hashes.get(dataset):
                raise ValueError("Manifest hash mismatch in re-scored row: {}".format(path))
            if not str(row.get("audit_source_file_sha256", "")).strip():
                raise ValueError("Missing source-file hash in re-scored row: {}".format(path))

    expected = {
        (dataset, variant, str(seed))
        for dataset in expected_datasets
        for variant in required_variants
        for seed in expected_seeds
    }
    if not expected.issubset(observed):
        raise ValueError(
            "Missing re-scored detail cells: {}".format(sorted(expected - observed))
        )


def verify_statistics(statistics_csv, expected_dataset_count,
                      expected_seed_count, min_pair_coverage):
    if not os.path.isfile(statistics_csv):
        raise FileNotFoundError("Missing statistical output: {}".format(statistics_csv))
    rows = read_csv(statistics_csv)
    rq3 = [row for row in rows if row.get("comparison") == RQ3_NAME]
    if len(rq3) != 1:
        raise ValueError("Expected exactly one RQ3 row, found {}".format(len(rq3)))
    row = rq3[0]
    if int(float(row.get("datasets", 0))) != int(expected_dataset_count):
        raise ValueError("RQ3 does not cover the complete dataset panel.")
    if int(float(row.get("training_seeds", 0))) != int(expected_seed_count):
        raise ValueError("RQ3 does not cover the complete seed panel.")
    expected_cells = int(expected_dataset_count) * int(expected_seed_count)
    if int(float(row.get("paired_dataset_seed_cells", 0))) != expected_cells:
        raise ValueError("RQ3 does not cover every dataset-seed cell.")
    min_cell_coverage = float(
        row.get("min_dataset_seed_pair_coverage", "nan")
    )
    if min_cell_coverage < float(min_pair_coverage):
        raise ValueError(
            "At least one dataset-seed cell is below the pair-coverage threshold."
        )
    coverage = float(row.get("pair_coverage", "nan"))
    if coverage < float(min_pair_coverage):
        raise ValueError(
            "RQ3 pair coverage {:.3f} is below the formal threshold {:.3f}."
            .format(coverage, min_pair_coverage)
        )
    if row.get("quality_status") != "primary_ready":
        raise ValueError(
            "RQ3 is descriptive_only, so it must not be reported as a primary inference."
        )
    if int(float(row.get("bootstrap_replicates", 0))) < 5000:
        raise ValueError("RQ3 was not analysed with at least 5000 bootstrap replicates.")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--rescored_dir", required=True)
    parser.add_argument("--statistics_csv", required=True)
    parser.add_argument("--datasets", nargs="+", default=PAPER_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 2026])
    parser.add_argument("--required_variants", nargs="+", default=["full", "no_dfa"])
    parser.add_argument("--min_pair_coverage", type=float, default=0.80)
    args = parser.parse_args()

    _, manifest_hashes = verify_manifests(args.manifest_dir, args.datasets)
    verify_rescore_summary(
        args.rescored_dir, args.datasets, args.seeds, args.required_variants
    )
    verify_rescored_details(
        args.rescored_dir, args.datasets, args.seeds,
        args.required_variants, manifest_hashes,
    )
    rq3 = verify_statistics(
        args.statistics_csv, len(args.datasets), len(args.seeds),
        args.min_pair_coverage,
    )
    print("AUDIT PIPELINE VERIFIED: paper-ready independent RQ3 evidence")
    print(
        "RQ3 effect={}; CI=[{}, {}]; coverage={}; Holm={}".format(
            rq3.get("mean_effect"), rq3.get("ci95_low"),
            rq3.get("ci95_high"), rq3.get("pair_coverage"),
            rq3.get("holm_p"),
        )
    )


if __name__ == "__main__":
    main()
