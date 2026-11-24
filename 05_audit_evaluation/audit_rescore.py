"""Re-score saved counterfactual traces with independent Audit-DFA manifests.

This utility does not retrain a predictive model or regenerate a
counterfactual.  It recompiles the rule formulas stored by
``experiment_runner.py`` and adds activation-aware audit fields to existing
``cf_details_*.csv`` files in a separate output directory.
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
from collections import defaultdict

from audit_constraints import compile_audit_rules, evaluate_audit_trace, stable_rule_set_hash


AUDIT_COLUMNS = [
    "audit_score",
    "audit_all_applicable",
    "audit_rule_count",
    "audit_applicable_rule_count",
    "audit_satisfied_rule_count",
    "audit_violated_rule_ids",
    "audit_compliant",
    "audit_compliant_label",
    "dfa_compliant",
    "audit_rescore_schema_version",
    "audit_manifest_schema_version",
    "audit_manifest_sha256",
    "audit_rule_set_sha256",
    "audit_source_file_sha256",
]


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifests(manifest_dir, dfa_max_states=500):
    compiled_by_dataset = {}
    diagnostics = {}
    paths = sorted(glob.glob(
        os.path.join(manifest_dir, "**", "*_audit_rules.json"),
        recursive=True,
    ))
    if not paths:
        raise FileNotFoundError(
            "No *_audit_rules.json manifests found in {!r}.".format(manifest_dir)
        )
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        dataset = str(payload.get("dataset", "")).strip()
        if not dataset:
            raise ValueError("Audit manifest has no dataset: {}".format(path))
        if dataset in compiled_by_dataset:
            raise ValueError(
                "Multiple audit manifests found for dataset {!r}; keep exactly one: {}"
                .format(dataset, path)
            )
        schema_version = int(payload.get("schema_version", 0) or 0)
        if schema_version != 2:
            raise ValueError(
                "Audit manifest {} uses schema {}; rebuild it with "
                "build_audit_manifests.py before paper analysis."
                .format(path, schema_version)
            )
        rules = payload.get("rules", [])
        stored_rule_hash = str(payload.get("rule_set_sha256", "")).strip()
        computed_rule_hash = stable_rule_set_hash(rules)
        if not stored_rule_hash or stored_rule_hash != computed_rule_hash:
            raise ValueError("Audit rule-set hash mismatch in {}.".format(path))
        manifest_diag = dict(payload.get("diagnostics", {}) or {})
        if manifest_diag.get("selection_uses_outcome_label") is not False:
            raise ValueError(
                "Manifest does not prove label-free rule selection: {}".format(path)
            )
        if manifest_diag.get("case_grain") != "complete_trace":
            raise ValueError(
                "Manifest was not built from complete traces: {}".format(path)
            )
        if int(manifest_diag.get("formula_overlap_count", -1)) != 0:
            raise ValueError(
                "Search/Audit formula overlap is non-zero in {}.".format(path)
            )
        audit_case_ids = [
            str(value) for value in manifest_diag.get("audit_case_ids", [])
        ]
        if not audit_case_ids or len(set(audit_case_ids)) != len(audit_case_ids):
            raise ValueError(
                "Manifest audit case IDs are missing or duplicated: {}".format(path)
            )
        if int(manifest_diag.get("audit_case_count", -1)) != len(audit_case_ids):
            raise ValueError(
                "Manifest audit case count disagrees with its ID list: {}".format(path)
            )
        compiled, compile_diag = compile_audit_rules(
            rules, dfa_max_states=dfa_max_states
        )
        if compile_diag.get("failed", 0) or len(compiled) != len(rules):
            raise RuntimeError(
                "Every audit rule must compile successfully in {}: {}".format(
                    path, compile_diag.get("failures", [])
                )
            )
        compiled_by_dataset[dataset] = compiled
        diagnostics[dataset] = {
            "manifest": path,
            "manifest_schema_version": schema_version,
            "manifest_sha256": _sha256_file(path),
            "rule_set_sha256": computed_rule_hash,
            "requested": compile_diag["requested"],
            "compiled": compile_diag["compiled"],
            "failed": compile_diag["failed"],
            "audit_case_ids": set(audit_case_ids),
            "audit_case_count": len(audit_case_ids),
            "dataset_case_count": int(manifest_diag.get("dataset_case_count", -1)),
            "split_seed": int(manifest_diag.get("split_seed", -1)),
            "audit_seed": int(manifest_diag.get("audit_seed", -1)),
            "audit_fraction": float(manifest_diag.get("audit_fraction", -1.0)),
        }
    return compiled_by_dataset, diagnostics


def _blank_audit(rule_count, provenance=None):
    blank = {
        "audit_score": "",
        "audit_all_applicable": "",
        "audit_rule_count": rule_count,
        "audit_applicable_rule_count": 0,
        "audit_satisfied_rule_count": 0,
        "audit_violated_rule_ids": "",
        "audit_compliant": "",
        "audit_compliant_label": "",
        "dfa_compliant": "",
    }
    blank.update(provenance or {})
    return blank


def _rescore_row(row, rules, provenance=None):
    if "candidate_returned" not in row:
        raise ValueError("CF detail row has no candidate_returned field.")
    try:
        returned = int(float(row.get("candidate_returned") or 0))
    except (TypeError, ValueError):
        raise ValueError("candidate_returned must be 0 or 1.")
    if returned not in (0, 1):
        raise ValueError("candidate_returned must be 0 or 1.")
    trace_text = str(row.get("cf_acts", "")).strip()
    if not returned:
        if trace_text:
            raise ValueError("A non-returned candidate unexpectedly contains cf_acts.")
        row.update(_blank_audit(len(rules), provenance))
        return row, None
    if not trace_text:
        raise ValueError("A returned candidate has an empty cf_acts trace.")

    trace = [part.strip() for part in trace_text.split(" -> ") if part.strip()]
    if not trace:
        raise ValueError("A returned candidate could not be decoded into activities.")
    result = evaluate_audit_trace(trace, rules)
    score = result["audit_score"]
    all_applicable = result["audit_all_applicable"]
    label = ""
    if not math.isnan(float(all_applicable)):
        label = "Y" if int(all_applicable) == 1 else "N"
    row.update({
        "audit_score": "" if math.isnan(float(score)) else score,
        "audit_all_applicable": (
            "" if math.isnan(float(all_applicable)) else int(all_applicable)
        ),
        "audit_rule_count": result["audit_rule_count"],
        "audit_applicable_rule_count": result["audit_applicable_rule_count"],
        "audit_satisfied_rule_count": result["audit_satisfied_rule_count"],
        "audit_violated_rule_ids": ";".join(result["audit_violated_rule_ids"]),
        "audit_compliant": (
            "" if math.isnan(float(all_applicable)) else int(all_applicable)
        ),
        "audit_compliant_label": label,
        "dfa_compliant": label,
    })
    row.update(provenance or {})
    return row, result


def _write_csv_atomic(rows, columns, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_protocol_rows(results_dir):
    path = os.path.join(results_dir, "all_results.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Paper re-scoring requires all_results.csv for split provenance: {}"
            .format(path)
        )
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_protocol_manifest(dataset, protocol_rows, manifest_diag):
    rows = [
        row for row in protocol_rows
        if str(row.get("dataset", "")).strip() == dataset
        and str(row.get("variant", "")).strip() == "full"
    ]
    if not rows:
        raise ValueError(
            "all_results.csv has no full-variant provenance for {!r}.".format(dataset)
        )

    def unique_numeric(field, cast):
        try:
            values = {cast(float(row[field])) for row in rows if str(row.get(field, "")).strip()}
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "Invalid {!r} provenance for dataset {!r}.".format(field, dataset)
            )
        if len(values) != 1:
            raise ValueError(
                "Expected one {!r} value for dataset {!r}, got {}."
                .format(field, dataset, sorted(values))
            )
        return next(iter(values))

    checks = [
        ("evaluation_protocol_version", unique_numeric("evaluation_protocol_version", int), 2),
        ("split_seed", unique_numeric("split_seed", int), manifest_diag["split_seed"]),
        ("num_cases", unique_numeric("num_cases", int), manifest_diag["dataset_case_count"]),
        ("pattern_audit_case_count", unique_numeric("pattern_audit_case_count", int), manifest_diag["audit_case_count"]),
    ]
    for field, observed, expected in checks:
        if observed != expected:
            raise ValueError(
                "Manifest/results provenance mismatch for {} {}: {} != {}"
                .format(dataset, field, observed, expected)
            )
    observed_fraction = unique_numeric("audit_fraction", float)
    if not math.isclose(observed_fraction, manifest_diag["audit_fraction"], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Manifest/results audit_fraction mismatch for {}: {} != {}"
            .format(dataset, observed_fraction, manifest_diag["audit_fraction"])
        )


def _validate_detail_run_provenance(dataset, variant, seed, protocol_rows,
                                    manifest_diag):
    matches = [
        row for row in protocol_rows
        if str(row.get("dataset", "")).strip() == dataset
        and str(row.get("variant", "")).strip() == variant
        and int(float(row.get("seed", -1))) == int(float(seed))
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one all_results row for ({}, {}, {}), found {}."
            .format(dataset, variant, seed, len(matches))
        )
    row = matches[0]
    checks = {
        "evaluation_protocol_version": (2, int),
        "split_seed": (manifest_diag["split_seed"], int),
        "num_cases": (manifest_diag["dataset_case_count"], int),
        "pattern_audit_case_count": (manifest_diag["audit_case_count"], int),
        "audit_fraction": (manifest_diag["audit_fraction"], float),
    }
    for field, (expected, cast) in checks.items():
        try:
            observed = cast(float(row[field]))
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "Invalid {} in all_results for ({}, {}, {})."
                .format(field, dataset, variant, seed)
            )
        equal = (
            math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
            if cast is float else observed == expected
        )
        if not equal:
            raise ValueError(
                "Run/manifest provenance mismatch for ({}, {}, {}) {}: {} != {}"
                .format(dataset, variant, seed, field, observed, expected)
            )

def rescore_directory(results_dir, manifest_dir, output_dir, dfa_max_states=500):
    rules_by_dataset, manifest_diagnostics = load_manifests(
        manifest_dir, dfa_max_states=dfa_max_states
    )
    protocol_rows = _load_protocol_rows(results_dir)
    paths = sorted(glob.glob(
        os.path.join(results_dir, "**", "cf_details_*.csv"), recursive=True
    ))
    if not paths:
        raise FileNotFoundError(
            "No cf_details_*.csv files found in {!r}.".format(results_dir)
        )

    aggregates = defaultdict(lambda: {
        "attempted_queries": set(),
        "returned_queries": set(),
        "returned_candidate_count": 0,
        "applicable_candidate_count": 0,
        "scores": [],
        "all_applicable": [],
        "applicable_rule_counts": [],
        "query_scores": defaultdict(list),
        "query_all_applicable": defaultdict(list),
        "input_detail_files": set(),
    })
    written = []
    validated_datasets = set()
    seen_observation_keys = set()
    for path in paths:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        required = {
            "dataset", "variant", "seed", "protocol_version", "case_id",
            "sample_idx", "cf_rank", "candidate_returned", "cf_acts",
        }
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(
                "CF detail file {} is missing required columns: {}"
                .format(path, missing)
            )
        datasets = {str(row.get("dataset", "")).strip() for row in rows}
        datasets.discard("")
        if len(datasets) != 1:
            raise ValueError(
                "Expected one dataset per details file, got {} in {}".format(
                    sorted(datasets), path
                )
            )
        dataset = next(iter(datasets))
        if dataset not in rules_by_dataset:
            raise KeyError(
                "No independent audit manifest for dataset {!r} (file {}).".format(
                    dataset, path
                )
            )
        rules = rules_by_dataset[dataset]
        manifest_diag = manifest_diagnostics[dataset]
        if dataset not in validated_datasets:
            _validate_protocol_manifest(dataset, protocol_rows, manifest_diag)
            validated_datasets.add(dataset)
        versions = {int(float(row["protocol_version"])) for row in rows}
        if versions != {2}:
            raise ValueError(
                "Only protocol-version 2 CF details may be re-scored: {}".format(path)
            )
        variants = {str(row["variant"]).strip() for row in rows}
        seeds = {str(row["seed"]).strip() for row in rows}
        if len(variants) != 1 or len(seeds) != 1 or "" in variants or "" in seeds:
            raise ValueError(
                "Each CF detail file must contain exactly one non-empty variant and seed: {}"
                .format(path)
            )
        _validate_detail_run_provenance(
            dataset, next(iter(variants)), next(iter(seeds)),
            protocol_rows, manifest_diag,
        )
        evaluation_ids = {str(row["case_id"]) for row in rows}
        overlap = sorted(evaluation_ids.intersection(manifest_diag["audit_case_ids"]))
        if overlap:
            raise ValueError(
                "Audit discovery/evaluation case overlap for {}: {}"
                .format(dataset, overlap[:10])
            )
        source_sha256 = _sha256_file(path)
        provenance = {
            "audit_rescore_schema_version": 2,
            "audit_manifest_schema_version": manifest_diag["manifest_schema_version"],
            "audit_manifest_sha256": manifest_diag["manifest_sha256"],
            "audit_rule_set_sha256": manifest_diag["rule_set_sha256"],
            "audit_source_file_sha256": source_sha256,
        }
        rescored = []
        for row in rows:
            key = (
                dataset,
                str(row.get("variant", "")),
                str(row.get("seed", "")),
            )
            group = aggregates[key]
            query_key = (
                str(row.get("case_id", "")),
                str(row.get("sample_idx", "")),
            )
            observation_key = key + query_key + (str(row.get("cf_rank", "")),)
            if observation_key in seen_observation_keys:
                raise ValueError(
                    "Duplicate CF observation across input files: {}".format(observation_key)
                )
            seen_observation_keys.add(observation_key)
            group["attempted_queries"].add(query_key)
            group["input_detail_files"].add(path)
            row, audit = _rescore_row(row, rules, provenance)
            rescored.append(row)
            if audit is None:
                continue
            group["returned_candidate_count"] += 1
            group["returned_queries"].add(query_key)
            if not math.isnan(float(audit["audit_score"])):
                group["applicable_candidate_count"] += 1
                group["scores"].append(float(audit["audit_score"]))
                group["all_applicable"].append(float(audit["audit_all_applicable"]))
                group["query_scores"][query_key].append(
                    float(audit["audit_score"])
                )
                group["query_all_applicable"][query_key].append(
                    float(audit["audit_all_applicable"])
                )
                group["applicable_rule_counts"].append(
                    float(audit["audit_applicable_rule_count"])
                )

        columns = fieldnames + [name for name in AUDIT_COLUMNS if name not in fieldnames]
        relative = os.path.relpath(path, results_dir)
        destination = os.path.join(output_dir, relative)
        _write_csv_atomic(rescored, columns, destination)
        written.append(destination)

    summary_rows = []
    for (dataset, variant, seed), values in sorted(aggregates.items()):
        scores = values["scores"]
        strict = values["all_applicable"]
        counts = values["applicable_rule_counts"]
        query_scores = [
            sum(items) / len(items)
            for items in values["query_scores"].values() if items
        ]
        query_strict = [
            sum(items) / len(items)
            for items in values["query_all_applicable"].values() if items
        ]
        summary_rows.append({
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "attempted_query_count": len(values["attempted_queries"]),
            "returned_query_count": len(values["returned_queries"]),
            "cf_return_rate": (
                len(values["returned_queries"]) / len(values["attempted_queries"])
                if values["attempted_queries"] else ""
            ),
            "returned_candidate_count": values["returned_candidate_count"],
            "audit_applicable_candidate_count": values["applicable_candidate_count"],
            "audit_applicability_rate": (
                values["applicable_candidate_count"] / values["returned_candidate_count"]
                if values["returned_candidate_count"] else ""
            ),
            "audit_metric_conditioning": "returned candidates with >=1 applicable rule",
            "audit_compliance_rate": (
                sum(query_scores) / len(query_scores) if query_scores else ""
            ),
            "audit_all_applicable_rate": (
                sum(query_strict) / len(query_strict) if query_strict else ""
            ),
            "audit_candidate_weighted_rate": (
                sum(scores) / len(scores) if scores else ""
            ),
            "audit_applicable_query_count": len(query_scores),
            "mean_applicable_rule_count": sum(counts) / len(counts) if counts else "",
            "audit_rule_count": len(rules_by_dataset[dataset]),
            "audit_manifest": manifest_diagnostics[dataset]["manifest"],
            "audit_manifest_sha256": manifest_diagnostics[dataset]["manifest_sha256"],
            "audit_rule_set_sha256": manifest_diagnostics[dataset]["rule_set_sha256"],
            "input_detail_file_count": len(values["input_detail_files"]),
        })
    summary_path = os.path.join(output_dir, "audit_rescore_summary.csv")
    summary_columns = list(summary_rows[0]) if summary_rows else [
        "dataset", "variant", "seed", "candidate_count"
    ]
    _write_csv_atomic(summary_rows, summary_columns, summary_path)
    return written, summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Re-score saved CF detail files with independent Audit-DFA rules."
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--audit_manifest_dir", default="./audit_manifests")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dfa_max_states", type=int, default=500)
    args = parser.parse_args()
    results_path = os.path.abspath(args.results_dir)
    output_path = os.path.abspath(args.output_dir)
    try:
        common = os.path.commonpath([results_path, output_path])
    except ValueError:
        common = ""
    if common in (results_path, output_path):
        parser.error(
            "--results_dir and --output_dir must be disjoint; neither may contain the other"
        )
    if not os.path.isdir(results_path):
        parser.error("--results_dir does not exist: {}".format(results_path))
    if not os.path.isdir(os.path.abspath(args.audit_manifest_dir)):
        parser.error(
            "--audit_manifest_dir does not exist: {}"
            .format(os.path.abspath(args.audit_manifest_dir))
        )
    if os.path.isdir(output_path) and os.listdir(output_path):
        parser.error(
            "--output_dir must be new or empty to prevent stale-file mixing: {}"
            .format(output_path)
        )
    written, summary = rescore_directory(
        args.results_dir,
        args.audit_manifest_dir,
        args.output_dir,
        dfa_max_states=args.dfa_max_states,
    )
    print("Re-scored {} detail files.".format(len(written)))
    print("Summary: {}".format(summary))


if __name__ == "__main__":
    main()
