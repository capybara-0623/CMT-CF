"""Independent saved P6-B acceptance: replay frozen languages and manual math.

Does not call protocol preparation, statistical indicators/interval helpers,
candidate generators, a predictor, Audit compiler, reference recovery or fit.
"""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import numpy as np
from cf_search.suite import METHODS, checked_json, seal
from evaluation.cf_formal_protocol import guard, DATASETS
from evaluation.cf_experiment_results import read_jsonl, write_json
from evaluation.experiment_protocol import artifact, check_artifact, load_protocol, stable_hash
from explaination.visualizations.utils import normalize_activity_name


def accepts(snapshot, letters):
    state = snapshot["initial_state"]
    for letter in letters:
        if state is None:
            return False
        outgoing = dict(snapshot["transitions"][state])
        state = outgoing.get(letter, outgoing.get("*"))
    return state in snapshot["accepting_states"]


def factual_strict(source, original, context):
    names = [normalize_activity_name(original["model"]["activity_vocabulary"][a-1]) for a in context["activities"]]
    atomic = {rid: accepts(s, names) for rid, s in source["atomic_snapshots"].items()}
    rules = source["search"]
    return all(atomic[r] for r in rules["protected_rule_ids"]) and (not rules["clusters"] or
        any(all(atomic[r] for r in c["rule_ids"]) for c in rules["clusters"]))


def manual_pilot(protocol, statistics_dir):
    """Manual binary extraction + Python case loops, separate from stats helpers."""
    from evaluation.cf_suite_results import load_unified, audit_rows
    root = Path(protocol["config"]["accepted_p6a_dir"])
    old, _, queries, candidates = load_unified(root)
    audits = audit_rows(root, queries, candidates)
    known = {r["input_sha256"]: r for r in old["cohort"]}
    grouped = defaultdict(list)
    for c in candidates:
        grouped[c["query_id"]].append(c)
    records = []
    for q in queries:
        r = known[q["input_sha256"]]
        goals = [c for c in grouped[q["query_id"]] if c["probability"] >= .5]
        failure = r["original_class"] == "Failure"
        records.append(dict(dataset=q["dataset"], method=q["method"], input_sha256=q["input_sha256"],
            case_id=r["context"]["case_id"], source_unit="sepsis" if q["dataset"].startswith("sepsis_cases_") else q["dataset"],
            case_key=stable_hash(dict(source_unit="sepsis" if q["dataset"].startswith("sepsis_cases_") else q["dataset"],
                case_id=r["context"]["case_id"])), eligible_failure=failure,
            common_buliga_applicable=r["common_buliga_applicable"], unavailable=q["status"] == "method_unavailable",
            coverage=int(len(grouped[q["query_id"]]) > 0), goal_coverage=int(len(goals) > 0),
            improvement_success_at_k=int(failure and len(goals) > 0),
            strict_improvement_success_at_k=int(failure and any(c["search"]["strict"] for c in goals)),
            audit_joint_improvement_success_at_k=int(failure and any(c["search"]["strict"] and
                audits[c["candidate_id"]]["status"] == "audit_pass" and audits[c["candidate_id"]]["applicable_rule_count"] > 0 for c in goals)),
            relaxed_improvement_success_at_k=int(failure and any(c["search"]["relaxation_count"] > 0 for c in goals))))
    stats_root = Path(statistics_dir).resolve()
    report = checked_json(stats_root/"statistics.json")
    if (report["purpose"] != "P6_B_engineering_pilot_statistics_only" or not report["engineering_pilot_only"] or
            report["protocol"] != artifact(Path(protocol["preparation_root"])/"p6b_protocol.json") or
            report["input"] != protocol["accepted_p6a"] or report["query_count"] != 6 or report["query_method_count"] != 66 or
            report["candidate_count"] != len(candidates) or report["statistics_plan"] != protocol["statistics_plan"]):
        raise ValueError("Pilot not explicitly labeled, or source/protocol drift")
    for item in report["saved_files"].values():
        check_artifact(item, stats_root)
    saved = read_jsonl(stats_root/"query_indicators.jsonl")
    sort_key = lambda r: (r["input_sha256"], r["method"])
    if sorted(saved, key=sort_key) != sorted(records, key=sort_key):
        raise ValueError("Independently extracted query indicator mismatch")
    intervals = read_jsonl(stats_root/"paired_intervals.jsonl")
    expected_keys = {(d, s, c["a"], c["b"], m) for d in {r["dataset"] for r in records}
        for s in protocol["statistics_plan"]["scopes"] for c in protocol["statistics_plan"]["contrasts"]
        for m in protocol["statistics_plan"]["metrics"]}
    actual_keys = [(r["dataset"], r["scope"], r["a"], r["b"], r["metric"]) for r in intervals]
    if set(actual_keys) != expected_keys or len(actual_keys) != len(expected_keys):
        raise ValueError("Predeclared contrast grid missing/duplicate")
    independently_bootstrapped = 0
    for row in intervals:
        rs = [r for r in records if r["dataset"] == row["dataset"] and
            (row["scope"] == "all_queries" or r["common_buliga_applicable"])]
        a = {r["input_sha256"]: r for r in rs if r["method"] == row["a"]}
        b = {r["input_sha256"]: r for r in rs if r["method"] == row["b"]}
        eligible = [k for k in a if row["metric"] in ("coverage", "goal_coverage") or a[k]["eligible_failure"]]
        cases = defaultdict(list)
        for k in eligible:
            cases[a[k]["case_key"]].append(k)
        total_a = sum(a[k][row["metric"]] for k in eligible)
        total_b = sum(b[k][row["metric"]] for k in eligible)
        if (row["query_count"] != len(a) or row["denominator"] != len(eligible) or
                row["case_cluster_count"] != len(cases) or row["numerator_a"] != total_a or row["numerator_b"] != total_b):
            raise ValueError("Independent paired numerator/denominator mismatch")
        for method_records, field, numerator in ((a, "rate_a", total_a), (b, "rate_b", total_b)):
            expected_rate = numerator/len(eligible) if eligible and not all(r["unavailable"] for r in method_records.values()) else None
            if row[field] != expected_rate:
                raise ValueError("Zero/missing/unavailable rate mismatch")
        if row["rate_a"] is None or row["rate_b"] is None:
            if row["difference"] is not None or row["confidence_interval"] is not None:
                raise ValueError("Ineligible/unavailable cannot have a difference/interval")
            continue
        if abs(row["difference"]-(total_a-total_b)/len(eligible)) > 1e-15:
            raise ValueError("Independent point difference mismatch")
        if len(cases) < 2:
            if row["confidence_interval"] is not None:
                raise ValueError("Single case cannot have an inferential interval")
            continue
        keys = sorted(cases)
        rng = np.random.default_rng(protocol["statistics_plan"]["bootstrap_seed"])
        samples = rng.integers(0, len(keys), size=(5000, len(keys)))
        differences = []
        for sample in samples:
            query_ids = [k for i in sample for k in cases[keys[int(i)]]]
            differences.append(sum(a[k][row["metric"]]-b[k][row["metric"]] for k in query_ids)/len(query_ids))
        ci = np.quantile(differences, [.025, .975], method="linear").tolist()
        if not np.allclose(ci, row["confidence_interval"], atol=1e-15, rtol=0):
            raise ValueError("Independent intact case-loop interval mismatch")
        expected_status = "degenerate_bootstrap" if min(differences) == max(differences) else "ok"
        if row["status"] != expected_status or row["completed_resamples"] != 5000:
            raise ValueError("Bootstrap completion/degeneracy disclosure missing")
        independently_bootstrapped += 1
    return dict(statistics=artifact(stats_root/"statistics.json"), indicator_rows=len(saved),
        contrast_rows=len(intervals), independently_recomputed_case_block_intervals=independently_bootstrapped,
        zero_new_model_evaluations=True)


def verify(output, statistics_dir=None):
    root = Path(output).resolve()
    protocol, report = guard(root)
    frozen = []
    missing = []
    per_dataset = []
    previously_inspected = {r["input_sha256"] for r in checked_json(check_artifact(protocol["accepted_p6a_protocol"], root))["cohort"]}
    for source, dataset in zip(protocol["sources"], DATASETS):
        if source["dataset"] != dataset:
            raise ValueError("Seven task order changed")
        original = load_protocol(check_artifact(source["protocol"], root))
        if source["search"] != original["search"] or source["budget"] != original["budget"] or source["intervention"] != original["intervention"]:
            raise ValueError("Source Search/budget/action policy changed")
        reference = None
        if source["reference"]["available"]:
            reference = checked_json(check_artifact(source["reference"]["artifact"], root))
            prov = original["preprocessing"]
            permitted = set(prov["search_discovery_case_ids"]+prov["search_calibration_case_ids"])
            prohibited = set(prov["audit_case_ids"]+prov["predictive_validation_case_ids"]+prov["predictive_test_case_ids"])
            if set(reference["case_ids"]) != permitted or permitted & prohibited or not permitted <= set(prov["predictive_train_case_ids"]):
                raise ValueError("Reference isolation mismatch")
        else:
            missing.append(dataset)
        rows = [r for r in protocol["cohort"] if r["dataset"] == dataset]
        if len(rows) != 50:
            raise ValueError("Every task must retain all 50 queries")
        for i, (r, old) in enumerate(zip(rows, original["queries"])):
            if any(r[k] != v for k, v in old.items()) or r["offset"] != i or stable_hash(r["context"]) != r["input_sha256"]:
                raise ValueError("Complete original frozen public input changed")
            strict = factual_strict(source, original, r["context"])
            length = len(r["context"]["activities"])
            ref_long = None if reference is None else any(len(x["activities"]) >= length for x in reference["rows"])
            common = False if not strict else ref_long
            if (r["factual_search_strict"] is not strict or r["permitted_length_reference_available"] is not ref_long or
                    r["common_buliga_applicable"] is not common or r["previously_inspected_p6a"] != (r["input_sha256"] in previously_inspected)):
                raise ValueError("Independent full-sequence DFA/reference membership mismatch")
        known = [r for r in rows if r["common_buliga_applicable"] is True]
        unknown = [r for r in rows if r["common_buliga_applicable"] is None]
        saved_count = next(r for r in report["per_dataset"] if r["dataset"] == dataset)
        if (saved_count["factual_failure_queries"] != sum(r["original_class"] == "Failure" for r in rows) or
                saved_count["common_query_count"] != (None if unknown else len(known)) or
                saved_count["common_failure_queries"] != (None if unknown else sum(r["original_class"] == "Failure" for r in known)) or
                saved_count["common_unknown_queries"] != len(unknown)):
            raise ValueError("Independent common/Failure count mismatch")
        per_dataset.append(saved_count)
        frozen.extend(rows)
    if len(frozen) != 350 or len({r["input_sha256"] for r in frozen}) != 350 or report["missing_references"] != missing:
        raise ValueError("Complete 350 input coverage or missing-reference report mismatch")
    if (report["factual_preflight_model_evaluations"] != 350 or
            sum(e["factual_model_evaluations"] for e in report["preparation_events"]) != 350 or
            report["factual_failure_queries"] != sum(r["original_class"] == "Failure" for r in frozen) or
            any(not math.isfinite(e["walltime_seconds"]) or e["walltime_seconds"] < 0 for e in report["preparation_events"]) or
            any(report[k] != 0 for k in ("generated_candidate_count", "counterfactual_model_evaluations",
                "cf_head_forward_evaluations", "KD_build_count", "audit_compile_count", "reference_recovery_count",
                "training_count", "statistical_model_fit_count")) or report["formal_experiment_started"] or report["formal_execution_authorized"]):
        raise ValueError("Preparation cost/no-execution scope mismatch")
    if any((root/name).exists() for name in ("queries.jsonl", "candidates.jsonl", "children", "search_manifest.json", "generation_manifest.json")):
        raise ValueError("Preparation directory must not contain executed matrix output")
    pilot = manual_pilot(dict(protocol, preparation_root=str(root)), statistics_dir) if statistics_dir else None
    accepted = seal(dict(schema_version=1, status="passed", purpose="P6_B_preparation_independent_acceptance",
        protocol=artifact(root/"p6b_protocol.json"), preflight=artifact(root/"preflight.json"),
        verification_code=artifact(Path(__file__)), query_count=350, planned_query_method_rows=3850,
        factual_failure_queries=report["factual_failure_queries"], missing_references=missing,
        per_dataset=per_dataset, saved_pilot_statistics=pilot, independent_case_language_replays=350,
        training_started=False, reference_recovery_started=False, formal_experiment_started=False,
        independent_verification_model_evaluations=0, full_external_matrix_artifacts_ready=report["full_external_matrix_artifacts_ready"],
        interpretation="protocol/statistical engineering accepted; missing external references not repaired; matrix not run"))
    write_json(root/"independent_acceptance.json", accepted)
    print("P6-B independently accepted: 350 frozen queries; 3850 planned rows; formal matrix not executed", flush=True)
    return accepted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--statistics-dir")
    args = p.parse_args()
    verify(args.protocol_dir, args.statistics_dir)


if __name__ == "__main__":
    main()
