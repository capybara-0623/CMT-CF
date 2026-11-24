"""P6-B read-only protocol/statistics entry. There is deliberately no search mode."""
import argparse
import json
from pathlib import Path
from cf_search.suite import METHODS, checked_json, seal
from evaluation.cf_formal_protocol import validate, prepare, guard
from evaluation.cf_experiment_results import write_json, write_jsonl, read_jsonl
from evaluation.cf_paired_statistics import indicators, statistics_rows, STATISTICS_PLAN
from evaluation.experiment_protocol import artifact, check_artifact


def load_formal_saved(root, prepared, protocol_dir):
    """Future normalized save contract; never manufactures missing search rows.

The upstream formal generation/acceptance bridge is NOT enabled by P6-B.
Saved files must be independently accepted before this statistical consumer.
"""
    root = Path(root).resolve()
    manifest = checked_json(root/"statistical_input_manifest.json")
    if (manifest.get("schema_version") != 1 or manifest.get("purpose") != "P6_formal_accepted_statistical_input_v1" or
            manifest.get("protocol") != artifact(Path(protocol_dir)/"p6b_protocol.json") or
            manifest.get("query_count") != 350 or manifest.get("query_method_count") != 3850 or
            manifest.get("methods") != METHODS or manifest.get("audit_used_in_generation") is not False or
            manifest.get("audit_used_in_selection") is not False or manifest.get("audit_used_in_tuning") is not False or
            manifest.get("all_candidates_saved_before_audit") is not True):
        raise ValueError("Full accepted predeclared saved-result contract required; pilot cannot be formal")
    if not all(r["common_buliga_applicable"] is not None for r in prepared["cohort"]):
        raise ValueError("Formal saved-results consumer requires all common memberships frozen")
    acceptance = checked_json(check_artifact(manifest["independent_acceptance"], root))
    if (acceptance.get("purpose") != "P6_formal_generation_independent_acceptance_v1" or
            acceptance.get("status") != "passed" or acceptance.get("protocol") != manifest["protocol"] or
            acceptance.get("complete_denominators_and_same_candidate_joint_verified") is not True or
            acceptance.get("budget_and_intervention_verified") is not True or
            acceptance.get("search_and_audit_evidence_independently_verified") is not True):
        raise ValueError("Independent formal generation acceptance missing")
    for item in acceptance["saved_files"].values():
        check_artifact(item, root)
    if acceptance["saved_files"] != manifest["saved_files"]:
        raise ValueError("Accepted and statistical file set differ")
    check_artifact(acceptance["verification_code"], root)
    if set(manifest["saved_files"]) != {"queries", "candidates", "audits", "generation_manifest", "audit_manifest"}:
        raise ValueError("All generation, candidate and post-save Audit evidence required")
    paths = {k: check_artifact(v, root) for k, v in manifest["saved_files"].items()}
    generation = checked_json(paths["generation_manifest"])
    audit = checked_json(paths["audit_manifest"])
    if (generation.get("protocol") != manifest["protocol"] or generation.get("audit_compilation_count") != 0 or
            generation.get("all_children_saved_before_audit") is not True or
            generation.get("queries") != manifest["saved_files"]["queries"] or
            generation.get("candidates") != manifest["saved_files"]["candidates"] or
            audit.get("generation_manifest") != manifest["saved_files"]["generation_manifest"] or
            audit.get("candidate_file") != manifest["saved_files"]["candidates"] or
            audit.get("audits") != manifest["saved_files"]["audits"] or
            audit.get("audit_used_in_generation") is not False or audit.get("audit_used_in_tuning") is not False):
        raise ValueError("Generation/save/Audit barrier evidence drift")
    queries, candidates, audit_list = [read_jsonl(paths[k]) for k in ("queries", "candidates", "audits")]
    if len(queries) != 3850 or len({r["candidate_id"] for r in audit_list}) != len(audit_list):
        raise ValueError("Full saved denominator / unique candidate Audit required")
    audits = {r["candidate_id"]: r for r in audit_list}
    return prepared["cohort"], queries, candidates, audits, artifact(root/"statistical_input_manifest.json")


def statistics(protocol_dir, output_dir, pilot=False, saved_run=None):
    prepared, _ = guard(protocol_dir)
    if pilot:
        from evaluation.cf_suite_results import load_unified, audit_rows
        root = Path(prepared["config"]["accepted_p6a_dir"]).resolve()
        old, _, qs, cs = load_unified(root)
        frozen = {r["input_sha256"]: r for r in prepared["cohort"]}
        cohort = []
        for r in old["cohort"]:
            current = frozen[r["input_sha256"]]
            if any(current[k] != r[k] for k in ("context", "original_probability", "original_class", "common_buliga_applicable")):
                raise ValueError("Previously saved pilot differs from new frozen source")
            cohort.append(current)
        audits = audit_rows(root, qs, cs)
        input_artifact = prepared["accepted_p6a"]
        purpose = "P6_B_engineering_pilot_statistics_only"
    else:
        if not saved_run:
            raise ValueError("--saved-run required; statistics never runs or fills the matrix")
        cohort, qs, cs, audits, input_artifact = load_formal_saved(saved_run, prepared, protocol_dir)
        purpose = "P6_B_predeclared_post_save_statistics"
    binary = indicators(cohort, qs, cs, audits)
    rows = statistics_rows(binary)
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Statistical output is immutable; use a new empty directory")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out/"query_indicators.jsonl", binary)
    write_jsonl(out/"paired_intervals.jsonl", rows)
    report = seal(dict(schema_version=1, purpose=purpose, protocol=artifact(Path(protocol_dir)/"p6b_protocol.json"),
        input=input_artifact, query_count=len(cohort), query_method_count=len(qs), candidate_count=len(cs),
        contrast_metric_scope_rows=len(rows), statistics_plan=STATISTICS_PLAN,
        saved_files={k: artifact(out/k) for k in ("query_indicators.jsonl", "paired_intervals.jsonl")},
        formal_matrix_executed=False, new_candidate_generation=False, model_evaluations=0,
        training_started=False, statistical_model_fit_started=False, audit_tuning_performed=False,
        engineering_pilot_only=pilot, interpretation="not dataset-level performance evidence" if pilot else
            "fixed-model conditional, marginal intervals; not training-seed robustness or untouched holdout",
        no_significance_or_equivalence_claim=True))
    write_json(out/"statistics.json", report)
    print(f"P6-B statistics: {len(cohort)} saved inputs, {len(rows)} paired interval rows; pilot_only={pilot}; no search", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "prepare", "verify", "statistics-pilot", "statistics"))
    parser.add_argument("--config", default="configs/cf_p6b_prepare.json")
    parser.add_argument("--protocol-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--saved-run")
    parser.add_argument("--statistics-dir")
    args = parser.parse_args()
    if args.mode in ("validate", "prepare"):
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        validate(config)
        if args.mode == "validate":
            print("P6-B config valid: seven tasks, fixed statistics; formal search/train/recovery forbidden")
            return
        if not args.output_dir:
            parser.error("prepare requires --output-dir")
        prepare(config, args.output_dir)
        return
    if not args.protocol_dir:
        parser.error("--protocol-dir required")
    if args.mode == "verify":
        from tools.verify_p6b_protocol import verify
        verify(args.protocol_dir, args.statistics_dir)
    else:
        if not args.output_dir:
            parser.error("statistics requires --output-dir")
        if args.mode == "statistics-pilot" and args.saved_run:
            parser.error("pilot source is fixed to accepted P6-A; --saved-run not allowed")
        statistics(args.protocol_dir, args.output_dir, args.mode == "statistics-pilot", args.saved_run)


if __name__ == "__main__":
    main()
