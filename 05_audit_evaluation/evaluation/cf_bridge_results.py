"""Full matrix persistence contract and post-save Audit; no search is imported."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from audit_constraints import compile_audit_rules, rule_applicable, rule_satisfied
from cf_search.suite import FAMILIES, METHODS, GROUPS, SCOPES, checked_json, seal, json_form
from evaluation.cf_experiment_results import read_jsonl, write_json, write_jsonl, validate_results, validate_search_pools
from evaluation.cf_suite_results import metric_rows, cost_rows
from evaluation.experiment_protocol import artifact, check_artifact, load_protocol, stable_hash
from explaination.visualizations.utils import normalize_activity_name

EXTRA = {"query_id", "candidate_id", "rank", "input_sha256", "method", "family", "dataset"}
STATUSES = {"completed", "model_budget_exhausted", "attempt_budget_exhausted", "walltime_exhausted",
            "search_error", "proposal_pool_limit_reached", "precondition_not_met", "method_unavailable"}
SAVED_KEYS = {"queries", "candidates", "audits", "generation_manifest", "audit_manifest"}


def bare(row):
    return {k: v for k, v in row.items() if k not in EXTRA}


def compatibility_manifest(protocol, cohort, family):
    """Reuse already accepted per-family validation schemas, not smoke runners.

    This in-memory view is never emitted as an engineering smoke manifest and
    its legacy purpose/schema describes validation dispatch ONLY.
    """
    sources, domains, names, snapshots = [], {}, {}, {}
    for source in protocol["sources"]:
        original = load_protocol(source["protocol"]["path"])
        dataset = source["dataset"]
        ids = [i for i, raw in enumerate(original["model"]["activity_vocabulary"], 1)
               if raw.upper() not in ("__UNK__", "<UNK>", "UNK", "__PAD__", "<PAD>")]
        sources.append(dict(dataset=dataset, search=source["search"], budget=source["budget"], model=original["model"],
            proposal_activity_ids=ids, reference_case_ids=source["reference"]["allowed_case_ids"]))
        domains[dataset] = list(range(1, len(original["model"]["activity_vocabulary"])+1))
        names[dataset] = {str(i): normalize_activity_name(v) for i, v in enumerate(original["model"]["activity_vocabulary"], 1)}
        snapshots[dataset] = source["atomic_snapshots"]
    expected = [dict(query_id=stable_hash(dict(input_sha256=r["input_sha256"], method=m)),
        context=r["context"], search_method=m, original_probability=r["original_probability"], original_class=r["original_class"])
        for r in cohort for m in FAMILIES[family]]
    schema, purpose = dict(search=(1, "P2_P3_engineering_smoke"), relaxation=(2, "P4_A_engineering_smoke"),
        proprietary=(4, "P4_B_engineering_smoke"), buliga=(5, "P5_Buliga_completion_smoke"))[family]
    return dict(schema_version=schema, purpose=purpose, methods=FAMILIES[family], expected_queries=expected,
        topk=5, threshold=.5, sources=sources, activity_domains=domains, activity_names=names,
        runtime_search_dfa_snapshots=snapshots, pool_checks_required=True, preselection_artifacts={})


def validate_saved(protocol, cohort, queries, candidates, physical):
    frozen = {r["input_sha256"]: r for r in cohort}
    if len(frozen) != len(cohort) or not cohort:
        raise ValueError("Unique nonempty predeclared cohort required")
    pairs = [(q["input_sha256"], q["method"]) for q in queries]
    if len(set(pairs)) != len(pairs) or set(pairs) != {(k, m) for k in frozen for m in METHODS}:
        raise ValueError("Complete frozen query-by-method denominator required; never fill missing rows")
    qs = {q["query_id"]: q for q in queries}
    if len(qs) != len(queries) or len({c["candidate_id"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate query/candidate IDs")
    physical_map = {r["execution_id"]: r for r in physical}
    if len(physical_map) != len(physical) or len(physical) != 9*len(cohort):
        raise ValueError("Exactly nine physical generation records per input required")
    attributed = set()
    candidate_counts = Counter(c["query_id"] for c in candidates)
    for q in queries:
        r = frozen[q["input_sha256"]]
        family = next(f for f, ms in FAMILIES.items() if q["method"] in ms)
        if (q["query_id"] != stable_hash(dict(input_sha256=q["input_sha256"], method=q["method"])) or
                q["family"] != family or any(q[k] != r[k] for k in ("dataset", "original_probability", "original_class", "common_buliga_applicable")) or
                q["time_policy"] != ("learned_normalized_elapsed_coordinate" if q["method"] == "O2" else "fixed") or
                q["comparison_groups"] != [g for g, ms in GROUPS.items() if q["method"] in ms] or q["status"] not in STATUSES):
            raise ValueError("Frozen query, reporting-group, or terminal-state drift")
        if q["candidate_count"] != candidate_counts[q["query_id"]]:
            raise ValueError("Candidate count differs from durable records")
        if q["method"] == "created_unavailable":
            if (q["status"] != "method_unavailable" or q["candidate_count"] or not q["statistics"].get("reason") or
                    any(q["statistics"]["oracle"].values()) or q.get("error")):
                raise ValueError("CREATED remains explicitly unavailable and zero-cost, not a measured baseline")
            continue
        p = physical_map.get(q["execution_id"])
        if p is None or p["input_sha256"] != q["input_sha256"] or p["family"] != family or p["dataset"] != q["dataset"]:
            raise ValueError("Missing/cross-input physical cost/pool evidence")
        expected_method = "R1_R2_shared_expanded" if q["method"] in ("R1_expanded_cap0", "R2_expanded_cap1") else q["method"]
        if (p["method"] != expected_method or p["pool_sha256"] != stable_hash(p["rows"]) or p["audit_input_present"] is not False or
                p["statistics"]["p6a_profile"] != p["costs"] or q["statistics"].get("p6a_profile") != p["costs"]):
            raise ValueError("Physical identity, immutable pool, profile, or Audit barrier mismatch")
        if q["status"] != p["status"] and not (family == "relaxation" and q["status"] == "search_error" and q.get("error")):
            raise ValueError("Physical terminal status differs from attributed view")
        if family != "relaxation" and (q.get("error") != p.get("error") or
                [bare(c) for c in sorted((c for c in candidates if c["query_id"] == q["query_id"]), key=lambda c: c["rank"])] != p["returned"]):
            raise ValueError("Normalized view differs from the algorithm's persisted return")
        if family == "relaxation" and p["saved_before_policy_selection"] is not True:
            raise ValueError("Fixed pool was not saved before policy selection")
        attributed.add(q["execution_id"])
    if attributed != set(physical_map):
        raise ValueError("Orphan physical execution/cost record")
    for c in candidates:
        q = qs.get(c["query_id"])
        if q is None or any(c[k] != q[k] for k in ("method", "family", "dataset", "input_sha256")) or c["candidate_id"] != stable_hash(
                dict(query_id=c["query_id"], activities=c["activities"], times=c["times"])):
            raise ValueError("Candidate identity/attribution mismatch")
    for family in ("search", "relaxation", "proprietary", "buliga"):
        manifest = compatibility_manifest(protocol, cohort, family)
        family_qs = [q for q in queries if q["family"] == family]
        family_cs = [{k: v for k, v in c.items() if k not in EXTRA or k in {"query_id", "candidate_id", "rank"}}
                     for c in candidates if c["family"] == family]
        pools = []
        ids_by_dataset = {s["dataset"]: set(s["proposal_activity_ids"]) for s in manifest["sources"]}
        for q in family_qs:
            p = physical_map[q["execution_id"]]
            context = frozen[q["input_sha256"]]["context"]
            for row in p["rows"]:
                if len(row["activities"]) != len(context["activities"]) or any(a not in ids_by_dataset[q["dataset"]]
                        for a, old in zip(row["activities"], context["activities"]) if a != old):
                    raise ValueError("Scored proposal outside common known-activity replacement domain")
            if q["status"] == "search_error":
                continue
            pools.append(dict(query_id=q["query_id"], rows=p["rows"], scored_pool_sha256=p["pool_sha256"], pool_id=p["local_execution_id"]))
            if family == "relaxation":
                manifest["preselection_artifacts"][p["local_execution_id"]] = dict(saved_before_selection=True)
        validate_results(manifest, family_qs, family_cs)
        validate_search_pools(manifest, family_qs, family_cs, pools)
    return True


def save_generation(root, bridge, protocol, cohort, queries, candidates, physical_artifacts, units, authorization, preparation):
    root = Path(root)
    physical = [checked_json(check_artifact(p, root)) for p in physical_artifacts]
    validate_saved(protocol, cohort, queries, candidates, physical)
    write_jsonl(root/"queries.jsonl", queries)
    write_jsonl(root/"candidates.jsonl", candidates)
    manifest = seal(dict(schema_version=1, purpose="P6_B_formal_saved_generation_v1", protocol=bridge["protocol"],
        bridge_preflight=artifact(bridge["preflight_path"]), scope=authorization["scope"],
        authorization=authorization["artifact"], cohort_input_sha256=[r["input_sha256"] for r in cohort],
        query_count=len(cohort), query_method_count=len(queries), candidate_count=len(candidates),
        queries=artifact(root/"queries.jsonl"), candidates=artifact(root/"candidates.jsonl"),
        physical_artifacts=physical_artifacts, unit_artifacts=units, physical_generation_count=len(physical),
        preparation_events=preparation, audit_compilation_count=0, all_children_saved_before_audit=True,
        audit_used_in_generation=False, audit_used_in_selection=False, audit_used_in_tuning=False,
        training_started=False, reference_recovery_started=False, statistical_model_fit_started=False,
        historical_incremental_cf_training_seconds=None,
        formal_experiment_started=authorization["scope"] == "seven_task_matrix"))
    write_json(root/"generation_manifest.json", manifest)
    return manifest


def load_generation(root):
    from evaluation.cf_bridge_protocol import guard, selected_cohort, authorize
    root = Path(root).resolve()
    generation = checked_json(root/"generation_manifest.json")
    preflight_path = check_artifact(generation["bridge_preflight"], root)
    bridge, protocol = guard(preflight_path.parent)
    authorization_path = check_artifact(generation["authorization"], root)
    authorization, _ = authorize(preflight_path.parent, root, authorization_path)
    cohort = selected_cohort(protocol, authorization["scope"])
    if (generation["purpose"] != "P6_B_formal_saved_generation_v1" or generation["scope"] != authorization["scope"] or
            generation["protocol"] != bridge["protocol"] or generation["cohort_input_sha256"] != [r["input_sha256"] for r in cohort] or
            generation["query_count"] != len(cohort) or generation["query_method_count"] != 11*len(cohort) or
            generation["physical_generation_count"] != 9*len(cohort) or generation["audit_compilation_count"] != 0 or
            generation["all_children_saved_before_audit"] is not True or
            any(generation[k] for k in ("training_started", "reference_recovery_started", "statistical_model_fit_started",
                                       "audit_used_in_generation", "audit_used_in_selection", "audit_used_in_tuning"))):
        raise ValueError("Full saved-generation contract/isolation mismatch")
    queries = read_jsonl(check_artifact(generation["queries"], root))
    candidates = read_jsonl(check_artifact(generation["candidates"], root))
    physical = [checked_json(check_artifact(p, root)) for p in generation["physical_artifacts"]]
    units = [checked_json(check_artifact(p, root)) for p in generation["unit_artifacts"]]
    if ([r for u in units for r in u["queries"]] != queries or [r for u in units for r in u["candidates"]] != candidates or
            [p for u in units for p in u["physical_artifacts"]] != generation["physical_artifacts"] or len(units) != len(cohort) or
            generation["candidate_count"] != len(candidates)):
        raise ValueError("Normalized records differ from durable per-input saves")
    validate_saved(protocol, cohort, queries, candidates, physical)
    return bridge, protocol, cohort, generation, queries, candidates, physical


def audit(root):
    """Compile Audit only AFTER complete immutable candidate persistence."""
    root = Path(root).resolve()
    _, protocol, _, generation, queries, candidates, _ = load_generation(root)
    if any((root/p).exists() for p in ("audit_manifest.json", "audit_results.jsonl", "audit_started.json")):
        raise ValueError("Audit immutable/previously interrupted; no silent retry")
    write_json(root/"audit_started.json", seal(dict(generation_manifest=artifact(root/"generation_manifest.json"),
        candidate_file=generation["candidates"], all_candidates_saved_before_audit=True)))
    started = time.perf_counter()
    rules, policy_hashes, policy_artifacts, errors, phases, names = {}, {}, {}, {}, [], {}
    for source in protocol["sources"]:
        dataset = source["dataset"]
        if not any(q["dataset"] == dataset for q in queries):
            continue
        phase = time.perf_counter()
        try:
            p = json.loads(check_artifact(source["audit_policy"], root).read_text(encoding="utf-8"))
            if stable_hash(p["rules"]) != p["rule_set_sha256"]:
                raise ValueError("Frozen Audit rule-set checksum mismatch")
            rules[dataset], diagnostic = compile_audit_rules(p["rules"])
            if diagnostic["failed"]:
                raise ValueError("Audit compilation failed")
            policy_hashes[dataset] = dict(policy_sha256=stable_hash(p), rule_set_sha256=p["rule_set_sha256"])
            policy_artifacts[dataset] = source["audit_policy"]
            original = load_protocol(source["protocol"]["path"])
            names[dataset] = {i: normalize_activity_name(v) for i, v in enumerate(original["model"]["activity_vocabulary"], 1)}
        except Exception as exc:
            errors[dataset] = f"{type(exc).__name__}: {exc}"
        phases.append(dict(dataset=dataset, operation="post_save_Audit_compile", walltime_seconds=time.perf_counter()-phase))
    rows = []
    for candidate in candidates:
        dataset = candidate["dataset"]
        row = dict(candidate_id=candidate["candidate_id"], query_id=candidate["query_id"],
            candidate_file_sha256=generation["candidates"]["sha256"],
            audit_policy_sha256=policy_hashes.get(dataset, {}).get("policy_sha256"),
            audit_rule_set_sha256=policy_hashes.get(dataset, {}).get("rule_set_sha256"),
            rule_checks=[], applicable_rule_count=0, violated_rule_ids=[], error=None)
        try:
            if dataset in errors:
                raise ValueError(errors[dataset])
            trace = [names[dataset][a] for a in candidate["activities"]]
            row["rule_checks"] = [dict(rule_id=r["rule_id"], applicable=rule_applicable(trace, r),
                satisfied=bool(rule_satisfied(trace, r)) if rule_applicable(trace, r) else None) for r in rules[dataset]]
            active = [r for r in row["rule_checks"] if r["applicable"]]
            row["applicable_rule_count"] = len(active)
            row["violated_rule_ids"] = [r["rule_id"] for r in active if not r["satisfied"]]
            row["status"] = "not_applicable" if not active else "audit_fail" if row["violated_rule_ids"] else "audit_pass"
        except Exception as exc:
            row["status"], row["error"] = "audit_error", f"{type(exc).__name__}: {exc}"
        rows.append(row)
    for record in (generation["queries"], generation["candidates"]):
        check_artifact(record, root)
    write_jsonl(root/"audit_results.jsonl", rows)
    manifest = seal(dict(schema_version=1, purpose="P6_B_post_save_Audit_v1", generation_manifest=artifact(root/"generation_manifest.json"),
        candidate_file=generation["candidates"], audits=artifact(root/"audit_results.jsonl"), candidate_count=len(candidates),
        all_candidates_saved_before_audit=True, audit_used_in_generation=False, audit_used_in_selection=False,
        audit_used_in_tuning=False, policy_artifacts=policy_artifacts, policy_errors=errors,
        phases=phases, inclusive_walltime_seconds=time.perf_counter()-started, model_evaluations=0, training_started=False))
    write_json(root/"audit_manifest.json", manifest)
    return manifest


def load_audits(root, generation, candidates):
    root = Path(root)
    manifest = checked_json(root/"audit_manifest.json")
    if (manifest["generation_manifest"] != artifact(root/"generation_manifest.json") or manifest["candidate_file"] != generation["candidates"] or
            manifest["all_candidates_saved_before_audit"] is not True or any(manifest[k] for k in
            ("audit_used_in_generation", "audit_used_in_selection", "audit_used_in_tuning", "model_evaluations", "training_started"))):
        raise ValueError("Post-save Audit provenance/isolation drift")
    rows = read_jsonl(check_artifact(manifest["audits"], root))
    audits = {r["candidate_id"]: r for r in rows}
    if len(audits) != len(rows) or set(audits) != {c["candidate_id"] for c in candidates} or manifest["candidate_count"] != len(candidates):
        raise ValueError("Exactly one Audit per saved candidate required")
    for r in rows:
        if r["candidate_file_sha256"] != generation["candidates"]["sha256"]:
            raise ValueError("Candidate/Audit hash mismatch")
    return manifest, audits


def summarize(root):
    root = Path(root)
    bridge, protocol, cohort, generation, queries, candidates, _ = load_generation(root)
    audit_manifest, audits = load_audits(root, generation, candidates)
    view = dict(protocol, cohort=cohort, sources=[dict(s,
        historical_reference_build_walltime_seconds=s["reference"]["historical_build_walltime_seconds"],
        historical_reference_model_evaluations=0) for s in protocol["sources"]])
    rows = metric_rows(view, queries, candidates, audits)
    for row in rows:
        row["relaxed_improvement_success_at_k"] = row["relaxed_improved_queries"]/row["eligible_failure_queries"] if (
            row["eligible_failure_queries"] and row["improvement_success_at_k"] is not None) else None
    report = seal(dict(purpose="P6_B_two_scope_saved_metrics_v1", protocol=bridge["protocol"],
        generation_manifest=artifact(root/"generation_manifest.json"), audit_manifest=artifact(root/"audit_manifest.json"),
        groups=GROUPS, report_scopes=SCOPES, query_count=len(cohort), query_method_count=len(queries), candidate_count=len(candidates),
        rows=rows, costs=cost_rows(view, queries), prerequisite_costs=bridge["prerequisite_preparation_costs"],
        runtime_preparation_events=generation["preparation_events"], post_save_audit_phases=audit_manifest["phases"],
        historical_incremental_cf_training_seconds=None, training_started=False, audit_tuning_performed=False,
        formal_experiment_started=generation["formal_experiment_started"], no_performance_claim_from_smoke=generation["scope"] != "seven_task_matrix"))
    write_json(root/"metrics.json", report)
    return report


def export_statistical_input(root):
    """Only independently accepted complete 350x11 saves reach frozen statistics."""
    root = Path(root).resolve()
    bridge, _, cohort, generation, _, candidates, _ = load_generation(root)
    audit_manifest, _ = load_audits(root, generation, candidates)
    acceptance = checked_json(root/"independent_acceptance.json")
    saved = dict(queries=generation["queries"], candidates=generation["candidates"], audits=audit_manifest["audits"],
        generation_manifest=artifact(root/"generation_manifest.json"), audit_manifest=artifact(root/"audit_manifest.json"))
    if (generation["scope"] != "seven_task_matrix" or len(cohort) != 350 or generation["query_method_count"] != 3850 or
            acceptance["purpose"] != "P6_formal_generation_independent_acceptance_v1" or acceptance["status"] != "passed" or
            acceptance["scope"] != "seven_task_matrix" or acceptance["protocol"] != bridge["protocol"] or
            acceptance["saved_files"] != saved or any(acceptance[k] is not True for k in (
                "complete_denominators_and_same_candidate_joint_verified", "budget_and_intervention_verified", "search_and_audit_evidence_independently_verified"))):
        raise ValueError("Six-query/partial/unaccepted run cannot enter full-matrix statistics")
    for record in saved.values():
        check_artifact(record, root)
    for record in acceptance["bridge_code"]+[acceptance["verification_code"], acceptance["metrics"]]:
        check_artifact(record, root)
    report = seal(dict(schema_version=1, purpose="P6_formal_accepted_statistical_input_v1", protocol=bridge["protocol"],
        query_count=350, query_method_count=3850, methods=METHODS, independent_acceptance=artifact(root/"independent_acceptance.json"),
        saved_files=saved, all_candidates_saved_before_audit=True, audit_used_in_generation=False,
        audit_used_in_selection=False, audit_used_in_tuning=False))
    target = root/"statistical_input_manifest.json"
    if target.exists():
        if checked_json(target) != report:
            raise ValueError("Statistical input export immutable")
    else:
        write_json(target, report)
    return report
