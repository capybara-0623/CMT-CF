"""Read-only independent bridge acceptance. Never runs a CF search/selector.

Preflight acceptance does not restore/forward any model. Future saved-result
acceptance scores already persisted rows, with separately accounted cost.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

from cf_search.suite import METHODS, GROUPS, SCOPES, ROOT, checked_json, seal
from evaluation.cf_bridge_protocol import guard, CODE
from evaluation.cf_bridge_results import load_generation, load_audits, bare, SAVED_KEYS
from evaluation.cf_experiment_results import write_json
from evaluation.experiment_protocol import artifact, check_artifact, context_from_record, load_protocol, stable_hash
from explaination.visualizations.utils import normalize_activity_name


def independent_search(source, vocabulary, activities):
    names = [normalize_activity_name(vocabulary[a-1]) for a in activities]
    truth = {}
    for rid, graph in source["atomic_snapshots"].items():
        state = graph["initial_state"]
        for name in names:
            outgoing = {} if state is None else dict(graph["transitions"][state])
            state = outgoing.get(name, outgoing.get("*"))
        truth[rid] = state in graph["accepting_states"]
    protected = all(truth[r] for r in source["search"]["protected_rule_ids"])
    options = []
    for c in source["search"]["clusters"]:
        violations = tuple(sorted(r for r in c["rule_ids"] if not truth[r]))
        weights = {r["rule_id"]: r["relaxation_weight"] for r in source["search"]["rules"]}
        cost = math.fsum(weights[r] for r in violations)
        options.append((len(violations), cost, violations, c["cluster_id"]))
    minimum = min(options, default=(0, 0., (), None)) if protected else None
    return dict(strict=protected and (not options or any(o[0] == 0 for o in options)),
                protected=protected, minimum=minimum, truth=truth)


def independent_audit_checks(trace, rules):
    """Separate Declare implementation, including activation; no Audit compiler."""
    checks = []
    for rule in rules:
        a, b, template = rule.get("activation"), rule.get("target"), rule.get("template")
        apos, bpos = [i for i, x in enumerate(trace) if x == a], [i for i, x in enumerate(trace) if x == b]
        if template == "response":
            active, satisfied = bool(apos), all(any(j > i for j in bpos) for i in apos)
        elif template == "precedence":
            active, satisfied = bool(bpos), all(any(i < j for i in apos) for j in bpos)
        elif template == "chain_response":
            active, satisfied = bool(apos), all(i+1 < len(trace) and trace[i+1] == b for i in apos)
        elif template == "not_coexistence":
            active, satisfied = bool(apos or bpos), not (apos and bpos)
        else:
            raise ValueError("Unsupported independent Audit template; do not assume pass")
        checks.append(dict(rule_id=rule["rule_id"], applicable=active, satisfied=bool(satisfied) if active else None))
    return checks


def independent_intervention(context, row, method, vocabulary):
    activities, times = row["activities"], row["times"]
    if (len(activities) != len(context["activities"]) or len(times) != len(activities) or
            any(type(a) is not int or not 1 <= a <= len(vocabulary) for a in activities)):
        raise ValueError("Independent finite prefix/activity domain mismatch")
    edits = [i for i, (a, old) in enumerate(zip(activities, context["activities"])) if a != old]
    known = {i for i, raw in enumerate(vocabulary, 1) if raw.upper() not in ("__UNK__", "<UNK>", "UNK", "__PAD__", "<PAD>")}
    if (not edits or edits != row["edit_positions"] or row["edit_count"] != len(edits) or
            not set(edits) <= set(context["allowed_positions"]) or len(edits) > context["max_changed_positions"] or
            any(activities[i] not in known for i in edits)):
        raise ValueError("Independent frozen substitution/edit/position budget mismatch")
    if method != "O2":
        if times != context["times"]:
            raise ValueError("Independent fixed-time intervention mismatch")
        return
    if (any(not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t) or not 0 <= t <= 1 for t in times) or
            any(b < a for a, b in zip(times, times[1:])) or set(row["cf_time_raw"]) != {str(i) for i in edits}):
        raise ValueError("Independent O2 chronological/raw-coordinate evidence mismatch")
    expected = list(context["times"])
    for i in edits:
        raw = row["cf_time_raw"][str(i)]
        if not isinstance(raw, (int, float)) or isinstance(raw, bool) or not math.isfinite(raw) or not 0 <= raw <= 1:
            raise ValueError("Independent O2 raw time coordinate invalid")
        lower = expected[i-1] if i else 0.
        upper = next((context["times"][j] for j in range(i+1, len(activities)) if j not in edits), 1.)
        expected[i] = max(lower, min(upper, raw))
    if any(abs(a-b) > 1e-6 for a, b in zip(times, expected)):
        raise ValueError("Independent O2 neighbour projection mismatch")


def independent_reference_projection(reference, context, known_ids):
    rows, seen = [], set()
    for row in reference["rows"]:
        if len(row["activities"]) < len(context["activities"]):
            continue
        acts = list(context["activities"])
        for position in context["allowed_positions"]:
            if row["activities"][position] in known_ids:
                acts[position] = row["activities"][position]
        if tuple(acts) not in seen:
            rows.append(dict(case_id=row["case_id"], activities=acts))
            seen.add(tuple(acts))
        if len(rows) == 64:
            break
    return rows


def independent_selection(q, physical, source, vocabulary, candidates):
    pool = physical["rows"]
    method = q["method"]
    truths = [independent_search(source, vocabulary, r["activities"]) for r in pool]
    for row, truth in zip(pool, truths):
        if row["search"]["strict"] is not truth["strict"]:
            raise ValueError("Independent complete finite Search evidence mismatch")
    order = lambda r: (-r["probability"], r["edit_count"], tuple(r["activities"]))
    if q["family"] == "search":
        eligible = [r for r, truth in zip(pool, truths) if method == "enum_outcome" or truth["strict"]]
        expected = sorted(eligible, key=order)[:5]
    elif q["family"] == "relaxation":
        protected = [(r, t) for r, t in zip(pool, truths) if t["protected"]]
        strict = [(r, t) for r, t in protected if t["strict"]]
        capped = [(r, t) for r, t in protected if t["minimum"][0] <= int(method == "R2_expanded_cap1")]
        layer = (0, 0.) if strict else min((t["minimum"][:2] for _, t in capped), default=None)
        eligible = strict if strict else [(r, t) for r, t in capped if t["minimum"][:2] == layer]
        expected = [r for r, _ in sorted(eligible, key=lambda pair: order(pair[0]))[:5]]
        decision = q["selection"]
        if (decision["selected_layer"] != (list(layer) if layer is not None else None) or
                decision["strict_pool_size"] != len(strict) or decision["pool_sha256"] != physical["pool_sha256"] or
                decision["pool_size"] != len(pool) or decision["globally_protected_rejected"] != len(pool)-len(protected) or
                decision["minimum_layer_size"] != len(eligible)):
            raise ValueError("Independent strict-first explored-pool minimum layer mismatch")
        for candidate in candidates:
            truth = independent_search(source, vocabulary, candidate["activities"])
            count, cost, ids, cluster = truth["minimum"]
            e = candidate["search"]
            if (e["relaxation_count"] != count or e["relaxation_cost"] != cost or tuple(e["relaxed_constraint_ids"]) != ids or
                    e["cluster_id"] != cluster or not truth["protected"] or
                    any(not truth["truth"][r] for r in next((c["rule_ids"] for c in source["search"]["clusters"] if c["cluster_id"] == cluster), []) if r not in ids)):
                raise ValueError("Independent waiver/protected/retained rule mismatch")
        # Expected raw rows carry pre-waiver Search evidence; compare other fields.
        actual = [bare(c) for c in candidates]
        if [{k: v for k, v in r.items() if k != "search"} for r in expected] != [{k: v for k, v in r.items() if k != "search"} for r in actual]:
            raise ValueError("Independent fixed-pool Top-K mismatch")
        return
    elif q["family"] == "proprietary":
        proposed = sorted(pool, key=lambda r: (-r["proposal_score"], r["edit_count"], tuple(r["activities"])))[:5]
        if method != "O0" and q["status"] == "walltime_exhausted" and all(r["probability"] is None for r in proposed):
            proposed = []
        expected = [dict(r, probability_delta=r["probability"]-q["statistics"]["original_probability"]) for r in sorted(
            proposed, key=lambda r: (-r["probability"], r["edit_count"], tuple(r["activities"]), tuple(r["times"])))]
    else:
        if q["status"] in ("method_unavailable", "precondition_not_met"):
            expected = []
        else:
            by_acts = {tuple(r["activities"]): r for r in pool}
            anchor = q["statistics"]["projected_reference"][0]["activities"]
            for row in pool:
                plausibility = sum(a != b for a, b in zip(row["activities"], anchor))/len(anchor)
                loss = max(0., 1-2*row["probability"])+.5*row["edit_count"]/len(anchor)+.5*plausibility+3*(not row["search"]["strict"])
                if abs(plausibility-row["plausibility"]) > 1e-10 or abs(loss-row["native_loss"]) > 3e-7:
                    raise ValueError("Independent official plausibility/native objective mismatch")
            expected, qualifying = [], 0
            for acts in q["statistics"]["committed_population"]:
                row = by_acts.get(tuple(map(int, acts)))
                if row and row["probability"] >= .5 and row["search"]["strict"]:
                    qualifying += 1
                    if row not in expected:
                        expected.append(row)
                if qualifying >= 5:
                    break
            if not q["statistics"]["committed_population_fully_scored"]:
                expected = []
    if [bare(c) for c in candidates] != expected:
        raise ValueError("Independent full-pool/native-population selection mismatch")


def independent_costs(cohort, queries, physical):
    physical_map = {r["execution_id"]: r for r in physical}
    used, units = set(), Counter()
    for q in queries:
        s = q["statistics"]
        counts = dict(outcome_sequences=s["oracle"]["model_evaluations"], context_encodings=s.get("context_encodings", 0),
                      cf_outcome_evaluations=s.get("cf_outcome_evaluations", 0), cf_time_evaluations=s.get("cf_time_evaluations", 0))
        if any(type(v) is not int or v < 0 for v in counts.values()) or counts["outcome_sequences"]+counts["context_encodings"] > 500:
            raise ValueError("Sequence/head counter invalid or exceeds frozen budget")
        nodes = s.get("attempted_nodes", 0)
        if type(nodes) is not int or not 0 <= nodes <= 10000:
            raise ValueError("Actual node budget exceeded")
        if q["method"] not in ("O1", "O2") and any(counts[k] for k in ("context_encodings", "cf_outcome_evaluations", "cf_time_evaluations")):
            raise ValueError("Forbidden privileged-head use")
        if q["method"] != "O2" and counts["cf_time_evaluations"]:
            raise ValueError("Forbidden time-head use")
        if q["method"] == "created_unavailable":
            if any(counts.values()) or nodes:
                raise ValueError("CREATED must remain unavailable with zero execution cost")
            continue
        record = physical_map[q["execution_id"]]
        profile = record["costs"]
        if any(profile[k] != v for k, v in counts.items()) or s["p6a_profile"] != profile or record["error"] or q.get("error") or q["status"] == "search_error":
            raise ValueError("Actual call profile/cost trace or preserved error blocks acceptance")
        for k, v in profile.items():
            if k.endswith("seconds") and (not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0):
                raise ValueError("Invalid inclusive call timing")
        if q["execution_id"] not in used:
            units.update(counts)
            used.add(q["execution_id"])
    if used != set(physical_map) or len(used) != 9*len(cohort):
        raise ValueError("Shared physical execution missing or double-counted")
    return dict(physical_generation_count=len(used), **units)


def independent_indicators(cohort, queries, candidates, audits):
    grouped = defaultdict(list)
    for c in candidates:
        grouped[c["query_id"]].append(c)
    frozen = {r["input_sha256"]: r for r in cohort}
    rows = []
    for q in queries:
        r, cs = frozen[q["input_sha256"]], grouped[q["query_id"]]
        failure = r["original_class"] == "Failure"
        goals = [c for c in cs if c["probability"] >= .5]
        rows.append(dict(input_sha256=q["input_sha256"], method=q["method"], coverage=int(bool(cs)), goal_coverage=int(bool(goals)),
            improvement_success_at_k=int(failure and bool(goals)),
            strict_improvement_success_at_k=int(failure and any(c["search"]["strict"] for c in goals)),
            audit_joint_improvement_success_at_k=int(failure and any(c["search"]["strict"] and audits[c["candidate_id"]]["status"] == "audit_pass" for c in goals)),
            relaxed_improvement_success_at_k=int(failure and any(c["search"]["relaxation_count"] for c in goals))))
    return rows


def verify_preflight(preflight_dir):
    """Recheck complete factual membership and native-language proofs, no oracle."""
    started = time.perf_counter()
    root = Path(preflight_dir).resolve()
    if (root/"independent_acceptance.json").exists():
        raise ValueError("Independent acceptance immutable")
    bridge, protocol = guard(root)
    from tools.verify_p6b_prerequisites import complete_language_equivalence
    proofs, counts = [], []
    for source in protocol["sources"]:
        original = load_protocol(source["protocol"]["path"])
        vocabulary = original["model"]["activity_vocabulary"]
        proof = complete_language_equivalence(source, vocabulary)
        proofs.append(proof)
        reference = checked_json(check_artifact(source["reference"]["artifact"], root))
        allowed = set(original["preprocessing"]["search_discovery_case_ids"]+original["preprocessing"]["search_calibration_case_ids"])
        excluded = set().union(*(set(original["preprocessing"][k]) for k in
            ("audit_case_ids", "predictive_validation_case_ids", "predictive_test_case_ids")))
        if allowed & excluded or set(reference["case_ids"]) != allowed or reference["model_evaluations"]:
            raise ValueError("Independent training reference isolation failure")
        rows = [r for r in protocol["cohort"] if r["dataset"] == source["dataset"]]
        common, failures, common_failures = [], [], []
        for row in rows:
            original_row = original["queries"][row["offset"]]
            if any(row[k] != original_row[k] for k in ("context", "input_sha256", "original_probability", "original_class")):
                raise ValueError("Independent original query/prediction drift")
            truth = independent_search(source, vocabulary, row["context"]["activities"])
            has_length = any(len(r["activities"]) >= len(row["context"]["activities"]) for r in reference["rows"])
            member = truth["strict"] and has_length
            if truth["strict"] != row["factual_search_strict"] or member != row["common_buliga_applicable"]:
                raise ValueError("Independent common cohort membership drift")
            if member:
                common.append(row)
            if row["original_class"] == "Failure":
                failures.append(row)
                if member:
                    common_failures.append(row)
        counts.append(dict(dataset=source["dataset"], query_count=len(rows), factual_failure_queries=len(failures),
            common_query_count=len(common), common_failure_queries=len(common_failures),
            common_failure_case_count=len({r["context"]["case_id"] for r in common_failures})))
    if (sum(r["query_count"] for r in counts) != 350 or sum(r["factual_failure_queries"] for r in counts) != 226 or
            sum(r["common_query_count"] for r in counts) != 248 or sum(r["common_failure_queries"] for r in counts) != 174 or
            any((root/p).exists() for p in ("queries.jsonl", "candidates.jsonl", "generation_manifest.json", "audit_manifest.json", "physical", "inputs"))):
        raise ValueError("Full/common denominator or read-only scope violated")
    result = seal(dict(purpose="P6_B_bridge_read_only_independent_acceptance_v1", status="passed", bridge_preflight=artifact(root/"bridge_preflight.json"),
        protocol=bridge["protocol"], query_count=350, factual_failure_queries=226, common_query_count=248, common_failure_queries=174,
        query_method_count=3850, per_dataset=counts, complete_language_proofs=proofs, verification_code=artifact(__file__),
        model_evaluations=0, cf_head_forward_evaluations=0, KD_build_count=0, audit_compile_count=0, generated_candidate_count=0,
        training_started=False, formal_experiment_started=False, reference_recovery_started=False, statistical_model_fit_started=False,
        audit_tuning_performed=False, execution_authorized=False, inclusive_walltime_seconds=time.perf_counter()-started,
        interpretation="Readiness and interfaces accepted only; true query candidates and matrix have not been executed"))
    write_json(root/"independent_acceptance.json", result)
    return result


def verify_saved(run_dir):
    """Future independent acceptance consumes saved pools; creates NO new CF.

    Recompute full-pool selection, complete Search and activation-aware Audit;
    separately score all physical pools and original CF layers. Acceptance
    inference is not attributed to search. No statistical fit or training.
    """
    started = time.perf_counter()
    root = Path(run_dir).resolve()
    if (root/"independent_acceptance.json").exists():
        raise ValueError("Saved generation acceptance immutable")
    bridge, protocol, cohort, generation, queries, candidates, physical = load_generation(root)
    audit_manifest, audits = load_audits(root, generation, candidates)
    if audit_manifest["policy_errors"]:
        raise ValueError("Preserved Audit errors block independent acceptance")
    costs = independent_costs(cohort, queries, physical)
    if any(q["status"] == "method_unavailable" for q in queries if q["method"] in ("O1", "O2") and
           next(s for s in protocol["sources"] if s["dataset"] == q["dataset"])["cf_head_inventory"][q["method"]]["available"]):
        raise ValueError("Present frozen heads cannot be reported as unavailable")
    from tools.verify_p4b_smoke import score_rows, independent_head_table
    from evaluation.freeze_protocol import load_frozen_oracle
    from counterfactual import CounterfactualRecommender
    import torch
    torch.set_num_threads(1)
    maximum, head_maximum, factual_maximum, evaluations, head_contexts, pool_checks = 0., 0., 0., 0, 0, 0
    candidate_map, query_map = defaultdict(list), defaultdict(list)
    for c in candidates:
        candidate_map[c["query_id"]].append(c)
    for q in queries:
        query_map[q["dataset"]].append(q)
    physical_map = {p["execution_id"]: p for p in physical}
    for source in protocol["sources"]:
        dataset = source["dataset"]
        if dataset not in query_map:
            continue
        original = load_protocol(source["protocol"]["path"])
        vocabulary = original["model"]["activity_vocabulary"]
        base, contexts = load_frozen_oracle(source["protocol"]["path"])
        ckpt = torch.load(check_artifact(source["model"], root), map_location="cpu", weights_only=False)
        legacy = CounterfactualRecommender(base.model.cfg, base.model).eval().requires_grad_(False)
        state = ckpt["cf_model_state"]
        # Do not let cf_model_state's optional base_model.* overwrite the oracle.
        expected_state = legacy.state_dict()
        required = [k for k in expected_state if k.startswith(("action_embed.", "cf_outcome_head.", "cf_time_head."))]
        if any(k not in state or state[k].shape != expected_state[k].shape or state[k].dtype != expected_state[k].dtype or
               not torch.isfinite(state[k]).all() for k in required):
            raise ValueError("Independent original CF layers missing or invalid; no random fallback")
        legacy.load_state_dict({k: state[k] for k in required}, strict=False)
        policy = json.loads(check_artifact(source["audit_policy"], root).read_text(encoding="utf-8"))
        if stable_hash(policy["rules"]) != policy["rule_set_sha256"]:
            raise ValueError("Independent Audit policy checksum mismatch")
        ids = [i for i, raw in enumerate(vocabulary, 1) if raw.upper() not in ("__UNK__", "<UNK>", "UNK", "__PAD__", "<PAD>")]
        reference = checked_json(check_artifact(source["reference"]["artifact"], root))
        scored_executions, tables, factuals = set(), {}, set()
        frozen_map = {r["input_sha256"]: r for r in cohort if r["dataset"] == dataset}
        for q in query_map[dataset]:
            frozen = frozen_map[q["input_sha256"]]
            context = contexts[frozen["offset"]]
            if stable_hash(context.__dict__) != frozen["input_sha256"]:
                raise ValueError("Independent public prefix projection mismatch")
            if q["input_sha256"] not in factuals:
                value = score_rows(context, base.model, [dict(activities=context.activities, times=context.times)])[0]
                factual_maximum = max(factual_maximum, abs(value-frozen["original_probability"]))
                if abs(value-frozen["original_probability"]) > 2e-6 or (value >= .5) != (frozen["original_class"] == "Success"):
                    raise ValueError("Independent factual predictor mismatch")
                factuals.add(q["input_sha256"])
                evaluations += 1
            cs = sorted(candidate_map[q["query_id"]], key=lambda c: c["rank"])
            for c in cs:
                independent_intervention(frozen["context"], c, q["method"], vocabulary)
            if q["family"] != "availability":
                p = physical_map[q["execution_id"]]
                if q["family"] == "buliga":
                    strict_factual = independent_search(source, vocabulary, frozen["context"]["activities"])["strict"]
                    projected = independent_reference_projection(reference, frozen["context"], set(ids))
                    if (q["status"] == "precondition_not_met" and strict_factual or
                            q["status"] == "method_unavailable" and strict_factual and projected):
                        raise ValueError("Independent Buliga factual/reference prerequisite falsely reported unavailable")
                    if q["status"] not in ("precondition_not_met", "method_unavailable") and (
                            not strict_factual or q["statistics"]["projected_reference"] != projected or
                            q["statistics"]["projected_reference_sha256"] != stable_hash(projected)):
                        raise ValueError("Independent isolated reference projection/native factual precondition mismatch")
                independent_selection(q, p, source, vocabulary, cs)
                if q["execution_id"] not in scored_executions:
                    scores = score_rows(context, base.model, p["rows"])
                    evaluations += len(scores)
                    for row, value in zip(p["rows"], scores):
                        independent_intervention(frozen["context"], row, q["method"], vocabulary)
                        if row["probability"] is not None:
                            maximum = max(maximum, abs(value-row["probability"]))
                            if abs(value-row["probability"]) > 1e-6:
                                raise ValueError("Independent full-pool backbone probability mismatch")
                        if q["method"] in ("O0", "O1", "O2"):
                            if q["method"] == "O0":
                                expected_score = value
                            else:
                                cache_key = (context.sample_idx, q["method"] == "O2")
                                if cache_key not in tables:
                                    tables[cache_key] = independent_head_table(context, base.model, legacy, ids, q["method"] == "O2")
                                    head_contexts += len(context.allowed_positions)
                                table = tables[cache_key]
                                expected_score = math.fsum(table[i, row["activities"][i]][0] for i in row["edit_positions"])/row["edit_count"]
                                if q["method"] == "O2" and any(abs(row["cf_time_raw"][str(i)]-table[i, row["activities"][i]][1]) > 1e-6 for i in row["edit_positions"]):
                                    raise ValueError("Independent original time-head output mismatch")
                            head_maximum = max(head_maximum, abs(expected_score-row["proposal_score"]))
                            if abs(expected_score-row["proposal_score"]) > 1e-6:
                                raise ValueError("Independent original CF proposal score mismatch")
                        pool_checks += 1
                    scored_executions.add(q["execution_id"])
            for c in cs:
                trace = [normalize_activity_name(vocabulary[a-1]) for a in c["activities"]]
                checks = independent_audit_checks(trace, policy["rules"])
                active = [r for r in checks if r["applicable"]]
                violations = [r["rule_id"] for r in active if not r["satisfied"]]
                status = "not_applicable" if not active else "audit_fail" if violations else "audit_pass"
                a = audits[c["candidate_id"]]
                if (a["query_id"] != q["query_id"] or a["rule_checks"] != checks or a["status"] != status or
                        a["applicable_rule_count"] != len(active) or a["violated_rule_ids"] != violations or a.get("error") or
                        a["audit_policy_sha256"] != stable_hash(policy) or a["audit_rule_set_sha256"] != policy["rule_set_sha256"]):
                    raise ValueError("Independent per-rule activation-aware Audit mismatch")
        del legacy, ckpt, base
    from evaluation.cf_paired_statistics import indicators
    binary = indicators(cohort, queries, candidates, audits)
    manual = independent_indicators(cohort, queries, candidates, audits)
    if any(any(a[k] != b[k] for k in b) for a, b in zip(binary, manual)):
        raise ValueError("Independent complete-denominator same-candidate statistical indicator mismatch")
    # Metrics are required before export; retain errors rather than report success.
    metrics = checked_json(root/"metrics.json")
    for record in (metrics["generation_manifest"], metrics["audit_manifest"], metrics["protocol"]):
        check_artifact(record, root)
    expected_rows = {(r["dataset"], scope, method) for r in cohort for scope in SCOPES for method in METHODS}
    actual_rows = [(r["dataset"], r["scope"], r["method"]) for r in metrics["rows"]]
    if len(actual_rows) != len(set(actual_rows)) or set(actual_rows) != expected_rows:
        raise ValueError("Complete unique two-scope method summary required")
    for row in metrics["rows"]:
        selected = [q for q in queries if q["dataset"] == row["dataset"] and q["method"] == row["method"] and
                    (row["scope"] == "all_queries" or q["common_buliga_applicable"])]
        failures = [q for q in selected if q["original_class"] == "Failure"]
        byid = {(r["input_sha256"], r["method"]): r for r in manual}
        for metric in ("improvement_success_at_k", "strict_improvement_success_at_k", "audit_joint_improvement_success_at_k", "relaxed_improvement_success_at_k", "coverage", "goal_coverage"):
            denominator = selected if metric in ("coverage", "goal_coverage") else failures
            expected = sum(byid[q["input_sha256"], q["method"]][metric] for q in denominator)/len(denominator) if (
                denominator and any(q["status"] != "method_unavailable" for q in selected)) else None
            if row[metric] != expected:
                raise ValueError("Independent two-scope summary denominator/rate mismatch")
    if metrics["groups"] != GROUPS or metrics["report_scopes"] != SCOPES or len(metrics["rows"]) != 22*len({r["dataset"] for r in cohort}):
        raise ValueError("Predeclared group/scope report incomplete")
    expected_costs = dict(actual_unique_outcome_evaluations=costs["outcome_sequences"],
        actual_unique_context_encodings=costs["context_encodings"], actual_unique_cf_outcome_evaluations=costs["cf_outcome_evaluations"],
        actual_unique_cf_time_evaluations=costs["cf_time_evaluations"])
    if any(metrics["costs"][k] != v for k, v in expected_costs.items()) or len(metrics["costs"]["per_query"]) != len(queries):
        raise ValueError("Reported unique execution costs double-counted/missing")
    saved = dict(queries=generation["queries"], candidates=generation["candidates"], audits=audit_manifest["audits"],
        generation_manifest=artifact(root/"generation_manifest.json"), audit_manifest=artifact(root/"audit_manifest.json"))
    result = seal(dict(purpose="P6_formal_generation_independent_acceptance_v1", status="passed", protocol=bridge["protocol"],
        scope=generation["scope"], query_count=len(cohort), query_method_count=len(queries), candidate_count=len(candidates), saved_files=saved,
        metrics=artifact(root/"metrics.json"), complete_denominators_and_same_candidate_joint_verified=True,
        budget_and_intervention_verified=True, search_and_audit_evidence_independently_verified=True,
        verification_code=artifact(__file__), bridge_code=[artifact(ROOT/p) for p in CODE],
        independently_rechecked_physical_pool_rows=pool_checks, acceptance_outcome_evaluations=evaluations,
        acceptance_cf_predecessor_encodings=head_contexts, backbone_max_abs_delta=maximum, head_max_abs_delta=head_maximum,
        factual_max_abs_delta=factual_maximum, actual_unique_generation_costs=costs, acceptance_cost_separate=True,
        new_candidate_generation=False, training_started=False, statistical_model_fit_started=False,
        audit_tuning_performed=False, inclusive_walltime_seconds=time.perf_counter()-started))
    write_json(root/"independent_acceptance.json", result)
    return result
