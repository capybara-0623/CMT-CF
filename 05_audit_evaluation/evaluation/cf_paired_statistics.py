"""Post-save query indicators and paired, intact source-case block intervals.

No search, prediction, Audit compilation, training, p-values or tuning. The
estimand is a QUERY-weighted success rate, not a mean of case-level rates.
"""
from collections import defaultdict
import math
import numpy as np
from cf_search.suite import GROUPS, METHODS, SCOPES
from evaluation.experiment_protocol import stable_hash

METRICS = ["improvement_success_at_k", "strict_improvement_success_at_k",
    "audit_joint_improvement_success_at_k", "coverage", "goal_coverage",
    "relaxed_improvement_success_at_k"]
FAILURE_METRICS = set(METRICS) - {"coverage", "goal_coverage"}
CONTRASTS = [
    dict(a="strict_dfa_outcome", b="enum_outcome", group="fixed_time_search_main"),
    dict(a="strict_dfa_outcome", b="postfilter_outcome", group="fixed_time_search_main"),
    dict(a="R2_expanded_cap1", b="R1_expanded_cap0", group="same_pool_minimal_relaxation"),
    dict(a="O1", b="O0", group="fixed_time_privileged_head_ablation"),
    dict(a="O2", b="O1", group="time_intervention_auxiliary"),
    dict(a="strict_dfa_outcome", b="buliga_official_protocol_v1", group="fixed_time_search_main"),
]
STATISTICS_PLAN = dict(schema_version=1, estimator="query_weighted_ratio",
    resampling_unit="source_unit_and_original_case_id", retain_all_case_prefixes=True,
    synchronize_sepsis_derived_tasks=True, resample_tasks=False, resample_prefixes=False,
    paired=True, interval="percentile", confidence_level=.95, n_resamples=5000,
    bootstrap_seed=20260918, minimum_case_clusters=2, small_cluster_warning_below=10,
    conditional_on="frozen_predictor_and_model_seed42_search_seed42",
    p_values=False, multiplicity_adjusted=False, inferential_winner_selection=False,
    primary_aggregation="per_dataset", supplementary_aggregation="sepsis_source_case_synchronized",
    contrasts=CONTRASTS, metrics=METRICS, scopes=SCOPES,
    unavailable_method_rate=None, missing_common_membership="block_scope_not_zero",
    failure_denominator="all_factual_Failure_queries_in_scope_including_no_candidate",
    joint_definition="one_same_candidate_goal_AND_Search_strict_AND_activated_Audit_pass")


def source_unit(dataset):
    return "sepsis" if dataset.startswith("sepsis_cases_") else dataset


def _audit_pass(row):
    checks = row.get("rule_checks")
    if not isinstance(checks, list):
        raise ValueError("Per-rule Audit evidence is required")
    if len({r["rule_id"] for r in checks}) != len(checks):
        raise ValueError("Duplicate Audit rule evidence")
    if any(type(r.get("applicable")) is not bool or
           (r["applicable"] and type(r.get("satisfied")) is not bool) or
           (not r["applicable"] and r.get("satisfied") is not None) for r in checks):
        raise ValueError("Invalid Audit activation/satisfaction evidence")
    active = [r for r in checks if r["applicable"]]
    violations = [r["rule_id"] for r in active if not r["satisfied"]]
    expected = "not_applicable" if not active else "audit_fail" if violations else "audit_pass"
    if (row.get("status") != expected or row.get("applicable_rule_count") != len(active) or
            sorted(row.get("violated_rule_ids", [])) != sorted(violations) or row.get("error")):
        raise ValueError("Audit summary disagrees with per-rule evidence; zero activation cannot pass")
    return expected == "audit_pass"


def indicators(cohort, queries, candidates, audits, methods=METHODS, threshold=.5, topk=5):
    """Reject missing pairs instead of intersecting methods' successful rows."""
    frozen = {r["input_sha256"]: r for r in cohort}
    if len(frozen) != len(cohort) or not cohort or len(set(methods)) != len(methods):
        raise ValueError("Unique nonempty frozen inputs/methods required")
    actual = [(q["input_sha256"], q["method"]) for q in queries]
    expected = {(key, m) for key in frozen for m in methods}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Complete frozen query by method denominator required")
    by_q = {q["query_id"]: q for q in queries}
    if len(by_q) != len(queries) or len({c["candidate_id"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate query/candidate identity")
    if set(audits) != {c["candidate_id"] for c in candidates}:
        raise ValueError("Exactly one checked Audit per saved candidate required")
    grouped = defaultdict(list)
    for c in candidates:
        if c["query_id"] not in by_q:
            raise ValueError("Unknown candidate query")
        q = by_q[c["query_id"]]
        if any(c[k] != q[k] for k in ("method", "dataset", "input_sha256")):
            raise ValueError("Cross-query/method candidate attribution")
        probability = c["probability"]
        search = c["search"]
        if (not isinstance(probability, (float, int)) or isinstance(probability, bool) or
                not math.isfinite(probability) or not 0 <= probability <= 1 or
                type(search.get("strict")) is not bool or
                type(search.get("relaxation_count")) is not int or search["relaxation_count"] < 0 or
                search["strict"] and search["relaxation_count"]):
            raise ValueError("Invalid outcome/Search evidence")
        audit = audits[c["candidate_id"]]
        if audit.get("candidate_id") != c["candidate_id"] or audit.get("query_id") != q["query_id"]:
            raise ValueError("Cross-candidate Audit attribution")
        grouped[q["query_id"]].append((c, _audit_pass(audit)))
    rows = []
    for q in queries:
        r = frozen[q["input_sha256"]]
        if (q["dataset"] != r["dataset"] or q["original_class"] != r["original_class"] or
                q["original_probability"] != r["original_probability"] or
                q.get("common_buliga_applicable") != r["common_buliga_applicable"] or
                q["query_id"] != stable_hash(dict(input_sha256=q["input_sha256"], method=q["method"]))):
            raise ValueError("Factual/cohort/query identity drift")
        if (r["original_class"] not in ("Failure", "Success") or
                r["original_class"] != ("Success" if r["original_probability"] >= threshold else "Failure") or
                r["common_buliga_applicable"] is not None and type(r["common_buliga_applicable"]) is not bool):
            raise ValueError("Invalid factual class/cohort membership")
        cs = grouped[q["query_id"]]
        if (type(q["candidate_count"]) is not int or q["candidate_count"] != len(cs) or
                len(cs) > topk or sorted(c["rank"] for c, _ in cs) != list(range(1, len(cs)+1))):
            raise ValueError("Candidate count/Top-K/rank mismatch")
        if q.get("error") or q["status"] == "search_error":
            raise ValueError("Search errors cannot enter accepted statistics")
        if q["status"] not in {"completed", "model_budget_exhausted", "attempt_budget_exhausted",
                "walltime_exhausted", "proposal_pool_limit_reached", "precondition_not_met", "method_unavailable"}:
            raise ValueError("Unknown terminal search status")
        unavailable = q["status"] == "method_unavailable"
        if q["method"] == "created_unavailable" and not unavailable:
            raise ValueError("CREATED remains explicitly unavailable, not measured zero")
        if unavailable and cs:
            raise ValueError("Unavailable method cannot have measured candidates")
        failure = r["original_class"] == "Failure"
        goal = [(c, audit) for c, audit in cs if c["probability"] >= threshold]
        rows.append(dict(dataset=r["dataset"], input_sha256=q["input_sha256"], method=q["method"],
            case_id=r["context"]["case_id"], source_unit=source_unit(r["dataset"]),
            case_key=stable_hash(dict(source_unit=source_unit(r["dataset"]), case_id=r["context"]["case_id"])),
            eligible_failure=failure, common_buliga_applicable=r["common_buliga_applicable"],
            unavailable=unavailable, coverage=int(bool(cs)), goal_coverage=int(bool(goal)),
            improvement_success_at_k=int(failure and bool(goal)),
            strict_improvement_success_at_k=int(failure and any(c["search"]["strict"] for c, _ in goal)),
            audit_joint_improvement_success_at_k=int(failure and any(c["search"]["strict"] and audit for c, audit in goal)),
            relaxed_improvement_success_at_k=int(failure and any(c["search"]["relaxation_count"] > 0 for c, _ in goal))))
    return rows


def paired_case_interval(a, b, metric, plan=STATISTICS_PLAN):
    """Pair BEFORE masking; resample intact case vectors with common weights."""
    if metric not in METRICS:
        raise ValueError("Unknown predeclared metric")
    if (type(plan["n_resamples"]) is not int or plan["n_resamples"] < 1 or
            type(plan["bootstrap_seed"]) is not int or not 0 < plan["confidence_level"] < 1 or
            plan["minimum_case_clusters"] < 2):
        raise ValueError("Invalid case-bootstrap settings")
    left = {r["input_sha256"]: r for r in a}
    right = {r["input_sha256"]: r for r in b}
    if len(left) != len(a) or len(right) != len(b) or set(left) != set(right):
        raise ValueError("Paired query sets must be identical, no intersections")
    for key, r in left.items():
        if any(r[k] != right[key][k] for k in ("case_key", "eligible_failure", "common_buliga_applicable")):
            raise ValueError("Pair metadata drift")
        if any(type(x[metric]) is not int or x[metric] not in (0, 1) or
               type(x["unavailable"]) is not bool for x in (r, right[key])):
            raise ValueError("Binary indicator/availability required")
    eligible = [k for k in left if metric not in FAILURE_METRICS or left[k]["eligible_failure"]]
    case_counts = defaultdict(lambda: [0, 0, 0])
    for key in eligible:
        x = case_counts[left[key]["case_key"]]
        x[0] += left[key][metric]
        x[1] += right[key][metric]
        x[2] += 1
    n = len(eligible)
    counts = np.asarray([case_counts[k] for k in sorted(case_counts)], dtype=np.int64).reshape(-1, 3)
    unavailable_a = bool(a) and all(r["unavailable"] for r in a)
    unavailable_b = bool(b) and all(r["unavailable"] for r in b)
    result = dict(query_count=len(a), denominator=n, case_cluster_count=len(counts),
        numerator_a=int(counts[:, 0].sum()) if len(counts) else 0,
        numerator_b=int(counts[:, 1].sum()) if len(counts) else 0,
        unavailable_queries_a=sum(r["unavailable"] for r in a), unavailable_queries_b=sum(r["unavailable"] for r in b),
        rate_a=None, rate_b=None, difference=None, confidence_interval=None,
        confidence_level=plan["confidence_level"], bootstrap_seed=plan["bootstrap_seed"],
        requested_resamples=plan["n_resamples"], completed_resamples=0, warnings=[],
        multiplicity_adjusted=False, p_value=None, status="no_eligible_queries")
    if n:
        result["rate_a"] = None if unavailable_a else result["numerator_a"]/n
        result["rate_b"] = None if unavailable_b else result["numerator_b"]/n
    if unavailable_a or unavailable_b:
        result["status"] = "method_unavailable"
        return result
    if not n:
        return result
    result["difference"] = result["rate_a"] - result["rate_b"]
    if len(counts) < plan["small_cluster_warning_below"]:
        result["warnings"].append("few_case_clusters_engineering_or_exploratory_only")
    if len(counts) < plan["minimum_case_clusters"]:
        result["status"] = "insufficient_case_clusters"
        return result
    rng = np.random.default_rng(plan["bootstrap_seed"])
    draws = rng.integers(0, len(counts), size=(plan["n_resamples"], len(counts)))
    sampled = counts[draws].sum(axis=1)
    differences = (sampled[:, 0]-sampled[:, 1])/sampled[:, 2]
    tail = (1-plan["confidence_level"])/2
    result["confidence_interval"] = np.quantile(differences, [tail, 1-tail], method="linear").tolist()
    result["completed_resamples"] = len(differences)
    result["status"] = "degenerate_bootstrap" if np.all(differences == differences[0]) else "ok"
    if result["status"] == "degenerate_bootstrap":
        result["warnings"].append("zero_width_interval_is_not_evidence_of_equivalence")
    return result


def statistics_rows(rows, plan=STATISTICS_PLAN):
    if plan != STATISTICS_PLAN:
        raise ValueError("Only predeclared statistics plan is accepted")
    datasets = sorted({r["dataset"] for r in rows})
    aggregates = [(d, [d], "primary_per_dataset") for d in datasets]
    sepsis = [d for d in datasets if source_unit(d) == "sepsis"]
    if len(sepsis) > 1:
        aggregates.append(("sepsis_source_family", sepsis, "supplementary_query_task_weighted"))
    output = []
    for name, members, aggregation in aggregates:
        full = [r for r in rows if r["dataset"] in members]
        for scope in SCOPES:
            unknown = scope != "all_queries" and any(r["common_buliga_applicable"] is None for r in full)
            selected = [r for r in full if scope == "all_queries" or r["common_buliga_applicable"] is True]
            for contrast in CONTRASTS:
                if any(m not in GROUPS[contrast["group"]] for m in (contrast["a"], contrast["b"])):
                    raise ValueError("Invalid contrast information/time regime")
                for metric in METRICS:
                    result = dict(status="cohort_not_frozen", denominator=None, confidence_interval=None,
                        difference=None, rate_a=None, rate_b=None) if unknown else paired_case_interval(
                        [r for r in selected if r["method"] == contrast["a"]],
                        [r for r in selected if r["method"] == contrast["b"]], metric, plan)
                    output.append(dict(dataset=name, datasets=members, aggregation=aggregation,
                        scope=scope, metric=metric, **contrast, **result))
    return output
