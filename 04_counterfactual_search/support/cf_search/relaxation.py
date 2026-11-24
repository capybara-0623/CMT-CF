"""Pure fixed-pool strict-first selection. Audit and prediction thresholds absent."""
import copy
import math

from pythomata import SimpleDFA

from counterfactual import evaluate_minimal_constraint_relaxation, verify_retained_constraints
from evaluation.experiment_protocol import stable_hash
from cf_search.methods import SearchPolicy, ranking


RELAXATION_METHODS = ("R0_strict", "R1_expanded_cap0", "R2_expanded_cap1")
POLICY_ID = "strict_pool_first_minimum_violations_v1"


def minimum_evidence(activities, policy):
    if not policy.clusters:
        return dict(policy.evidence(activities), global_protected_verified=True,
                    protected_rule_count=0, retained_cluster_rule_count=0,
                    retained_rule_status="unconstrained", relaxed_rule_weights={},
                    compile_weakened_rule_ids=[], compile_omitted_rule_ids=[])
    found = evaluate_minimal_constraint_relaxation(
        policy.names(activities), policy.clusters, non_relaxable_rule_ids=policy.protected_rule_ids,
        exact_cost_order=True)
    if found is None:
        return None
    retained = found["satisfied_constraint_count"]
    verified = verify_retained_constraints(policy.names(activities), policy.clusters,
                                           found["cluster_id"], found["relaxed_constraint_ids"])
    if retained and verified != 1:
        raise ValueError("Retained rules failed final verification")
    atomic = {str(r.get("rule_id", r.get("pid"))): r
              for c in policy.clusters for r in c["rule_dfas"]}
    return dict(strict=found["strict"], cluster_id=found["cluster_id"],
                retained_rules_verified=True if retained else None,
                retained_cluster_rule_count=retained,
                retained_rule_status="verified" if retained else "vacuous",
                global_protected_verified=True, protected_rule_count=len(policy.protected_rule_ids),
                relaxed_constraint_ids=list(found["relaxed_constraint_ids"]),
                relaxed_rule_weights={rid:float(atomic[rid]["relaxation_weight"])
                                     for rid in found["relaxed_constraint_ids"]},
                relaxation_count=found["relaxation_count"], relaxation_cost=found["relaxation_cost"],
                compile_weakened_rule_ids=list(found["compile_relaxed_constraint_ids"]),
                compile_omitted_rule_ids=list(found["compile_omitted_constraint_ids"]))


def select_minimal_relaxation(pool, policy, max_relaxed_constraints=0, topk=5):
    """Minimum is over this explored pool, not all possible interventions.

    A strict non-improving candidate still blocks relaxation. Factual zero-edit
    traces are not candidate recommendations and cannot establish a strict pool.
    """
    if (type(max_relaxed_constraints) is not int or max_relaxed_constraints < 0
            or type(topk) is not int or topk < 1):
        raise ValueError("Invalid waiver cap or Top-K")
    fingerprint = stable_hash(pool)
    annotated, protected_rejected, ids = [], 0, set()
    for row in pool:
        acts = tuple(row["activities"])
        if acts in ids or row["edit_count"] < 1:
            raise ValueError("Pool contains duplicate or factual candidate")
        ids.add(acts)
        evidence = minimum_evidence(acts, policy)
        if evidence is None:
            protected_rejected += 1
            continue
        if evidence["strict"] != row["search"]["strict"]:
            raise ValueError("Pool strict evidence differs from full rule verification")
        annotated.append(dict(copy.deepcopy(row), search=evidence))
    strict = [r for r in annotated if r["search"]["strict"]]
    capped = [r for r in annotated if r["search"]["relaxation_count"] <= max_relaxed_constraints]
    layer = (0, 0.) if strict else min(
        ((r["search"]["relaxation_count"], r["search"]["relaxation_cost"]) for r in capped), default=None)
    eligible = strict if strict else [r for r in capped if
        (r["search"]["relaxation_count"], r["search"]["relaxation_cost"]) == layer]
    selected = sorted(eligible, key=ranking)[:topk]
    for row in selected:
        if minimum_evidence(row["activities"], policy) != row["search"]:
            raise ValueError("Output retained-rule recheck mismatch")
    if stable_hash(pool) != fingerprint:
        raise ValueError("Fixed pool was mutated during selection")
    return selected, dict(policy_id=POLICY_ID, minimum_scope="explored_pool_only",
        max_relaxed_constraints=max_relaxed_constraints, pool_sha256=fingerprint,
        pool_size=len(pool), strict_pool_size=len(strict), globally_protected_rejected=protected_rejected,
        cap_admissible_pool_size=len(capped), selected_layer=list(layer) if layer is not None else None,
        minimum_layer_size=len(eligible), returned_count=len(selected))


def policy_from_manifest(manifest, dataset):
    """Restore the saved actual automata, without compiling or reading Audit."""
    source = next(s for s in manifest["sources"] if s["dataset"] == dataset)
    snapshot = manifest["runtime_search_dfa_snapshots"][dataset]
    rules = {}
    for rule in source["search"]["rules"]:
        graph = snapshot[rule["rule_id"]]
        outgoing = {i:dict(row) for i, row in enumerate(graph["transitions"])}
        letters = {letter for row in outgoing.values() for letter in row}
        dfa = SimpleDFA(set(outgoing), letters, graph["initial_state"],
                        set(graph["accepting_states"]), outgoing)
        rules[rule["rule_id"]] = dict(rule, dfa=dfa)
    clusters = [dict(cluster_id=c["cluster_id"], rule_dfas=[rules[r] for r in c["rule_ids"]])
                for c in source["search"]["clusters"]]
    return SearchPolicy(clusters, {int(k):v for k,v in manifest["activity_names"][dataset].items()},
                        tuple(source["search"]["protected_rule_ids"]))


def validate_relaxation_candidate(evidence, method):
    cap = 1 if method == "R2_expanded_cap1" else 0
    if method not in RELAXATION_METHODS:
        raise ValueError("Unknown P4-A method")
    ids, weights = evidence["relaxed_constraint_ids"], evidence["relaxed_rule_weights"]
    if (type(evidence["strict"]) is not bool or type(evidence["relaxation_count"]) is not int
            or not 0 <= evidence["relaxation_count"] <= cap
            or len(set(ids)) != len(ids) or len(ids) != evidence["relaxation_count"]
            or set(weights) != set(ids) or any(not math.isfinite(w) or w <= 0 for w in weights.values())
            or math.fsum(weights.values()) != evidence["relaxation_cost"]
            or evidence["strict"] != (not ids) or evidence["global_protected_verified"] is not True):
        raise ValueError("Invalid or unprotected waiver evidence")
    status, count = evidence["retained_rule_status"], evidence["retained_cluster_rule_count"]
    if (status == "verified" and (count < 1 or evidence["retained_rules_verified"] is not True)
            or status == "vacuous" and (count != 0 or evidence["retained_rules_verified"] is not None)
            or status == "unconstrained" and (count != 0 or evidence["retained_rules_verified"] is not True)
            or status not in ("verified", "vacuous", "unconstrained")):
        raise ValueError("Invalid retained-rule verification evidence")
