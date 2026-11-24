"""Deterministic edit-layer enumeration with sound left-to-right DFA pruning."""
from dataclasses import dataclass
from collections import defaultdict, deque

from counterfactual import evaluate_minimal_constraint_relaxation, verify_retained_constraints
from evaluation.cf_types import Candidate, SearchResult
from evaluation.experiment_protocol import stable_hash
from evaluation.outcome_oracle import OracleBudgetExceeded
from explaination.visualizations.utils import normalize_activity_name
from cf_search.budget import BudgetExecutor


METHODS = ("enum_outcome", "postfilter_outcome", "strict_dfa_outcome")


@dataclass
class SearchPolicy:
    clusters: list
    activity_names: dict
    protected_rule_ids: tuple = ()

    def __post_init__(self):
        atomic = {str(r.get("rule_id", r.get("pid"))): r
                  for c in self.clusters for r in c["rule_dfas"]}
        if (set(map(str, self.protected_rule_ids)) - set(atomic) or
                any(atomic[str(r)].get("compile_mode", "strict") != "strict"
                    for r in self.protected_rule_ids)):
            raise ValueError("Protected rule missing or compilation-weakened")

    def names(self, activities):
        return [self.activity_names[a] for a in activities]

    def evidence(self, activities):
        if not self.clusters:
            if self.protected_rule_ids:
                raise ValueError("Protected rules require compiled policy")
            return dict(strict=True, retained_rules_verified=True, cluster_id=None,
                        relaxed_constraint_ids=[], relaxation_count=0, relaxation_cost=0.)
        names = self.names(activities)
        found = evaluate_minimal_constraint_relaxation(
            names, self.clusters, max_relaxed_constraints=0,
            non_relaxable_rule_ids=self.protected_rule_ids)
        return dict(strict=found is not None,
                    retained_rules_verified=bool(found is not None and verify_retained_constraints(
                        names, self.clusters, found["cluster_id"], ()) == 1),
                    cluster_id=found["cluster_id"] if found else None,
                    relaxed_constraint_ids=[], relaxation_count=0, relaxation_cost=0.)


class _Pruner:
    def __init__(self, policy):
        self.rules = {}
        self.groups = []
        for cluster in policy.clusters:
            ids = []
            for rule in cluster["rule_dfas"]:
                rid = str(rule.get("rule_id", rule.get("pid")))
                self.rules[rid] = rule["dfa"]
                ids.append(rid)
            self.groups.append(ids)
        self.protected = tuple(map(str, policy.protected_rule_ids))
        if set(self.protected) - set(self.rules):
            raise ValueError("Protected rules are missing")
        self.live = {}
        for rid, dfa in self.rules.items():
            reverse = defaultdict(set)
            for state, outgoing in dfa.transition_function.items():
                for target in outgoing.values():
                    reverse[target].add(state)
            live = set(dfa.accepting_states)
            pending = deque(live)
            while pending:
                for previous in reverse[pending.popleft()]:
                    if previous not in live:
                        live.add(previous)
                        pending.append(previous)
            self.live[rid] = live

    def initial(self):
        return {rid: dfa.initial_state for rid, dfa in self.rules.items()}

    def advance(self, states, name):
        updated = {}
        for rid, dfa in self.rules.items():
            outgoing = dfa.transition_function.get(states[rid], {})
            atom = normalize_activity_name(name)
            updated[rid] = outgoing.get(name, outgoing.get(atom, outgoing.get("*")))
        if any(updated[rid] not in self.live[rid] for rid in self.protected):
            return None
        if self.groups and not any(all(updated[rid] in self.live[rid] for rid in group)
                                   for group in self.groups):
            return None
        return updated


def ranking(row):
    return (-row["probability"], row["edit_count"], tuple(row["activities"]))


def select_candidates(pool, method, topk):
    """Filtering precedes truncation and consumes the entire scored pool."""
    if method not in METHODS or type(topk) is not int or topk < 1:
        raise ValueError("Invalid method or Top-K")
    eligible = pool if method == "enum_outcome" else [r for r in pool if r["search"]["strict"]]
    return sorted(eligible, key=ranking)[:topk]


def generate(query, oracle, search_policy, budget, method="strict_dfa_outcome", topk=5,
             activity_ids=None, pool_only=False):
    if method not in METHODS:
        raise ValueError("Unknown search method")
    if type(topk) is not int or topk < 1:
        raise ValueError("Top-K must be positive")
    # Fresh instances prevent cross-method or cross-query cache/budget leakage.
    if any(oracle.statistics().values()):
        raise ValueError("Search requires a fresh per-query oracle")
    if oracle.budget.max_model_evaluations != budget.model_evaluations:
        raise ValueError("Oracle and search sequence budgets must match")
    ids = tuple(sorted(search_policy.activity_names if activity_ids is None else activity_ids))
    if (not ids or len(set(ids)) != len(ids) or
            any(type(a) is not int or a <= 0 or a not in search_policy.activity_names for a in ids)):
        raise ValueError("Invalid shared activity domain")
    executor = BudgetExecutor(budget)
    pool, pruned, leaves = [], 0, 0
    pruner = _Pruner(search_policy) if method == "strict_dfa_outcome" else None
    original = float(oracle.predict_proba([Candidate(query.activities)], query)[0, 1])
    allowed = set(query.allowed_positions)
    remaining = [sum(p >= i for p in allowed) for i in range(query.prefix_len + 1)]

    def walk(index, prefix, edits, target, states):
        nonlocal pruned, leaves
        if executor.status != "completed" or not executor.check():
            return
        if edits > target or edits + remaining[index] < target:
            return
        if index == query.prefix_len:
            leaves += 1
            evidence = search_policy.evidence(prefix)
            if method == "strict_dfa_outcome" and not evidence["strict"]:
                return
            if oracle.stats.model_evaluations >= budget.model_evaluations:
                executor.status = "model_budget_exhausted"
                return
            try:
                probability = float(oracle.predict_proba([Candidate(tuple(prefix))], query)[0, 1])
            except OracleBudgetExceeded:
                executor.status = "model_budget_exhausted"
                return
            changed = [i for i, (a, b) in enumerate(zip(query.activities, prefix)) if a != b]
            pool.append(dict(activities=list(prefix), times=list(query.times), probability=probability,
                             probability_delta=probability-original, edit_positions=changed,
                             edit_count=len(changed), search=evidence))
            return
        options = (query.activities[index],) if index not in allowed else tuple(
            [query.activities[index]] + [a for a in ids if a != query.activities[index]])
        for activity in options:
            new_edits = edits + (activity != query.activities[index])
            if new_edits > target or new_edits + remaining[index+1] < target:
                continue
            if not executor.visit():
                return
            updated = pruner.advance(states, search_policy.activity_names[activity]) if pruner else None
            if pruner and updated is None:
                pruned += 1
                continue
            walk(index+1, prefix+[activity], new_edits, target, updated)
            if executor.status != "completed":
                return

    for target in range(1, min(query.max_changed_positions, len(allowed)) + 1):
        walk(0, [], 0, target, pruner.initial() if pruner else None)
        if executor.status != "completed":
            break
    # P4 persists this raw pool before applying either waiver policy.
    selected = [] if pool_only else select_candidates(pool, method, topk)
    for row in selected:
        if search_policy.evidence(row["activities"]) != row["search"]:
            raise ValueError("Final full Search verification mismatch")
    # Re-score through the same cache; these requests are explicitly counted.
    if selected and executor.check():
        rescored = oracle.predict_proba([Candidate(tuple(r["activities"])) for r in selected], query)
        for row, score in zip(selected, rescored):
            if float(score[1]) != row["probability"]:
                raise ValueError("Final scoring mismatch")
    return SearchResult(selected, executor.status, dict(
        oracle=oracle.statistics(), original_probability=original, attempted_nodes=executor.attempts,
        completed_sequences=leaves, pruned_branches=pruned, scored_pool_size=len(pool),
        scored_pool_sha256=stable_hash(pool), scored_pool=pool, walltime_seconds=executor.elapsed,
        cf_head_calls=0, time_policy="fixed", attempt_unit="assigned_prefix_nodes"))
