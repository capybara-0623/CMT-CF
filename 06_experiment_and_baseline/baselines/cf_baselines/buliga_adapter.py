"""Author ONLINE crossover and loss, adapted to the frozen outcome-only oracle.

Not an end-to-end replication of the paper: initialization, bounded orchestration,
fixed action space, disabled reference plausibility, and result policy are disclosed.
"""
import math
import re
from types import MethodType, SimpleNamespace
import numpy as np
from cf_search.budget import BudgetExecutor
from evaluation.cf_types import Candidate, SearchResult
from evaluation.experiment_protocol import stable_hash
from evaluation.outcome_oracle import OracleBudgetExceeded
from cf_baselines.source import AUTHOR_METHODS, load_author_methods


class _Stop(Exception):
    pass


class _InvalidOffspring(Exception):
    pass


class _BoundedRNG:
    """Author ONLINE draws random() once per actually constructed gene."""
    def __init__(self, generator, executor):
        self.generator, self.executor = generator, executor

    def random(self):
        if not self.executor.visit():
            raise _Stop()
        return self.generator.random()

    def __getattr__(self, name):
        return getattr(self.generator,name)


def select_native(pool, topk, threshold):
    eligible = [r for r in pool if r["search"]["strict"] and r["probability"] >= threshold]
    return sorted(eligible, key=lambda r:(r["native_loss"], tuple(r["activities"])))[:topk]


def _symbol(index):
    # Pure letters avoid the upstream Utils digit-to-letter normalization collision.
    value, chars = index, []
    while value:
        value, rem = divmod(value-1, 26)
        chars.append(chr(97+rem))
    return "event"+"".join(reversed(chars))


class _Encoder:
    def __init__(self, query, ids, symbols, features):
        self.forward = symbols
        self.reverse = {v:k for k,v in symbols.items()}
        self._label_dict = {f:{symbols[a]:a for a in (ids if i in query.allowed_positions else
            (query.activities[i],))} for i,f in enumerate(features)}

    def decode(self, frame):
        for column in frame:
            frame[column] = frame[column].map(lambda x:self.forward[int(x)])

    def encode(self, frame):
        for column in frame:
            frame[column] = frame[column].map(self.reverse)


class PolicyAutomaton:
    """Lazy product facade: global protection AND OR-of-conjunctive clusters.

    Original ONLINE asks which letters preserve the factual next automaton state;
    this facade provides that exact state equivalence over all atomic DFAs.
    No rule relaxation and no reachable-accept-state heuristic is substituted.
    """
    def __init__(self, policy, symbols, executor=None):
        self.policy = policy
        self.rules = {str(r["rule_id"]):r["dfa"] for c in policy.clusters for r in c["rule_dfas"]}
        self.ids = tuple(sorted(self.rules))
        self.initial_state = tuple(self.rules[r].initial_state for r in self.ids)
        self.symbol_names = {symbols[i]:n for i,n in policy.activity_names.items()}
        self.visited = {self.initial_state}
        self.executor = executor
        self.transition_checks = 0

    def _advance(self, state, letter):
        self.transition_checks += 1
        if self.executor and not self.executor.check():
            raise _Stop()
        name = self.symbol_names[letter]
        return tuple(self.rules[r].transition_function.get(s, {}).get(name,
            self.rules[r].transition_function.get(s, {}).get("*")) for r,s in zip(self.ids,state))

    def get_successors(self, state, interpretation):
        letter = next(k for k,v in interpretation.items() if v)
        if letter not in self.symbol_names:
            # The pinned ONLINE body can leave non-constraint genes as NaN.
            # Reject that proposal; do not repair/score it or silently patch author code.
            raise _InvalidOffspring()
        successor = self._advance(state, letter)
        self.visited.add(successor)
        return {successor}

    def get_transitions(self):
        # Lazily enumerate only states already requested by the author check_state.
        return [(s, letter, self._advance(s,letter)) for s in tuple(self.visited)
                for letter in self.symbol_names]

    def is_accepting(self, state):
        accepted = {r:s in self.rules[r].accepting_states for r,s in zip(self.ids,state)}
        return (all(accepted[str(r)] for r in self.policy.protected_rule_ids) and
                (not self.policy.clusters or any(all(accepted[str(r["rule_id"])] for r in c["rule_dfas"])
                 for c in self.policy.clusters)))


def generate(query, oracle, search_policy, budget, topk=5, activity_ids=None,
             threshold=.5, seed=42, population_size=40, max_generations=50,
             source_root=".runtime/p5_sources/buliga"):
    if (type(topk) is not int or topk < 1 or type(seed) is not int or
            type(population_size) is not int or population_size < 4 or
            type(max_generations) is not int or max_generations < 1 or
            not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError("Invalid external search configuration")
    if any(oracle.statistics().values()) or oracle.budget.max_model_evaluations != budget.model_evaluations:
        raise ValueError("Requires a fresh budget-matched oracle")
    ids = tuple(sorted(search_policy.activity_names if activity_ids is None else activity_ids))
    if (not ids or len(ids) != len(set(ids)) or
            any(type(i) is not int or i not in search_policy.activity_names for i in ids)):
        raise ValueError("Invalid shared activity domain")
    executor = BudgetExecutor(budget)
    pool, seen = [], set()
    stats = dict(cf_head_calls=0, time_policy="fixed", attempt_unit="assigned_prefix_nodes",
                 author_operator_calls=0, author_loss_calls=0, edit_budget_rejections=0,
                 strict_rejections=0, duplicate_offspring=0, search_seed=seed,
                 source_encoding_rejections=0,
                 population_size=population_size, max_generations=max_generations,
                 initialization="factual_clones_then_author_online_variation",
                 reference_plausibility_enabled=False, original_pipeline_reproduced=False,
                 usable_external_baseline=True, completed_generations=0)

    def finish(status=None):
        selected = select_native(pool,topk,threshold)
        if selected:
            probabilities = oracle.predict_proba([Candidate(tuple(r["activities"])) for r in selected],query)
            for row,p in zip(selected,probabilities):
                if float(p[1]) != row["probability"] or row["search"] != search_policy.evidence(row["activities"]):
                    raise ValueError("Final external candidate recheck mismatch")
        return SearchResult(selected,status or executor.status,dict(stats, oracle=oracle.statistics(),
            attempted_nodes=executor.attempts, walltime_seconds=executor.elapsed,
            scored_pool=pool, scored_pool_size=len(pool), scored_pool_sha256=stable_hash(pool)))

    if not search_policy.evidence(query.activities)["strict"]:
        stats["reason"] = "Original ONLINE precondition not met: factual finite prefix is not Search compliant"
        stats["usable_external_baseline"] = False
        return finish("precondition_not_met")
    namespace = load_author_methods(source_root)
    original = float(oracle.predict_proba([Candidate(query.activities)],query)[0,1])
    stats["original_probability"] = original
    features = [f"prefix_{i}" for i in range(query.prefix_len)]
    symbols = {i:_symbol(i) for i in search_policy.activity_names}
    encoder = _Encoder(query,ids,symbols,features)
    dfa = PolicyAutomaton(search_policy,symbols,executor)
    original_symbols = {n:symbols[i] for i,n in search_policy.activity_names.items()}
    formulas = " ".join(str(r["formula"]) for c in search_policy.clusters for r in c["rule_dfas"])
    atoms = set(re.findall(r"[a-z_][a-z_0-9]*",formulas))
    activities = [symbols[i] for i,n in search_policy.activity_names.items() if n in atoms]
    varied = [features[i] for i in query.allowed_positions]
    engine = SimpleNamespace(data_interface=SimpleNamespace(feature_names=features,
        number_of_features=query.prefix_len, continuous_feature_indexes=[]),
        rng=_BoundedRNG(np.random.default_rng(seed),executor), model=SimpleNamespace(model_type="classifier"),
        num_output_nodes=2, yloss_type="hinge_loss", x1=np.array(query.activities),
        sparsity_weight=.5, proximity_weight=0., plausibility_weight=0., conformance_weight=3.,
        conformance_score=np.ones(1), query_instance_normalized=None)
    for method in AUTHOR_METHODS:
        setattr(engine,method,MethodType(namespace[method],engine))
    # This term is explicitly disabled: no leaked test/reference distribution.
    engine.compute_plausibility = lambda cfs:np.zeros(len(cfs))
    engine.feature_range = {f:np.array(tuple(encoder._label_dict[f].values())) for f in features}
    engine.predict_fn_scores = lambda cfs:oracle.predict_proba(
        [Candidate(tuple(map(int,row))) for row in cfs],query)

    def evaluate(sequence):
        sequence = tuple(map(int,sequence))
        evidence = search_policy.evidence(sequence)
        if not evidence["strict"]:
            stats["strict_rejections"] += 1
            return None
        if oracle.stats.model_evaluations >= budget.model_evaluations:
            executor.status = "model_budget_exhausted"
            raise _Stop()
        if not executor.check():
            raise _Stop()
        stats["author_loss_calls"] += 1
        try:
            loss = float(engine.compute_loss(None,np.array([sequence]),None,1)[0,1])
        except OracleBudgetExceeded:
            executor.status = "model_budget_exhausted"
            raise _Stop()
        probability = float(oracle.predict_proba([Candidate(sequence)],query)[0,1])
        edits = [i for i,(a,b) in enumerate(zip(query.activities,sequence)) if a != b]
        row = dict(activities=list(sequence), times=list(query.times), probability=probability,
            probability_delta=probability-original, edit_positions=edits, edit_count=len(edits),
            search=evidence,native_loss=loss)
        if edits and sequence not in seen:
            pool.append(row)
            seen.add(sequence)
        return row

    factual = dict(activities=list(query.activities),native_loss=max(0.,1.-2.*original))
    population = [factual]*population_size
    try:
        for generation in range(max_generations):
            ranked = sorted(population,key=lambda r:(r["native_loss"],tuple(r["activities"])))
            # Author retains total_CFs elites; here bounded by available distinct parents.
            next_population = ranked[:min(topk,len(ranked))]
            for _ in range(population_size-len(next_population)):
                half = max(1,len(population)//2)
                ix = engine.rng.integers(0,half,size=2)
                if half > 1:
                    while ix[0] == ix[1]:
                        ix[1] = engine.rng.integers(0,half)
                # Keep the upstream unsorted-population parent indexing, not silently fix it.
                stats["author_operator_calls"] += 1
                try:
                    child = engine.mate_2(population[int(ix[0])]["activities"],
                        population[int(ix[1])]["activities"],varied,np.array([query.activities]),encoder,
                        activities,dfa,"lydia",original_symbols,dfa)
                except _InvalidOffspring:
                    stats["source_encoding_rejections"] += 1
                    continue
                raw = child.to_numpy().reshape(-1)
                if any(not np.isfinite(a) or int(a) != a or int(a) not in search_policy.activity_names for a in raw):
                    stats["source_encoding_rejections"] += 1
                    continue
                sequence = tuple(map(int,raw))
                edits = [i for i,(a,b) in enumerate(zip(query.activities,sequence)) if a != b]
                if any(sequence[i] not in ids for i in edits):
                    stats["source_encoding_rejections"] += 1
                    continue
                if not set(edits) <= set(query.allowed_positions) or len(edits) > query.max_changed_positions:
                    stats["edit_budget_rejections"] += 1
                    continue
                if sequence == query.activities or sequence in seen:
                    stats["duplicate_offspring"] += 1
                    next_population.append(factual if sequence == query.activities else
                        next(r for r in pool if tuple(r["activities"]) == sequence))
                    continue
                row = evaluate(sequence)
                if row is not None:
                    next_population.append(row)
            population = next_population or [factual]
            stats["completed_generations"] = generation+1
    except _Stop:
        pass
    stats["dfa_transition_checks"] = dfa.transition_checks
    return finish()
