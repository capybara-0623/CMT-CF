"""Official initialization/plausibility/GA/final selection, protocol adaptation."""
import time
from types import MethodType,SimpleNamespace
import numpy as np
import pandas as pd
from pythomata import SimpleDFA
from cf_baselines.buliga_adapter import _symbol,_Encoder,PolicyAutomaton
from cf_baselines.buliga_official_source import load_methods,METHODS
from cf_baselines.buliga_reference import project_reference
from cf_search.budget import BudgetExecutor
from evaluation.cf_types import Candidate,SearchResult
from evaluation.experiment_protocol import stable_hash,canonical_dfa_snapshot
from evaluation.outcome_oracle import OracleBudgetExceeded

METHOD = "buliga_official_protocol_v1"


class Stop(RuntimeError):
    pass


class PolicyDFA:
    """Complete minimized policy; state 1 is initial (upstream ltlf2dfa convention)."""
    def __init__(self,policy,max_states=10000):
        started = time.perf_counter()
        self.symbols = {i:_symbol(i) for i in policy.activity_names}
        product = PolicyAutomaton(policy,self.symbols)
        states = [product.initial_state]
        index = {states[0]:0}
        transitions,accepting = {},set()
        for state in states:
            if len(states) > max_states or time.perf_counter()-started > 30:
                raise ValueError("Combined Search DFA engineering compilation cap")
            i = index[state]
            if product.is_accepting(state):
                accepting.add(i)
            outgoing = {}
            for symbol in sorted(self.symbols.values()):
                target = product._advance(state,symbol)
                if target not in index:
                    index[target] = len(states)
                    states.append(target)
                outgoing[symbol] = index[target]
            transitions[i] = outgoing
        # Pythomata's all-state-pairs bisimulation is quadratic in memory and
        # excessively slow on the policy product. Equivalent Moore refinement
        # preserves the minimal language-state equivalence needed by ONLINE.
        letters = sorted(self.symbols.values())
        blocks = [int(i in accepting) for i in range(len(states))]
        refinements = 0
        while True:
            if time.perf_counter()-started > 30:
                raise ValueError("Combined Search DFA minimization engineering cap")
            signatures,refined = {},[]
            for i in range(len(states)):
                signature = (blocks[i],*(blocks[transitions[i][a]] for a in letters))
                refined.append(signatures.setdefault(signature,len(signatures)))
            refinements += 1
            if refined == blocks:
                break
            blocks = refined
        representatives = {}
        for state,block in enumerate(blocks):
            representatives.setdefault(block,state)
        minimized = SimpleDFA(set(blocks),set(letters),blocks[0],{blocks[i] for i in accepting},
            {b:{a:blocks[transitions[s][a]] for a in letters} for b,s in representatives.items()})
        snapshot = canonical_dfa_snapshot(minimized)
        # Snapshot canonical state numbers start at zero; renumber to one.
        mapping = {minimized.initial_state:1}
        queue = [minimized.initial_state]
        for state in queue:
            for symbol in sorted(minimized.alphabet):
                target = minimized.transition_function[state][symbol]
                if target not in mapping:
                    mapping[target] = len(mapping)+1
                    queue.append(target)
        self.initial_state = 1
        self.transitions = {mapping[s]:{a:mapping[t] for a,t in minimized.transition_function[s].items()} for s in queue}
        self.accepting = {mapping[s] for s in minimized.accepting_states}
        self.snapshot = snapshot
        self.build_statistics = dict(product_states=len(states),minimized_states=len(queue),refinement_rounds=refinements,
            minimization="equivalent bounded Moore partition refinement",
            walltime_seconds=time.perf_counter()-started,scope="shared policy preparation, not search")

    def get_successors(self,state,interpretation):
        letter = next(k for k,v in interpretation.items() if v)
        return {self.transitions[state][letter]}

    def get_transitions(self):
        return [(s,a,t) for s,outgoing in self.transitions.items() for a,t in outgoing.items()]


class BoundedRNG:
    def __init__(self,rng,executor):
        self.rng,self.executor = rng,executor
    def random(self):
        if not self.executor.visit():
            raise Stop()
        return self.rng.random()
    def __getattr__(self,name):
        def call(*args,**kwargs):
            if not self.executor.check():
                raise Stop()
            return getattr(self.rng,name)(*args,**kwargs)
        return call


def generate(query,oracle,policy,budget,reference,policy_dfa=None,topk=5,threshold=.5,
             activity_ids=None,seed=42,max_generations=50,max_reference_rows=64):
    if any(oracle.statistics().values()) or oracle.budget.max_model_evaluations != budget.model_evaluations:
        raise ValueError("Fresh matching-budget oracle required")
    if type(topk) is not int or topk < 1 or type(max_generations) is not int or not 1 <= max_generations <= 50:
        raise ValueError("Invalid official baseline bounded settings")
    ids = tuple(sorted(policy.activity_names if activity_ids is None else activity_ids))
    if not ids or len(set(ids)) != len(ids) or any(i not in policy.activity_names for i in ids):
        raise ValueError("Invalid shared activity domain")
    namespace = load_methods()
    executor = BudgetExecutor(budget)
    stats = dict(cf_head_calls=0,time_policy="fixed",attempt_unit="assigned_activity_genes",
        author_calls={},initialization="official_kdtree_with_bounded_official_random_fallback",
        reference_plausibility_enabled=True,source_parent_pool_degenerate=False,
        rejected_offspring=0,final_gate_rejections=0,reference_rows=0,
        main_flow_integrated=True,original_pipeline_reproduced=False,
        comparison_scope="official_source_protocol_adaptation_not_original_experiment_reproduction")
    pool,seen = [],set()
    def finish(status,candidates=()):
        return SearchResult(list(candidates),status,dict(stats,oracle=oracle.statistics(),attempted_nodes=executor.attempts,
            walltime_seconds=executor.elapsed,scored_pool=pool,scored_pool_size=len(pool),scored_pool_sha256=stable_hash(pool)))
    if not policy.evidence(query.activities)["strict"]:
        stats.update(reason="Original ONLINE factual finite-prefix Search precondition not met",usable_external_baseline=False)
        return finish("precondition_not_met")
    projected = project_reference(reference,query,ids,max_reference_rows)
    stats.update(reference_rows=len(projected),projected_reference=projected,projected_reference_sha256=stable_hash(projected),
        usable_external_baseline=True)
    if not projected:
        stats.update(reason="No permitted training case has the required prefix length",usable_external_baseline=False)
        return finish("method_unavailable")
    dfa = policy_dfa or PolicyDFA(policy)
    features = [f"prefix_{i}" for i in range(query.prefix_len)]
    varied = [features[i] for i in query.allowed_positions]
    frame = pd.DataFrame([r["activities"] for r in projected],columns=features)
    data = SimpleNamespace(feature_names=features,number_of_features=query.prefix_len,
        continuous_feature_names=[],continuous_feature_indexes=[],categorical_feature_names=features,
        categorical_feature_indexes=list(range(query.prefix_len)),data_df=frame,outcome_name="outcome",
        prepare_query_instance=lambda query_instance:pd.DataFrame(query_instance,columns=features),
        get_decimal_precisions=lambda:[0]*query.prefix_len,normalize_data=lambda x:np.asarray(x,dtype=float))
    engine = SimpleNamespace(data_interface=data,rng=BoundedRNG(np.random.default_rng(seed),executor),
        num_output_nodes=2,yloss_type="hinge_loss",x1=np.asarray(query.activities),
        sparsity_weight=.5,proximity_weight=0.,plausibility_weight=.5,conformance_weight=3.,
        conformance_score=np.ones(1),query_instance_normalized=None,population_size=5*topk,total_CFs=topk,
        cfs=[],predicted_outcome_name="outcome_pred",target_cf_class=1,stopping_threshold=threshold,
        committed_population=None,committed_iterations=0,termination_reason="max_generations_or_native_convergence",
        start_time=time.perf_counter())
    encoder = _Encoder(query,ids,dfa.symbols,features)
    engine.feature_range = {f:np.asarray(list(encoder._label_dict[f].values())) for f in features}
    for name in METHODS+("build_KD_tree","is_cf_valid","round_to_precision","finalize"):
        original_function = namespace[name]
        def bound(*args,_function=original_function,_name=name,**kwargs):
            stats["author_calls"][_name] = stats["author_calls"].get(_name,0)+1
            started = time.perf_counter()
            before = oracle.stats.model_evaluations
            result = None
            try:
                result = _function(engine,*args,**kwargs)
            finally:
                if _name == "build_KD_tree":
                    stats["kd_build_walltime_seconds"] = time.perf_counter()-started
                    stats["kd_reference_model_evaluations"] = oracle.stats.model_evaluations-before
                    stats["kd_index_built"] = result is not None and result[1] is not None
            return result
        setattr(engine,name,bound)
    def assigned_gene():
        if not executor.visit():
            raise Stop()
    engine._assigned_gene = assigned_gene
    def admissible(value):
        raw = np.asarray(value).reshape(-1)
        if (len(raw) != query.prefix_len or not np.isfinite(raw).all()
                or any(int(v) != v or int(v) not in policy.activity_names for v in raw)):
            stats["rejected_offspring"] += 1
            return False
        sequence = tuple(map(int,raw))
        edits = [i for i,(a,b) in enumerate(zip(query.activities,sequence)) if a != b]
        valid = (set(edits) <= set(query.allowed_positions) and len(edits) <= query.max_changed_positions
                 and all(sequence[i] in ids for i in edits))
        if not valid:
            stats["rejected_offspring"] += 1
        return valid
    engine._admissible = admissible
    def final_admissible(value):
        valid = admissible(value) and tuple(map(int,np.asarray(value).reshape(-1))) != query.activities and policy.evidence(
            tuple(map(int,np.asarray(value).reshape(-1))))["strict"]
        if not valid:
            stats["final_gate_rejections"] += 1
        return valid
    engine._final_admissible = final_admissible
    def checkpoint(population,iterations):
        engine.committed_population = np.asarray(population).copy()
        engine.committed_iterations = iterations
        if not executor.check():
            raise Stop()
    engine._checkpoint_population = checkpoint
    original = float(oracle.predict_proba([Candidate(query.activities)],query)[0,1])
    engine.test_pred = np.array([int(original >= threshold)])
    stats["original_probability"] = original
    def scores(values):
        if not executor.check():
            raise Stop()
        array = np.asarray(values).reshape(-1,query.prefix_len)
        if any(not admissible(row) for row in array):
            raise ValueError("Illegal candidate escaped official initializer/operator gate")
        candidates = [Candidate(tuple(map(int,row))) for row in array]
        try:
            output = oracle.predict_proba(candidates,query)
        except OracleBudgetExceeded:
            executor.status = "model_budget_exhausted"
            raise Stop()
        for candidate,probability in zip(candidates,output):
            if candidate.activities == query.activities or candidate.activities in seen:
                continue
            seen.add(candidate.activities)
            changed = [i for i,(a,b) in enumerate(zip(query.activities,candidate.activities)) if a != b]
            pool.append(dict(activities=list(candidate.activities),times=list(query.times),probability=float(probability[1]),
                probability_delta=float(probability[1])-original,edit_positions=changed,edit_count=len(changed),
                search=policy.evidence(candidate.activities)))
        return output
    engine.predict_fn_scores = scores
    engine.model = SimpleNamespace(model_type="classifier",get_output=lambda values,model_score=True:
        scores(values) if model_score else np.argmax(scores(values),axis=1))
    def conformance(population,encoder,ltlf_model,dfa):
        accepted = np.asarray([policy.evidence(tuple(map(int,row)))["strict"] for row in population],dtype=float)
        return accepted,float(np.mean(accepted))
    engine.compute_conformance_new = conformance
    # Same per-row original loss values (zero continuous dimension); recomputed
    # for all saved scores, including initialization rows, at the end, no oracle calls.
    def attach_losses():
        for row in pool:
            plausible = float(np.asarray(engine.compute_plausibility(np.asarray([row["activities"]]))).reshape(-1)[0])
            row["plausibility"] = plausible
            row["native_loss"] = max(0.,1.-2*row["probability"])+.5*row["edit_count"]/query.prefix_len+\
                .5*plausible+3*(not row["search"]["strict"])
    try:
        engine.do_cf_initializations(topk,"kdtree","DiverseCF",varied,None,1,np.asarray(query.activities),
            pd.DataFrame([query.activities],columns=features),False)
        stats["initial_population"] = np.asarray(engine.cfs).tolist()
        engine.find_counterfactuals(np.asarray([query.activities]),None,1,varied,max_generations,.01,False,
            encoder,query.dataset,None,SimpleNamespace(formula="frozen_search_policy"),"loss_function",
            "heuristic_2",list(dfa.symbols.values()),[],[],True,dfa.symbols,dfa)
    except Stop:
        stats["interrupted_phase"] = "GA" if engine.committed_population is not None else "initialization"
        # Native tail only. Do not return best candidates from the explored pool.
        if engine.committed_population is not None:
            cached = all(oracle._key(oracle._candidate(Candidate(tuple(map(int,row))),query),query) in oracle._cache
                         for row in engine.committed_population)
            if cached:
                saved_status = executor.status
                # No new model call can occur; bypass elapsed-only check for cached
                # final verification while keeping all oracle sequence budget checks.
                original_scores = engine.predict_fn_scores
                def cached_scores(values):
                    return oracle.predict_proba([Candidate(tuple(map(int,row))) for row in np.asarray(values).reshape(-1,query.prefix_len)],query)
                engine.predict_fn_scores = cached_scores
                try:
                    engine.finalize(engine.committed_population,np.asarray([query.activities]))
                finally:
                    engine.predict_fn_scores = original_scores
                    executor.status = saved_status
            else:
                # Do not publish a partially executed native final selector.
                engine.final_cfs = []
    attach_losses()
    stats.update(completed_generations=engine.committed_iterations,termination_reason=engine.termination_reason,
        source_parent_pool_degenerate=engine.termination_reason == "source_parent_pool_degenerate",
        committed_population=engine.committed_population.tolist() if engine.committed_population is not None else [],
        final_population_selection="native committed-population order; goal+strict+nonzero-edit final gate",
        initial_population=stats.get("initial_population",[]))
    stats["committed_population_fully_scored"] = all(
        oracle._key(oracle._candidate(Candidate(tuple(map(int,row))),query),query) in oracle._cache
        for row in stats["committed_population"])
    selected = []
    for candidate in getattr(engine,"final_cfs",[]):
        key = tuple(map(int,candidate))
        row = next((r for r in pool if tuple(r["activities"]) == key),None)
        if row is not None and row not in selected:
            selected.append(row)
    return finish(executor.status,selected)
