"""P4-B: identical bounded strict proposals, different preselection scorers.

No training, labels, future events, or Audit input. Learned heads condition on
factual causal hidden[p-1]; multi-edit proposal score is their arithmetic mean.
Only the proposal Top-K is rescored/reranked by the frozen outcome backbone.
"""
from dataclasses import asdict, replace
import math
import torch
from torch import nn
from cf_search.budget import BudgetExecutor
from cf_search.methods import _Pruner
from evaluation.cf_types import Candidate, SearchResult
from evaluation.experiment_protocol import stable_hash
from evaluation.outcome_oracle import OutcomeOracle

METHODS = ("O0", "O1", "O2")
TIME_POLICY = "normalized_elapsed_coordinate_neighbor_projection_v1"


class MissingCFWeights(ValueError):
    pass


class FrozenCFHeads(nn.Module):
    """Same layers/names as CounterfactualRecommender, strict trusted restore."""
    def __init__(self, cfg, state, require_time=False):
        super().__init__()
        self.action_embed = nn.Embedding(cfg.num_activities, cfg.hidden_dim)
        self.cf_outcome_head = nn.Sequential(nn.Linear(cfg.hidden_dim*2,cfg.hidden_dim),nn.ReLU(),
                                             nn.Linear(cfg.hidden_dim,2))
        if require_time:
            self.cf_time_head = nn.Sequential(nn.Linear(cfg.hidden_dim*2,cfg.hidden_dim),nn.ReLU(),
                                              nn.Linear(cfg.hidden_dim,1),nn.Sigmoid())
        expected = self.state_dict()
        missing = [k for k in expected if k not in (state or {})]
        if missing:
            raise MissingCFWeights("Missing trained CF weights: " + ", ".join(missing))
        restored = {}
        for key, target in expected.items():
            value = state[key]
            if (not isinstance(value,torch.Tensor) or value.shape != target.shape
                    or value.dtype != target.dtype or not torch.isfinite(value).all()):
                raise MissingCFWeights("Invalid trained CF weight: " + key)
            restored[key] = value
        self.load_state_dict(restored,strict=True)
        self.eval().requires_grad_(False)


def head_inventory(checkpoint):
    report = {}
    for mode in ("O1","O2"):
        try:
            head = FrozenCFHeads(checkpoint["cfg"],checkpoint.get("cf_model_state"),mode == "O2")
            report[mode] = dict(available=True,missing_reason=None,weight_keys=list(head.state_dict()))
        except MissingCFWeights as exc:
            report[mode] = dict(available=False,missing_reason=str(exc),weight_keys=[])
    history = checkpoint.get("train_history",[])
    report["training"] = dict(source="existing cf_model_state checkpoint; trainer.py joint UWL losses",
        recorded_epoch_count=len(history),outcome_loss_recorded=any("loss_cf_outcome" in h for h in history),
        time_loss_recorded=any("loss_cf_time" in h for h in history),
        incremental_training_seconds=None,incremental_training_cost_status="not recorded; no retraining",
        causal_effect_identified=False)
    return report


def check_times(activities, times, query):
    if (len(times) != query.prefix_len or any(not math.isfinite(t) or not 0 <= t <= 1 for t in times)
            or any(b < a for a,b in zip(times,times[1:]))):
        raise ValueError("Elapsed coordinates must be aligned, finite, [0,1], nondecreasing")
    changed = {i for i,(a,b) in enumerate(zip(query.activities,activities)) if a != b}
    if any(t != query.times[i] and i not in changed for i,t in enumerate(times)):
        raise ValueError("Time edits must accompany allowed activity replacements")


class TimeOutcomeOracle(OutcomeOracle):
    """Explicit opt-in; normal OutcomeOracle retains fixed-time rejection."""
    def _candidate(self, value, query):
        candidate = value if isinstance(value,Candidate) else Candidate(tuple(value))
        acts = super()._candidate(Candidate(candidate.activities),query)
        times = query.times if candidate.times is None else candidate.times
        check_times(acts,times,query)
        return Candidate(acts,tuple(times))

    def _key(self, candidate, query):
        payload = asdict(query)
        payload.update(activities=candidate.activities,times=candidate.times,
            model_identity=self.model_identity,feature_identity=self.feature_identity,time_policy=TIME_POLICY)
        return stable_hash(payload)


def project_times(activities, raw_by_position, query):
    check_times(query.activities,query.times,query)
    changed = {i for i,(a,b) in enumerate(zip(query.activities,activities)) if a != b}
    times = list(query.times)
    for i in sorted(changed):
        raw = raw_by_position[i]
        if not math.isfinite(raw) or not 0 <= raw <= 1:
            raise ValueError("CF time head returned invalid coordinate")
        left = times[i-1] if i else 0.
        right = next((query.times[j] for j in range(i+1,query.prefix_len) if j not in changed),1.)
        times[i] = min(max(raw,left),right)
    check_times(activities,times,query)
    return times


def strict_activity_pool(query, policy, budget, activity_ids):
    """Common proposal count cap reserves factual + one learned-context encoding.

    Reserve applies to O0 too, keeping the activity pool identical. This is NOT
    P4-A's full 500-evaluation enumeration. Node/time caps remain cooperative.
    """
    ids = tuple(sorted(activity_ids))
    if (len(set(ids)) != len(ids) or not ids or
            any(type(a) is not int or a <= 0 or a not in policy.activity_names for a in ids)):
        raise ValueError("Invalid proposal activity domain")
    if budget.model_evaluations < 3:
        raise ValueError("P4-B requires at least three sequence budget units")
    cap = budget.model_evaluations-2
    executor,pruner = BudgetExecutor(budget),_Pruner(policy)
    allowed = set(query.allowed_positions)
    remaining = [sum(p >= i for p in allowed) for i in range(query.prefix_len+1)]
    rows,pruned,leaves = [],0,0

    def walk(index,prefix,edits,target,states):
        nonlocal pruned,leaves
        if executor.status != "completed" or not executor.check():
            return
        if index == query.prefix_len:
            leaves += 1
            evidence = policy.evidence(prefix)
            if evidence["strict"]:
                rows.append(dict(activities=prefix,search=evidence))
                if len(rows) >= cap:
                    executor.status = "proposal_pool_limit_reached"
            return
        options = [query.activities[index]]
        if index in allowed:
            options += [a for a in ids if a != query.activities[index]]
        for a in options:
            new = edits+(a != query.activities[index])
            if new > target or new+remaining[index+1] < target:
                continue
            if not executor.visit():
                return
            states_new = pruner.advance(states,policy.activity_names[a])
            if states_new is None:
                pruned += 1
                continue
            walk(index+1,prefix+[a],new,target,states_new)
            if executor.status != "completed":
                return
    for target in range(1,min(query.max_changed_positions,len(allowed))+1):
        walk(0,[],0,target,pruner.initial())
        if executor.status != "completed":
            break
    return rows,dict(status=executor.status,attempted_nodes=executor.attempts,pruned_branches=pruned,
        completed_sequences=leaves,activity_pool_sha256=stable_hash(rows),proposal_pool_cap=cap,
        walltime_seconds=executor.elapsed)


def proposal_ranking(row):
    return (-row["proposal_score"],row["edit_count"],tuple(row["activities"]))


def final_ranking(row):
    return (-row["probability"],row["edit_count"],tuple(row["activities"]),tuple(row["times"]))


def proposal_table(query, model, heads, ids, include_time):
    if any(p == 0 for p in query.allowed_positions):
        raise ValueError("Learned CF requires observed predecessor; position zero is protected")
    device = next(model.parameters()).device
    cfg = model.cfg
    def static(values,width,dtype):
        return torch.tensor([values],device=device,dtype=dtype).reshape(1,width)
    def dynamic(values,width,dtype):
        if not width:
            return torch.zeros(1,query.prefix_len,1,device=device,dtype=dtype)
        return torch.tensor([values],device=device,dtype=dtype)
    activities = torch.tensor([query.activities],device=device,dtype=torch.long)
    with torch.inference_mode():
        hidden = model.encode(activities,torch.tensor([query.times],device=device),
            torch.ones_like(activities,dtype=torch.bool),
            static(query.static_cat,cfg.n_static_cat,torch.long),static(query.static_num,cfg.n_static_num,torch.float),
            dynamic(query.dynamic_cat,cfg.n_dynamic_cat,torch.long),dynamic(query.dynamic_num,cfg.n_dynamic_num,torch.float))
        pairs = [(p,a) for p in sorted(query.allowed_positions) for a in ids if a != query.activities[p]]
        if not pairs:
            return {},dict(context_encodings=1,cf_outcome_evaluations=0,cf_time_evaluations=0,cf_head_batches=0)
        context = torch.stack([hidden[0,p-1] for p,a in pairs])
        embedding = heads.action_embed(torch.tensor([a for p,a in pairs],device=device))
        joined = torch.cat([context,embedding],dim=-1)
        probabilities = torch.softmax(heads.cf_outcome_head(joined),dim=-1)[:,1].cpu().tolist()
        times = heads.cf_time_head(joined).flatten().cpu().tolist() if include_time else [None]*len(pairs)
    return {pair:(p,t) for pair,p,t in zip(pairs,probabilities,times)},dict(
        context_encodings=1,cf_outcome_evaluations=len(pairs),cf_time_evaluations=len(pairs) if include_time else 0,
        cf_head_batches=1)


def generate(query, oracle, policy, budget, method, heads=None, topk=5, activity_ids=None):
    import time
    started = time.monotonic()
    if method not in METHODS or type(topk) is not int or topk < 1:
        raise ValueError("Invalid P4-B method/Top-K")
    if any(oracle.statistics().values()) or oracle.budget.max_model_evaluations != budget.model_evaluations:
        raise ValueError("Fresh matching-budget oracle required")
    if method != "O0" and heads is None:
        return SearchResult([],"method_unavailable",dict(oracle=oracle.statistics(),scored_pool=[],
            scored_pool_size=0,scored_pool_sha256=stable_hash([]),missing_reason="Trained CF weights unavailable",
            context_encodings=0,cf_outcome_evaluations=0,cf_time_evaluations=0,total_backbone_evaluations=0,
            time_policy=TIME_POLICY if method == "O2" else "fixed"))
    if method == "O2" and not isinstance(oracle,TimeOutcomeOracle):
        raise ValueError("O2 requires explicit time-intervention oracle")
    ids = tuple(sorted(policy.activity_names if activity_ids is None else activity_ids))
    original = float(oracle.predict_proba([Candidate(query.activities)],query)[0,1])
    remaining_seconds = max(1e-9,budget.walltime_seconds-(time.monotonic()-started))
    activity_pool,statistics = strict_activity_pool(query,policy,replace(budget,walltime_seconds=remaining_seconds),ids)
    status = statistics.pop("status")
    costs = dict(context_encodings=0,cf_outcome_evaluations=0,cf_time_evaluations=0,cf_head_batches=0)
    table = {}
    if method != "O0" and activity_pool and time.monotonic()-started < budget.walltime_seconds:
        table,costs = proposal_table(query,oracle.model,heads,ids,method == "O2")
    pool = []
    for base in activity_pool:
        if time.monotonic()-started >= budget.walltime_seconds:
            status = "walltime_exhausted"
            break
        acts = base["activities"]
        edits = [i for i,(a,b) in enumerate(zip(query.activities,acts)) if a != b]
        raw = {i:table[i,acts[i]][1] for i in edits} if method == "O2" else {}
        times = project_times(acts,raw,query) if method == "O2" else list(query.times)
        probability = (float(oracle.predict_proba([Candidate(tuple(acts))],query)[0,1])
                       if method == "O0" else None)
        score = probability if method == "O0" else sum(table[i,acts[i]][0] for i in edits)/len(edits)
        pool.append(dict(activities=acts,times=times,probability=probability,proposal_score=score,
            edit_positions=edits,edit_count=len(edits),search=base["search"],
            cf_time_raw={str(i):t for i,t in raw.items()}))
    selected = sorted(pool,key=proposal_ranking)[:min(topk,budget.model_evaluations-2)]
    if method != "O0" and time.monotonic()-started >= budget.walltime_seconds:
        status = "walltime_exhausted"
        selected = []
    if selected:
        scores = oracle.predict_proba([Candidate(tuple(r["activities"]),tuple(r["times"])) for r in selected],query)
        for row,score in zip(selected,scores):
            row["probability"] = float(score[1])
    returned = []
    for row in sorted(selected,key=final_ranking):
        if policy.evidence(row["activities"]) != row["search"]:
            raise ValueError("Final full Search evidence mismatch")
        returned.append(dict(row,probability_delta=row["probability"]-original))
    total = oracle.stats.model_evaluations+costs["context_encodings"]
    if total > budget.model_evaluations:
        raise ValueError("Outcome + learned-context sequence budget exceeded")
    statistics.update(costs,oracle=oracle.statistics(),original_probability=original,scored_pool=pool,
        scored_pool_size=len(pool),scored_pool_sha256=stable_hash(pool),total_backbone_evaluations=total,
        time_policy=TIME_POLICY if method == "O2" else "fixed",attempt_unit="assigned_prefix_nodes",
        cf_outcome_aggregation="mean_at_factual_predecessors",cf_time_penalty_enabled=False,
        walltime_seconds=time.monotonic()-started,
        proposal_walltime_cap_scope="cooperative whole-search checks; in-flight inference cannot be interrupted")
    return SearchResult(returned,status,statistics)
