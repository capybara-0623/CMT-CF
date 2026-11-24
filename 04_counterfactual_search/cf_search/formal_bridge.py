"""Dormant seven-task execution adapter. No Audit input in the generator API."""
import copy
from dataclasses import asdict
from pathlib import Path
import time
import torch

from cf_search import SearchBudget, generate as internal_generate
from cf_search.proprietary import FrozenCFHeads, TimeOutcomeOracle, head_inventory, generate as proprietary_generate
from cf_search.relaxation import policy_from_manifest, select_minimal_relaxation
from cf_search.suite import FAMILIES, GROUPS, Profile, checked_json, json_form, seal
from cf_baselines.buliga_official import PolicyDFA, generate as buliga_generate
from cf_baselines.buliga_adapter import _symbol
from cf_baselines.buliga_reference import load_reference
from evaluation.cf_types import Candidate, OracleBudget, SearchResult
from evaluation.cf_experiment_results import write_json
from evaluation.experiment_protocol import artifact, check_artifact, context_from_record, load_protocol, stable_hash
from evaluation.freeze_protocol import load_frozen_oracle
from evaluation.outcome_oracle import OutcomeOracle
from explaination.visualizations.utils import normalize_activity_name


def restore_policy(source, original):
    names = {str(i): normalize_activity_name(v) for i, v in enumerate(original["model"]["activity_vocabulary"], 1)}
    manifest = dict(sources=[source], runtime_search_dfa_snapshots={source["dataset"]: source["atomic_snapshots"]},
                    activity_names={source["dataset"]: names})
    policy = policy_from_manifest(manifest, source["dataset"])
    graph = source["combined_snapshot"]
    combined = PolicyDFA.__new__(PolicyDFA)
    combined.symbols = {i: _symbol(i) for i in policy.activity_names}
    combined.initial_state = graph["initial_state"]+1
    combined.transitions = {i+1: {a: t+1 for a, t in row} for i, row in enumerate(graph["transitions"])}
    combined.accepting = {i+1 for i in graph["accepting_states"]}
    combined.snapshot = graph
    combined.build_statistics = dict(source["combined_build_statistics"], restored_from_accepted_snapshot=True, new_build_seconds=0.)
    return policy, combined


def runtime(source):
    """Called ONLY beyond authorization barrier. Preparation has no forward calls."""
    started = time.perf_counter()
    path = Path(source["protocol"]["path"])
    original = load_protocol(path)
    base, contexts = load_frozen_oracle(path)
    checkpoint = torch.load(check_artifact(source["model"], path.parent), map_location="cpu", weights_only=False)
    inventory = head_inventory(checkpoint)
    if inventory != source["cf_head_inventory"]:
        raise ValueError("Head inventory differs from prerequisite freeze")
    heads = {m: FrozenCFHeads(checkpoint["cfg"], checkpoint["cf_model_state"], m == "O2") if inventory[m]["available"] else None
             for m in ("O1", "O2")}
    policy, combined = restore_policy(source, original)
    reference = load_reference(source["reference"]["artifact"]["path"])
    ids = tuple(i for i, raw in enumerate(original["model"]["activity_vocabulary"], 1)
                if raw.upper() not in ("__UNK__", "<UNK>", "UNK", "__PAD__", "<PAD>"))
    return dict(base=base, contexts=contexts, heads=heads, policy=policy, combined=combined, reference=reference,
        activity_ids=ids, budget=SearchBudget(**{k: source["budget"][k] for k in asdict(SearchBudget())}),
        model_identity=base.model_identity, preparation=dict(dataset=source["dataset"],
            operation="predictor_heads_reference_and_accepted_DFA_restore_no_forward", walltime_seconds=time.perf_counter()-started,
            model_evaluations=0, new_reference_build_seconds=0., new_DFA_compile_seconds=0., KD_build_count=0, audit_compile_count=0))


def oracle_for(item, method):
    base = item["base"]
    cls = TimeOutcomeOracle if method == "O2" else OutcomeOracle
    return cls(base.model, base.class_mapping, base.model_identity, base.pattern_encoder, base.feature_identity,
               budget=OracleBudget(item["budget"].model_evaluations))


def call_algorithm(query, oracle, item, method):
    args = dict(topk=5, activity_ids=item["activity_ids"])
    if method in FAMILIES["search"]:
        return internal_generate(query, oracle, item["policy"], item["budget"], method, **args)
    if method in ("R0_strict", "R1_R2_shared_expanded"):
        return internal_generate(query, oracle, item["policy"], item["budget"],
            "strict_dfa_outcome" if method == "R0_strict" else "enum_outcome", pool_only=True, **args)
    if method in FAMILIES["proprietary"]:
        return proprietary_generate(query, oracle, item["policy"], item["budget"], method, item["heads"].get(method), **args)
    if method in FAMILIES["buliga"]:
        return buliga_generate(query, oracle, item["policy"], item["budget"], item["reference"], item["combined"],
            threshold=.5, seed=42, max_generations=50, max_reference_rows=64, **args)
    raise ValueError("Unknown physical method")


def execution_id(frozen, item, method):
    if method in ("R0_strict", "R1_R2_shared_expanded"):
        return stable_hash(dict(dataset=frozen["dataset"], sample_idx=frozen["context"]["sample_idx"],
            input_sha256=frozen["input_sha256"], proposal_space_mode="strict_pruned" if method == "R0_strict" else "expanded",
            model=item["model_identity"]))
    return stable_hash(dict(input_sha256=frozen["input_sha256"], method=method))


def physical_generation(frozen, item, method, root):
    query = context_from_record(frozen["context"])
    oracle = oracle_for(item, method)
    profile, started, error = Profile(), time.perf_counter(), None
    try:
        with profile.observe(oracle.model, list(item["heads"].values())):
            result = call_algorithm(query, oracle, item, method)
        if "original_probability" in result.statistics and (
                abs(result.statistics["original_probability"]-frozen["original_probability"]) > 2e-6 or
                (result.statistics["original_probability"] >= .5) != (frozen["original_class"] == "Success")):
            raise ValueError("Frozen factual predictor drift")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result = SearchResult([], "search_error", dict(oracle=oracle.statistics(), scored_pool=[],
            context_encodings=profile.values["context_encodings"], cf_outcome_evaluations=profile.values["cf_outcome_evaluations"],
            cf_time_evaluations=profile.values["cf_time_evaluations"]))
    stats = copy.deepcopy(result.statistics)
    pool = stats.pop("scored_pool")
    costs = dict(profile.values, generation_inclusive_walltime_seconds=time.perf_counter()-started)
    stats["p6a_profile"] = costs  # Existing call-profile contract, unchanged meanings.
    eid = execution_id(frozen, item, method)
    family = next(f for f, ms in FAMILIES.items() if method in ms) if method != "R1_R2_shared_expanded" else "relaxation"
    payload = seal(dict(purpose="P6_B_saved_physical_generation_v1", execution_id=family+"/"+eid,
        local_execution_id=eid, input_sha256=frozen["input_sha256"], dataset=frozen["dataset"], method=method, family=family,
        rows=pool, pool_sha256=stable_hash(pool), returned=result.candidates, status=result.status, statistics=stats,
        error=error, costs=costs, saved_before_policy_selection=family == "relaxation", audit_input_present=False))
    path = Path(root)/"physical"/(eid+".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)  # Durable barrier precedes either R1/R2 selector.
    return payload, artifact(path), oracle


def normalize_view(frozen, method, result, rows, stats, error=None, decision=None, selection_delta=None):
    key = frozen["input_sha256"]
    qid = stable_hash(dict(input_sha256=key, method=method))
    family = next(f for f, ms in FAMILIES.items() if method in ms)
    q = dict(query_id=qid, input_sha256=key, method=method, family=family, dataset=frozen["dataset"],
        sample_idx=frozen["context"]["sample_idx"], original_probability=frozen["original_probability"],
        original_class=frozen["original_class"], common_buliga_applicable=frozen["common_buliga_applicable"],
        execution_id=result["execution_id"], status=result["status"] if error is None else "search_error", error=error or result["error"],
        candidate_count=len(rows), statistics=stats, time_policy="learned_normalized_elapsed_coordinate" if method == "O2" else "fixed",
        information_regime="unavailable" if family == "availability" else "trained_cf_heads" if method in ("O1", "O2") else "outcome_oracle_only",
        comparison_groups=[g for g, ms in GROUPS.items() if method in ms])
    if family == "relaxation":
        q.update(selection=decision, proposal_space_mode="strict_pruned" if method == "R0_strict" else "expanded",
                 exploration_execution_id=result["local_execution_id"])
        q["statistics"]["selection_oracle_delta"] = selection_delta or {}
    cs = [dict(r, query_id=qid, rank=rank, candidate_id=stable_hash(dict(query_id=qid, activities=r["activities"], times=r["times"])),
               input_sha256=key, method=method, family=family, dataset=frozen["dataset"]) for rank, r in enumerate(rows, 1)]
    return q, cs


def generate_input(frozen, item, root):
    """One input, nine physical executions and eleven attributed policy rows."""
    qs, cs, artifacts = [], [], []
    for method in FAMILIES["search"]+["R0_strict", "R1_R2_shared_expanded"]+FAMILIES["proprietary"]+FAMILIES["buliga"]:
        result, resource, oracle = physical_generation(frozen, item, method, root)
        artifacts.append(resource)
        views = ("R1_expanded_cap0", "R2_expanded_cap1") if method == "R1_R2_shared_expanded" else (method,)
        for view in views:
            stats, decision, error, delta = copy.deepcopy(result["statistics"]), None, None, None
            rows = copy.deepcopy(result["returned"])
            if result["family"] == "relaxation" and result["status"] != "search_error":
                started = time.perf_counter()
                try:
                    saved = checked_json(resource["path"])
                    rows, decision = select_minimal_relaxation(saved["rows"], item["policy"],
                        max_relaxed_constraints=int(view == "R2_expanded_cap1"), topk=5)
                    before = oracle.statistics()
                    probabilities = oracle.predict_proba([Candidate(tuple(r["activities"])) for r in rows], context_from_record(frozen["context"]))
                    after = oracle.statistics()
                    delta = {k: after[k]-before[k] for k in before}
                    if (any(delta[k] for k in ("model_evaluations", "model_batches", "model_errors", "unique_candidates")) or
                            any(float(p[1]) != r["probability"] for p, r in zip(probabilities, rows))):
                        raise ValueError("Policy selection triggered uncached scoring or changed cached probabilities")
                    stats["oracle"] = {k: result["statistics"]["oracle"][k]+delta[k] for k in before}
                except Exception as exc:
                    rows, error = [], f"{type(exc).__name__}: {exc}"
                stats["selection_walltime_seconds"] = time.perf_counter()-started
                stats["shared_exploration_full_cost_attributed"] = method == "R1_R2_shared_expanded"
            q, candidates = normalize_view(frozen, view, result, rows, stats, error, decision, delta)
            qs.append(q)
            cs.extend(candidates)
    unavailable = dict(execution_id="availability/"+frozen["input_sha256"], status="method_unavailable", error=None)
    stats = dict(reason="CREATED integration remains unavailable; not measured zero", oracle=dict(
        requested_sequences=0, unique_candidates=0, cache_hits=0, model_evaluations=0, model_batches=0, model_errors=0))
    q, _ = normalize_view(frozen, "created_unavailable", unavailable, [], stats)
    qs.append(q)
    return qs, cs, artifacts
