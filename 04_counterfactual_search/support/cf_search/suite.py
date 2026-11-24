"""P6-A registry, read-only preflight and non-semantic cost instrumentation."""
from contextlib import ExitStack,contextmanager
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from unittest.mock import patch
import torch
from cf_baselines.buliga_official import METHOD,PolicyDFA
from cf_baselines.buliga_official_source import load_methods
from cf_baselines.buliga_reference import load_reference
from cf_search.proprietary import FrozenCFHeads,head_inventory
from evaluation.experiment_protocol import artifact,check_artifact,canonical_dfa_snapshot,load_protocol,stable_hash
from evaluation.freeze_protocol import load_frozen_oracle
from run_cf_benchmark import load_search_policy

PURPOSE = "P6_A_unified_engineering_smoke"
FAMILIES = {
    "search":["enum_outcome","postfilter_outcome","strict_dfa_outcome"],
    "relaxation":["R0_strict","R1_expanded_cap0","R2_expanded_cap1"],
    "proprietary":["O0","O1","O2"],
    "buliga":[METHOD],
    "availability":["created_unavailable"],
}
METHODS = [m for methods in FAMILIES.values() for m in methods]
GROUPS = {
    "fixed_time_search_main":["enum_outcome","postfilter_outcome","strict_dfa_outcome",METHOD],
    "same_pool_minimal_relaxation":["R0_strict","R1_expanded_cap0","R2_expanded_cap1"],
    "fixed_time_privileged_head_ablation":["O0","O1"],
    "time_intervention_auxiliary":["O1","O2"],
    "availability_only":["created_unavailable"],
}
ROOT = Path(__file__).resolve().parents[1]
SCOPES = ["all_queries","common_buliga_applicable_queries"]


def json_form(value):
    return json.loads(json.dumps(value,allow_nan=False))


def seal(value):
    value = json_form(value)
    return dict(value,content_sha256=stable_hash(value))


def checked_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("content_sha256") != stable_hash({k:v for k,v in value.items() if k != "content_sha256"}):
        raise ValueError("Suite JSON fingerprint mismatch: "+str(path))
    return value


def validate(config):
    required = {"purpose","formal_experiment","training_allowed","audit_tuning_allowed","methods", "groups",
        "report_scopes","search_seed","max_generations","max_reference_rows","reference_dir","tasks"}
    if set(config) != required or config["purpose"] != PURPOSE or any(config[k] is not False for k in
        ("formal_experiment","training_allowed","audit_tuning_allowed")):
        raise ValueError("Only frozen P6-A engineering scope; no formal/train/Audit tuning")
    if config["methods"] != METHODS or config["groups"] != GROUPS or config["report_scopes"] != SCOPES:
        raise ValueError("Method/group/scope registry changed; O2 is auxiliary only")
    if (type(config["search_seed"]) is not int or config["search_seed"] != 42 or
        type(config["max_generations"]) is not int or config["max_generations"] != 50 or
        type(config["max_reference_rows"]) is not int or config["max_reference_rows"] != 64):
        raise ValueError("P6-A preserves accepted Buliga bounded settings")
    if len(config["tasks"]) != 2:
        raise ValueError("Exactly two datasets/six queries required")
    seen = set()
    for task in config["tasks"]:
        if set(task) != {"protocol","query_offsets"} or task["query_offsets"] != [0,1,2] or any(type(i) is not int for i in task["query_offsets"]):
            raise ValueError("Fixed query offsets cannot change")
        path = Path(task["protocol"]).resolve()
        protocol = load_protocol(path)
        dataset = protocol["dataset"]
        accepted = ROOT/"evaluation_manifests/cf_comparison_v1"/(dataset+"_full_seed42")/"protocol.json"
        if dataset not in ("sepsis_cases_1","BPIC15_2_f2") or dataset in seen or path != accepted.resolve():
            raise ValueError("Only accepted frozen full/seed42 protocols allowed")
        seen.add(dataset)
        if (protocol["budget"]["model_evaluations"] != 500 or protocol["budget"]["candidate_attempts"] != 10000
                or protocol["budget"]["walltime_seconds"] != 60 or protocol["target"]["threshold"] != .5
                or protocol["intervention"]["topk"] != 5):
            raise ValueError("Frozen common target/engineering caps changed")
    return True


def child_configs(config):
    tasks = json_form(config["tasks"])
    return {
        "search":dict(purpose="P2_P3_engineering_smoke",formal_experiment=False,methods=FAMILIES["search"],tasks=tasks),
        "relaxation":dict(purpose="P4_A_engineering_smoke",formal_experiment=False,
            policy_id="strict_pool_first_minimum_violations_v1",methods=FAMILIES["relaxation"],tasks=tasks),
        "proprietary":dict(purpose="P4_B_engineering_smoke",formal_experiment=False,methods=FAMILIES["proprietary"],tasks=tasks),
        "buliga":dict(purpose="P5_Buliga_completion_smoke",formal_experiment=False,methods=FAMILIES["buliga"],tasks=tasks,
            reference_dir=config["reference_dir"],search_seed=42,max_generations=50,max_reference_rows=64),
    }


def prepare(config):
    """No candidate generation, labels, Audit compilation, fitting, or training."""
    validate(config)
    load_methods()  # Verifies pinned source and both licenses, imports no pipeline.
    torch.set_num_threads(1)
    prepared,sources,cohort,events = {},[],[],[]
    for task in config["tasks"]:
        path = Path(task["protocol"]).resolve()
        protocol = load_protocol(path)
        dataset = protocol["dataset"]
        def timed(operation,call):
            start = time.perf_counter()
            try:
                result = call()
            except Exception as exc:
                events.append(dict(dataset=dataset,operation=operation,walltime_seconds=time.perf_counter()-start,model_evaluations=0,
                    error=f"{type(exc).__name__}: {exc}"))
                exc.p6a_preparation_events = events
                raise
            events.append(dict(dataset=dataset,operation=operation,walltime_seconds=time.perf_counter()-start,model_evaluations=0))
            return result
        base,contexts = timed("frozen_model_and_query_restore",lambda:load_frozen_oracle(path))
        checkpoint_path = check_artifact(protocol["model"]["checkpoint"],path.parent)
        checkpoint = timed("cf_checkpoint_read",lambda:torch.load(checkpoint_path,map_location="cpu",weights_only=False))
        inventory = timed("cf_weight_inventory",lambda:head_inventory(checkpoint))
        heads = timed("cf_head_restore",lambda:{mode:FrozenCFHeads(checkpoint["cfg"],checkpoint["cf_model_state"],mode == "O2")
            if inventory[mode]["available"] else None for mode in ("O1","O2")})
        policy = timed("atomic_search_dfa_compile",lambda:load_search_policy(protocol))
        combined = timed("combined_search_dfa_prepare",lambda:PolicyDFA(policy))
        reference_path = Path(config["reference_dir"])/(dataset+".json")
        reference = timed("existing_reference_read_and_verify",lambda:load_reference(reference_path))
        provenance = protocol["preprocessing"]
        allowed = set(provenance["search_discovery_case_ids"]) | set(provenance["search_calibration_case_ids"])
        excluded = set().union(*(set(provenance[k]) for k in ("audit_case_ids","predictive_validation_case_ids","predictive_test_case_ids")))
        if (reference["protocol"] != artifact(path) or set(reference["case_ids"]) != allowed or allowed & excluded
                or not allowed <= set(provenance["predictive_train_case_ids"])):
            raise ValueError("Isolated reference protocol/case whitelist drift")
        check_artifact(reference["raw_source"],reference_path.parent)
        check_artifact(reference["encoder"],reference_path.parent)
        ids = [i for i,raw in enumerate(protocol["model"]["activity_vocabulary"],1)
            if raw.upper() not in ("__UNK__","<UNK>","UNK","__PAD__","<PAD>")]
        snapshots = {r["rule_id"]:canonical_dfa_snapshot(r["dfa"]) for c in policy.clusters for r in c["rule_dfas"]}
        for offset in task["query_offsets"]:
            query,frozen = contexts[offset],protocol["queries"][offset]
            strict = policy.evidence(query.activities)["strict"]
            has_reference = any(len(r["activities"]) >= query.prefix_len for r in reference["rows"])
            cohort.append(dict(dataset=dataset,offset=offset,context=asdict(query),input_sha256=frozen["input_sha256"],
                original_probability=frozen["original_probability"],original_class=frozen["original_class"],
                factual_search_strict=strict,permitted_length_reference_available=has_reference,
                common_buliga_applicable=strict and has_reference))
        sources.append(dict(dataset=dataset,protocol=artifact(path),model=protocol["model"],search=protocol["search"],
            audit_policy=protocol["audit_policy"],budget=protocol["budget"],intervention=protocol["intervention"],
            proposal_activity_ids=ids,reference=artifact(reference_path),reference_case_ids=reference["case_ids"],
            reference_reused=True,new_reference_build_walltime_seconds=0.,
            historical_reference_build_walltime_seconds=reference["build_walltime_seconds"],historical_reference_model_evaluations=reference["model_evaluations"],
            cf_head_inventory=inventory,atomic_snapshots=snapshots,combined_snapshot=combined.snapshot,
            combined_build_statistics=combined.build_statistics))
        prepared[str(path)] = dict(base=base,contexts=contexts,checkpoint=checkpoint,checkpoint_path=str(checkpoint_path),
            inventory=inventory,heads=heads,policy=policy,combined=combined,reference=reference)
        print(f"P6-A preflight {dataset}: references={reference['case_count']}, heads="+
            str({m:inventory[m]["available"] for m in ("O1","O2")}),flush=True)
    return prepared,json_form(sources),json_form(cohort),events


class Profile:
    def __init__(self):
        self.depth = 0
        self.values = dict(outcome_sequences=0,outcome_batches=0,outcome_walltime_seconds=0.,
            context_encodings=0,context_walltime_seconds=0.,cf_outcome_evaluations=0,cf_outcome_batches=0,cf_outcome_walltime_seconds=0.,
            cf_time_evaluations=0,cf_time_batches=0,cf_time_walltime_seconds=0.)
    @contextmanager
    def observe(self,model,heads):
        original_forward,original_encode = model.forward_prefix_outcome,model.encode
        def forward(*args,**kwargs):
            values = args[0] if args else kwargs["activities"]
            self.values["outcome_sequences"] += len(values)
            self.values["outcome_batches"] += 1
            self.depth += 1
            start = time.perf_counter()
            try:
                return original_forward(*args,**kwargs)
            finally:
                self.depth -= 1
                self.values["outcome_walltime_seconds"] += time.perf_counter()-start
        def encode(*args,**kwargs):
            if self.depth:
                return original_encode(*args,**kwargs)
            start = time.perf_counter()
            self.values["context_encodings"] += len(args[0] if args else kwargs["activities"])
            try:
                return original_encode(*args,**kwargs)
            finally:
                self.values["context_walltime_seconds"] += time.perf_counter()-start
        handles = []
        for head in heads:
            if head is None:
                continue
            for name in ("cf_outcome","cf_time"):
                module = getattr(head,name+"_head",None)
                if module is None:
                    continue
                clock = []
                def before(module,args,_name=name,_clock=clock):
                    _clock.append(time.perf_counter())
                    self.values[_name+"_evaluations"] += len(args[0])
                    self.values[_name+"_batches"] += 1
                def after(module,args,result,_name=name,_clock=clock):
                    self.values[_name+"_walltime_seconds"] += time.perf_counter()-_clock.pop()
                handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after,always_call=True)])
        with patch.object(model,"forward_prefix_outcome",forward),patch.object(model,"encode",encode):
            try:
                yield self
            finally:
                for handle in handles:
                    handle.remove()


@contextmanager
def instrument(module,family,prepared,traces):
    """Cache immutable preparation and observe actual calls; outputs unchanged."""
    original_generate,original_load = module.generate,torch.load
    by_model = {id(p["base"].model):p for p in prepared.values()}
    by_checkpoint = {id(p["checkpoint"]):p for p in prepared.values()}
    def cached_oracle(path,*args,**kwargs):
        if args or kwargs:
            raise ValueError("Generation cache supports only unchanged oracle restore")
        item = prepared[str(Path(path).resolve())]
        return item["base"],item["contexts"]
    def cached_policy(protocol):
        item = next(p for p in prepared.values() if p["contexts"][0].dataset == protocol["dataset"])
        return item["policy"]
    def cached_load(path,*args,**kwargs):
        item = next((p for p in prepared.values() if str(Path(path).resolve()) == p["checkpoint_path"]),None)
        return item["checkpoint"] if item else original_load(path,*args,**kwargs)
    def generated(*args,**kwargs):
        query,oracle = args[:2]
        if family == "buliga":
            method = METHOD
        else:
            method = args[4] if len(args) > 4 else kwargs["method"]
            if family == "relaxation":
                method = "R0_strict" if method == "strict_dfa_outcome" else "R1_R2_shared_expanded"
        item = by_model[id(oracle.model)]
        profile = Profile()
        start = time.perf_counter()
        error = None
        try:
            with profile.observe(oracle.model,list(item["heads"].values())):
                result = original_generate(*args,**kwargs)
            result.statistics["p6a_profile"] = dict(profile.values,generation_inclusive_walltime_seconds=time.perf_counter()-start)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            traces.append(dict(family=family,method=method,input_sha256=stable_hash(asdict(query)),
                costs=dict(profile.values,generation_inclusive_walltime_seconds=time.perf_counter()-start),error=error))
    with ExitStack() as stack:
        stack.enter_context(patch.object(module,"load_frozen_oracle",cached_oracle))
        stack.enter_context(patch.object(module,"load_search_policy",cached_policy))
        stack.enter_context(patch.object(module,"generate",generated))
        if family == "proprietary":
            stack.enter_context(patch.object(torch,"load",cached_load))
            stack.enter_context(patch.object(module,"head_inventory",lambda checkpoint:by_checkpoint[id(checkpoint)]["inventory"]))
            stack.enter_context(patch.object(module,"FrozenCFHeads",lambda cfg,state,require_time=False:
                next(p for p in prepared.values() if p["checkpoint"]["cfg"] is cfg)["heads"]["O2" if require_time else "O1"]))
        if family == "buliga":
            stack.enter_context(patch.object(module,"PolicyDFA",lambda policy:next(p for p in prepared.values() if p["policy"] is policy)["combined"]))
        yield
