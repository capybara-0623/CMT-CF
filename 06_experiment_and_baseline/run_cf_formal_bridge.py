"""P6-B additive bridge. Default real-generation scope is LOCKED, not a smoke run."""
import argparse
import json
from pathlib import Path
import time

from cf_search.suite import checked_json, seal
from evaluation.cf_bridge_protocol import validate, preflight, guard, authorize, selected_cohort
from evaluation.cf_experiment_results import write_json
from evaluation.cf_bridge_results import save_generation, audit, summarize
from evaluation.experiment_protocol import artifact, stable_hash


def search(preflight_dir, run_dir, authorization_path=None):
    # Fail before creating output, loading any model, or compiling Audit.
    authorization, receipt = authorize(preflight_dir, run_dir, authorization_path)
    bridge, protocol = guard(preflight_dir)
    root = Path(run_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Generation immutable, including partial runs; use a new empty directory")
    cohort = selected_cohort(protocol, authorization["scope"])
    from cf_search.formal_bridge import runtime, generate_input
    import torch
    torch.set_num_threads(1)
    root.mkdir(parents=True, exist_ok=True)
    bridge = dict(bridge, preflight_path=str(Path(preflight_dir).resolve()/"bridge_preflight.json"))
    authorization = dict(authorization, artifact=receipt)
    write_json(root/"generation_started.json", seal(dict(bridge_preflight=artifact(bridge["preflight_path"]),
        authorization=receipt, scope=authorization["scope"], started_before_Audit=True)))
    queries, candidates, physical, units, preparation = [], [], [], [], []
    started = time.perf_counter()
    try:
        for source in protocol["sources"]:
            inputs = [r for r in cohort if r["dataset"] == source["dataset"]]
            if not inputs:
                continue
            item = runtime(source)
            preparation.append(item["preparation"])
            for frozen in inputs:
                if stable_hash(item["contexts"][frozen["offset"]].__dict__) != frozen["input_sha256"]:
                    raise ValueError("Public query projection differs from frozen source")
                qs, cs, resources = generate_input(frozen, item, root)
                directory = root/"inputs"
                directory.mkdir(exist_ok=True)
                path = directory/(frozen["input_sha256"]+".json")
                write_json(path, seal(dict(input_sha256=frozen["input_sha256"], queries=qs, candidates=cs,
                    physical_artifacts=resources, all_candidates_saved_before_audit=True, audit_used_in_generation=False)))
                queries.extend(qs)
                candidates.extend(cs)
                physical.extend(resources)
                units.append(artifact(path))
                print(f"Bridge saved {source['dataset']} offset={frozen['offset']}; Audit not compiled", flush=True)
            del item
        manifest = save_generation(root, bridge, protocol, cohort, queries, candidates, physical, units, authorization, preparation)
        return manifest
    except Exception as exc:
        write_json(root/"generation_failure.json", seal(dict(status="failed_incomplete_not_statistically_accepted",
            error=f"{type(exc).__name__}: {exc}", completed_query_method_rows=len(queries),
            saved_candidate_rows=len(candidates), physical_artifacts=physical, unit_artifacts=units,
            preparation_events=preparation, inclusive_walltime_seconds=time.perf_counter()-started,
            audit_compilation_count=0, training_started=False, reference_recovery_started=False)))
        raise


def statistics(preflight_dir, run_dir, output_dir):
    from evaluation.cf_bridge_results import export_statistical_input, load_generation
    from run_cf_protocol import statistics as frozen_statistics
    bridge, _ = guard(preflight_dir)
    if load_generation(run_dir)[3]["protocol"] != bridge["protocol"]:
        raise ValueError("CLI preflight differs from saved generation")
    export_statistical_input(run_dir)
    protocol_dir = Path(bridge["protocol"]["path"]).parent
    return frozen_statistics(protocol_dir, output_dir, pilot=False, saved_run=run_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "preflight", "verify-preflight", "search", "audit", "summarize", "verify", "statistics"))
    parser.add_argument("--config", default="configs/cf_p6b_bridge.json")
    parser.add_argument("--preflight-dir")
    parser.add_argument("--run-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--authorization")
    args = parser.parse_args(argv)
    if args.mode in ("validate", "preflight"):
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        validate(config)
        if args.mode == "validate":
            print("Bridge config valid; real generation locked; no inference/search/fit/train")
            return
        if not args.preflight_dir:
            parser.error("--preflight-dir required")
        report = preflight(config, args.preflight_dir)
    elif args.mode == "verify-preflight":
        if not args.preflight_dir:
            parser.error("--preflight-dir required")
        from tools.verify_p6b_bridge import verify_preflight
        report = verify_preflight(args.preflight_dir)
    else:
        if not args.run_dir:
            parser.error("--run-dir required")
        if args.mode in ("search", "statistics") and not args.preflight_dir:
            parser.error("--preflight-dir required")
        if args.mode == "search":
            report = search(args.preflight_dir, args.run_dir, args.authorization)
        elif args.mode == "audit":
            report = audit(args.run_dir)
        elif args.mode == "summarize":
            report = summarize(args.run_dir)
        elif args.mode == "verify":
            from tools.verify_p6b_bridge import verify_saved
            report = verify_saved(args.run_dir)
        else:
            if not args.output_dir:
                parser.error("--output-dir required")
            report = statistics(args.preflight_dir, args.run_dir, args.output_dir)
    print(json.dumps({k: v for k, v in report.items() if k in ("purpose", "status", "query_count", "query_method_count", "candidate_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
