"""Read-only independent recovery/DFA acceptance, no oracle or CF generators.

Reconstructs FIVE allowed raw activity histories using frozen vocabulary (not
the recovery encoder), and proves complete DFA language equivalence by joint
state BFS with a separately implemented atomic-AND/cluster-OR predicate.
"""
from collections import defaultdict, deque
from pathlib import Path
import time
import numpy as np
import pandas as pd
from dataset_confs import DatasetConfs
from cf_search.suite import checked_json, seal
from evaluation.cf_prerequisites import guard, RECOVER
from evaluation.cf_formal_protocol import DATASETS
from evaluation.cf_experiment_results import write_json, read_jsonl
from evaluation.experiment_protocol import artifact, check_artifact, load_protocol, stable_hash
from explaination.visualizations.utils import normalize_activity_name


def reference_reconstruction(protocol_path, reference):
    """Read only the three permitted fields; select cases before time/activity work."""
    path = Path(protocol_path).resolve()
    protocol = load_protocol(path)
    prov = protocol["preprocessing"]
    allowed = set(prov["search_discovery_case_ids"]) | set(prov["search_calibration_case_ids"])
    excluded = set(prov["audit_case_ids"]+prov["predictive_validation_case_ids"]+prov["predictive_test_case_ids"])
    if (not allowed or allowed & excluded or not allowed <= set(prov["predictive_train_case_ids"]) or
            set(reference["case_ids"]) != allowed or reference["protocol"] != artifact(path)):
        raise ValueError("Independent reference whitelist/isolation failure")
    raw = [r for r in protocol["source_artifacts"] if str(r["path"]).replace("\\", "/").startswith("../../../logs/")
        and str(r["path"]).endswith(".csv")]
    if len(raw) != 1:
        raise ValueError("One authoritative frozen raw log required")
    source = check_artifact(raw[0], path.parent)
    conf = DatasetConfs(protocol["dataset"])
    cols = [conf.case_id_col[protocol["dataset"]], conf.activity_col[protocol["dataset"]], conf.timestamp_col[protocol["dataset"]]]
    frame = None
    for sep in (';', ','):
        if set(cols) <= set(pd.read_csv(source, sep=sep, nrows=0).columns):
            frame = pd.read_csv(source, sep=sep, usecols=cols)
            break
    if frame is None:
        raise ValueError("Three authorized source columns missing")
    case, activity, timestamp = cols
    frame[case] = frame[case].map(str)
    frame = frame.loc[frame[case].isin(allowed), cols].copy()
    if frame.isna().any().any() or frame.empty:
        raise ValueError("Missing case/activity/timestamp in permitted training histories")
    frame[timestamp] = pd.to_datetime(frame[timestamp], utc=True)
    if frame[timestamp].isna().any():
        raise ValueError("Missing parsed training timestamp")
    vocabulary = {str(name): i+1 for i, name in enumerate(protocol["model"]["activity_vocabulary"])}
    records = []
    tied_rows, tied_cases = 0, 0
    for case_id in sorted(allowed):
        group = frame.loc[frame[case] == case_id]
        if group.empty:
            raise ValueError("A permitted training case is absent from raw source")
        # Independent array ordering, intentionally same existing quicksort tie
        # semantics; no new tie-breaking policy is introduced into references.
        nanos = group[timestamp].array.asi8
        # NumPy can choose a different unstable quicksort implementation for
        # int64 than datetime64, changing tie permutations. Preserve temporal
        # dtype as well as quicksort kind, exactly as the legacy reader does.
        temporal = nanos.view("datetime64["+group[timestamp].array.dtype.unit+"]")
        order = np.argsort(temporal, kind="quicksort")
        names = group[activity].map(str).to_numpy()[order].tolist()
        if any(name not in vocabulary for name in names):
            raise ValueError("Unseen raw activity in permitted training history")
        records.append(dict(case_id=case_id, activities=[vocabulary[name] for name in names]))
        duplicate_times = len(nanos)-len(set(nanos.tolist()))
        tied_rows += duplicate_times
        tied_cases += int(duplicate_times > 0)
    if records != reference["rows"] or reference["rows_sha256"] != stable_hash(records):
        raise ValueError("Independent raw-history reconstruction differs from recovered artifact")
    if reference["raw_source"] != artifact(source):
        raise ValueError("Reference does not link the authoritative raw source")
    return dict(dataset=protocol["dataset"], checked_case_count=len(records), checked_event_count=len(frame),
        input_columns=cols, prohibited_case_overlap=0, missing_permitted_case_count=0,
        timestamp_duplicate_excess_rows=tied_rows, timestamp_tied_case_count=tied_cases,
        raw_reconstruction_matches=True,
        tie_order_limitation="Equal timestamps retain legacy runtime quicksort ordering; not a verified business ordering" if tied_rows else None)


def official_symbol(index):
    letters = []
    while index > 0:
        index, rem = divmod(index-1, 26)
        letters.insert(0, chr(ord('a')+rem))
    return "event"+"".join(letters)


def complete_language_equivalence(source, vocabulary, max_pairs=100000):
    """Prove equivalence over ALL finite strings, not just observed prefixes."""
    started = time.perf_counter()
    snapshots = source["atomic_snapshots"]
    ids = sorted(snapshots)
    index = {rid: i for i, rid in enumerate(ids)}
    rules = source["search"]
    if set(ids) != {r["rule_id"] for r in rules["rules"]}:
        raise ValueError("Atomic snapshot/rule identity mismatch")
    protected = [index[r] for r in rules["protected_rule_ids"]]
    groups = [[index[r] for r in c["rule_ids"]] for c in rules["clusters"]]
    atomic_accept = [set(snapshots[r]["accepting_states"]) for r in ids]
    atomic_transitions = [[dict(t) for t in snapshots[r]["transitions"]] for r in ids]
    native = source["combined_snapshot"]
    if native is None or source["combined_dfa_status"] != "verified":
        raise ValueError("Official combined DFA not prepared")
    native_transitions = [dict(t) for t in native["transitions"]]
    native_accept = set(native["accepting_states"])
    alphabet = [(normalize_activity_name(name), official_symbol(i+1)) for i, name in enumerate(vocabulary)]
    required = {letter for _, letter in alphabet}
    if any(set(t) != required for t in native_transitions):
        raise ValueError("Combined DFA must be complete on the entire frozen activity vocabulary")
    initial = (tuple(snapshots[r]["initial_state"] for r in ids), native["initial_state"])
    seen = {initial}
    pending = deque([initial])
    transitions_checked = 0
    while pending:
        atomic, state = pending.popleft()
        values = [s in a for s, a in zip(atomic, atomic_accept)]
        expected = all(values[i] for i in protected) and (not groups or any(all(values[i] for i in g) for g in groups))
        if expected != (state in native_accept):
            raise ValueError("Combined DFA language mismatch, including cross-cluster protection")
        for name, symbol in alphabet:
            next_atomic = []
            for i, s in enumerate(atomic):
                outgoing = {} if s is None else atomic_transitions[i][s]
                next_atomic.append(outgoing.get(name, outgoing.get('*')))
            target = (tuple(next_atomic), native_transitions[state][symbol])
            transitions_checked += 1
            if target not in seen:
                if len(seen) >= max_pairs:
                    raise ValueError("Independent full-language proof pair cap; never substitute sampled acceptance")
                seen.add(target)
                pending.append(target)
    return dict(dataset=source.get("dataset"), equivalent=True, proof="exhaustive_joint_state_language_equivalence",
        reachable_state_pairs=len(seen), transitions_checked=transitions_checked,
        native_states=len(native_transitions), protected_rules=len(protected),
        walltime_seconds=time.perf_counter()-started, model_evaluations=0, generated_candidate_count=0)


def verify(root):
    root = Path(root).resolve()
    if (root/"independent_acceptance.json").exists():
        raise ValueError("Independent acceptance is immutable; no overwrite")
    base, protocol, recovery, completion = guard(root)
    begin = time.perf_counter()
    reference_checks, language_proofs = [], []
    for record, source, old in zip(recovery["records"], protocol["sources"], base["sources"]):
        ref = checked_json(check_artifact(record["reference"], root))
        if (record["case_count"] != ref["case_count"] or record["event_count"] != sum(len(r["activities"]) for r in ref["rows"]) or
                ref["label_columns_read"] or ref["attribute_columns_read"] or ref["predictor_training_started"] or
                ref["statistical_model_fit_started"] or ref["model_evaluations"]):
            raise ValueError("Reference count/leakage/no-model contract failed")
        if source["reference"]["artifact"] != record["reference"]:
            raise ValueError("Protocol consumed a different reference from the recovery manifest")
        if source["dataset"] in RECOVER:
            check = reference_reconstruction(check_artifact(source["protocol"], root), ref)
        else:
            if record["reference"]["sha256"] != old["reference"]["artifact"]["sha256"]:
                raise ValueError("Existing references must be preserved byte-for-byte")
            check = dict(dataset=source["dataset"], existing_reference_byte_identical=True,
                checked_case_count=ref["case_count"], checked_event_count=record["event_count"], prohibited_case_overlap=0)
        reference_checks.append(check)
        original = load_protocol(check_artifact(source["protocol"], root))
        language_proofs.append(complete_language_equivalence(source, original["model"]["activity_vocabulary"]))
    # Reuse the separate P6-B saved-case checker, NOT a generation function.
    from tools.verify_p6b_protocol import verify as verify_protocol
    if not (root/"protocol/independent_acceptance.json").exists():
        child = verify_protocol(root/"protocol")
    else:
        child = checked_json(root/"protocol/independent_acceptance.json")
    if child["status"] != "passed" or child["protocol"] != completion["protocol"] or child["missing_references"]:
        raise ValueError("Refrozen child protocol not independently accepted")
    # Independent membership/Factual Failure aggregation, no cohort_counts.
    per_dataset = []
    for dataset in DATASETS:
        rows = [r for r in protocol["cohort"] if r["dataset"] == dataset]
        common = [r for r in rows if r["common_buliga_applicable"]]
        failures = [r for r in rows if r["original_class"] == "Failure"]
        common_failures = [r for r in common if r["original_class"] == "Failure"]
        count = dict(dataset=dataset, query_count=len(rows), factual_failure_queries=len(failures),
            reference_case_count=next(r["case_count"] for r in recovery["records"] if r["dataset"] == dataset),
            common_query_count=len(common), common_failure_queries=len(common_failures),
            common_original_case_count=len({r["context"]["case_id"] for r in common}),
            common_failure_case_count=len({r["context"]["case_id"] for r in common_failures}),
            unknown_common_membership_count=sum(r["common_buliga_applicable"] is None for r in rows))
        frozen_count = next(r for r in completion["per_dataset"] if r["dataset"] == dataset)
        if any(count[k] != frozen_count[k] for k in ("query_count", "factual_failure_queries", "common_query_count", "common_failure_queries")):
            raise ValueError("Independent all/common Failure numerator mismatch")
        per_dataset.append(count)
    if (sum(r["query_count"] for r in per_dataset) != 350 or sum(r["factual_failure_queries"] for r in per_dataset) != 226 or
            sum(r["common_query_count"] for r in per_dataset) != completion["common_query_count"] or
            sum(r["common_failure_queries"] for r in per_dataset) != completion["common_failure_queries"] or
            any(r["unknown_common_membership_count"] for r in per_dataset)):
        raise ValueError("Complete population or common membership freeze failed")
    reference_cost = sum(r["new_cold_build_walltime_seconds"] for r in recovery["records"])
    if reference_cost != completion["new_cold_reference_build_walltime_seconds"]:
        raise ValueError("Cold recovery costs missing or double-counted")
    prohibited = ("candidates.jsonl", "queries.jsonl", "search_manifest.json", "generation_manifest.json", "children")
    if any((directory/name).exists() for directory in (root, root/"protocol") for name in prohibited):
        raise ValueError("No new CF matrix artifacts permitted")
    accepted = seal(dict(schema_version=1, purpose="P6_B_prerequisite_independent_acceptance", status="passed",
        completion_manifest=artifact(root/"completion_manifest.json"), recovery_manifest=artifact(root/"recovery_manifest.json"),
        protocol=completion["protocol"], child_acceptance=artifact(root/"protocol/independent_acceptance.json"),
        verification_code=artifact(Path(__file__)), reference_checks=reference_checks, complete_language_proofs=language_proofs,
        per_dataset=per_dataset, query_count=350, factual_failure_queries=226,
        common_query_count=completion["common_query_count"], common_failure_queries=completion["common_failure_queries"],
        common_cohort_fully_frozen=True, missing_references=[], new_reference_build_count=5, reused_reference_count=2,
        new_cold_reference_build_walltime_seconds=reference_cost, reference_recovery_started=True,
        full_external_matrix_artifacts_ready=completion["full_external_matrix_artifacts_ready"],
        current_preparation_factual_model_evaluations=350, independent_verification_model_evaluations=0,
        training_started=False, statistical_model_fit_started=False, candidate_generation_started=False,
        formal_experiment_started=False, formal_execution_authorized=False, generated_candidate_count=0,
        inclusive_walltime_seconds=time.perf_counter()-begin,
        interpretation="Prerequisite artifacts and common cohort accepted; no formal generation bridge or matrix execution"))
    write_json(root/"independent_acceptance.json", accepted)
    print(f"P6-B prerequisites accepted: 7 references, 7 complete language proofs; common={completion['common_query_count']}, Failure={completion['common_failure_queries']}; no CF generation", flush=True)
    return accepted
