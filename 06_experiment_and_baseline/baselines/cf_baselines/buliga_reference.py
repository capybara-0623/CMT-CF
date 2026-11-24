"""Recover only permitted training activity histories, never labels/Audit rules."""
import json
import pickle
import time
from pathlib import Path
import pandas as pd
from dataset_confs import DatasetConfs
from evaluation.experiment_protocol import artifact,check_artifact,load_protocol,stable_hash
from evaluation.cf_experiment_results import write_json


def build_reference(protocol_path,output_path):
    if Path(output_path).exists():
        raise ValueError("Training reference artifacts are immutable; use a new path")
    started = time.perf_counter()
    path = Path(protocol_path).resolve()
    protocol = load_protocol(path)
    provenance = protocol["preprocessing"]
    allowed = set(provenance["search_discovery_case_ids"]) | set(provenance["search_calibration_case_ids"])
    excluded = set(provenance["audit_case_ids"]) | set(provenance["predictive_validation_case_ids"]) | set(provenance["predictive_test_case_ids"])
    if not allowed or allowed & excluded or not allowed <= set(provenance["predictive_train_case_ids"]):
        raise ValueError("Training reference isolation not proven")
    dataset = protocol["dataset"]
    raw = [r for r in protocol["source_artifacts"] if str(r["path"]).replace("\\","/").startswith("../../../logs/")
           and str(r["path"]).endswith(".csv")]
    if len(raw) != 1:
        raise ValueError("Need one frozen raw CSV activity source")
    raw_path = check_artifact(raw[0],path.parent)
    conf = DatasetConfs(dataset)
    columns = [conf.case_id_col[dataset],conf.activity_col[dataset],conf.timestamp_col[dataset]]
    # Only these columns are read. Outcomes and attributes are not reference input.
    separators = [';',',']
    frame = None
    for separator in separators:
        try:
            header = pd.read_csv(raw_path,sep=separator,nrows=0).columns
            if not set(columns) <= set(header):
                continue
            frame = pd.read_csv(raw_path,sep=separator,usecols=columns)
            break
        except ValueError:
            continue
    if frame is None or not set(columns) <= set(header):
        raise ValueError("Frozen raw CSV does not match configured columns")
    case_col,act_col,time_col = columns
    frame[case_col] = frame[case_col].astype(str)
    frame = frame[frame[case_col].isin(allowed)].copy()
    frame[time_col] = pd.to_datetime(frame[time_col],utc=True)
    encoder_path = check_artifact(protocol["model"]["oracle_context"],path.parent)
    with encoder_path.open("rb") as handle:
        encoder = pickle.load(handle)["activity_encoder"]
    mapping = {str(name):i+1 for i,name in enumerate(encoder.classes_)}
    rows = []
    for case,group in frame.groupby(case_col,sort=True):
        names = group.sort_values(time_col)[act_col].astype(str).tolist()
        if any(name not in mapping for name in names):
            raise ValueError("Unseen activity in purported training reference")
        rows.append(dict(case_id=str(case),activities=[mapping[n] for n in names]))
    if {r["case_id"] for r in rows} != allowed:
        raise ValueError("Permitted training cases missing from raw source")
    report = dict(schema_version=1,dataset=dataset,protocol=artifact(path),raw_source=artifact(raw_path),
        encoder=artifact(encoder_path),case_ids=sorted(allowed),case_count=len(rows),rows=rows,
        excluded_case_ids=sorted(excluded),label_columns_read=False,attribute_columns_read=False,
        rows_sha256=stable_hash(rows),build_walltime_seconds=time.perf_counter()-started,
        model_evaluations=0,predictor_training_started=False,statistical_model_fit_started=False,
        scope="Search discovery/calibration training activities only; per-query reference projection disclosed")
    report["content_sha256"] = stable_hash(report)
    write_json(output_path,report)
    return report


def load_reference(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (report["content_sha256"] != stable_hash({k:v for k,v in report.items() if k != "content_sha256"})
            or report["rows_sha256"] != stable_hash(report["rows"]) or report["label_columns_read"]
            or report["attribute_columns_read"] or report["predictor_training_started"] or report["statistical_model_fit_started"]
            or report["case_count"] != len(report["rows"]) or len(set(report["case_ids"])) != report["case_count"]
            or [r["case_id"] for r in report["rows"]] != report["case_ids"]
            or any(set(r) != {"case_id","activities"} or not r["activities"]
                   or any(type(a) is not int or a < 1 for a in r["activities"]) for r in report["rows"])
            or set(report["case_ids"]) & set(report["excluded_case_ids"])):
        raise ValueError("Reference integrity/isolation failure")
    return report


def project_reference(reference,query,activity_ids,max_rows=64):
    if type(max_rows) is not int or max_rows < 1 or max_rows > 64:
        raise ValueError("Engineering reference cap is frozen at at most 64 rows")
    ids = set(activity_ids)
    rows,seen = [],set()
    for row in reference["rows"]:
        if len(row["activities"]) < query.prefix_len:
            continue
        projected = list(query.activities)
        for p in query.allowed_positions:
            if row["activities"][p] in ids:
                projected[p] = row["activities"][p]
        key = tuple(projected)
        if key not in seen:
            seen.add(key)
            rows.append(dict(case_id=row["case_id"],activities=projected))
        if len(rows) >= max_rows:
            break
    return rows
