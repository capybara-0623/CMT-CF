"""Export accepted P6-B metrics/statistics as paper tables, without rerunning them."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from cf_search.suite import checked_json, seal
from evaluation.cf_experiment_results import read_jsonl, write_json
from evaluation.experiment_protocol import artifact, check_artifact


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                         if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def export(run_dir, statistics_dir, output_dir):
    root, stats, out = [Path(p).resolve() for p in (run_dir, statistics_dir, output_dir)]
    acceptance = checked_json(root / "independent_acceptance.json")
    metrics = checked_json(root / "metrics.json")
    statistics = checked_json(stats / "statistics.json")
    if (acceptance["status"] != "passed" or acceptance["scope"] != "seven_task_matrix" or
            acceptance["metrics"] != artifact(root / "metrics.json") or
            acceptance["protocol"] != metrics["protocol"] or statistics["protocol"] != metrics["protocol"] or
            statistics["engineering_pilot_only"] or statistics["query_count"] != 350):
        raise ValueError("Accepted full matrix and matching formal statistics required")
    for record in list(acceptance["saved_files"].values()) + list(statistics["saved_files"].values()) + [statistics["input"]]:
        check_artifact(record, root)
    rows = [r for r in metrics["rows"] if r["method"] != "created_unavailable"]
    intervals = read_jsonl(stats / "paired_intervals.jsonl")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new table output directory")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    write_csv(out / "all_methods.csv", rows)
    files["all_methods"] = artifact(out / "all_methods.csv")
    for group, methods in metrics["groups"].items():
        if group == "availability_only":
            continue
        path = out / (group + ".csv")
        write_csv(path, [r for r in rows if r["method"] in methods])
        files[group] = artifact(path)
    write_csv(out / "paired_intervals.csv", intervals)
    files["paired_intervals"] = artifact(out / "paired_intervals.csv")
    query_costs = [r for r in metrics["costs"]["per_query"] if r["method"] != "created_unavailable"]
    write_csv(out / "query_costs.csv", query_costs)
    files["query_costs"] = artifact(out / "query_costs.csv")
    costs = dict(costs=metrics["costs"], prerequisite_costs=metrics["prerequisite_costs"],
        runtime_preparation_events=metrics["runtime_preparation_events"],
        post_save_audit_phases=metrics["post_save_audit_phases"],
        final_acceptance_seconds=acceptance["inclusive_walltime_seconds"],
        caveat="per-query views may share physical cost; KD/model and inclusive timers overlap; do not sum them")
    write_json(out / "costs.json", costs)
    files["costs"] = artifact(out / "costs.json")
    lines = ["# 七任务正式对比结果", "", "Buliga：官方核心算法的统一协议适配版。CREATED 未参与实测对比。", "",
        "率保持原指标定义；改善率分母包括所有原预测 Failure 查询，无候选查询不删除。联合成功要求同一个候选达标、Search 严格满足且激活的 Audit 全部通过。", "",
        "主比较时间固定；O0/O1 与 O1/O2 分别见专有头消融和时间干预辅助表。统计条件于既有模型及 seed42，不作训练随机性稳健或自动显著胜出声明。", ""]
    def rate(value):
        return "—" if value is None else f"{100*value:.2f}%"
    for scope in metrics["report_scopes"]:
        lines += ["## " + scope, "", "| 任务 | 方法 | 查询数 | Failure 分母 | 预测改善 | 严格改善 | 严格＋Audit 联合改善 | 覆盖率 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in rows:
            if row["scope"] == scope and row["method"] in metrics["groups"]["fixed_time_search_main"]:
                lines.append("| " + " | ".join([row["dataset"], row["method"], str(row["query_count"]),
                    str(row["eligible_failure_queries"]), rate(row["improvement_success_at_k"]),
                    rate(row["strict_improvement_success_at_k"]), rate(row["audit_joint_improvement_success_at_k"]),
                    rate(row["coverage"])]) + " |")
        lines.append("")
    lines += ["## 保存后 Audit 及最小放宽", "",
        "| 任务 | 方法 | 候选数 | Audit 通过 | Audit 违反 | 零激活 | 严格改善查询数 | 联合改善查询数 | 放宽改善查询数 | 豁免总数 | 豁免总代价 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if row["scope"] == "all_queries" and row["method"] in ("strict_dfa_outcome", "R1_expanded_cap0", "R2_expanded_cap1", "buliga_official_protocol_v1"):
            keys = ("candidate_count", "audit_pass", "audit_fail", "audit_not_applicable", "strict_improved_queries",
                    "audit_joint_improved_queries", "relaxed_improved_queries", "waiver_count", "waiver_cost")
            lines.append("| " + " | ".join([row["dataset"], row["method"]] + [str(row[k]) for k in keys]) + " |")
    lines += ["", "## 工件", ""]
    lines += [f"- [{name}]({Path(record['path']).as_posix()})" for name, record in files.items()]
    with (out / "RESULTS.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    files["report"] = artifact(out / "RESULTS.md")
    receipt = seal(dict(purpose="P6_B_accepted_formal_table_export_v1", status="passed",
        code=artifact(__file__), acceptance=artifact(root / "independent_acceptance.json"),
        metrics=artifact(root / "metrics.json"), statistics=artifact(stats / "statistics.json"),
        metric_rows=len(rows), interval_rows=len(intervals), query_cost_rows=len(query_costs),
        interval_status_counts=dict(Counter(r["status"] for r in intervals)), files=files,
        new_candidate_generation=False, model_evaluations=0, audit_tuning_performed=False))
    write_json(out / "tables_manifest.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--statistics-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = export(args.run_dir, args.statistics_dir, args.output_dir)
    print(json.dumps({k: report[k] for k in ("status", "metric_rows", "interval_rows")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
