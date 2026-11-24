"""Query-level recourse metrics. No-candidate queries remain in the denominator."""
import argparse
import math
from pathlib import Path


def _number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def query_key(row):
    return (str(row.get("case_id", "")), int(row["prefix_len"]), int(row["sample_idx"]))


def summarize_queries(rows, query_keys=None, expected_query_keys=None):
    if expected_query_keys is not None:
        actual = {query_key(row) for row in rows}
        if actual != set(expected_query_keys):
            raise ValueError("Incomplete query manifest: missing or unexpected queries")
    groups = {}
    for row in rows:
        key = query_key(row)
        if query_keys is None or key in query_keys:
            groups.setdefault(key, []).append(row)
    eligible = improved = strict = joint = unknown = audit_unknown = 0
    for candidates in groups.values():
        labels = {r.get("orig_pred_label") for r in candidates
                  if r.get("orig_pred_label") in ("Success", "Failure")}
        if len(labels) > 1:
            raise ValueError("Inconsistent original predictions within a query")
        if not labels:
            unknown += 1
            continue
        if labels == {"Success"}:
            continue
        eligible += 1
        successes = [r for r in candidates if _number(r.get("candidate_returned")) == 1
                     and r.get("cf_pred_label") == "Success"]
        strict_successes = [r for r in successes if _number(r.get("search_compliant")) == 1]
        improved += bool(successes)
        strict += bool(strict_successes)
        joint += any(_number(r.get("audit_all_applicable")) == 1
                     and _number(r.get("audit_applicable_rule_count")) > 0
                     for r in strict_successes)
        audit_unknown += bool(strict_successes) and not any(
            math.isfinite(_number(r.get("audit_all_applicable")))
            and _number(r.get("audit_applicable_rule_count")) > 0
            for r in strict_successes
        )
    def rate(count):
        return count / eligible if eligible and not unknown else float("nan")
    return {
        "query_count": len(groups),
        "original_prediction_unknown_queries": unknown,
        "eligible_failure_queries": eligible,
        "improved_queries": improved,
        "strict_improved_queries": strict,
        "audit_joint_improved_queries": joint,
        "strict_improved_audit_unverified_queries": audit_unknown,
        "improvement_success_at_k": rate(improved),
        "strict_improvement_success_at_k": rate(strict),
        "audit_joint_improvement_success_at_k": rate(joint),
    }


def main():
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    source, dest = Path(args.results_dir).resolve(), Path(args.output_dir).resolve()
    if source == dest:
        parser.error("Use a separate output directory to preserve original results")
    tables = []
    seen = set()
    for path in sorted(source.rglob("cf_details_*.csv")):
        frame = pd.read_csv(path, dtype={"case_id": str})
        if frame.empty:
            continue
        for (dataset, variant, seed), group in frame.groupby(["dataset", "variant", "seed"]):
            key = (dataset, variant, seed)
            if key in seen:
                raise ValueError("Multiple detail files for the same experiment: " + str(key))
            seen.add(key)
            tables.append((dataset, variant, seed, str(path.relative_to(source)), group.to_dict("records")))
    if not tables:
        parser.error("No counterfactual detail rows found")
    summaries = []
    for dataset, variant, seed, name, rows in tables:
        peers = [r for d, v, s, _, r in tables if d == dataset and s == seed]
        universes = [{query_key(r) for r in peer} for peer in peers]
        if any(keys != universes[0] for keys in universes):
            raise ValueError("Variant query manifests differ: " + str(dataset))
        # Unknown original predictions prevent claiming a complete common-failure cohort.
        complete = all(all(r.get("orig_pred_label") in ("Success", "Failure") for r in peer)
                       for peer in peers)
        common = set.intersection(*[
            {query_key(r) for r in peer if r.get("orig_pred_label") == "Failure"}
            for peer in peers
        ])
        summary = {"dataset": dataset, "variant": variant, "seed": seed,
                   "source_file": name, **summarize_queries(rows)}
        summary["comparison_variants"] = ";".join(sorted(str(v) for d, v, s, _, _ in tables
                                                        if d == dataset and s == seed))
        summary["common_failure_cohort_complete"] = complete
        if complete:
            summary.update({"common_" + k: v for k, v in summarize_queries(rows, common).items()})
        summaries.append(summary)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "query_metrics.csv"
    if path.exists():
        raise FileExistsError(path)
    pd.DataFrame(summaries).to_csv(path, index=False)
    print(path)


if __name__ == "__main__":
    main()
