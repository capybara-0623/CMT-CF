"""Independent, activation-aware Declare auditing for counterfactual traces.

The search constraints in this project are outcome-associated positive
subsequences.  This module intentionally uses a different, pre-specified
family of global Declare templates and discovers them from complete cases in a
case-disjoint audit fold.  Outcome labels are never read.

Two distinctions are important:

* A rule can be *applicable* (its activation occurs) or inapplicable.  An
  inapplicable implication is not counted as evidence of compliance.
* A trace-level audit score is the fraction of applicable rules it satisfies.
  It is undefined when no compiled rule is applicable.
"""

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SUPPORTED_TEMPLATES = (
    "response",
    "precedence",
    "chain_response",
    "not_coexistence",
)


def normalize_activity_name(name: Any) -> str:
    """Mirror ``explaination.visualizations.utils.normalize_activity_name``."""
    raw = "" if name is None else str(name)
    base = re.split(r"__tb", raw, flags=re.IGNORECASE)[0]
    atom = re.sub(r"[^0-9a-zA-Z_]+", "_", base.strip().lower())
    atom = re.sub(r"_+", "_", atom).strip("_")
    if not atom:
        digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:8]
        atom = f"act_{digest}"
    if not re.match(r"^[a-z]", atom):
        atom = f"a_{atom}"
    return atom


def _case_key(sample: Mapping[str, Any], fallback_index: int) -> str:
    for key in ("source_case_id", "case_id", "Case ID", "case:concept:name"):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return f"case-{fallback_index}"


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _decode_activity_value(value: Any, act_encoder=None) -> Optional[str]:
    """Decode one stored activity without applying the LTLf atom mapping."""
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None and act_encoder is not None:
        if numeric <= 0:
            return None
        name = act_encoder.inverse_transform([numeric - 1])[0]
    else:
        if value in (None, "", 0, "0"):
            return None
        name = value
    return str(name)


def decode_complete_trace(sample: Mapping[str, Any], act_encoder=None) -> List[str]:
    """Decode one complete case and reject prefix-level records explicitly."""
    if "prefix_len" in sample:
        raise ValueError(
            "Audit discovery requires raw complete cases before prefix expansion."
        )
    activities = _to_list(sample.get("activities"))
    length = int(sample.get("length", len(activities)))
    names: List[str] = []
    for value in activities[:length]:
        name = _decode_activity_value(value, act_encoder)
        if name is not None:
            names.append(normalize_activity_name(name))
    return names


def find_activity_normalization_collisions(
    cases: Sequence[Mapping[str, Any]], act_encoder=None
) -> Dict[str, List[str]]:
    """Return raw activity names that collapse to the same LTLf atom."""
    mapped: Dict[str, set] = defaultdict(set)
    for sample in cases or []:
        if "prefix_len" in sample:
            raise ValueError(
                "Audit discovery requires raw complete cases before prefix expansion."
            )
        activities = _to_list(sample.get("activities"))
        length = int(sample.get("length", len(activities)))
        for value in activities[:length]:
            raw = _decode_activity_value(value, act_encoder)
            if raw is not None:
                mapped[normalize_activity_name(raw)].add(raw)
    return {
        atom: sorted(raw_names)
        for atom, raw_names in mapped.items()
        if len(raw_names) > 1
    }


def prepare_complete_traces(
    cases: Sequence[Mapping[str, Any]], act_encoder=None
) -> List[Dict[str, Any]]:
    """Deduplicate by case and retain only complete, non-empty traces."""
    prepared: List[Dict[str, Any]] = []
    seen = set()
    for index, sample in enumerate(cases or []):
        key = _case_key(sample, index)
        if key in seen:
            raise ValueError(
                f"Audit fold contains duplicate case {key!r}; pass complete cases, not prefixes."
            )
        seen.add(key)
        trace = decode_complete_trace(sample, act_encoder)
        if trace:
            prepared.append({"case_id": key, "trace": trace})
    return prepared


def complete_case_trace_fingerprint(
    cases: Sequence[Mapping[str, Any]], act_encoder=None
) -> str:
    """Hash case IDs and complete activity traces; outcomes are excluded."""
    prepared = prepare_complete_traces(cases, act_encoder)
    canonical = sorted(
        (item["case_id"], tuple(item["trace"])) for item in prepared
    )
    payload = json.dumps(
        canonical, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_rule_set_hash(rules: Sequence[Mapping[str, Any]]) -> str:
    """Return an order-independent fingerprint of an audit rule set."""
    canonical = sorted(
        (
            str(rule.get("rule_id", "")),
            _canonical_formula(rule.get("formula", "")),
        )
        for rule in (rules or [])
    )
    payload = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rule_formula(template: str, activation: str, target: str) -> str:
    template = str(template).strip().lower()
    if template == "response":
        return f"G({activation} -> F({target}))"
    if template == "precedence":
        # Future-time encoding: either B never occurs, or A is reached before any B.
        return f"(G(!({target}))) | ((!({target})) U ({activation}))"
    if template == "chain_response":
        return f"G({activation} -> X({target}))"
    if template == "not_coexistence":
        return f"!((F({activation})) & (F({target})))"
    raise ValueError(f"Unsupported audit template: {template}")


def _canonical_formula(formula: str) -> str:
    return "".join(str(formula).split()).lower()


def _positions(trace: Sequence[str], activity: str) -> List[int]:
    return [idx for idx, value in enumerate(trace) if value == activity]


def rule_applicable(trace: Sequence[str], rule: Mapping[str, Any]) -> bool:
    template = str(rule.get("template", "")).lower()
    activation = rule.get("activation")
    target = rule.get("target")
    if template in ("response", "chain_response"):
        return activation in trace
    if template == "precedence":
        return target in trace
    if template == "not_coexistence":
        return activation in trace or target in trace
    # A generic external LTLf formula has no known activation semantics; it is
    # therefore treated as globally applicable and must be compiled.
    return True


def rule_satisfied(trace: Sequence[str], rule: Mapping[str, Any]) -> bool:
    """Evaluate supported templates directly; generic rules require a DFA."""
    template = str(rule.get("template", "")).lower()
    activation = rule.get("activation")
    target = rule.get("target")
    if template == "response":
        target_positions = _positions(trace, target)
        return all(any(j > i for j in target_positions) for i in _positions(trace, activation))
    if template == "precedence":
        activation_positions = _positions(trace, activation)
        return all(any(i < j for i in activation_positions) for j in _positions(trace, target))
    if template == "chain_response":
        return all(i + 1 < len(trace) and trace[i + 1] == target
                   for i in _positions(trace, activation))
    if template == "not_coexistence":
        return not (activation in trace and target in trace)

    dfa = rule.get("dfa")
    if dfa is None:
        raise ValueError(
            f"External rule {rule.get('rule_id', '<unknown>')} has no supported template or DFA."
        )
    from explaination.visualizations.utils import trace_accepts_dfa
    return bool(trace_accepts_dfa(list(trace), dfa))


def _rule_statistics(
    traces: Sequence[Sequence[str]], template: str, activation: str, target: str
) -> Dict[str, Any]:
    rule = {"template": template, "activation": activation, "target": target}
    applicable = [trace for trace in traces if rule_applicable(trace, rule)]
    satisfied = sum(rule_satisfied(trace, rule) for trace in applicable)
    n_cases = len(traces)
    n_applicable = len(applicable)
    return {
        "audit_case_count": n_cases,
        "activation_cases": n_applicable,
        "activation_coverage": n_applicable / n_cases if n_cases else float("nan"),
        "satisfied_activation_cases": int(satisfied),
        "violation_cases": int(n_applicable - satisfied),
        "confidence": satisfied / n_applicable if n_applicable else float("nan"),
        "support": satisfied / n_cases if n_cases else float("nan"),
    }


def _stable_rule_id(template: str, activation: str, target: str, source: str) -> str:
    payload = f"{source}|{template}|{activation}|{target}".encode("utf-8")
    return "AUD_" + hashlib.sha256(payload).hexdigest()[:12]


def _rank_rules_balanced(
    rules: Sequence[Dict[str, Any]], templates: Sequence[str], limit: int
) -> List[Dict[str, Any]]:
    """Round-robin templates so one easy family cannot occupy all slots."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        buckets[rule["template"]].append(rule)
    for template in buckets:
        buckets[template].sort(
            key=lambda r: (
                -float(r["confidence"]),
                -float(r["activation_coverage"]),
                -int(r["violation_cases"] > 0),
                str(r["formula"]),
            )
        )
    selected: List[Dict[str, Any]] = []
    while len(selected) < int(limit):
        added = False
        for template in templates:
            if buckets.get(template):
                selected.append(buckets[template].pop(0))
                added = True
                if len(selected) >= int(limit):
                    break
        if not added:
            break
    return selected


def discover_audit_rules(
    cases: Sequence[Mapping[str, Any]],
    act_encoder=None,
    *,
    templates: Sequence[str] = SUPPORTED_TEMPLATES,
    min_activation_cases: int = 10,
    min_activation_fraction: float = 0.02,
    min_confidence: float = 0.80,
    max_activities: int = 40,
    max_constraints: int = 20,
    search_formulas: Optional[Iterable[str]] = None,
    source: str = "heldout_complete_cases",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Discover label-free global Declare rules on complete audit cases."""
    collisions = find_activity_normalization_collisions(cases, act_encoder)
    if collisions:
        examples = list(sorted(collisions.items()))[:5]
        raise ValueError(
            "Distinct activity labels collapse to the same LTLf atom; "
            "audit semantics would be ambiguous: {}".format(examples)
        )
    prepared = prepare_complete_traces(cases, act_encoder)
    traces = [item["trace"] for item in prepared]
    if not traces:
        return [], {
            "audit_case_count": 0,
            "candidate_count": 0,
            "eligible_count": 0,
            "formula_overlap_count": 0,
            "reason": "no_complete_audit_cases",
            "activity_normalization_collision_count": 0,
        }

    selected_templates = []
    for value in templates:
        template = str(value).strip().lower()
        if template not in SUPPORTED_TEMPLATES:
            raise ValueError(
                f"Unsupported audit template {value!r}; choose from {SUPPORTED_TEMPLATES}."
            )
        if template not in selected_templates:
            selected_templates.append(template)

    activity_support = Counter()
    for trace in traces:
        activity_support.update(set(trace))
    activities = [
        activity for activity, _ in sorted(
            activity_support.items(), key=lambda item: (-item[1], item[0])
        )[: max(2, int(max_activities))]
    ]

    required_activation = max(
        int(min_activation_cases),
        int(math.ceil(float(min_activation_fraction) * len(traces))),
    )
    search_keys = {_canonical_formula(value) for value in (search_formulas or []) if value}
    candidate_count = 0
    overlap_count = 0
    eligible: List[Dict[str, Any]] = []
    for template in selected_templates:
        for left_index, activation in enumerate(activities):
            for right_index, target in enumerate(activities):
                if activation == target:
                    continue
                if template == "not_coexistence" and right_index <= left_index:
                    continue
                candidate_count += 1
                stats = _rule_statistics(traces, template, activation, target)
                if stats["activation_cases"] < required_activation:
                    continue
                if not math.isfinite(float(stats["confidence"])) or float(stats["confidence"]) < float(min_confidence):
                    continue
                formula = rule_formula(template, activation, target)
                if _canonical_formula(formula) in search_keys:
                    overlap_count += 1
                    continue
                rule = {
                    "rule_id": _stable_rule_id(template, activation, target, source),
                    "template": template,
                    "activation": activation,
                    "target": target,
                    "formula": formula,
                    "formula_hash": hashlib.sha256(
                        _canonical_formula(formula).encode("utf-8")
                    ).hexdigest(),
                    "source": source,
                    "outcome_label_used": False,
                    "complete_case_discovery": True,
                    "search_formula_overlap": False,
                    **stats,
                }
                eligible.append(rule)

    selected = _rank_rules_balanced(eligible, selected_templates, int(max_constraints))
    diagnostics = {
        "audit_case_count": len(traces),
        "audit_case_ids": [item["case_id"] for item in prepared],
        "activity_count": len(activity_support),
        "candidate_activity_count": len(activities),
        "candidate_count": candidate_count,
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "formula_overlap_count": overlap_count,
        "required_activation_cases": required_activation,
        "templates": selected_templates,
        "outcome_label_used": False,
        "complete_case_discovery": True,
        "activity_normalization_collision_count": 0,
        "audit_case_trace_sha256": complete_case_trace_fingerprint(
            cases, act_encoder
        ),
    }
    return selected, diagnostics


def _external_activity(value: Any) -> str:
    value = "" if value is None else str(value).strip()
    return normalize_activity_name(value) if value else ""


def load_external_audit_rules(path: str, dataset_name: str = "") -> List[Dict[str, Any]]:
    """Load pre-registered domain rules from JSON or CSV."""
    if not path:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audit rule file does not exist: {path}")
    source_hash = hashlib.sha256()
    with open(path, "rb") as source_handle:
        for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
            source_hash.update(chunk)
    source_file_sha256 = source_hash.hexdigest()
    extension = os.path.splitext(path)[1].lower()
    if extension == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("rules", []) if isinstance(payload, dict) else payload
    elif extension == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("External audit rules must be a .json or .csv file.")

    loaded: List[Dict[str, Any]] = []
    for index, raw in enumerate(rows or []):
        row = dict(raw)
        row_dataset = str(row.get("dataset", "*")).strip()
        if row_dataset not in ("", "*", dataset_name):
            continue
        template = str(row.get("template", "")).strip().lower()
        activation = _external_activity(row.get("activation"))
        target = _external_activity(row.get("target"))
        formula = str(row.get("formula", "")).strip()
        if not formula:
            if template not in SUPPORTED_TEMPLATES or not activation or not target:
                raise ValueError(
                    f"External audit rule row {index + 1} needs formula or a supported template with activation and target."
                )
            formula = rule_formula(template, activation, target)
        source = str(row.get("source", "external_preregistered")).strip() or "external_preregistered"
        rule_id = str(row.get("rule_id", "")).strip() or _stable_rule_id(
            template or "ltlf", activation, target or formula, source
        )
        loaded.append({
            "rule_id": rule_id,
            "template": template or "external_ltlf",
            "activation": activation,
            "target": target,
            "formula": formula,
            "formula_hash": hashlib.sha256(
                _canonical_formula(formula).encode("utf-8")
            ).hexdigest(),
            "source": source,
            "source_file": os.path.abspath(path),
            "source_file_sha256": source_file_sha256,
            "outcome_label_used": False,
            "complete_case_discovery": False,
            "search_formula_overlap": False,
        })
    return loaded


def merge_audit_rules(
    discovered: Sequence[Dict[str, Any]], external: Sequence[Dict[str, Any]],
    search_formulas: Optional[Iterable[str]] = None, max_constraints: int = 20,
) -> Tuple[List[Dict[str, Any]], int]:
    """Prefer external rules, deduplicate formulas, and enforce isolation."""
    search_keys = {_canonical_formula(value) for value in (search_formulas or []) if value}
    seen = set()
    merged: List[Dict[str, Any]] = []
    overlap_count = 0
    for rule in list(external or []) + list(discovered or []):
        key = _canonical_formula(rule.get("formula", ""))
        if not key or key in seen:
            continue
        if key in search_keys:
            overlap_count += 1
            continue
        seen.add(key)
        merged.append(dict(rule))
        if len(merged) >= int(max_constraints):
            break
    return merged, overlap_count


def attach_external_rule_statistics(
    rules: Sequence[Dict[str, Any]], cases: Sequence[Mapping[str, Any]], act_encoder=None
) -> List[Dict[str, Any]]:
    """Evaluate external template rules on the audit fold without selecting them."""
    traces = [item["trace"] for item in prepare_complete_traces(cases, act_encoder)]
    enriched = []
    for rule in rules:
        current = dict(rule)
        if current.get("template") in SUPPORTED_TEMPLATES:
            current.update(_rule_statistics(
                traces,
                current["template"],
                current["activation"],
                current["target"],
            ))
        else:
            current.update({
                "audit_case_count": len(traces),
                "activation_cases": len(traces),
                "activation_coverage": 1.0 if traces else float("nan"),
                "satisfied_activation_cases": None,
                "violation_cases": None,
                "confidence": None,
                "support": None,
            })
        enriched.append(current)
    return enriched


def compile_audit_rules(
    rules: Sequence[Dict[str, Any]], dfa_max_states: int = 500
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compile every audit formula separately; no OR-cluster shortcut is used."""
    if not rules:
        return [], {"requested": 0, "compiled": 0, "failed": 0}
    from explaination.visualizations.utils import alphabet, parse, symbolic_to_singleton, to_dfa

    compiled: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for source_rule in rules:
        rule = dict(source_rule)
        try:
            parsed = parse(rule["formula"])
            symbolic = to_dfa(parsed)
            state_count = len(symbolic.states)
            if state_count <= 0 or state_count > int(dfa_max_states):
                raise ValueError(
                    f"DFA state count {state_count} outside 1..{int(dfa_max_states)}"
                )
            letters = alphabet(parsed)
            letters.add("*")
            rule["dfa"] = symbolic_to_singleton(symbolic, letters)
            if rule.get("template") in SUPPORTED_TEMPLATES:
                from explaination.visualizations.utils import trace_accepts_dfa
                activation = rule["activation"]
                target = rule["target"]
                other = "audit_semantic_check_other"
                while other in (activation, target):
                    other += "_x"
                witnesses = [
                    list(word)
                    for length in range(1, 5)
                    for word in product((activation, target, other), repeat=length)
                ]
                for witness in witnesses:
                    direct = bool(rule_satisfied(witness, rule))
                    automated = bool(trace_accepts_dfa(witness, rule["dfa"]))
                    if direct != automated:
                        raise ValueError(
                            "DFA semantics disagree with direct template semantics "
                            f"for witness {witness}: direct={direct}, dfa={automated}"
                        )
                rule["semantic_check_status"] = "passed"
            rule["compile_status"] = "compiled"
            rule["dfa_state_count"] = state_count
            rule["compile_error"] = ""
            compiled.append(rule)
        except Exception as exc:
            failures.append({
                "rule_id": str(rule.get("rule_id", "")),
                "error": f"{type(exc).__name__}: {exc}",
            })
    return compiled, {
        "requested": len(rules),
        "compiled": len(compiled),
        "failed": len(failures),
        "failures": failures,
    }


def evaluate_audit_trace(
    trace: Sequence[str], rules: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Return activation-aware macro rule satisfaction for one trace."""
    normalized = [normalize_activity_name(value) for value in trace]
    applicable = [rule for rule in (rules or []) if rule_applicable(normalized, rule)]
    if not applicable:
        return {
            "audit_score": float("nan"),
            "audit_all_applicable": float("nan"),
            "audit_rule_count": len(rules or []),
            "audit_applicable_rule_count": 0,
            "audit_satisfied_rule_count": 0,
            "audit_violated_rule_ids": [],
        }
    satisfied_flags = [bool(rule_satisfied(normalized, rule)) for rule in applicable]
    violated = [
        str(rule.get("rule_id", ""))
        for rule, satisfied in zip(applicable, satisfied_flags)
        if not satisfied
    ]
    satisfied_count = int(sum(satisfied_flags))
    return {
        "audit_score": satisfied_count / len(applicable),
        "audit_all_applicable": int(satisfied_count == len(applicable)),
        "audit_rule_count": len(rules or []),
        "audit_applicable_rule_count": len(applicable),
        "audit_satisfied_rule_count": satisfied_count,
        "audit_violated_rule_ids": violated,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if key != "dfa"}
    return str(value)


def persist_audit_manifest(
    rules: Sequence[Mapping[str, Any]], diagnostics: Mapping[str, Any],
    output_dir: str, dataset_name: str,
) -> Dict[str, str]:
    """Persist an auditable JSON manifest plus a flat rule CSV."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", dataset_name or "dataset")
    json_path = os.path.join(output_dir, f"{safe_name}_audit_rules.json")
    csv_path = os.path.join(output_dir, f"{safe_name}_audit_rules.csv")
    serial_rules = [_json_safe(dict(rule)) for rule in rules]
    rule_set_sha256 = stable_rule_set_hash(serial_rules)
    payload = {
        "schema_version": 2,
        "dataset": dataset_name,
        "rule_set_sha256": rule_set_sha256,
        "diagnostics": _json_safe(dict(diagnostics)),
        "rules": serial_rules,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    columns = sorted({key for rule in serial_rules for key in rule})
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for rule in serial_rules:
            writer.writerow({key: rule.get(key) for key in columns})
    return {"json": json_path, "csv": csv_path}
