#!/usr/bin/env python
"""
Batch experiment runner: multiple datasets x multiple configurations x full evaluation.

Usage:
    python experiment_runner.py                          # Default: all datasets, all variants
    python experiment_runner.py --datasets sepsis_cases_1 BPIC17_O_ACCEPTED
    python experiment_runner.py --no-pattern             
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from config import Config
from constant import set_seed
from dataset import ProcessDataset, collate_fn, build_reference_matrix
from models import MultiTaskProcessModel
from counterfactual import (
    CounterfactualRecommender,
    annotate_dfa_next_activity_supervision,
    evaluate_minimal_constraint_relaxation,
    flatten_cluster_rule_dfas,
    verify_retained_constraints,
)
from evaluation.cf_query_metrics import summarize_queries
from trainer import Trainer
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from data_loader import (
    load_public_dataset, encode_activities,
    encode_activities_train_only, fit_transform_attributes_train_only,
)

# Available dataset list
# ---------------------------------------------------------------------------
ALL_DATASETS = [
    "sepsis_cases_1",
    "sepsis_cases_2",
    "sepsis_cases_4",
    "BPIC15_2_f2",
    "BPIC15_3_f2",
    "BPIC15_4_f2",
    "BPIC15_5_f2",
    "BPIC17_O_ACCEPTED",
    "BPIC17_O_CANCELLED",
    "BPIC17_O_REFUSED",
]

# The thesis evaluation protocol covers these seven logs.  BPIC17 remains
# available through ``--datasets`` but is not silently mixed into paper tables.
PAPER_DATASETS = ALL_DATASETS[:7]

# Experiment configuration variants (for ablation)
CONFIG_VARIANTS = {
    "full": {
        "desc": "Full model",
        "use_pattern": True,
        "use_uwl": True,
        "use_dfa": True,
        "use_dfa_aux_loss": False,
    },
    "no_pattern": {
        "desc": "No pattern features",
        "use_pattern": False,
        "use_uwl": True,
        "use_dfa": True,
    },
    "no_uwl": {
        "desc": "Fixed weights (no UWL)",
        "use_pattern": True,
        "use_uwl": False,
        "use_dfa": True,
    },
    "no_dfa": {
        "desc": "No search constraints",
        "use_pattern": True,
        "use_uwl": True,
        "use_dfa": False,
        "use_soft_constraints": False,
    },
    "no_dfa_aux": {
        "desc": "Deprecated alias of the aux-free full model",
        "use_pattern": True,
        "use_uwl": True,
        "use_dfa": True,
        "use_dfa_aux_loss": False,
    },
    "with_dfa_aux": {
        "desc": "Opt-in Search-DFA next-activity auxiliary-loss ablation",
        "use_pattern": True,
        "use_uwl": True,
        "use_dfa": True,
        "use_dfa_aux_loss": True,
        "dfa_next_act_loss_weight_override": 0.10,
    },
    "soft_constraint_cf": {
        "desc": "Soft Search-Patterns only",
        "use_pattern": True,
        "use_uwl": True,
        "use_dfa": False,
        "use_soft_constraints": True,
    },
    "fixed_constraint_cf": {
        "desc": "Fixed-support Search-DFA ablation",
        "use_pattern": True,
        "use_uwl": True,
        "use_dfa": True,
        "use_soft_constraints": True,
        "constraint_policy": "fixed",
    },
    "model_score_cf": {
        "desc": "Unconstrained model-score CF baseline",
        "use_pattern": True,
        "use_uwl": True,
        "use_dfa": False,
        "use_soft_constraints": False,
        "is_model_score_cf": True,
        "cf_generator": "model_score",
        "cf_ranking_override": "score",
    },    "single_point_cf": {
        "desc": "Single-point CF ablation",
        "use_pattern": True,
        "use_uwl": True,
        "use_dfa": True,
        "cf_generator": "single_point",
        "cf_intervention_points_override": 1,
        "cf_ranking_override": "pareto",
    },
    "multi_point_cf": {
        "desc": "Multi-point CF ablation",
        "use_pattern": True,
        "use_uwl": True,
        "use_dfa": True,
        "cf_generator": "multi_point",
        "cf_intervention_points_override": 3,
        "cf_ranking_override": "pareto",
    },
    "score_ranking_cf": {
        "desc": "Score-ranking CF ablation",
        "use_pattern": True,
        "use_uwl": True,
        "use_dfa": True,
        "cf_generator": "score_ranking",
        "cf_ranking_override": "score",
    },
    "pareto_ranking_cf": {
        "desc": "Pareto-ranking CF ablation",
        "use_pattern": True,
        "use_uwl": True,
        "use_dfa": True,
        "cf_generator": "pareto_ranking",
        "cf_ranking_override": "pareto",
    },
    "random_baseline": {
        "desc": "Random baseline",
        "use_pattern": False,
        "use_uwl": False,
        "use_dfa": False,
        "use_soft_constraints": False,
        "is_random": True,
        "cf_generator": "random",
    },
    "xgboost": {
        "desc": "XGBoost baseline",
        "use_pattern": False,
        "use_uwl": False,
        "use_dfa": False,
        "is_baseline": True,
        "baseline_type": "xgboost",
    },
    "lstm": {
        "desc": "LSTM baseline",
        "use_pattern": False,
        "use_uwl": False,
        "use_dfa": False,
        "is_baseline": True,
        "baseline_type": "lstm",
    },
    "lstm_mtl": {
        "desc": "Task-aligned shared LSTM multi-task baseline (v1)",
        "use_pattern": False,
        "use_uwl": False,
        "use_dfa": False,
        "is_baseline": True,
        "baseline_type": "lstm_mtl",
    },
    "process_transformer": {
        "desc": "ProcessTransformer baseline",
        "use_pattern": False,
        "use_uwl": False,
        "use_dfa": False,
        "is_baseline": True,
        "baseline_type": "process_transformer",
    },
    "declare_dfa": {
        "desc": "Declare template DFA",
        "use_pattern": False,
        "use_uwl": True,
        "use_dfa": True,
        "is_declare": True,
    },
}


def build_outcome_sampler(train_samples):
    labels = [
        int(s.get("outcome", -1))
        for s in train_samples
    ]
    valid_labels = [y for y in labels if y in (0, 1)]
    if len(valid_labels) != len(labels):
        return None
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2).astype(float)
    if np.any(counts <= 0):
        return None

    case_prefix_counts = {}
    for s in train_samples:
        cid = s.get("source_case_id", s.get("case_id", None))
        case_prefix_counts[cid] = case_prefix_counts.get(cid, 0) + 1

    weights = []
    for s, y in zip(train_samples, labels):
        cid = s.get("source_case_id", s.get("case_id", None))
        prefix_factor = 1.0 / max(case_prefix_counts.get(cid, 1), 1)
        weights.append((1.0 / counts[y]) * prefix_factor)

    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def build_dataloaders(train_samples, val_samples, test_samples, cfg, batch_size, pattern_feat_dim,
                      num_workers=0, pin_memory=False, balanced_outcome_sampler=False,
                      persistent_workers=False):
    """Build train/val/test dataloaders"""
    ds_kwargs = dict(
        max_seq_len=cfg.max_seq_len, pattern_feat_dim=pattern_feat_dim,
        n_static_cat=cfg.n_static_cat, n_static_num=cfg.n_static_num,
        n_dynamic_cat=cfg.n_dynamic_cat, n_dynamic_num=cfg.n_dynamic_num,
        dfa_action_mask_dim=max(0, cfg.num_activities - 1),
    )
    train_ds = ProcessDataset(train_samples, **ds_kwargs)
    val_ds = ProcessDataset(val_samples, **ds_kwargs)
    test_ds = ProcessDataset(test_samples, **ds_kwargs)

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
    )
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = 1
    train_sampler = build_outcome_sampler(train_samples) if balanced_outcome_sampler else None
    if train_sampler is not None:
        print("  Train sampler: balanced outcome + per-case prefix weighting")
    train_loader = DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    train_eval_loader = DataLoader(train_ds, shuffle=False, **loader_kwargs)

    return train_loader, train_eval_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def shuffle_samples(samples, seed=42):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    return [samples[i] for i in order]


def summarize_outcome_split(name, samples):
    if not samples:
        print(f"  Split {name}: 0 cases")
        return
    labels = np.array([s["outcome"] for s in samples], dtype=int)
    valid = labels[labels >= 0]
    pos_rate = float(valid.mean()) if len(valid) > 0 else float("nan")
    print(f"  Split {name}: {len(samples)} cases, positive_rate={pos_rate:.2%}")


def _slice_dynamic_rows(rows, length):
    return [list(r) for r in list(rows or [])[:length]]


def expand_prefix_samples(case_samples, mode="all", max_prefixes_per_case=None):
    """Expand complete case traces into prefix-level PPM samples.

    Each expanded sample keeps activities[:L], where activities[L-1] is the
    held-out next-event target used by the current causal model heads.  Cases
    must already be split before this function is called.
    """
    if mode == "last":
        expanded = []
        for s in case_samples:
            full_len = int(s.get("length", len(s.get("activities", []))))
            activities = list(s.get("activities", []))
            if full_len < 2 or len(activities) < full_len:
                continue
            item = dict(
                s,
                prefix_len=full_len - 1,
                source_case_id=s.get("case_id"),
                next_activity=activities[full_len - 1],
            )
            item["pattern_sample_id"] = (
                f"{item['source_case_id']}::p{item['prefix_len']}"
            )
            expanded.append(item)
        return expanded
    if mode != "all":
        raise ValueError(f"Unsupported prefix mode: {mode}")

    expanded = []
    for s in case_samples:
        full_len = int(s.get("length", len(s.get("activities", []))))
        if full_len < 2:
            continue
        prefix_lengths = list(range(2, full_len + 1))
        if max_prefixes_per_case is not None and max_prefixes_per_case > 0:
            if len(prefix_lengths) > max_prefixes_per_case:
                # Deterministic spread over the trace, always keeping the final prefix.
                idx = np.linspace(0, len(prefix_lengths) - 1, int(max_prefixes_per_case))
                prefix_lengths = [prefix_lengths[int(round(i))] for i in idx]
                prefix_lengths = sorted(set(prefix_lengths))
        for L in prefix_lengths:
            item = dict(s)
            item["activities"] = list(s.get("activities", []))[:L]
            item["times"] = list(s.get("times", []))[:L]
            item["length"] = L
            item["prefix_len"] = L - 1
            # Explicit target prevents supervised pattern discovery from
            # silently falling back to the unrelated outcome label.
            item["next_activity"] = item["activities"][L - 1]
            item["source_case_id"] = s.get("case_id")
            item["pattern_sample_id"] = (
                f"{item['source_case_id']}::p{item['prefix_len']}"
            )
            item["dynamic_cat"] = _slice_dynamic_rows(s.get("dynamic_cat", []), L)
            item["dynamic_num"] = _slice_dynamic_rows(s.get("dynamic_num", []), L)
            item["pattern_features"] = []
            expanded.append(item)
    return expanded


def warn_potential_label_proxies(dataset_name, attr_dims):
    if not dataset_name.startswith("BPIC17"):
        return
    suspicious = {"accepted", "selected", "cancelled", "declined", "approved", "rejected", "label", "outcome"}
    used_cols = []
    for key in ("static_cat_cols", "dynamic_cat_cols", "static_num_cols", "dynamic_num_cols"):
        used_cols.extend(attr_dims.get(key, []))
    removed = attr_dims.get("removed_bp17_cols", [])
    if removed:
        print("  [BPIC17 anti-leak] active; removed columns: " + ", ".join(removed))
    flagged = sorted({col for col in used_cols if str(col).strip().lower() in suspicious})
    if flagged:
        print("  [Leakage warning] BPIC17 uses potentially label-proxy attributes: "
              + ", ".join(flagged))
        print("  [Leakage warning] If validation stays near-perfect, rerun without these columns.")


def renormalize_times_from_train(samples, train_end):
    """Scale normalized times by the train split only to avoid test-set scale leakage."""
    train_times = [
        max(s.get("times", [0.0]))
        for s in samples[:train_end]
        if len(s.get("times", [])) > 0
    ]
    train_max = max(train_times) if train_times else 0.0
    if train_max <= 0:
        return
    for s in samples:
        s["times"] = [float(np.clip(t / train_max, 0.0, 1.0)) for t in s["times"]]
    print(f"  Time normalization: train-only scale applied (train_max={train_max:.4f})")


def select_monitor_score(val_logs, cfg=None):
    """Prefer outcome quality, with a small normalized-time term for multitask early stopping."""
    outcome_score = float("nan")
    outcome_name = "fallback_zero"
    for key in ("outcome_auc", "outcome_f1", "outcome_acc"):
        value = float(val_logs.get(key, float("nan")))
        if np.isfinite(value):
            outcome_score = value
            outcome_name = key
            break

    if np.isfinite(outcome_score):
        time_weight = float(getattr(cfg, "early_stop_time_weight", 0.0)) if cfg is not None else 0.0
        time_mae = float(val_logs.get("next_time_mae", float("nan")))
        if time_weight > 0.0 and np.isfinite(time_mae):
            time_quality = 1.0 - float(np.clip(time_mae, 0.0, 1.0))
            monitor = (1.0 - time_weight) * outcome_score + time_weight * time_quality
            return monitor, f"{outcome_name}+time_mae"
        return outcome_score, outcome_name
    return 0.0, "fallback_zero"


def compute_outcome_class_weights(train_samples, clip=5.0):
    labels = [
        int(s.get("outcome"))
        for s in train_samples
        if int(s.get("outcome", -1)) in (0, 1)
    ]
    if not labels:
        return None
    counts = np.bincount(labels, minlength=2).astype(float)
    if np.any(counts <= 0):
        return None
    total = counts.sum()
    weights = total / (2.0 * counts)
    weights = np.clip(weights, 1.0 / clip, clip)
    return [float(weights[0]), float(weights[1])]


def compute_next_act_class_weights(train_samples, num_activities, clip=5.0):
    n_classes = max(int(num_activities) - 1, 1)
    targets = []
    for s in train_samples:
        acts = s.get("activities", [])
        length = int(s.get("length", len(acts)))
        if length >= 2 and len(acts) >= length:
            aid = int(acts[length - 1])
            if 1 <= aid < num_activities:
                targets.append(aid - 1)
    if not targets:
        return None, None

    counts = np.bincount(np.asarray(targets, dtype=int), minlength=n_classes).astype(float)
    present = counts > 0
    if present.sum() <= 1:
        return None, counts

    weights = np.ones(n_classes, dtype=float)
    total = counts[present].sum()
    weights[present] = total / (present.sum() * counts[present])
    weights[present] = np.clip(weights[present], 1.0 / clip, clip)
    weights[present] = weights[present] / max(weights[present].mean(), 1e-8)
    return [float(w) for w in weights], counts


def compute_next_act_label_diagnostics(train_samples, num_activities, rare_threshold=5):
    """Train-split next-activity label diagnostics without validation/test leakage."""
    n_classes = max(int(num_activities) - 1, 0)
    targets = []
    for s in train_samples:
        acts = s.get("activities", [])
        length = int(s.get("length", len(acts)))
        if length >= 2 and len(acts) >= length:
            aid = int(acts[length - 1])
            if 1 <= aid < int(num_activities):
                targets.append(aid - 1)

    counts = np.bincount(np.asarray(targets, dtype=int), minlength=max(n_classes, 1)).astype(float)
    total = float(counts.sum())
    present = counts > 0
    if total <= 0:
        entropy = float("nan")
        entropy_norm = float("nan")
        rare_ratio = float("nan")
        majority_baseline = float("nan")
    else:
        probs = counts[present] / total
        entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
        entropy_norm = (
            float(entropy / np.log(max(int(present.sum()), 2)))
            if int(present.sum()) > 1 else 0.0
        )
        rare_mask = (counts > 0) & (counts < int(rare_threshold))
        rare_ratio = float(counts[rare_mask].sum() / total)
        majority_baseline = float(counts.max() / total)

    return {
        "num_activity_classes": int(n_classes),
        "activity_present_classes": int(present.sum()),
        "activity_label_entropy": entropy,
        "activity_label_entropy_norm": entropy_norm,
        "rare_activity_ratio": rare_ratio,
        "rare_activity_threshold": int(rare_threshold),
        "rare_activity_classes": int(((counts > 0) & (counts < int(rare_threshold))).sum()),
        "activity_majority_baseline_acc": majority_baseline,
    }


def build_next_act_transition_log_prior(train_samples, num_activities, smoothing=1.0):
    num_activities = int(num_activities)
    smoothing = max(float(smoothing), 1e-8)
    counts = np.zeros((num_activities, num_activities), dtype=float)
    counts[:, 1:] = smoothing
    transition_count = 0

    for s in train_samples:
        acts = s.get("activities", [])
        length = int(s.get("length", len(acts)))
        if length < 2 or len(acts) < length:
            continue
        prev_a = int(acts[length - 2])
        next_a = int(acts[length - 1])
        if 0 < prev_a < num_activities and 0 < next_a < num_activities:
            counts[prev_a, next_a] += 1.0
            transition_count += 1

    row_sums = counts[:, 1:].sum(axis=1, keepdims=True)
    probs = counts[:, 1:] / np.maximum(row_sums, 1e-12)
    log_prior = np.zeros((num_activities, num_activities), dtype=np.float32)
    log_prior[:, 0] = -20.0
    log_prior[:, 1:] = np.log(np.maximum(probs, 1e-12)).astype(np.float32)
    return log_prior.tolist(), transition_count


def _safe_filename_part(value):
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text)


def _write_csv_atomic(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def save_incremental_result(result, args):
    """Persist one completed dataset/variant row immediately.

    The final all_results.csv is still written at the end of the run, but this
    function keeps a crash/interruption-safe copy after every completed variant.
    """
    incremental_dir = getattr(args, "incremental_results_dir", "./all_results")
    os.makedirs(incremental_dir, exist_ok=True)

    row = dict(result)
    row["run_output_dir"] = getattr(args, "output_dir", "")
    row["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    dataset = row.get("dataset", "unknown_dataset")
    dataset_path = os.path.join(
        incremental_dir,
        f"{_safe_filename_part(dataset)}_all_results.csv",
    )
    combined_path = os.path.join(incremental_dir, "all_results.csv")

    for path in (dataset_path, combined_path):
        new_df = pd.DataFrame([row])
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                old_df = pd.read_csv(path)
                merged = pd.concat([old_df, new_df], ignore_index=True, sort=False)
            except Exception:
                merged = new_df
        else:
            merged = new_df

        dedupe_cols = ["dataset", "variant"]
        if "seed" in merged.columns:
            dedupe_cols.append("seed")
        if set(dedupe_cols).issubset(merged.columns):
            merged = merged.drop_duplicates(
                subset=dedupe_cols, keep="last"
            )
        _write_csv_atomic(merged, path)

    print(f"  Incremental all_results saved: {dataset_path}")


# Cross-variant pattern discovery cache (dataset_name -> result)
_PATTERN_CACHE: dict = {}


def _sample_case_key(sample, fallback_index=0):
    """Return a stable case key so prefixes of one case stay in one fold."""
    for key in ("source_case_id", "case_id", "Case ID", "case:concept:name"):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return f"sample-{fallback_index}"


def _symbolic_sample_signature(samples):
    """Fingerprint a symbolic fold so caches cannot cross split protocols."""
    digest = hashlib.sha256()
    rows = []
    for index, sample in enumerate(samples or []):
        rows.append((
            _sample_case_key(sample, index),
            str(sample.get("prefix_len", "complete")),
            str(sample.get("pattern_sample_id", "")),
        ))
    for case_key, prefix_len, pattern_sample_id in sorted(rows):
        digest.update(
            f"{case_key}\x1f{prefix_len}\x1f{pattern_sample_id}\x1e".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()[:16]


def split_pattern_discovery_calibration_audit_samples(
    train_samples,
    audit_fraction=0.25,    calibration_fraction=0.15,
    seed=2026,
):
    """Create disjoint complete-case discovery/calibration/audit folds."""
    audit_fraction = float(audit_fraction)
    calibration_fraction = float(calibration_fraction)
    if audit_fraction < 0.0 or calibration_fraction < 0.0:
        raise ValueError("Symbolic fold fractions must be non-negative")
    if audit_fraction + calibration_fraction >= 1.0:
        raise ValueError(
            "audit_fraction + calibration_fraction must be strictly below 1"
        )
    if not train_samples:
        return [], [], [], set(), set(), set()

    case_keys = sorted({
        _sample_case_key(sample, idx)
        for idx, sample in enumerate(train_samples)
    })
    if len(case_keys) < 2:
        return (
            list(train_samples), [], [], set(case_keys), set(), set()
        )

    rng = np.random.default_rng(int(seed))
    shuffled = list(
        np.asarray(case_keys, dtype=object)[rng.permutation(len(case_keys))]
    )
    n_cases = len(shuffled)
    n_audit = int(round(n_cases * audit_fraction)) if audit_fraction > 0 else 0
    n_audit = min(max(n_audit, 1 if audit_fraction > 0 else 0), n_cases - 1)
    remaining = n_cases - n_audit
    n_calibration = (
        int(round(n_cases * calibration_fraction))
        if calibration_fraction > 0 and remaining >= 2 else 0
    )
    if calibration_fraction > 0 and remaining >= 2:
        n_calibration = min(max(n_calibration, 1), remaining - 1)

    audit_cases = set(str(x) for x in shuffled[:n_audit])
    calibration_cases = set(
        str(x) for x in shuffled[n_audit:n_audit + n_calibration]
    )
    discovery_cases = set(
        str(x) for x in shuffled[n_audit + n_calibration:]
    )

    def take(case_set):
        return [
            sample for idx, sample in enumerate(train_samples)
            if _sample_case_key(sample, idx) in case_set
        ]

    return (
        take(discovery_cases),
        take(calibration_cases),
        take(audit_cases),
        discovery_cases,
        calibration_cases,
        audit_cases,
    )


def split_pattern_discovery_audit_samples(train_samples, audit_fraction=0.25,
                                          seed=2026):
    """Split complete training cases into disjoint discovery and audit folds.

    Call this before prefix expansion.  The predictive model still trains on
    all training cases; only symbolic knowledge construction is isolated.
    """
    if not train_samples:
        return [], [], set(), set()
    case_keys = sorted({
        _sample_case_key(sample, idx)
        for idx, sample in enumerate(train_samples)
    })
    if len(case_keys) < 2 or float(audit_fraction) <= 0.0:
        return list(train_samples), [], set(case_keys), set()

    rng = np.random.default_rng(int(seed))
    shuffled = list(np.asarray(case_keys, dtype=object)[rng.permutation(len(case_keys))])
    n_audit = int(round(len(shuffled) * float(audit_fraction)))
    n_audit = min(max(n_audit, 1), len(shuffled) - 1)
    audit_cases = set(str(x) for x in shuffled[:n_audit])
    discovery_cases = set(str(x) for x in shuffled[n_audit:])
    discovery = [
        sample for idx, sample in enumerate(train_samples)
        if _sample_case_key(sample, idx) in discovery_cases
    ]
    audit = [
        sample for idx, sample in enumerate(train_samples)
        if _sample_case_key(sample, idx) in audit_cases
    ]
    return discovery, audit, discovery_cases, audit_cases


def _compile_constraint_set(constraint_dict, act_encoder, label,
                            dfa_max_states=500):
    """Compile one disjoint constraint set into conflict-aware DFA clusters."""
    if not constraint_dict or act_encoder is None:
        return None, None

    from counterfactual import ConflictAwareDFAManager
    from pattern_to_ltlf import extract_ltlf_from_patterns, pattern_to_ltlf
    from explaination.visualizations.utils import (
        parse, to_dfa, symbolic_to_singleton, alphabet,
    )

    dfa = None
    clusters = None
    try:
        print(
            f"  Building {label} DFA clusters for {len(constraint_dict)} constraints...",
            flush=True,
        )
        manager = ConflictAwareDFAManager(
            constraint_dict,
            act_encoder,
            parse_fn=parse,
            to_dfa_fn=to_dfa,
            sym_to_singleton_fn=symbolic_to_singleton,
            alphabet_fn=alphabet,
            dfa_max_states=int(dfa_max_states),
        )
        if manager.num_clusters() > 0:
            clusters = manager.clusters
            if manager.num_clusters() == 1:
                dfa = clusters[0]["dfa"]
            print(
                f"  {label} DFA: {manager.num_clusters()} clusters, "
                f"covering {manager.num_patterns_covered()}/{len(constraint_dict)} patterns"
            )
            return dfa, clusters
    except Exception as exc:
        print(f"  {label} conflict clustering failed: {exc}; trying conjunction DFA")

    formula = extract_ltlf_from_patterns(constraint_dict, act_encoder)
    if formula is None:
        return None, None
    try:
        parsed = parse(formula)
        symbolic = to_dfa(parsed)
        if 0 < len(symbolic.states) <= int(dfa_max_states):
            letters = alphabet(parsed)
            letters.add("*")
            dfa = symbolic_to_singleton(symbolic, letters)
    except Exception as exc:
        print(f"  {label} conjunction DFA failed: {exc}")
    return dfa, clusters


def _dfa_status_parts(dfa, multi_dfa_clusters):
    multi_count = len(multi_dfa_clusters) if multi_dfa_clusters else 0
    single_ok = dfa is not None
    if multi_count > 1:
        effective = "multi"
    elif single_ok:
        effective = "single"
    else:
        effective = "none"
    return {
        "single": "OK" if single_ok else "N/A",
        "multi": f"{multi_count} clusters" if multi_count > 0 else "N/A",
        "effective": effective,
    }


def _format_dfa_status(dfa, multi_dfa_clusters):
    status = _dfa_status_parts(dfa, multi_dfa_clusters)
    return (f"single={status['single']}, "
            f"multi={status['multi']}, "
            f"effective={status['effective']}")


def _canonical_formula_key(value):
    """Use the same formula canonicalisation criterion as audit isolation."""
    return "".join(str(value or "").split()).lower()


def _isolate_search_constraints_from_audit(
    hard_constraints,
    soft_constraints,
    calibration_frame,
    audit_rules,
    max_soft_constraints=20,
):
    """Demote exact Audit-DFA overlaps on the search side.

    Audit rules remain fixed because they are selected only from the held-out
    complete-case fold.  Any coincidentally identical Search-Pattern is kept
    out of the hard generator and may only contribute bounded ranking evidence.
    """
    from pattern_to_ltlf import pattern_to_ltlf

    hard_constraints = dict(hard_constraints or {})
    soft_constraints = dict(soft_constraints or {})
    audit_keys = {
        _canonical_formula_key(rule.get("formula"))
        for rule in (audit_rules or [])
        if rule.get("formula")
    }
    overlap_ids = []
    hard_isolated = {}
    for pid, graph in hard_constraints.items():
        formula_key = _canonical_formula_key(pattern_to_ltlf(graph))
        copied = graph.copy()
        if formula_key and formula_key in audit_keys:
            overlap_ids.append(str(pid))
            copied.graph["audit_formula_overlap"] = True
            copied.graph["constraint_role"] = "soft"
            soft_constraints[str(pid)] = copied
        else:
            copied.graph["audit_formula_overlap"] = False
            copied.graph["constraint_role"] = "hard"
            hard_isolated[str(pid)] = copied

    frame = (
        calibration_frame.copy()
        if calibration_frame is not None else pd.DataFrame()
    )
    score_by_pid = {}
    if not frame.empty and "pattern_id" in frame.columns:
        score_column = (
            "constraint_selection_score"
            if "constraint_selection_score" in frame.columns
            else "calibration_score"
        )
        score_by_pid = {
            str(row["pattern_id"]): float(row.get(score_column, 0.0))
            for _, row in frame.iterrows()
        }

    soft_order = sorted(
        soft_constraints,
        key=lambda pid: (-score_by_pid.get(str(pid), 0.0), str(pid)),
    )
    soft_limit = max(0, int(max_soft_constraints))
    selected_soft_ids = set(soft_order[:soft_limit])
    soft_isolated = {}
    for pid in soft_order:
        if pid not in selected_soft_ids:
            continue
        graph = soft_constraints[pid].copy()
        graph.graph["constraint_role"] = "soft"
        graph.graph["audit_formula_overlap"] = str(pid) in set(overlap_ids)
        soft_isolated[str(pid)] = graph

    if not frame.empty and "pattern_id" in frame.columns:
        frame["pattern_id"] = frame["pattern_id"].astype(str)
        frame["audit_formula_overlap"] = frame["pattern_id"].isin(overlap_ids)
        frame["constraint_role"] = frame["pattern_id"].map(
            lambda pid: (
                "hard" if pid in hard_isolated
                else "soft" if pid in soft_isolated
                else "unused"
            )
        )
        frame["hard_selected_after_audit_isolation"] = frame[
            "pattern_id"
        ].isin(hard_isolated)

    return hard_isolated, soft_isolated, frame, overlap_ids


def run_pattern_pipeline(
    discovery_samples,
    act_encoder,
    dataset_name="",
    dfa_max_constraints=10,
    all_samples=None,
    calibration_samples=None,
    audit_seed=2026,
    constraint_policy="calibrated",
    prediction_top_k=32,
    search_pool_size=64,
    search_min_calibration_support=0.05,
    search_max_calibration_support=1.00,
    search_min_stability=0.85,
    search_soft_constraints=20,
    search_min_hard_constraints=3,
    search_fallback_min_case_support=0.50,
    search_fallback_min_prefix_support=0.10,
    search_fallback_min_stability=0.70,
    search_support_confidence_z=1.96,
    dfa_max_states=500,
):
    """Build role-separated Prediction-Patterns and Search-Patterns.

    Prediction-Patterns are selected against the next-activity label and only
    feed the next-activity head.  Search-Patterns are selected without labels,
    calibrated on disjoint cases, and split into hard DFA constraints versus
    soft ranking evidence.  Audit rules are intentionally outside this path.
    """
    from pattern_discovery.pattern_mining import (
        calibrate_search_patterns,
        encode_pattern_features,
        persist_constraint_manifest,
        run_pattern_discovery,
    )

    feature_samples = all_samples if all_samples is not None else discovery_samples
    calibration_samples = list(calibration_samples or [])
    discovery_signature = _symbolic_sample_signature(discovery_samples)
    calibration_signature = _symbolic_sample_signature(calibration_samples)
    feature_signature = _symbolic_sample_signature(feature_samples)
    protocol_signature = hashlib.sha256(
        json.dumps(
            {
                "constraint_policy": str(constraint_policy),
                "prediction_top_k": int(prediction_top_k),
                "search_pool_size": int(search_pool_size),
                "max_hard": int(dfa_max_constraints),
                "max_soft": int(search_soft_constraints),
                "support_bounds": [
                    float(search_min_calibration_support),
                    float(search_max_calibration_support),
                ],
                "min_stability": float(search_min_stability),
                "min_hard": int(search_min_hard_constraints),
                "fallback_min_case_support": float(search_fallback_min_case_support),
                "fallback_min_prefix_support": float(search_fallback_min_prefix_support),
                "fallback_min_stability": float(search_fallback_min_stability),
                "support_confidence_z": float(search_support_confidence_z),
                "dfa_max_states": int(dfa_max_states),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:10]
    cache_key = (
        6,
        dataset_name,
        len(discovery_samples),
        len(calibration_samples),
        len(feature_samples),
        discovery_signature,
        calibration_signature,
        feature_signature,
        int(audit_seed),
        str(constraint_policy),
        int(prediction_top_k),
        int(search_pool_size),
        int(dfa_max_constraints),
        int(search_soft_constraints),
        float(search_min_calibration_support),
        float(search_max_calibration_support),
        float(search_min_stability),
        int(search_min_hard_constraints),
        float(search_fallback_min_case_support),
        float(search_fallback_min_prefix_support),
        float(search_fallback_min_stability),
        float(search_support_confidence_z),
        int(dfa_max_states),
    )
    cached = _PATTERN_CACHE.get(cache_key)
    if cached is not None:
        result = cached["result"]
        feature_encoded = cached.get("feature_encoded")
        if int(result[5]) > 0 and feature_encoded is not None:
            encode_pattern_features(feature_samples, feature_encoded, act_encoder)
        print(
            "  Role-separated pattern protocol: cache hit "
            f"(prediction={len(result[0] or {})}, "
            f"hard={len(result[1] or {})}, soft={len(result[2] or {})})"
        )
        return result

    t0 = time.time()
    safe_dataset_name = "".join(
        ch if ch.isalnum() or ch in ("_", "-") else "_"
        for ch in (dataset_name or "default")
    )
    output_path = os.path.join(
        "./patterns",
        f"{safe_dataset_name}_kv2_d{len(discovery_samples)}"
        f"_c{len(calibration_samples)}_s{int(audit_seed)}"
        f"_p{str(constraint_policy).strip().lower()}"
        f"_f{discovery_signature[:8]}-{calibration_signature[:8]}_h{protocol_signature}",
    )
    prediction_path = os.path.join(output_path, "prediction")
    search_path = os.path.join(output_path, "search")
    prediction_patterns = {}
    feature_encoded = None
    pattern_encoder = None
    pattern_feat_dim = 0
    try:
        (
            prediction_train_encoded,
            prediction_patterns,
            feature_encoded,
            pattern_encoder,
        ) = run_pattern_discovery(
            discovery_samples,
            act_encoder,
            output_path=prediction_path,
            return_dict=True,
            transform_samples=feature_samples,
            return_transform=True,
            return_miner=True,
            target_col="next_activity",
            selection_mode="supervised_categorical",
            top_k=int(prediction_top_k),
            min_feature_coverage=0.03,
            max_feature_coverage=0.90,
            min_abs_hit_rate_diff=0.02,
        )
        prediction_patterns = prediction_patterns or {}
        if prediction_train_encoded is not None and prediction_patterns:
            pattern_feat_dim = int(prediction_train_encoded.shape[1])
            encode_pattern_features(feature_samples, feature_encoded, act_encoder)
    except Exception as exc:
        prediction_patterns = {}
        feature_encoded = None
        pattern_encoder = None
        pattern_feat_dim = 0
        print(f"  Prediction-Pattern discovery failed: {exc}")

    search_patterns = {}
    search_train_encoded = None
    calibration_encoded = None
    search_attributes = pd.DataFrame()
    try:
        (
            search_train_encoded,
            search_patterns,
            calibration_encoded,
            search_miner,
        ) = run_pattern_discovery(
            discovery_samples,
            act_encoder,
            output_path=search_path,
            return_dict=True,
            transform_samples=calibration_samples,
            return_transform=True,
            return_miner=True,
            target_col="__label_free__",
            selection_mode="label_free",
            top_k=int(search_pool_size),
            min_feature_coverage=0.03,
            max_feature_coverage=1.00,
            include_unigrams=True,
            min_abs_hit_rate_diff=0.0,
        )
        search_patterns = search_patterns or {}
        if search_miner is not None:
            search_attributes = search_miner.get_pattern_table()
    except Exception as exc:
        search_patterns = {}
        calibration_encoded = None
        print(f"  label-free Search-Pattern discovery failed: {exc}")

    hard_constraints, soft_constraints, calibration_frame = (
        calibrate_search_patterns(
            search_patterns,
            search_attributes,
            calibration_encoded,
            discovery_encoded=search_train_encoded,
            policy=constraint_policy,
            min_calibration_support=float(search_min_calibration_support),
            max_calibration_support=float(search_max_calibration_support),
            min_stability=float(search_min_stability),
            max_hard_constraints=int(dfa_max_constraints),
            max_soft_constraints=int(search_soft_constraints),
            min_hard_constraints=int(search_min_hard_constraints),
            fallback_min_case_support=float(search_fallback_min_case_support),
            fallback_min_prefix_support=float(search_fallback_min_prefix_support),
            fallback_min_stability=float(search_fallback_min_stability),
            support_confidence_z=float(search_support_confidence_z),
        )
    )
    selected_reasons = (
        calibration_frame["hard_selection_reason"]
        if not calibration_frame.empty
        and "hard_selection_reason" in calibration_frame.columns
        else pd.Series(dtype=str)
    )
    protocol_metadata = {
        "knowledge_protocol_version": 2,
        "prediction_pattern_target": "next_activity",
        "protocol_signature": protocol_signature,
        "search_selection_mode": "label_free",
        "search_selection_uses_outcome_label": False,
        "constraint_policy": str(constraint_policy),
        "discovery_prefix_count": len(discovery_samples),
        "calibration_prefix_count": len(calibration_samples),
        "discovery_fold_signature": discovery_signature,
        "calibration_fold_signature": calibration_signature,
        "search_pattern_pool_count": len(search_patterns),
        "hard_constraint_count": len(hard_constraints),
        "soft_constraint_count": len(soft_constraints),
        "strict_hard_constraint_count": int((selected_reasons == "strict").sum()),
        "fallback_hard_constraint_count": int(
            (selected_reasons == "stable_anchor_fallback").sum()
        ),
        "support_granularity": "complete_case_any_prefix",
        "min_calibration_support": float(search_min_calibration_support),
        "max_calibration_support": float(search_max_calibration_support),
        "min_stability": float(search_min_stability),
        "min_hard_constraints": int(search_min_hard_constraints),
        "fallback_min_case_support": float(search_fallback_min_case_support),
        "fallback_min_prefix_support": float(search_fallback_min_prefix_support),
        "fallback_min_stability": float(search_fallback_min_stability),
        "support_confidence_z": float(search_support_confidence_z),
    }
    manifest_paths = persist_constraint_manifest(
        search_path, calibration_frame, protocol_metadata
    )
    dfa, multi_dfa_clusters = _compile_constraint_set(
        hard_constraints,
        act_encoder,
        label="calibrated search",
        dfa_max_states=int(dfa_max_states),
    )
    result = (
        prediction_patterns,
        hard_constraints,
        soft_constraints,
        dfa,
        calibration_frame,
        pattern_feat_dim,
        multi_dfa_clusters,
        manifest_paths,
        protocol_metadata,
        pattern_encoder,
    )
    _PATTERN_CACHE[cache_key] = {
        "result": result,
        "feature_encoded": feature_encoded,
    }
    print(
        "  Role-separated pattern protocol done: "
        f"prediction={len(prediction_patterns)}, "
        f"search_pool={len(search_patterns)}, hard={len(hard_constraints)}, "
        f"soft={len(soft_constraints)}, "
        f"{_format_dfa_status(dfa, multi_dfa_clusters)}, "
        f"elapsed={time.time() - t0:.0f}s"
    )
    return result


def generate_random_cf(acts, times, mask, num_activities, topk=5):
    """Random baseline: replace the last observed activity, not its target."""
    raw_len = mask.sum().int().item()
    if raw_len < 2:
        return [[]]
    orig_acts = acts[0, :raw_len].tolist()
    orig_times = times[0, :raw_len].tolist()
    edit_pos = raw_len - 2
    candidates = []
    seen = set()
    for _ in range(topk * 3):
        a = np.random.randint(1, num_activities)
        if a == orig_acts[edit_pos]:
            continue
        cf_acts_list = list(orig_acts)
        cf_acts_list[edit_pos] = a
        cf_acts = tuple(cf_acts_list)
        if cf_acts in seen:
            continue
        seen.add(cf_acts)
        candidates.append((list(cf_acts), list(orig_times), float(np.random.random())))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return [candidates[:topk]]


def build_intervention_positions(n_points: int):
    n_points = max(1, int(n_points))
    return [-(idx + 1) for idx in range(n_points)]


def infer_dataset_family(dataset_name):
    name = str(dataset_name).lower()
    if name.startswith("sepsis"):
        return "Sepsis"
    if name.startswith("bpic15"):
        return "BPIC15"
    if name.startswith("bpic17"):
        return "BPIC17"
    return str(dataset_name).split("_")[0]


def build_cf_evaluation_manifest(test_samples, dataset_name, manifest_dir,
                                 max_samples=0, split_seed=42,
                                 evaluation_seed=2026):
    """Persist and reuse an evaluation sample list shared by every variant.

    A local RNG makes the list independent of model training and other random
    operations.  ``max_samples <= 0`` evaluates all test prefixes.
    """
    os.makedirs(manifest_dir, exist_ok=True)
    safe_name = _safe_filename_part(dataset_name)
    cap_label = "all" if int(max_samples or 0) <= 0 else str(int(max_samples))
    manifest_path = os.path.join(
        manifest_dir,
        f"{safe_name}_split{int(split_seed)}_eval{int(evaluation_seed)}_n{cap_label}.csv",
    )
    n_test = len(test_samples)
    required = {
        "sample_idx", "n_test_samples", "split_seed", "evaluation_seed",
        "protocol_version",
    }
    if os.path.exists(manifest_path) and os.path.getsize(manifest_path) > 0:
        existing = pd.read_csv(manifest_path)
        if required.issubset(existing.columns) and not existing.empty:
            valid_metadata = (
                int(existing["n_test_samples"].iloc[0]) == n_test
                and int(existing["split_seed"].iloc[0]) == int(split_seed)
                and int(existing["evaluation_seed"].iloc[0]) == int(evaluation_seed)
                and int(existing["protocol_version"].iloc[0]) == 2
            )
            indices = existing["sample_idx"].astype(int).tolist()
            if valid_metadata and all(0 <= idx < n_test for idx in indices):
                print(f"  CF evaluation manifest reused: {manifest_path} ({len(indices)} samples)")
                return np.asarray(indices, dtype=int), manifest_path

    if int(max_samples or 0) <= 0 or int(max_samples) >= n_test:
        indices = np.arange(n_test, dtype=int)
    else:
        rng = np.random.default_rng(int(evaluation_seed))
        indices = np.sort(
            rng.choice(n_test, size=int(max_samples), replace=False).astype(int)
        )

    rows = []
    for idx in indices.tolist():
        sample = test_samples[idx]
        rows.append({
            "dataset": dataset_name,
            "dataset_family": infer_dataset_family(dataset_name),
            "sample_idx": idx,
            "case_id": _sample_case_key(sample, idx),
            "prefix_len": int(sample.get("prefix_len", sample.get("length", 0))),
            "outcome": int(sample.get("outcome", -1)),
            "n_test_samples": n_test,
            "split_seed": int(split_seed),
            "evaluation_seed": int(evaluation_seed),
            "protocol_version": 2,
        })
    _write_csv_atomic(pd.DataFrame(rows), manifest_path)
    print(f"  CF evaluation manifest saved: {manifest_path} ({len(indices)} samples)")
    return indices, manifest_path


def _spread_indices(n_items, max_rows):
    if n_items <= 0:
        return []
    if max_rows is None or int(max_rows) <= 0 or int(max_rows) >= n_items:
        return list(range(n_items))
    idx = np.linspace(0, n_items - 1, int(max_rows))
    return sorted(set(int(round(i)) for i in idx))


def _activity_name(act_encoder, activity_id):
    if act_encoder is None or activity_id <= 0:
        return "N/A"
    try:
        return act_encoder.inverse_transform([int(activity_id) - 1])[0]
    except Exception:
        return "N/A"


def _calibration_rows(pred_rows, dataset_name, variant, split_name="test", n_bins=10,
                      seed=42):
    rows = []
    if not pred_rows:
        return rows
    probs = np.asarray([r["outcome_prob"] for r in pred_rows], dtype=float)
    labels = np.asarray([r["true_outcome_label"] for r in pred_rows], dtype=float)
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if b == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            mean_pred = float("nan")
            observed = float("nan")
            gap = float("nan")
        else:
            mean_pred = float(np.mean(probs[mask]))
            observed = float(np.mean(labels[mask]))
            gap = abs(mean_pred - observed)
        rows.append({
            "dataset": dataset_name,
            "variant": variant,
            "seed": int(seed),
            "split": split_name,
            "bin": b,
            "bin_lower": lo,
            "bin_upper": hi,
            "n": n,
            "mean_pred_prob": mean_pred,
            "observed_success_rate": observed,
            "abs_calibration_gap": gap,
        })
    return rows


def save_prediction_outputs(backbone, dataset, cfg, act_encoder, dataset_name,
                            variant, output_dir, max_rows=5000,
                            split_name="test", batch_size=256, seed=42):
    """Save sampled per-prefix predictions and calibration bins for visualization."""
    if act_encoder is None or len(dataset) == 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    indices = _spread_indices(len(dataset), max_rows)
    pred_rows = []
    backbone.eval()
    pred_ds = Subset(dataset, indices)
    pred_loader = DataLoader(
        pred_ds,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        collate_fn=collate_fn,
    )

    with torch.no_grad():
        row_offset = 0
        for cpu_batch in pred_loader:
            B = int(cpu_batch["activities"].size(0))
            batch_indices = indices[row_offset:row_offset + B]
            row_offset += B
            batch = {k: v.to(cfg.device) for k, v in cpu_batch.items()}
            out = backbone(
                batch["activities"], batch["times"], batch["mask"],
                pattern_features=batch.get("pattern_features", None),
                static_cat=batch.get("static_cat"),
                static_num=batch.get("static_num"),
                dynamic_cat=batch.get("dynamic_cat"),
                dynamic_num=batch.get("dynamic_num"),
            )

            outcome_probs = torch.softmax(out["outcome_logits"], dim=-1)[:, 1].detach().cpu().numpy()
            next_time_preds = out["next_time_pred"].detach().cpu().numpy()
            if "next_time_delta_pred" in out:
                next_delta_preds = out["next_time_delta_pred"].detach().cpu().numpy()
            else:
                next_delta_preds = np.full(B, np.nan)
            act_logits_all = out["next_act_logits"].detach().cpu()

            for j, idx in enumerate(batch_indices):
                raw_sample = (
                    dataset.samples[idx]
                    if hasattr(dataset, "samples") else {}
                )
                raw_len = int(cpu_batch["lengths"][j].item())
                true_outcome_label = int(cpu_batch["outcome"][j].item())
                outcome_prob = float(outcome_probs[j])
                pred_outcome_label = int(outcome_prob >= 0.5)

                act_logits = act_logits_all[j]
                if act_logits.numel() > 1:
                    real_act_probs = torch.softmax(act_logits[1:], dim=-1)
                    pred_offset = int(real_act_probs.argmax().item())
                    next_act_pred = pred_offset + 1
                    next_act_prob = float(real_act_probs[pred_offset].item())
                    topk = min(5, real_act_probs.numel())
                    top_vals, top_idx = real_act_probs.topk(topk)
                    top5_ids = [int(i.item()) + 1 for i in top_idx]
                    top5_probs = [float(v.item()) for v in top_vals]
                    top3_ids = top5_ids[:3]
                    top3_probs = top5_probs[:3]
                else:
                    next_act_pred = 0
                    next_act_prob = float("nan")
                    top3_ids = []
                    top3_probs = []
                    top5_ids = []
                    top5_probs = []

                next_time_pred = float(next_time_preds[j])
                next_time_delta_pred = float(next_delta_preds[j])

                true_act_id = 0
                true_time = float("nan")
                prev_time = float("nan")
                true_time_delta = float("nan")
                if raw_len >= 2:
                    true_act_id = int(cpu_batch["activities"][j, raw_len - 1].item())
                    true_time = float(cpu_batch["times"][j, raw_len - 1].item())
                    prev_time = float(cpu_batch["times"][j, raw_len - 2].item())
                    true_time_delta = max(true_time - prev_time, 0.0)

                pred_rows.append({
                    "dataset": dataset_name,
                    "variant": variant,
                    "dataset_family": infer_dataset_family(dataset_name),
                    "seed": int(seed),
                    "protocol_version": 2,
                    "split": split_name,
                    "sample_idx": idx,
                    "case_id": _sample_case_key(raw_sample, idx),
                    "prefix_len": int(
                        raw_sample.get("prefix_len", raw_sample.get("length", 0))
                    ),
                    "raw_len": raw_len,
                    "true_outcome": "Success" if true_outcome_label == 1 else "Failure",
                    "true_outcome_label": true_outcome_label,
                    "pred_outcome": "Success" if pred_outcome_label == 1 else "Failure",
                    "pred_outcome_label": pred_outcome_label,
                    "outcome_prob": round(outcome_prob, 6),
                    "outcome_brier_error": round((outcome_prob - true_outcome_label) ** 2, 6),
                    "outcome_correct": int(pred_outcome_label == true_outcome_label),
                    "true_next_act": _activity_name(act_encoder, true_act_id),
                    "pred_next_act": _activity_name(act_encoder, next_act_pred),
                    "next_act_prob": round(next_act_prob, 6) if np.isfinite(next_act_prob) else float("nan"),
                    "next_act_correct": int(next_act_pred == true_act_id),
                    "next_act_top3_correct": int(true_act_id in top3_ids) if true_act_id > 0 else 0,
                    "next_act_top3": " | ".join(_activity_name(act_encoder, a) for a in top3_ids),
                    "next_act_top3_probs": " | ".join(f"{p:.6f}" for p in top3_probs),
                    "next_act_top5_correct": int(true_act_id in top5_ids) if true_act_id > 0 else 0,
                    "next_act_top5": " | ".join(_activity_name(act_encoder, a) for a in top5_ids),
                    "next_act_top5_probs": " | ".join(f"{p:.6f}" for p in top5_probs),
                    "prev_time": round(prev_time, 6) if np.isfinite(prev_time) else float("nan"),
                    "true_next_time": round(true_time, 6) if np.isfinite(true_time) else float("nan"),
                    "pred_next_time": round(next_time_pred, 6),
                    "next_time_error": round(next_time_pred - true_time, 6) if np.isfinite(true_time) else float("nan"),
                    "next_time_abs_error": round(abs(next_time_pred - true_time), 6) if np.isfinite(true_time) else float("nan"),
                    "next_time_sq_error": round((next_time_pred - true_time) ** 2, 6) if np.isfinite(true_time) else float("nan"),
                    "true_next_delta": round(true_time_delta, 6) if np.isfinite(true_time_delta) else float("nan"),
                    "pred_next_delta": round(next_time_delta_pred, 6),
                })

    if not pred_rows:
        return

    pred_csv = os.path.join(
        output_dir, f"predictions_{dataset_name}_{variant}_seed{int(seed)}.csv"
    )
    pd.DataFrame(pred_rows).to_csv(pred_csv, index=False)
    print(f"  Prediction details saved:  {pred_csv} ({len(pred_rows)} samples)")

    calibration_rows = _calibration_rows(
        pred_rows,
        dataset_name,
        variant,
        split_name=split_name,
        seed=seed,
    )
    cal_csv = os.path.join(
        output_dir,
        f"prediction_calibration_{dataset_name}_{variant}_seed{int(seed)}.csv",
    )
    pd.DataFrame(calibration_rows).to_csv(cal_csv, index=False)
    print(f"  Prediction calibration saved:  {cal_csv} ({len(calibration_rows)} bins)")


def _detail_grouped_mean(rows, key):
    groups = {}
    for row in rows or []:
        if key not in row:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        groups.setdefault(row.get("sample_idx", len(groups)), []).append(value)
    if not groups:
        return float("nan")
    return float(np.mean([np.mean(values) for values in groups.values() if values]))


def _finite_metric(metrics, key):
    try:
        value = float(metrics.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def add_standard_cf_quality_metrics(metrics, cf_detail_rows=None,
                                    dfa_eval_available=False,
                                    dfa_enforced=False,
                                    search_dfa_eval_available=False):
    """Add paper-facing CF quality metrics while keeping legacy columns.

    Direction convention:
    - higher is better: validity/success, diversity, compliance, plausibility_score
    - lower is better: proximity, sparsity, plausibility_error, generation_time
    """
    metrics = dict(metrics or {})

    count = _finite_metric(metrics, "nbr_cfs")
    success_rate = _detail_grouped_mean(cf_detail_rows, "cf_success")
    validity_rate = _detail_grouped_mean(cf_detail_rows, "cf_valid")
    flip_rate = _detail_grouped_mean(cf_detail_rows, "prediction_flipped")
    prob_shift = _detail_grouped_mean(cf_detail_rows, "prob_shift")

    # These event-rate metrics have a real zero meaning when no CF is returned.
    # Distance/diversity/plausibility metrics remain undefined unless computed
    # from actual counterfactual candidates.
    if (not cf_detail_rows) and np.isfinite(count) and count <= 0:
        success_rate = 0.0
        validity_rate = 0.0
        flip_rate = 0.0
        prob_shift = 0.0

    implaus = _finite_metric(metrics, "implausibility")
    plausibility_score = 1.0 / (1.0 + implaus) if np.isfinite(implaus) and implaus >= 0 else float("nan")

    metrics.update({
        "metric_schema_version": 6.0,
        "count": count,
        "validity_rate": validity_rate,
        "success_rate": success_rate,
        "target_class_rate": success_rate,
        "prediction_flip_rate": flip_rate,
        "bidirectional_flip_rate": flip_rate,
        "mean_prob_shift": prob_shift,
        "proximity_l2j": _finite_metric(metrics, "distance_l2j"),
        "proximity_mh": _finite_metric(metrics, "distance_mh"),
        "sparsity_rate": _finite_metric(metrics, "avg_nbr_changes"),
        "sparsity_count": _finite_metric(metrics, "avg_n_changed_positions"),
        "diversity_score": _finite_metric(metrics, "diversity_l2j"),
        "plausibility_error": implaus,
        "plausibility_score": plausibility_score,
        "generation_time_sec": _finite_metric(metrics, "gen_time_per_sample"),
        # Primary audit metric: mean satisfaction across activated rules.  The
        # stricter all-applicable rate is retained as a secondary diagnostic.
        "audit_compliance_rate": _detail_grouped_mean(
            cf_detail_rows, "audit_score"
        ),
        "audit_all_applicable_rate": _detail_grouped_mean(
            cf_detail_rows, "audit_all_applicable"
        ),
        "search_constraint_satisfaction_rate": _detail_grouped_mean(
            cf_detail_rows, "search_compliant"
        ),
        "search_runtime_relaxation_rate": _detail_grouped_mean(
            cf_detail_rows, "search_relaxed"
        ),
        "search_mean_relaxed_constraint_count": _detail_grouped_mean(
            cf_detail_rows, "search_relaxation_count"
        ),
        "search_mean_relaxation_cost": _detail_grouped_mean(
            cf_detail_rows, "search_relaxation_cost"
        ),
        "search_effective_compliance_rate": _detail_grouped_mean(
            cf_detail_rows, "search_effective_compliant"
        ),
        "search_combination_budget_exhaustion_rate": _detail_grouped_mean(
            cf_detail_rows, "search_combination_budget_exhausted"
        ),
        "audit_compliance_defined": float(bool(dfa_eval_available)),
        "search_compliance_defined": float(bool(search_dfa_eval_available)),
        "compliance_defined": float(bool(dfa_eval_available)),
        "compliance_enforced": float(bool(dfa_enforced)),
    })
    metrics.update(summarize_queries(cf_detail_rows or []))
    return metrics


def evaluate_cfs(cf_model, backbone, test_dataset, test_indices,
                 cfg, L, cont_idx, cat_idx, ref_matrix,
                 pattern_dict, constraint_dict, audit_rules, dfa, act_encoder,
                 save_prefix="", audit_dfa=None, audit_multi_dfa_clusters=None,
                 search_eval_dfa=None, search_eval_multi_dfa_clusters=None,
                 dfa_enforced=False, output_dir="results", dataset_name="",
                 variant="", seed=42, soft_constraint_dict=None):
    """Generate and evaluate counterfactuals for a test slice."""
    from evaluation.cf_evaluation import evaluate_counterfactuals
    from audit_constraints import evaluate_audit_trace
    from explaination.visualizations.utils import normalize_activity_name, trace_accepts_dfa
    all_metrics = []
    all_audit_dfa_flags = []
    all_search_dfa_flags = []
    total_time = 0.0
    n_generated = 0
    n_attempted = 0
    generation_failures = 0
    metric_failures = 0
    failure_examples = []
    cf_detail_rows = []
    backbone.eval()
    cf_model.eval()
    cf_model._dfa_action_prune_log = []
    cf_model._dfa_prefilter_log = []
    cf_model._dfa_full_log = []
    cf_model._constraint_relaxation_log = []
    intervention_positions = build_intervention_positions(cfg.cf_intervention_points)
    # Audit evidence comes only from independently discovered, separately
    # compiled rules.  Search-DFA clusters are never used as an audit fallback.
    audit_rules = list(audit_rules or [])
    dfa_eval_available = bool(audit_rules)
    search_eval_multi_dfa_clusters = search_eval_multi_dfa_clusters or []
    search_dfa_eval_available = bool(
        search_eval_dfa is not None or len(search_eval_multi_dfa_clusters) > 0
    )
    protected = getattr(cfg, "cf_non_relaxable_rule_ids", ())
    max_relaxed = getattr(cfg, "cf_max_relaxed_constraints", None)
    if protected or max_relaxed is not None:
        if not dfa_enforced or not flatten_cluster_rule_dfas(search_eval_multi_dfa_clusters):
            raise ValueError("Exemption policy requires enforced, individually compiled Search-DFA rules")
        # Validate configuration before the per-query exception handler.
        evaluate_minimal_constraint_relaxation([], search_eval_multi_dfa_clusters, max_relaxed, protected)

    def is_dfa_compliant(norm_names, single_dfa, clusters):
        if single_dfa is None and not clusters:
            return None
        if clusters:
            relaxation = evaluate_minimal_constraint_relaxation(norm_names, clusters)
            if relaxation is not None:
                return bool(relaxation["strict"])
            return any(
                trace_accepts_dfa(norm_names, cluster["dfa"])
                for cluster in clusters
                if cluster.get("dfa") is not None
            )
        return trace_accepts_dfa(norm_names, single_dfa)

    def sample_model_kwargs(sample, length, repeat=1):
        """Build model kwargs from one dataset sample and keep tensors on cfg.device."""
        kwargs = {}

        pattern_features = sample.get("pattern_features")
        if pattern_features is not None:
            t = pattern_features.unsqueeze(0).to(cfg.device)
            if repeat > 1:
                t = t.repeat(repeat, 1)
            kwargs["pattern_features"] = t

        for key in ("static_cat", "static_num"):
            tensor = sample.get(key)
            if tensor is None:
                continue
            t = tensor.unsqueeze(0).to(cfg.device)
            if repeat > 1:
                t = t.repeat(repeat, 1)
            kwargs[key] = t

        for key in ("dynamic_cat", "dynamic_num"):
            tensor = sample.get(key)
            if tensor is None:
                continue
            t = tensor[:length].unsqueeze(0).to(cfg.device)
            if repeat > 1:
                t = t.repeat(repeat, 1, 1)
            kwargs[key] = t

        return kwargs

    def repeat_model_kwargs(kwargs, repeat):
        if repeat <= 1:
            return kwargs
        repeated = {}
        for key, tensor in kwargs.items():
            if tensor is None:
                continue
            reps = [repeat] + [1] * (tensor.dim() - 1)
            repeated[key] = tensor.repeat(*reps)
        return repeated

    for idx in test_indices:
        sample = test_dataset[idx]
        raw_sample = (
            test_dataset.samples[idx]
            if hasattr(test_dataset, "samples") else {}
        )
        case_id = _sample_case_key(raw_sample, idx)
        prefix_len = int(raw_sample.get("prefix_len", raw_sample.get("length", 0)))
        raw_len = int(sample["length"].item())
        acts = sample["activities"][:raw_len].unsqueeze(0).to(cfg.device)
        times = sample["times"][:raw_len].unsqueeze(0).to(cfg.device)
        mask = torch.ones_like(acts, dtype=torch.bool)
        model_kwargs = sample_model_kwargs(sample, raw_len)
        # Evaluate before search so failed/no-candidate queries retain eligibility.
        true_label = sample["outcome"].item()
        with torch.no_grad():
            orig_out = backbone(acts, times, mask, **model_kwargs)["outcome_logits"]
            orig_prob = float(torch.softmax(orig_out, dim=-1)[0, 1].item())
        orig_pred_success = orig_prob >= 0.5
        n_attempted += 1

        t_start = time.time()
        generation_failed_current = False
        generation_relaxation_diag = None
        relaxation_log_before = len(
            getattr(cf_model, "_constraint_relaxation_log", [])
        )
        try:
            if getattr(cfg, "cf_generator", "default") == "random":
                traces = generate_random_cf(
                    acts, times, mask, cfg.num_activities, topk=cfg.cf_topk
                )
            else:
                traces = cf_model.generate_counterfactual_traces(
                    acts, times, mask, topk=cfg.cf_topk,
                    pattern_dict=soft_constraint_dict,
                    act_encoder=act_encoder,
                    pattern_dfa=dfa,
                    intervention_positions=intervention_positions,
                    **model_kwargs,
                )
            total_time += time.time() - t_start
            n_generated += 1
        except Exception as exc:
            total_time += time.time() - t_start
            generation_failures += 1
            generation_failed_current = True
            if len(failure_examples) < 3:
                failure_examples.append(
                    f"idx={idx}: {type(exc).__name__}: {exc}"
                )
            traces = [[]]
        relaxation_log = getattr(cf_model, "_constraint_relaxation_log", [])
        if len(relaxation_log) > relaxation_log_before:
            generation_relaxation_diag = relaxation_log[-1]
        generation_relaxation_diag = generation_relaxation_diag or {}
        generation_search_mode = generation_relaxation_diag.get(
            "search_mode", "no_dfa" if not dfa_enforced else "no_candidate"
        )

        if not traces or not traces[0]:
            # Preserve the fixed-manifest analysis unit even when a generator
            # returns no candidate.  Efficacy is then zero, while candidate-only
            # quantities (including compliance) remain undefined.
            cf_detail_rows.append({
                "dataset": dataset_name,
                "dataset_family": infer_dataset_family(dataset_name),
                "variant": variant,
                "seed": int(seed),
                "protocol_version": 2,
                "sample_idx": idx,
                "case_id": case_id,
                "prefix_len": prefix_len,
                "raw_len": raw_len,
                "orig_prob": orig_prob,
                "orig_pred_label": "Success" if orig_pred_success else "Failure",
                "true_outcome": (
                    "Success" if int(sample["outcome"].item()) == 1 else "Failure"
                ),
                "cf_rank": 0,
                "candidate_returned": 0,
                "cf_success": 0,
                "cf_valid": 0,
                "prediction_flipped": 0,
                "prob_shift": 0.0,
                "audit_score": float("nan"),
                "audit_all_applicable": float("nan"),
                "audit_rule_count": len(audit_rules),
                "audit_applicable_rule_count": 0,
                "audit_satisfied_rule_count": 0,
                "audit_violated_rule_ids": "",
                "audit_compliant": float("nan"),
                "search_compliant": float("nan"),
                "audit_compliant_label": "",
                "search_compliant_label": "",
                "dfa_compliant": "",
                "pattern_hits": "",
                "n_pattern_hits": 0,
                "hard_pattern_hits": "",
                "n_hard_pattern_hits": 0,
                "soft_pattern_hits": "",
                "n_soft_pattern_hits": 0,
                "soft_constraint_score": float("nan"),
                "search_mode": generation_search_mode,
                "search_relaxed": float("nan"),
                "search_relaxation_count": generation_relaxation_diag.get(
                    "relaxation_count", float("nan")
                ),
                "search_relaxation_cost": generation_relaxation_diag.get(
                    "relaxation_cost", float("nan")
                ),
                "search_relaxed_constraint_ids": ";".join(
                    generation_relaxation_diag.get(
                        "relaxed_constraint_ids", ()
                    )
                ),
                "search_selected_cluster_id": generation_relaxation_diag.get(
                    "cluster_id", ""
                ),
                "search_cluster_rule_count": generation_relaxation_diag.get(
                    "cluster_rule_count", float("nan")
                ),
                "search_compile_relaxed_constraint_ids": ";".join(
                    generation_relaxation_diag.get(
                        "compile_relaxed_constraint_ids", ()
                    )
                ),
                "search_compile_omitted_constraint_ids": ";".join(
                    generation_relaxation_diag.get(
                        "compile_omitted_constraint_ids", ()
                    )
                ),
                "search_effective_compliant": float("nan"),
                "search_verification_status": "no_candidate",
                "search_original_cluster_compliant": float("nan"),
                "search_combination_count": generation_relaxation_diag.get("combination_count", 0),
                "search_combination_space": generation_relaxation_diag.get("combination_space", 0),
                "search_truncated_option_count": generation_relaxation_diag.get("truncated_option_count", 0),
                "search_combination_budget_reached": generation_relaxation_diag.get("combination_budget_reached", False),
                "search_combination_budget_exhausted": bool(
                    generation_relaxation_diag.get(
                        "combination_budget_exhausted", False
                    )
                ),
                "generation_status": (
                    "error" if generation_failed_current else "no_candidate"
                ),
            })
            continue

        candidates = traces[0]
        query_acts = sample["activities"][:raw_len].tolist()
        query_times = sample["times"][:raw_len].tolist()

        query_names = [act_encoder.inverse_transform([a-1])[0] if a>0 else None for a in query_acts[:prefix_len]]
        query_names = [n for n in query_names if n is not None]

        recommended = []
        for act_seq, time_seq, _ in candidates:
            pad_act_seq = act_seq + [0] * (L - len(act_seq))
            pad_time_seq = time_seq + [0.0] * (L - len(time_seq))
            recommended.append((pad_act_seq, pad_time_seq))

        pad_acts = query_acts + [0] * (L - len(query_acts))
        pad_times = query_times + [0.0] * (L - len(query_times))

        cf_probs = []
        if candidates:
            try:
                cf_tensor_batch = torch.tensor(
                    [act_seq[:raw_len] for act_seq, _, _ in candidates],
                    dtype=torch.long,
                    device=cfg.device,
                )
                cf_time_batch = torch.tensor(
                    [time_seq[:raw_len] for _, time_seq, _ in candidates],
                    dtype=torch.float,
                    device=cfg.device,
                )
                cf_mask_batch = torch.ones_like(cf_tensor_batch, dtype=torch.bool)
                cf_model_kwargs = repeat_model_kwargs(model_kwargs, len(candidates))
                candidate_pattern_features = cf_model.encode_pattern_batch(
                    [act_seq[:raw_len] for act_seq, _, _ in candidates],
                    [time_seq[:raw_len] for _, time_seq, _ in candidates],
                    [prefix_len] * len(candidates),
                    device=cfg.device,
                )
                if candidate_pattern_features is not None:
                    cf_model_kwargs["pattern_features"] = candidate_pattern_features
                with torch.no_grad():
                    cf_logits = backbone(
                        cf_tensor_batch, cf_time_batch, cf_mask_batch,
                        **cf_model_kwargs,
                    )["outcome_logits"]
                    cf_probs = torch.softmax(cf_logits, dim=-1)[:, 1].detach().cpu().tolist()
            except Exception:
                cf_probs = []

        # ---- CF detail rows (one row per candidate) ----
        for rank, (act_seq, time_seq, score) in enumerate(candidates, 1):
            if rank - 1 < len(cf_probs):
                cf_prob = float(cf_probs[rank - 1])
            else:
                    cf_tensor = torch.tensor([act_seq[:raw_len]], dtype=torch.long, device=cfg.device)
                    single_kwargs = dict(model_kwargs)
                    candidate_pattern_features = cf_model.encode_pattern_batch(
                        [act_seq[:raw_len]], [time_seq[:raw_len]], [prefix_len],
                        device=cfg.device,
                    )
                    if candidate_pattern_features is not None:
                        single_kwargs["pattern_features"] = candidate_pattern_features
                    with torch.no_grad():
                        cf_prob = float(torch.softmax(
                            backbone(cf_tensor,
                                 torch.tensor([time_seq[:raw_len]], device=cfg.device),
                                 torch.ones_like(cf_tensor, dtype=torch.bool),
                                     **single_kwargs,
                        )["outcome_logits"], dim=-1)[0, 1].item())
            orig_pred_success = orig_prob >= 0.5
            cf_pred_success = cf_prob >= 0.5
            prob_shift = cf_prob - orig_prob

            cf_names = [act_encoder.inverse_transform([a-1])[0] if a>0 else None for a in act_seq[:prefix_len]]
            cf_names = [n for n in cf_names if n is not None]

            # Independent audit uses complete-case Declare rules and excludes
            # inapplicable implications from both numerator and denominator.
            search_dfa_ok = float("nan")
            search_dfa_label = ""
            norm_cf_names = [normalize_activity_name(n) for n in cf_names]
            search_relaxation = evaluate_minimal_constraint_relaxation(
                norm_cf_names, search_eval_multi_dfa_clusters,
                getattr(cfg, "cf_max_relaxed_constraints", None) if dfa_enforced else None,
                getattr(cfg, "cf_non_relaxable_rule_ids", ()) if dfa_enforced else (),
            )
            search_relaxation = search_relaxation or {}
            audit_result = evaluate_audit_trace(norm_cf_names, audit_rules)
            audit_score = audit_result["audit_score"]
            audit_all = audit_result["audit_all_applicable"]
            audit_dfa_label = (
                "Y" if audit_all == 1 else "N" if audit_all == 0 else ""
            )
            if np.isfinite(float(audit_score)):
                all_audit_dfa_flags.append(float(audit_score))
            is_search_ok = is_dfa_compliant(
                norm_cf_names,
                search_eval_dfa,
                search_eval_multi_dfa_clusters,
            )
            if is_search_ok is not None:
                all_search_dfa_flags.append(is_search_ok)
                search_dfa_ok = int(bool(is_search_ok))
                search_dfa_label = "Y" if is_search_ok else "N"

            effective = float("nan")
            original_cluster = float("nan")
            verification_status = "unverified"
            if dfa_enforced and search_relaxation:
                effective = verify_retained_constraints(
                    norm_cf_names, search_eval_multi_dfa_clusters,
                    search_relaxation["cluster_id"], search_relaxation["relaxed_constraint_ids"],
                )
                weakened = bool(search_relaxation["compile_relaxed_constraint_ids"]
                                or search_relaxation["compile_omitted_constraint_ids"])
                if not weakened:
                    original_cluster = int(search_relaxation["strict"])
                if effective == 1:
                    verification_status = (
                        "retained_compiled_rules_verified" if search_relaxation["relaxation_count"]
                        else "compiled_cluster_verified" if weakened
                        else "original_selected_cluster_verified"
                    )
                elif effective == 0:
                    verification_status = "retained_rules_violated"
            elif dfa_enforced and not search_eval_multi_dfa_clusters and search_eval_dfa is not None:
                effective = search_dfa_ok
                verification_status = "compiled_dfa_verified" if effective == 1 else "compiled_dfa_violated"

            # Report hard-DFA evidence and soft-ranking evidence separately.
            hard_hit_pids = []
            soft_hit_pids = []
            if (constraint_dict or soft_constraint_dict) and act_encoder:
                from trace_alignment import does_trace_contain_pattern_advanced

                for pid, graph in (constraint_dict or {}).items():
                    if does_trace_contain_pattern_advanced(
                        cf_names, time_seq[:prefix_len],
                        graph,
                        delta_time_seconds=cfg.pattern_delta_time,
                    ):
                        hard_hit_pids.append(pid)
                for pid, graph in (soft_constraint_dict or {}).items():
                    if does_trace_contain_pattern_advanced(
                        cf_names, time_seq[:prefix_len],
                        graph,
                        delta_time_seconds=cfg.pattern_delta_time,
                    ):
                        soft_hit_pids.append(pid)
            soft_constraint_score = cf_model.soft_constraint_score(
                cf_names, time_seq[:prefix_len], soft_constraint_dict
            )

            row = {
                "dataset": dataset_name,
                "dataset_family": infer_dataset_family(dataset_name),
                "variant": variant,
                "seed": int(seed),
                "protocol_version": 2,
                "sample_idx": idx,
                "case_id": case_id,
                "prefix_len": prefix_len,
                "raw_len": raw_len,
                "true_outcome": "Success" if true_label==1 else "Failure",
                "orig_prob": orig_prob,
                "orig_pred_label": "Success" if orig_pred_success else "Failure",
                "cf_rank": rank,
                "candidate_returned": 1,
                "generation_status": "ok",
                "orig_acts": " -> ".join(query_names),
                "cf_acts": " -> ".join(cf_names),
                "n_changed_positions": sum(
                    1 for a, b in zip(query_acts[:raw_len], act_seq[:raw_len]) if a != b
                ),
                "cf_score": round(score, 4),
                "cf_generator": getattr(cfg, "cf_generator", "default"),
                "cf_ranking": getattr(cfg, "cf_ranking", "score"),
                "cf_prob": cf_prob,
                "cf_pred_label": "Success" if cf_pred_success else "Failure",
                "cf_success": int(cf_pred_success),
                "cf_valid": int(cf_pred_success),
                "prediction_flipped": int(orig_pred_success != cf_pred_success),
                "prob_shift": round(prob_shift, 4),
                "audit_score": audit_score,
                "audit_all_applicable": audit_all,
                "audit_rule_count": audit_result["audit_rule_count"],
                "audit_applicable_rule_count": audit_result["audit_applicable_rule_count"],
                "audit_satisfied_rule_count": audit_result["audit_satisfied_rule_count"],
                "audit_violated_rule_ids": ";".join(audit_result["audit_violated_rule_ids"]),
                "audit_compliant": audit_all,
                "search_compliant": search_dfa_ok,
                "search_mode": generation_search_mode,
                "search_relaxed": (
                    int(search_relaxation.get("relaxation_count", 0) > 0)
                    if search_relaxation
                    else float("nan")
                ),
                "search_relaxation_count": search_relaxation.get(
                    "relaxation_count", float("nan")
                ),
                "search_relaxation_cost": search_relaxation.get(
                    "relaxation_cost", float("nan")
                ),
                "search_relaxed_constraint_ids": ";".join(
                    search_relaxation.get("relaxed_constraint_ids", ())
                ),
                "search_selected_cluster_id": search_relaxation.get(
                    "cluster_id", ""
                ),
                "search_cluster_rule_count": search_relaxation.get(
                    "cluster_rule_count", float("nan")
                ),
                "search_effective_compliant": effective,
                "search_verification_status": verification_status,
                "search_original_cluster_compliant": original_cluster,
                "search_combination_count": generation_relaxation_diag.get("combination_count", 0),
                "search_combination_space": generation_relaxation_diag.get("combination_space", 0),
                "search_truncated_option_count": generation_relaxation_diag.get("truncated_option_count", 0),
                "search_combination_budget_reached": generation_relaxation_diag.get("combination_budget_reached", False),
                "search_compile_relaxed_constraint_ids": ";".join(
                    search_relaxation.get(
                        "compile_relaxed_constraint_ids", ()
                    )
                ),
                "search_compile_omitted_constraint_ids": ";".join(
                    search_relaxation.get(
                        "compile_omitted_constraint_ids", ()
                    )
                ),
                "search_combination_budget_exhausted": bool(
                    generation_relaxation_diag.get(
                        "combination_budget_exhausted", False
                    )
                ),
                "audit_compliant_label": audit_dfa_label,
                "search_compliant_label": search_dfa_label,
                # Backward-compatible binary alias: all applicable audit rules.
                "dfa_compliant": audit_dfa_label,
                "pattern_hits": "; ".join(hard_hit_pids),
                "n_pattern_hits": len(hard_hit_pids),
                "hard_pattern_hits": "; ".join(hard_hit_pids),
                "n_hard_pattern_hits": len(hard_hit_pids),
                "soft_pattern_hits": "; ".join(soft_hit_pids),
                "n_soft_pattern_hits": len(soft_hit_pids),
                "soft_constraint_score": float(soft_constraint_score),
            }
            cf_detail_rows.append(row)

        sample_dfa_flags = [
            float(row["audit_score"])
            for row in cf_detail_rows
            if row["sample_idx"] == idx
            and np.isfinite(float(row["audit_score"]))
        ]

        try:
            metrics = evaluate_counterfactuals(
                query_activities=np.array(pad_acts),
                query_times=np.array(pad_times),
                recommended_sequences=recommended,
                cont_feat_idx=cont_idx,
                cat_feat_idx=cat_idx,
                reference_data=ref_matrix,
                dfa=None,
                activity_to_letter={name: name for name in act_encoder.classes_} if act_encoder else None,
                pattern_dict=None,
                    act_encoder=act_encoder,
                ratio_cont=None,
                comp_rate=float(np.mean(sample_dfa_flags)) if sample_dfa_flags else None,
            )
            all_metrics.append(metrics)
        except Exception as exc:
            metric_failures += 1
            if len(failure_examples) < 3:
                failure_examples.append(
                    f"metric idx={idx}: {type(exc).__name__}: {exc}"
                )

    if generation_failures or metric_failures:
        print(
            "  CF evaluation warnings: "
            f"generation_failures={generation_failures}/{max(n_attempted, 1)}, "
            f"metric_failures={metric_failures}"
        )
        for msg in failure_examples:
            print(f"    {msg}")

    # Save CF detail CSV
    if cf_detail_rows and save_prefix:
        import os as _os
        _os.makedirs(output_dir, exist_ok=True)
        cf_csv = _os.path.join(output_dir, f"cf_details_{save_prefix}.csv")
        pd.DataFrame(cf_detail_rows).to_csv(cf_csv, index=False)
        print(f"  CF details saved:  {cf_csv} ({len(cf_detail_rows)} rows)")

    def collect_dfa_filter_metrics():
        stats = {}
        prune_log = getattr(cf_model, '_dfa_action_prune_log', None)
        action_checked = 0
        action_passed = 0
        action_pruned = 0
        if prune_log:
            action_checked = sum(x[0] for x in prune_log)
            action_passed = sum(x[1] for x in prune_log)
            action_pruned = max(action_checked - action_passed, 0)
            stats["dfa_action_prune_rate"] = (
                action_pruned / action_checked if action_checked > 0 else 0.0
            )
            stats["dfa_prune_rate"] = stats["dfa_action_prune_rate"]
        else:
            stats["dfa_action_prune_rate"] = float('nan')
            stats["dfa_prune_rate"] = float('nan')

        prefilter_log = getattr(cf_model, '_dfa_prefilter_log', None)
        if prefilter_log:
            prefilter_checked = sum(x[0] for x in prefilter_log)
            prefilter_passed = sum(x[1] for x in prefilter_log)
            prefilter_pruned = max(prefilter_checked - prefilter_passed, 0)
            stats["dfa_candidate_prefilter_rate"] = (
                prefilter_pruned / prefilter_checked if prefilter_checked > 0 else 0.0
            )
        else:
            stats["dfa_candidate_prefilter_rate"] = float('nan')

        full_log = getattr(cf_model, '_dfa_full_log', None)
        full_checked = 0
        full_rejected = 0
        if full_log:
            full_checked = sum(x[0] for x in full_log)
            full_rejected = sum(x[1] for x in full_log)
            stats["dfa_full_reject_rate"] = (
                full_rejected / full_checked if full_checked > 0 else 0.0
            )
        else:
            stats["dfa_full_reject_rate"] = float('nan')

        effective_den = action_checked + full_checked
        if effective_den > 0:
            stats["dfa_effective_filter_rate"] = (
                action_pruned + full_rejected
            ) / effective_den
        else:
            stats["dfa_effective_filter_rate"] = float('nan')
        return stats

    dfa_filter_metrics = collect_dfa_filter_metrics()

    if not all_metrics:
        if generation_failures >= n_attempted and n_attempted > 0:
            no_cf_reason = "generation_failed"
        elif metric_failures > 0:
            no_cf_reason = "metric_failed"
        else:
            no_cf_reason = "no_candidates_returned"
        empty_metrics = {
            "gen_time_per_sample": total_time / max(n_attempted, 1),
            "cf_n_generated": n_generated,
            "generation_attempts": n_attempted,
            "generation_failures": generation_failures,
            "generation_failure_rate": generation_failures / max(n_attempted, 1),
            "metric_failures": metric_failures,
            "evaluable_samples": 0,
            "detail_count": len(cf_detail_rows),
            "no_cf_reason": no_cf_reason,
            "nbr_cfs": 0.0,
            "implausibility": float('nan'),
            "distance_l2j": float('nan'),
            "distance_mh": float('nan'),
            "diversity_l2j": float('nan'),
            "diversity_mh": float('nan'),
            "avg_nbr_changes": float('nan'),
            "avg_n_changed_positions": float('nan'),
            "compliance_rate": float('nan'),
            "audit_compliance_rate": float('nan'),
            "search_constraint_satisfaction_rate": float('nan'),
            "pattern_hit_rate": float('nan'),
            **dfa_filter_metrics,
            "dfa_eval_available": float(dfa_eval_available),
            "audit_dfa_eval_available": float(dfa_eval_available),
            "search_dfa_eval_available": float(search_dfa_eval_available),
            "dfa_enforced": float(bool(dfa_enforced)),
            "compliance_applicable": float(dfa_eval_available),
        }
        return add_standard_cf_quality_metrics(
            empty_metrics,
            cf_detail_rows,
            dfa_eval_available,
            dfa_enforced,
            search_dfa_eval_available,
        ), cf_detail_rows

    avg = {}
    for k in all_metrics[0]:
        vals = [m[k] for m in all_metrics if not np.isnan(m[k])]
        avg[k] = float(np.mean(vals)) if vals else float('nan')

    avg["gen_time_per_sample"] = total_time / max(n_attempted, 1)
    avg["nbr_cfs"] = (
        sum(int(row.get("candidate_returned", 1)) for row in cf_detail_rows)
        / max(n_attempted, 1)
    )
    avg["cf_n_generated"] = n_generated
    avg["generation_attempts"] = n_attempted
    avg["generation_failures"] = generation_failures
    avg["generation_failure_rate"] = generation_failures / max(n_attempted, 1)
    avg["metric_failures"] = metric_failures
    avg["evaluable_samples"] = len(all_metrics)
    avg["detail_count"] = len(cf_detail_rows)
    avg["no_cf_reason"] = ""

    avg.update(dfa_filter_metrics)
    avg["dfa_eval_available"] = float(dfa_eval_available)
    avg["audit_dfa_eval_available"] = float(dfa_eval_available)
    avg["search_dfa_eval_available"] = float(search_dfa_eval_available)
    avg["dfa_enforced"] = float(bool(dfa_enforced))
    avg["compliance_applicable"] = float(dfa_eval_available)
    avg = add_standard_cf_quality_metrics(
        avg,
        cf_detail_rows,
        dfa_eval_available,
        dfa_enforced,
        search_dfa_eval_available,
    )

    return avg, cf_detail_rows


def run_experiment(dataset_name: str, variant: str, args, seed=42):
    """Run one dataset/variant experiment and return the result row."""
    seed = int(seed)
    split_seed = int(getattr(args, "split_seed", 42))
    set_seed(seed)
    var_cfg = CONFIG_VARIANTS[variant]

    print(f"\n{'='*60}")
    print(f"  Experiment:  {dataset_name} | {var_cfg['desc']} | seed={seed}")
    print(f"{'='*60}")

    # ---- Loading data ----
    samples, attr_dims = load_public_dataset(
        dataset_name,
        data_path=args.data_path,
        bp17_anti_leak=args.bp17_anti_leak,
    )
    samples = shuffle_samples(samples, seed=split_seed)
    n = len(samples)
    train_end = int(0.7 * n)
    val_end = int(0.9 * n)
    samples, attr_dims = fit_transform_attributes_train_only(samples, attr_dims, train_end)
    warn_potential_label_proxies(dataset_name, attr_dims)
    samples, num_activities, act_encoder = encode_activities_train_only(samples, train_end)
    renormalize_times_from_train(samples, train_end)
    train_cases = samples[:train_end]
    val_cases = samples[train_end:val_end]
    test_cases = samples[val_end:]
    summarize_outcome_split("train", train_cases)
    summarize_outcome_split("val", val_cases)
    summarize_outcome_split("test", test_cases)

    # Symbolic knowledge is split at complete-case granularity before any
    # prefix expansion.  The predictive model still trains on all train cases.
    symbolic_discovery_cases = list(train_cases)
    symbolic_calibration_cases = []
    symbolic_audit_cases = []
    discovery_cases = set()
    calibration_cases = set()
    audit_cases = set()
    if not var_cfg.get("is_baseline"):
        (
            symbolic_discovery_cases, symbolic_calibration_cases,
            symbolic_audit_cases, discovery_cases, calibration_cases,
            audit_cases,
        ) = split_pattern_discovery_calibration_audit_samples(
            train_cases,
            audit_fraction=getattr(args, "audit_fraction", 0.25),
            calibration_fraction=getattr(
                args, "constraint_calibration_fraction", 0.15
            ),
            seed=getattr(args, "audit_seed", 2026),
        )
    prefix_mode = getattr(args, "prefix_mode", "all")
    max_prefixes_per_case = getattr(args, "max_prefixes_per_case", None)
    train_samples = expand_prefix_samples(train_cases, prefix_mode, max_prefixes_per_case)
    val_samples = expand_prefix_samples(val_cases, prefix_mode, max_prefixes_per_case)
    test_samples = expand_prefix_samples(test_cases, prefix_mode, max_prefixes_per_case)
    all_samples = train_samples + val_samples + test_samples
    print(f"  Prefix expansion: mode={prefix_mode}, "
          f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)} samples")
    if not train_samples:
        raise ValueError("No train prefix samples were generated; check trace lengths.")
    activity_diag = compute_next_act_label_diagnostics(train_samples, num_activities)
    print(
        "  Next-activity label diagnostics: "
        f"classes={activity_diag['num_activity_classes']}, "
        f"present={activity_diag['activity_present_classes']}, "
        f"entropy={activity_diag['activity_label_entropy']:.3f}, "
        f"rare_ratio={activity_diag['rare_activity_ratio']:.3f}"
    )

    max_seq_len = max(s["length"] for s in train_samples)
    cfg = Config(
        num_activities=num_activities, max_seq_len=max_seq_len,
        n_static_cat=attr_dims["n_static_cat"],
        n_static_num=attr_dims["n_static_num"],
        n_dynamic_cat=attr_dims["n_dynamic_cat"],
        n_dynamic_num=attr_dims["n_dynamic_num"],
        learning_rate=getattr(args, "learning_rate", 1e-3),
        weight_decay=getattr(args, "weight_decay", 1e-4),
        cf_topk=args.cf_topk,
        cf_intervention_points=args.cf_intervention_points,
        cf_ranking=getattr(args, "cf_ranking", "pareto"),
        cf_success_weight=getattr(args, "cf_success_weight", 1.0),
        cf_proximity_weight=getattr(args, "cf_proximity_weight", 0.15),
        cf_sparsity_weight=getattr(args, "cf_sparsity_weight", 0.10),
        cf_diversity_weight=getattr(args, "cf_diversity_weight", 0.10),
        cf_per_position_limit=getattr(args, "cf_per_position_limit", 5),
        cf_combination_budget=getattr(args, "cf_combination_budget", 100),
        cf_backbone_rerank=getattr(args, "cf_backbone_rerank", True),
        cf_prob_shift_weight=getattr(args, "cf_prob_shift_weight", 0.5),
        cf_soft_constraint_weight=getattr(args, "cf_soft_constraint_weight", 0.10),
        cf_minimal_relaxation=getattr(args, "cf_minimal_relaxation", True),
        cf_max_relaxed_constraints=getattr(args, "cf_max_relaxed_constraints", None),
        cf_non_relaxable_rule_ids=tuple(getattr(args, "cf_non_relaxable_rule_ids", ())),
        next_act_loss_weight=getattr(args, "next_act_loss_weight", 1.0),
        next_act_label_smoothing=getattr(args, "next_act_label_smoothing", 0.02),
        next_act_transition_prior_weight=getattr(args, "next_act_transition_prior_weight", 0.35),
        dfa_next_act_loss_weight=getattr(args, "dfa_next_act_loss_weight", 0.0),
        dfa_next_act_require_target_valid=getattr(
            args, "dfa_next_act_require_target_valid", True
        ),
        time_tail_weight=getattr(args, "time_tail_weight", 0.0),
        time_tail_quantile=getattr(args, "time_tail_quantile", 0.80),
        time_std_loss_weight=getattr(args, "time_std_loss_weight", 0.0),
        outcome_loss_weight=getattr(args, "outcome_loss_weight", 2.0),
        outcome_label_smoothing=getattr(args, "outcome_label_smoothing", 0.02),
        uwl_log_var_min=getattr(args, "uwl_log_var_min", -1.0),
        uwl_log_var_max=getattr(args, "uwl_log_var_max", 3.0),
    )
    cfg.early_stop_time_weight = float(getattr(args, "early_stop_time_weight", 0.0))
    if getattr(args, "next_act_transition_prior_weight", 0.35) > 0.0:
        cfg.next_act_transition_log_prior, n_transitions = build_next_act_transition_log_prior(
            train_samples,
            num_activities,
            smoothing=getattr(args, "next_act_transition_smoothing", 1.0),
        )
        print(
            "  Next-activity transition prior: "
            f"{n_transitions} train transitions, "
            f"weight={cfg.next_act_transition_prior_weight:.3f}"
        )
    if getattr(args, "next_act_class_weights", True):
        cfg.next_act_class_weights, next_act_counts = compute_next_act_class_weights(
            train_samples,
            num_activities,
            clip=getattr(args, "next_act_class_weight_clip", 5.0),
        )
        if cfg.next_act_class_weights is not None:
            present = int(np.sum(np.asarray(next_act_counts) > 0))
            print(
                "  Next-activity loss class weights: "
                f"{present}/{num_activities - 1} classes present, "
                f"min={min(cfg.next_act_class_weights):.3f}, "
                f"max={max(cfg.next_act_class_weights):.3f}"
            )
    cfg.outcome_class_weights = compute_outcome_class_weights(
        train_samples,
        clip=getattr(args, "outcome_class_weight_clip", 5.0),
    )
    if getattr(args, "balanced_outcome_sampler", False):
        cfg.outcome_class_weights = None
        print("  Outcome loss class weights: disabled (balanced sampler active)")
    if cfg.outcome_class_weights is not None:
        print("  Outcome loss class weights: "
              f"failure={cfg.outcome_class_weights[0]:.3f}, "
              f"success={cfg.outcome_class_weights[1]:.3f}")
    cfg.cf_generator = var_cfg.get("cf_generator", "default")
    if var_cfg.get("cf_intervention_points_override") is not None:
        cfg.cf_intervention_points = int(var_cfg["cf_intervention_points_override"])
    if var_cfg.get("cf_ranking_override"):
        cfg.cf_ranking = var_cfg["cf_ranking_override"]

    # ---- Search-DFA and independent Audit-DFA construction ----
    pattern_dict = None
    constraint_dict = None
    soft_constraint_dict = None
    search_calibration_df = pd.DataFrame()
    search_constraint_manifest_paths = {}
    knowledge_protocol_metadata = {}
    audit_rules = []
    audit_diagnostics = {}
    audit_manifest_paths = {}
    dfa = None
    multi_dfa_clusters = None
    audit_dfa = None                 # legacy argument; never used for paper audit
    audit_multi_dfa_clusters = None  # legacy argument; never used for paper audit
    act_encoder_for_dfa = act_encoder
    pattern_feat_dim = 0
    pattern_encoder = None
    pattern_discovery_samples = []
    pattern_calibration_samples = []

    if not var_cfg.get("is_baseline"):
        pattern_discovery_samples = expand_prefix_samples(
            symbolic_discovery_cases, prefix_mode, max_prefixes_per_case
        )
        pattern_calibration_samples = expand_prefix_samples(
            symbolic_calibration_cases, prefix_mode, max_prefixes_per_case
        )
        print(
            "  Symbolic protocol: "
            f"discovery={len(discovery_cases)} complete cases/"
            f"{len(pattern_discovery_samples)} prefixes, "
            f"calibration={len(calibration_cases)} complete cases/"
            f"{len(pattern_calibration_samples)} prefixes, "
            f"audit={len(audit_cases)} complete cases"
        )

        # Prediction and search knowledge have disjoint targets and roles.
        try:
            (
                pattern_dict,
                constraint_dict,
                soft_constraint_dict,
                dfa,
                search_calibration_df,
                discovered_pattern_feat_dim,
                multi_dfa_clusters,
                search_constraint_manifest_paths,
                knowledge_protocol_metadata,
                pattern_encoder,
            ) = run_pattern_pipeline(
                pattern_discovery_samples,
                act_encoder,
                dataset_name,
                dfa_max_constraints=cfg.dfa_max_constraints,
                all_samples=all_samples,
                calibration_samples=pattern_calibration_samples,
                audit_seed=getattr(args, "audit_seed", 2026),
                constraint_policy=var_cfg.get(
                    "constraint_policy",
                    getattr(args, "constraint_policy", "calibrated"),
                ),
                prediction_top_k=getattr(args, "prediction_pattern_top_k", 32),
                search_pool_size=getattr(args, "search_pattern_pool_size", 64),
                search_min_calibration_support=getattr(
                    args, "search_min_calibration_support", 0.05
                ),
                search_max_calibration_support=getattr(
                    args, "search_max_calibration_support", 1.00
                ),
                search_min_stability=getattr(args, "search_min_stability", 0.85),
                search_soft_constraints=getattr(args, "search_soft_constraints", 20),
                search_min_hard_constraints=getattr(
                    args, "search_min_hard_constraints", 3
                ),
                search_fallback_min_case_support=getattr(
                    args, "search_fallback_min_case_support", 0.50
                ),
                search_fallback_min_prefix_support=getattr(
                    args, "search_fallback_min_prefix_support", 0.10
                ),
                search_fallback_min_stability=getattr(
                    args, "search_fallback_min_stability", 0.70
                ),
                search_support_confidence_z=getattr(
                    args, "search_support_confidence_z", 1.96
                ),
                dfa_max_states=cfg.dfa_max_states,
            )
            if var_cfg["use_pattern"]:
                pattern_feat_dim = discovered_pattern_feat_dim
                cfg.pattern_feat_dim = discovered_pattern_feat_dim
            print(
                f"  Prediction patterns: {len(pattern_dict or {})}; "
                f"hard search constraints: {len(constraint_dict or {})}; "
                f"soft search patterns: {len(soft_constraint_dict or {})}; "
                f"{_format_dfa_status(dfa, multi_dfa_clusters)}"
            )
        except Exception as e:
            traceback.print_exc()
            print(f"  Role-separated pattern pipeline failed: {e}")
            cfg.pattern_feat_dim = 0

        # Audit knowledge uses a pre-specified, globally constraining Declare
        # family on complete held-out cases.  No result label or search score is
        # used to select these rules.
        try:
            from audit_constraints import (
                SUPPORTED_TEMPLATES,
                attach_external_rule_statistics,
                compile_audit_rules,
                discover_audit_rules,
                load_external_audit_rules,
                merge_audit_rules,
                persist_audit_manifest,
            )

            # Audit selection is deliberately independent of the current
            # search-policy variant. Exact overlaps are removed later from
            # the search side, never from the audit rule set.
            search_formulas = []
            audit_source = getattr(args, "audit_rule_source", "heldout")
            discovered_rules = []
            discovery_diag = {}
            if audit_source in ("heldout", "hybrid"):
                discovered_rules, discovery_diag = discover_audit_rules(
                    symbolic_audit_cases,
                    act_encoder,
                    templates=getattr(args, "audit_templates", SUPPORTED_TEMPLATES),
                    min_activation_cases=getattr(args, "audit_min_activation_cases", 10),
                    min_activation_fraction=getattr(args, "audit_min_activation_fraction", 0.02),
                    min_confidence=getattr(args, "audit_min_confidence", 0.80),
                    max_activities=getattr(args, "audit_max_activities", 40),
                    max_constraints=getattr(args, "audit_max_constraints", 20),
                    search_formulas=search_formulas,
                )

            external_rules = []
            external_path = getattr(args, "audit_rule_file", None)
            if audit_source in ("external", "hybrid"):
                if not external_path:
                    raise ValueError(
                        f"--audit_rule_source {audit_source} requires --audit_rule_file"
                    )
                external_rules = load_external_audit_rules(
                    external_path, dataset_name=dataset_name
                )
                external_rules = attach_external_rule_statistics(
                    external_rules, symbolic_audit_cases, act_encoder
                )

            selected_rules, merge_overlap = merge_audit_rules(
                discovered_rules,
                external_rules,
                search_formulas=search_formulas,
                max_constraints=getattr(args, "audit_max_constraints", 20),
            )
            audit_rules, compile_diag = compile_audit_rules(
                selected_rules,
                dfa_max_states=getattr(args, "audit_dfa_max_states", 500),
            )
            total_overlap = int(
                discovery_diag.get("formula_overlap_count", 0)
            ) + int(merge_overlap)
            if not selected_rules:
                raise RuntimeError(
                    "No independent audit rule met the pre-specified thresholds."
                )
            if compile_diag.get("failed", 0) or len(audit_rules) != len(selected_rules):
                raise RuntimeError(
                    "Independent Audit-DFA compilation must be all-or-nothing: "
                    f"{compile_diag.get('failures', [])}"
                )
            if total_overlap != 0:
                raise RuntimeError("Search/Audit formula overlap must be exactly zero.")
            audit_diagnostics = dict(discovery_diag)
            audit_diagnostics.update({
                "dataset": dataset_name,
                "source_mode": audit_source,
                "dataset_case_count": len(samples),
                "split_seed": split_seed,
                "audit_seed": int(getattr(args, "audit_seed", 2026)),
                "audit_fraction": float(getattr(args, "audit_fraction", 0.25)),
                "external_rule_count": len(external_rules),
                "selected_rule_count": len(selected_rules),
                "compiled_rule_count": len(audit_rules),
                "formula_overlap_count": total_overlap,
                "compile": compile_diag,
                "case_grain": "complete_trace",
                "selection_uses_outcome_label": False,
                "audit_selection_independent_of_search_policy": True,
            })
            (
                constraint_dict,
                soft_constraint_dict,
                search_calibration_df,
                demoted_overlap_ids,
            ) = _isolate_search_constraints_from_audit(
                constraint_dict,
                soft_constraint_dict,
                search_calibration_df,
                audit_rules,
                max_soft_constraints=getattr(args, "search_soft_constraints", 20),
            )
            audit_diagnostics.update({
                "search_formula_overlap_count_before_isolation": len(
                    demoted_overlap_ids
                ),
                "formula_overlap_count": 0,
                "search_audit_formula_sets_disjoint": True,
            })
            final_hard_mask = (
                search_calibration_df["constraint_role"].eq("hard")
                if not search_calibration_df.empty
                and "constraint_role" in search_calibration_df.columns
                else pd.Series(False, index=search_calibration_df.index)
            )
            final_selection_reason = (
                search_calibration_df["hard_selection_reason"]
                if not search_calibration_df.empty
                and "hard_selection_reason" in search_calibration_df.columns
                else pd.Series("", index=search_calibration_df.index)
            )
            knowledge_protocol_metadata.update({
                "hard_constraint_count": len(constraint_dict),
                "soft_constraint_count": len(soft_constraint_dict),
                "strict_hard_constraint_count": int(
                    (final_hard_mask & final_selection_reason.eq("strict")).sum()
                ),
                "fallback_hard_constraint_count": int(
                    (final_hard_mask & final_selection_reason.eq(
                        "stable_anchor_fallback"
                    )).sum()
                ),
                "audit_overlap_demoted_count": len(demoted_overlap_ids),
                "search_audit_formula_sets_disjoint": True,
            })
            if demoted_overlap_ids:
                dfa, multi_dfa_clusters = _compile_constraint_set(
                    constraint_dict,
                    act_encoder,
                    label="audit-isolated search",
                    dfa_max_states=cfg.dfa_max_states,
                )
                print(
                    "  Search/Audit isolation: demoted "
                    f"{len(demoted_overlap_ids)} exact hard-rule overlap(s)"
                )
            manifest_dir = os.path.dirname(
                search_constraint_manifest_paths.get("json", "")
            )
            if manifest_dir:
                from pattern_discovery.pattern_mining import persist_constraint_manifest

                search_constraint_manifest_paths = persist_constraint_manifest(
                    manifest_dir,
                    search_calibration_df,
                    knowledge_protocol_metadata,
                )

            audit_manifest_paths = persist_audit_manifest(
                audit_rules if audit_rules else selected_rules,
                audit_diagnostics,
                getattr(args, "audit_manifest_dir", "./audit_manifests"),
                dataset_name,
            )
            print(
                "  Independent audit rules: "
                f"selected={len(selected_rules)}, compiled={len(audit_rules)}, "
                f"overlap={audit_diagnostics['formula_overlap_count']}, "
                f"manifest={audit_manifest_paths.get('json', '')}"
            )
        except Exception as e:
            if getattr(args, "require_independent_audit", True):
                raise RuntimeError(
                    "Required independent Audit-DFA construction failed"
                ) from e
            traceback.print_exc()
            print(f"  Independent Audit-DFA construction failed: {e}")
            audit_rules = []
            audit_diagnostics = {
                "dataset": dataset_name,
                "source_mode": getattr(args, "audit_rule_source", "heldout"),
                "case_grain": "complete_trace",
                "selection_uses_outcome_label": False,
                "failure": f"{type(e).__name__}: {e}",
            }

    pattern_feature_density = float("nan")
    pattern_feature_nonconstant_count = 0
    # Zero out pattern features when Search pattern features are unavailable.
    if pattern_feat_dim == 0:
        for s in all_samples:
            s['pattern_features'] = []
    else:
        train_pattern_matrix = np.asarray(
            [s.get("pattern_features", []) for s in train_samples],
            dtype=np.float32,
        )
        expected_shape = (len(train_samples), int(pattern_feat_dim))
        if train_pattern_matrix.shape != expected_shape:
            raise ValueError(
                "Training pattern feature matrix has shape "
                f"{train_pattern_matrix.shape}, expected {expected_shape}"
            )
        column_rates = train_pattern_matrix.mean(axis=0)
        pattern_feature_density = float(train_pattern_matrix.mean())
        pattern_feature_nonconstant_count = int(
            np.sum((column_rates > 0.0) & (column_rates < 1.0))
        )
        print(
            "  Pattern feature diagnostics: "
            f"dim={pattern_feat_dim}, density={pattern_feature_density:.3f}, "
            f"nonconstant={pattern_feature_nonconstant_count}/{pattern_feat_dim}"
        )

    # ---- Declare DFA construction ----
    if var_cfg.get("is_declare"):
        print("  Building Declare template DFA...")
        try:
            from baselines import build_declare_dfa
            dfa, declare_clusters, n_f, decl_enc = build_declare_dfa(train_samples, act_encoder)
            if declare_clusters:
                multi_dfa_clusters = declare_clusters
            # Sanitized encoder for DFA action pruning (details use original encoder).
            if decl_enc is not None:
                act_encoder_for_dfa = decl_enc
            print(f"  Declare DFA: {'OK' if dfa or multi_dfa_clusters else 'N/A'} "
                  f"({n_f}  formulas")
        except Exception as e:
            print(f"  Declare DFA constructionfailed:  {e}")
            dfa, multi_dfa_clusters = None, None

    # ---- Train-only Search-DFA guidance for next-activity prediction ----
    configured_dfa_aux_weight = max(0.0, float(
        var_cfg.get(
            "dfa_next_act_loss_weight_override",
            getattr(args, "dfa_next_act_loss_weight", 0.0),
        )
    ))
    dfa_aux_variant_enabled = bool(
        var_cfg.get("use_dfa", True)
        and var_cfg.get("use_dfa_aux_loss", False)
        and not var_cfg.get("is_baseline")
        and not var_cfg.get("is_declare")
    )
    requested_dfa_aux_weight = (
        configured_dfa_aux_weight if dfa_aux_variant_enabled else 0.0
    )
    dfa_aux_clusters = []
    if multi_dfa_clusters:
        dfa_aux_clusters = multi_dfa_clusters
    elif dfa is not None:
        dfa_aux_clusters = [{"dfa": dfa, "pids": []}]

    dfa_aux_diagnostics = {
        "source": "search_dfa",
        "mode": "strict_acceptance_progress",
        "selection_uses_outcome_label": False,
        "uses_next_activity_target_for_safety_gate": bool(
            cfg.dfa_next_act_require_target_valid
        ),
        "prefix_count": len(train_samples),
        "state_applicable_count": 0,
        "no_progress_action_count": 0,
        "informative_mask_count": 0,
        "target_compatible_count": 0,
        "target_incompatible_count": 0,
        "aux_applicable_count": 0,
        "aux_applicable_fraction": 0.0,
        "mean_progress_action_fraction": float("nan"),
        "mean_valid_action_fraction": float("nan"),
    }
    if (
        requested_dfa_aux_weight > 0.0
        and cfg.next_act_loss_weight > 0.0
        and dfa_aux_variant_enabled
        and dfa_aux_clusters
    ):
        dfa_aux_diagnostics = annotate_dfa_next_activity_supervision(
            train_samples,
            dfa_aux_clusters,
            act_encoder,
            num_activities,
            require_target_valid=cfg.dfa_next_act_require_target_valid,
        )
        if dfa_aux_diagnostics["aux_applicable_count"] > 0:
            cfg.dfa_next_act_loss_weight = requested_dfa_aux_weight
        else:
            cfg.dfa_next_act_loss_weight = 0.0
    else:
        cfg.dfa_next_act_loss_weight = 0.0

    print(
        "  Search-DFA next-activity guidance: "
        f"mode={dfa_aux_diagnostics['mode']}, "
        f"active={cfg.dfa_next_act_loss_weight > 0.0}, "
        f"weight={cfg.dfa_next_act_loss_weight:.3f}, "
        f"eligible={dfa_aux_diagnostics['aux_applicable_count']}/"
        f"{len(train_samples)}, "
        f"no_progress={dfa_aux_diagnostics['no_progress_action_count']}, "
        f"target_conflicts={dfa_aux_diagnostics['target_incompatible_count']}"
    )
    # ---- Build DataLoader ----
    (train_loader, train_eval_loader, val_loader, test_loader,
     train_ds, val_ds, test_ds) = build_dataloaders(
        train_samples, val_samples, test_samples,
        cfg, args.batch_size, pattern_feat_dim,
        num_workers=getattr(args, "num_workers", 0),
        pin_memory=(cfg.device.startswith("cuda") and getattr(args, "pin_memory", False)),
        balanced_outcome_sampler=getattr(args, "balanced_outcome_sampler", False),
        persistent_workers=getattr(args, "persistent_workers", False),
    )

    ref_matrix = build_reference_matrix(train_ds, cfg.max_seq_len)
    L = cfg.max_seq_len
    cont_idx = list(range(L, 2 * L))
    cat_idx = list(range(L))

    # ---- External Baseline (XGBoost / LSTM) ----
    if var_cfg.get("is_baseline"):
        bt = var_cfg["baseline_type"]
        print(f"  Running external baseline: {bt}")

        if bt == "lstm_mtl":
            from lstm_baseline import LSTMMultitaskBaseline
            baseline = LSTMMultitaskBaseline(
                epochs=args.epochs, lr=cfg.learning_rate, device=cfg.device,
                batch_size=args.batch_size,
                patience=getattr(args, "lstm_patience", 5),
                hidden_dim=getattr(args, "lstm_hidden_dim", 128),
                num_layers=getattr(args, "lstm_num_layers", 2),
                dropout=getattr(args, "lstm_dropout", 0.1),
                use_attributes=getattr(args, "lstm_use_attributes", True),
            ).fit(train_samples, val_samples, cfg)
        else:
            from baselines import XGBoostBaseline, LSTMBaseline, ProcessTransformerBaseline
            if bt == "xgboost":
                baseline = XGBoostBaseline().fit(train_samples, val_samples, cfg)
            elif bt == "process_transformer":
                baseline = ProcessTransformerBaseline(
                    epochs=args.epochs, device=cfg.device, batch_size=args.batch_size
                ).fit(train_samples, val_samples, cfg)
            elif bt == "lstm":
                baseline = LSTMBaseline(epochs=args.epochs, device=cfg.device).fit(train_samples, val_samples, cfg)
            else:
                raise ValueError(f"Unknown external baseline: {bt}")

        test_logs = baseline.predict(test_samples, cfg)
        print(f"  Test:  acc={test_logs['outcome_acc']:.4f}  "
              f"mae={test_logs['next_time_mae']:.3f}  "
              f"act_acc={test_logs['next_act_acc']:.3f}  "
              f"act_top3={test_logs.get('next_act_top3_acc', float('nan')):.3f}  "
              f"act_top5={test_logs.get('next_act_top5_acc', float('nan')):.3f}")

        result = {
            "dataset": dataset_name, "variant": variant,
            "dataset_family": infer_dataset_family(dataset_name),
            "seed": seed,
            "split_seed": split_seed,
            "evaluation_protocol_version": 2,
            "num_cases": len(samples), "num_prefix_samples": len(all_samples),
            "train_prefix_samples": len(train_samples),
            "val_prefix_samples": len(val_samples),
            "test_prefix_samples": len(test_samples),
            "prefix_mode": prefix_mode,
            "max_prefixes_per_case": max_prefixes_per_case if max_prefixes_per_case else 0,
            "num_activities": num_activities,
            **activity_diag,
            "avg_trace_len": float(np.mean([s["length"] for s in samples])),
            "avg_prefix_len": float(np.mean([s["prefix_len"] for s in all_samples])) if all_samples else float("nan"),
            "positive_rate": float(np.mean([s["outcome"] for s in samples])),
            "pattern_count": 0, "constraint_count": 0,
            "pattern_feature_protocol_version": 5,
            "pattern_fusion_protocol_version": int(
                cfg.pattern_fusion_protocol_version
            ),
            "pattern_fusion_target": cfg.pattern_fusion_target,
            "pattern_feature_active": False,
            "pattern_feature_dim": 0,
            "pattern_feature_density": float("nan"),
            "pattern_feature_nonconstant_count": 0,
            "pattern_discovery_prefixes": 0,
            "has_dfa": False, "dfa_available": False, "has_multi_dfa": False,
            "dfa_next_act_aux_protocol_version": int(
                cfg.dfa_next_act_aux_protocol_version
            ),
            "dfa_next_act_aux_requested_weight": requested_dfa_aux_weight,
            "dfa_next_act_aux_effective_weight": 0.0,
            "dfa_next_act_aux_active": False,
            "dfa_next_act_aux_require_target_valid": bool(
                cfg.dfa_next_act_require_target_valid
            ),
            "dfa_next_act_aux_source": "search_dfa",
            "dfa_next_act_aux_mode": "strict_acceptance_progress",
            "dfa_next_act_aux_applicable_count": 0,
            "dfa_next_act_aux_applicable_fraction": 0.0,
            "dfa_next_act_aux_state_applicable_count": 0,
            "dfa_next_act_aux_no_progress_action_count": 0,
            "dfa_next_act_aux_informative_mask_count": 0,
            "dfa_next_act_aux_target_compatible_count": 0,
            "dfa_next_act_aux_target_incompatible_count": 0,
            "dfa_next_act_aux_mean_progress_action_fraction": float("nan"),
            "dfa_next_act_aux_mean_valid_action_fraction": float("nan"),
            "dfa_next_act_aux_final_train_loss": float("nan"),
            "dfa_next_act_aux_final_allowed_mass": float("nan"),
            "dfa_next_act_aux_selection_uses_outcome_label": False,
            "dfa_next_act_aux_uses_target_safety_gate": bool(
                cfg.dfa_next_act_require_target_valid
            ),
            "n_static_cat": cfg.n_static_cat, "n_static_num": cfg.n_static_num,
            "n_dynamic_cat": cfg.n_dynamic_cat, "n_dynamic_num": cfg.n_dynamic_num,
            "bp17_anti_leak": attr_dims.get("bp17_anti_leak", False),
            "train_only_preprocessing": attr_dims.get("train_only_preprocessing", False),
            "cf_generator": getattr(cfg, "cf_generator", "default"),
            "cf_topk": cfg.cf_topk,
            "cf_intervention_points": cfg.cf_intervention_points,
            "cf_ranking": cfg.cf_ranking,
            "cf_success_weight": cfg.cf_success_weight,
            "cf_proximity_weight": cfg.cf_proximity_weight,
        "cf_sparsity_weight": cfg.cf_sparsity_weight,
        "cf_diversity_weight": cfg.cf_diversity_weight,
        "cf_backbone_rerank": bool(cfg.cf_backbone_rerank),
        "cf_prob_shift_weight": cfg.cf_prob_shift_weight,
        "time_tail_weight": cfg.time_tail_weight,
        "time_tail_quantile": cfg.time_tail_quantile,
        "time_std_loss_weight": cfg.time_std_loss_weight,
        "epochs": args.epochs,
            "best_val_acc": getattr(baseline, "best_val_acc", 0.0),
            "best_val_auc": getattr(baseline, "best_val_auc", float("nan")),
            "stopped_early": getattr(baseline, "stopped_early", False),
            "actual_epochs": getattr(baseline, "actual_epochs", args.epochs),
            **{f"test_{k}": v for k, v in test_logs.items()},
        }
        if bt == "lstm_mtl":
            artifact_dir = os.path.join(args.output_dir, "lstm_artifacts")
            os.makedirs(artifact_dir, exist_ok=True)
            stem = f"{_safe_filename_part(dataset_name)}_seed{seed}"
            checkpoint = os.path.join(artifact_dir, stem + ".pt")
            baseline.save_checkpoint(checkpoint)
            pd.DataFrame(baseline.history).to_csv(
                os.path.join(artifact_dir, stem + "_history.csv"), index=False)
            pd.DataFrame(baseline.prediction_rows).to_csv(
                os.path.join(artifact_dir, stem + "_predictions.csv"), index=False)
            manifest = {
                "dataset": dataset_name, "seed": seed, "split_seed": split_seed,
                "prefix_mode": prefix_mode, "max_prefixes_per_case": max_prefixes_per_case,
                "max_seq_len": cfg.max_seq_len,
                "activity_classes": [str(x) for x in act_encoder.classes_],
                "time_target": "next-event case-relative timestamp, normalized with train maximum, clipped to [0,1]",
                "splits": {},
            }
            for name, cases, prefixes in (
                ("train", train_cases, train_samples), ("val", val_cases, val_samples),
                ("test", test_cases, test_samples),
            ):
                manifest["splits"][name] = {
                    "case_ids": [str(s["case_id"]) for s in cases],
                    "prefix_count": len(prefixes),
                    "truncated_prefix_count": sum(s["length"] > cfg.max_seq_len for s in prefixes),
                }
            with open(os.path.join(artifact_dir, stem + "_split.json"), "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
            result.update({
                "prediction_baseline_protocol_version": baseline.protocol_version,
                "baseline_model_class": "LSTMMultitaskModel",
                "baseline_hidden_dim": baseline.cfg.hidden_dim,
                "baseline_num_layers": baseline.cfg.num_layers,
                "baseline_dropout": baseline.cfg.dropout,
                "baseline_use_attributes": baseline.use_attributes,
                "baseline_batch_size": baseline.batch_size,
                "baseline_learning_rate": baseline.lr,
                "baseline_weight_decay": 0.0,
                "baseline_loss": "CE_outcome+CE_activity+SmoothL1_next_time",
                "baseline_monitor": "val_outcome_auc_then_macro_f1_then_acc",
                "baseline_patience": baseline.patience,
                "baseline_best_epoch": baseline.best_epoch,
                "baseline_parameter_count": sum(p.numel() for p in baseline.model.parameters()),
                "baseline_checkpoint": checkpoint,
                **{key: getattr(baseline.cfg, key) for key in
                   ("n_static_cat", "n_static_num", "n_dynamic_cat", "n_dynamic_num")},
            })
        # Fill CF columns with NaN (baseline does not generate CF)
        for cf_col in ["cf_nbr_cfs", "cf_compliance_rate", "cf_pattern_hit_rate", "cf_implausibility",
                        "cf_distance_l2j", "cf_diversity_l2j", "cf_avg_nbr_changes",
                        "cf_gen_time_per_sample", "cf_dfa_prune_rate", "cf_dfa_action_prune_rate",
                        "cf_dfa_candidate_prefilter_rate", "cf_dfa_full_reject_rate",
                        "cf_dfa_effective_filter_rate",
                        "cf_dfa_eval_available", "cf_dfa_enforced", "cf_compliance_applicable",
                        "cf_metric_schema_version", "cf_count", "cf_validity_rate",
                        "cf_success_rate", "cf_prediction_flip_rate", "cf_mean_prob_shift",
                        "cf_proximity_l2j", "cf_proximity_mh", "cf_sparsity_rate",
                        "cf_sparsity_count", "cf_diversity_score", "cf_plausibility_error",
                        "cf_plausibility_score", "cf_generation_time_sec",
                        "cf_audit_compliance_rate", "cf_audit_all_applicable_rate",
                        "cf_search_constraint_satisfaction_rate",
                        "cf_audit_compliance_defined", "cf_search_compliance_defined",
                        "cf_compliance_defined", "cf_compliance_enforced"]:
            if cf_col not in result:
                result[cf_col] = float("nan")
        return result

    # ---- Training ----
    if cfg.cf_non_relaxable_rule_ids or cfg.cf_max_relaxed_constraints is not None:
        if not var_cfg.get("use_dfa", True) or not flatten_cluster_rule_dfas(multi_dfa_clusters):
            raise ValueError("Exemption policy requires an enforced atomic Search-DFA variant")
        evaluate_minimal_constraint_relaxation(
            [], multi_dfa_clusters, cfg.cf_max_relaxed_constraints, cfg.cf_non_relaxable_rule_ids)
    backbone = MultiTaskProcessModel(cfg)
    cf_model = CounterfactualRecommender(cfg, backbone)
    if pattern_encoder is not None and var_cfg.get("use_pattern", False):
        cf_model.set_pattern_encoder(pattern_encoder)

    # Inject multi-DFA (must be after cf_model creation)
    if multi_dfa_clusters and var_cfg.get("use_dfa", True):
        _enc = act_encoder_for_dfa if var_cfg.get("is_declare") else act_encoder
        cf_model.set_multi_dfa(multi_dfa_clusters, _enc)

    trainer = Trainer(cfg, backbone, cf_model)

    # Freeze uncertainty weights when the no-UWL variant is selected.
    if not var_cfg["use_uwl"]:
        trainer.task_log_vars.requires_grad = False

    best_val_acc = 0.0
    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_logs = trainer.train_epoch(train_loader)
        val_logs = trainer.evaluate(val_loader)
        dur = time.time() - t0

        if args.verbose:
            print(f"  [Epoch {epoch:3d}] "
                  f"loss={train_logs['loss_total']:.4f} "
                  f"base={train_logs['loss_base_total']:.4f} "
                  f"val_acc={val_logs['outcome_acc']:.4f} "
                  f"val_f1={val_logs['outcome_f1']:.3f} "
                  f"val_auc={val_logs['outcome_auc']:.3f} "
                  f"val_auc_inv={val_logs.get('outcome_auc_inverted', float('nan')):.3f} "
                  f"val_brier={val_logs.get('outcome_brier', float('nan')):.3f} "
                  f"val_ece={val_logs.get('outcome_ece', float('nan')):.3f} "
                  f"val_pos={val_logs.get('outcome_true_pos_rate', float('nan')):.3f}/"
                  f"{val_logs.get('outcome_pred_pos_rate', float('nan')):.3f} "
                  f"val_mae={val_logs['next_time_mae']:.3f} "
                  f"val_r2={val_logs.get('next_time_r2', float('nan')):.3f} "
                  f"({dur:.1f}s)")
            print("             "
                  f"base_losses act={train_logs['loss_next_act']:.4f} "
                  f"dfa_act={train_logs['loss_dfa_next_act']:.4f} "
                  f"time={train_logs['loss_next_time']:.4f} "
                  f"out={train_logs['loss_outcome']:.4f} "
                  f"cf_out={train_logs['loss_cf_outcome']:.4f} "
                  f"cf_time={train_logs['loss_cf_time']:.4f}")
            print("             "
                  f"uwl_terms act={train_logs['uwl_next_act']:.4f} "
                  f"time={train_logs['uwl_next_time']:.4f} "
                  f"out={train_logs['uwl_outcome']:.4f} "
                  f"cf_out={train_logs['uwl_cf_outcome']:.4f} "
                  f"cf_time={train_logs['uwl_cf_time']:.4f} "
                  f"log_vars=[{train_logs['w_0']:.2f}, {train_logs['w_1']:.2f}, "
                  f"{train_logs['w_2']:.2f}, {train_logs['w_3']:.2f}, {train_logs['w_4']:.2f}]")

        if val_logs['outcome_acc'] > best_val_acc:
            best_val_acc = val_logs['outcome_acc']

        monitor_score, monitor_name = select_monitor_score(val_logs, cfg)
        trainer.scheduler.step(monitor_score)

        if trainer.check_early_stop(monitor_score):
            stopped_early = True
            print(f"  Early stop at epoch {epoch} "
                  f"({monitor_name}={monitor_score:.4f}, patience={cfg.early_stop_patience})")
            break

    if trainer.best_state is not None:
        trainer.restore_best()
        if stopped_early:
            print(f"  Restored best validation state (monitor={trainer.best_val_score:.4f})")
        else:
            print(f"  Restored best validation state after final epoch "
                  f"(monitor={trainer.best_val_score:.4f})")

    # ---- Test evaluation ----
    train_fit_logs = {}
    if getattr(args, "fit_diagnostics", False):
        train_fit_logs = trainer.evaluate(train_eval_loader, record_history=False)

    test_logs = trainer.evaluate(test_loader, record_history=False)
    print(f"  Test:  acc={test_logs['outcome_acc']:.4f}  "
          f"bal_acc={test_logs.get('outcome_balanced_acc', float('nan')):.3f}  "
          f"brier={test_logs.get('outcome_brier', float('nan')):.3f}  "
          f"ece={test_logs.get('outcome_ece', float('nan')):.3f}  "
          f"mae={test_logs['next_time_mae']:.3f}  "
          f"rmse={test_logs.get('next_time_rmse', float('nan')):.3f}  "
          f"r2={test_logs.get('next_time_r2', float('nan')):.3f}  "
          f"corr={test_logs.get('next_time_corr', float('nan')):.3f}  "
          f"act_acc={test_logs['next_act_acc']:.3f}  "
          f"act_top3={test_logs.get('next_act_top3_acc', float('nan')):.3f}  "
          f"act_top5={test_logs.get('next_act_top5_acc', float('nan')):.3f}  "
          f"act_f1={test_logs.get('next_act_f1', float('nan')):.3f}")
    if train_fit_logs:
        print("  Fit diagnostics: "
              f"outcome_gap={train_fit_logs['outcome_acc'] - test_logs['outcome_acc']:.3f} "
              f"act_gap={train_fit_logs['next_act_acc'] - test_logs['next_act_acc']:.3f} "
              f"train_outcome={train_fit_logs['outcome_acc']:.3f} "
              f"train_act={train_fit_logs['next_act_acc']:.3f}")

    # ---- CF evaluation ----
    test_indices, evaluation_manifest_path = build_cf_evaluation_manifest(
        test_samples,
        dataset_name,
        manifest_dir=getattr(args, "evaluation_manifest_dir", "./evaluation_manifests"),
        max_samples=getattr(args, "cf_eval_samples", 0),
        split_seed=split_seed,
        evaluation_seed=getattr(args, "evaluation_seed", 2026),
    )
    search_dfa_available_for_eval = dfa is not None or (
        multi_dfa_clusters is not None and len(multi_dfa_clusters) > 0
    )
    audit_dfa_available_for_eval = bool(audit_rules)

    _dfa_enforced = bool(
        var_cfg.get("use_dfa", True) and search_dfa_available_for_eval
    )
    _dfa_to_use = dfa if _dfa_enforced else None
    _constraint_to_report = constraint_dict
    _soft_constraints_to_use = (
        soft_constraint_dict
        if var_cfg.get("use_soft_constraints", True) else None
    )
    if var_cfg.get("is_model_score_cf") or var_cfg.get("is_random"):
        _dfa_to_use = None
        _soft_constraints_to_use = None
        _dfa_enforced = False
    save_tag = f"{dataset_name}_{variant}_seed{seed}"
    cf_metrics, _ = evaluate_cfs(
        cf_model, backbone, test_ds, test_indices,
        cfg, L, cont_idx, cat_idx, ref_matrix,
        pattern_dict, _constraint_to_report, audit_rules, _dfa_to_use, act_encoder,
        save_prefix=save_tag,
        search_eval_dfa=dfa if search_dfa_available_for_eval else None,
        search_eval_multi_dfa_clusters=(
            multi_dfa_clusters if search_dfa_available_for_eval else None
        ),
        dfa_enforced=_dfa_enforced,
        output_dir=args.output_dir,
        dataset_name=dataset_name,
        variant=variant,
        seed=seed,
        soft_constraint_dict=_soft_constraints_to_use,
    )

    # Ablation: no_dfa
    if (not var_cfg["use_dfa"] and var_cfg.get("use_pattern", False)
            and not var_cfg.get("is_model_score_cf")):
        cf_metrics["cf_nbr_cfs_nodfa"] = cf_metrics.get("nbr_cfs", 0)

    # ---- Save prediction diagnostics for later visualization ----
    if not var_cfg.get("is_random"):
        save_prediction_outputs(
            backbone, test_ds, cfg, act_encoder, dataset_name, variant,
            args.output_dir,
            max_rows=getattr(args, "prediction_detail_limit", 5000),
            batch_size=getattr(args, "prediction_detail_batch_size", 256),
            split_name="test",
            seed=seed,
        )

    dfa_available = dfa is not None or (multi_dfa_clusters is not None and len(multi_dfa_clusters) > 0)
    audit_dfa_available = bool(audit_rules)
    dfa_active = bool(var_cfg.get("use_dfa", True) and dfa_available)
    compiled_search_rules = flatten_cluster_rule_dfas(multi_dfa_clusters)
    compile_relaxed_rule_ids = sorted({
        str(rule.get("rule_id", rule.get("pid", "")))
        for rule in compiled_search_rules
        if rule.get("compile_mode", "strict") != "strict"
    })
    compile_omitted_rule_ids = sorted({
        str(pid)
        for cluster in (multi_dfa_clusters or [])
        for pid in cluster.get("compile_omitted_pids", ())
    })
    if constraint_dict and compiled_search_rules:
        search_dfa_compile_coverage = (
            len(compiled_search_rules) / len(constraint_dict)
        )
        search_dfa_compile_complete = bool(
            len(compiled_search_rules) == len(constraint_dict)
        )
    elif constraint_dict and dfa is not None:
        search_dfa_compile_coverage = 1.0
        search_dfa_compile_complete = True
    elif constraint_dict:
        search_dfa_compile_coverage = 0.0
        search_dfa_compile_complete = False
        if not compile_omitted_rule_ids:
            compile_omitted_rule_ids = sorted(str(pid) for pid in constraint_dict)
    else:
        search_dfa_compile_coverage = float("nan")
        search_dfa_compile_complete = True
    if multi_dfa_clusters:
        search_constraint_semantics = "disjunction_of_conjunctive_clusters"
    elif dfa is not None:
        search_constraint_semantics = "single_conjunctive_dfa"
    else:
        search_constraint_semantics = "none"
    finite_val_aucs = [
        h.get("outcome_auc", float("nan"))
        for h in getattr(trainer, "val_history", [])
        if np.isfinite(h.get("outcome_auc", float("nan")))
    ]
    best_val_auc = max(finite_val_aucs) if finite_val_aucs else float("nan")

    # ---- Summary----
    result = {
        "dataset": dataset_name,
        "variant": variant,
        "dataset_family": infer_dataset_family(dataset_name),
        "seed": seed,
        "split_seed": split_seed,
        "evaluation_protocol_version": 2,
        "num_cases": len(samples),
        "num_prefix_samples": len(all_samples),
        "train_prefix_samples": len(train_samples),
        "val_prefix_samples": len(val_samples),
        "test_prefix_samples": len(test_samples),
        "prefix_mode": prefix_mode,
        "max_prefixes_per_case": max_prefixes_per_case if max_prefixes_per_case else 0,
        "num_activities": num_activities,
        **activity_diag,
        "avg_trace_len": float(np.mean([s["length"] for s in samples])),
        "avg_prefix_len": float(np.mean([s["prefix_len"] for s in all_samples])) if all_samples else float("nan"),
        "positive_rate": float(np.mean([s["outcome"] for s in samples])),
        "pattern_count": len(pattern_dict) if pattern_dict else 0,
        "constraint_count": len(constraint_dict) if constraint_dict else 0,
        "prediction_pattern_count": len(pattern_dict or {}),
        "prediction_pattern_target": "next_activity",
        "search_pattern_pool_count": knowledge_protocol_metadata.get(
            "search_pattern_pool_count", 0
        ),
        "search_hard_constraint_count": len(constraint_dict or {}),
        "search_soft_constraint_count": len(soft_constraint_dict or {}),
        "search_constraint_policy": knowledge_protocol_metadata.get(
            "constraint_policy", getattr(args, "constraint_policy", "calibrated")
        ),
        "search_selection_mode": "label_free",
        "search_selection_uses_outcome_label": False,
        "search_soft_constraint_active": bool(_soft_constraints_to_use),
        "search_protocol_signature": knowledge_protocol_metadata.get(
            "protocol_signature", ""
        ),
        "search_audit_overlap_demoted_count": knowledge_protocol_metadata.get(
            "audit_overlap_demoted_count", 0
        ),
        "search_audit_formula_sets_disjoint": bool(
            knowledge_protocol_metadata.get(
                "search_audit_formula_sets_disjoint", False
            )
        ),
        "pattern_discovery_fold_signature": knowledge_protocol_metadata.get(
            "discovery_fold_signature", ""
        ),
        "pattern_calibration_fold_signature": knowledge_protocol_metadata.get(
            "calibration_fold_signature", ""
        ),
        "search_constraint_manifest": search_constraint_manifest_paths.get(
            "json", ""
        ),
        "knowledge_protocol_version": 2,
        "search_support_granularity": knowledge_protocol_metadata.get(
            "support_granularity", "complete_case_any_prefix"
        ),
        "search_strict_hard_constraint_count": knowledge_protocol_metadata.get(
            "strict_hard_constraint_count", 0
        ),
        "search_fallback_hard_constraint_count": knowledge_protocol_metadata.get(
            "fallback_hard_constraint_count", 0
        ),
        "pattern_feature_protocol_version": 5,
        "pattern_fusion_protocol_version": int(
            cfg.pattern_fusion_protocol_version
        ),
        "pattern_fusion_target": cfg.pattern_fusion_target,
        "pattern_feature_active": bool(
            var_cfg.get("use_pattern", False) and pattern_feat_dim > 0
        ),
        "pattern_feature_dim": int(pattern_feat_dim),
        "pattern_feature_density": pattern_feature_density,
        "pattern_feature_nonconstant_count": pattern_feature_nonconstant_count,
        "pattern_discovery_prefixes": len(pattern_discovery_samples),
        "pattern_calibration_prefixes": len(pattern_calibration_samples),
        "pattern_dropout": float(cfg.pattern_dropout),
        "pattern_gate_init_bias": float(cfg.pattern_gate_init_bias),
        "audit_constraint_count": len(audit_rules),
        "audit_rule_source": audit_diagnostics.get(
            "source_mode", getattr(args, "audit_rule_source", "heldout")
        ),
        "audit_rule_candidate_count": audit_diagnostics.get("candidate_count", 0),
        "audit_rule_eligible_count": audit_diagnostics.get("eligible_count", 0),
        "audit_rule_compiled_count": audit_diagnostics.get("compiled_rule_count", 0),
        "audit_formula_overlap_count": audit_diagnostics.get("formula_overlap_count", 0),
        "audit_case_grain": audit_diagnostics.get("case_grain", "complete_trace"),
        "audit_selection_uses_outcome_label": False,
        "audit_selection_independent_of_search_policy": bool(
            audit_diagnostics.get("audit_selection_independent_of_search_policy", False)
        ),
        "audit_rule_manifest": audit_manifest_paths.get("json", ""),
        "pattern_discovery_case_count": len(discovery_cases),
        "pattern_calibration_case_count": len(calibration_cases),
        "pattern_audit_case_count": len(audit_cases),
        "symbolic_folds_case_disjoint": bool(
            discovery_cases.isdisjoint(calibration_cases)
            and discovery_cases.isdisjoint(audit_cases)
            and calibration_cases.isdisjoint(audit_cases)
        ),
        "constraint_calibration_fraction": float(
            getattr(args, "constraint_calibration_fraction", 0.15)
        ),
        "search_min_calibration_support": float(
            getattr(args, "search_min_calibration_support", 0.05)
        ),
        "search_max_calibration_support": float(
            getattr(args, "search_max_calibration_support", 1.00)
        ),
        "search_min_stability": float(getattr(args, "search_min_stability", 0.85)),
        "search_min_hard_constraints": int(
            getattr(args, "search_min_hard_constraints", 3)
        ),
        "search_fallback_min_case_support": float(
            getattr(args, "search_fallback_min_case_support", 0.50)
        ),
        "search_fallback_min_prefix_support": float(
            getattr(args, "search_fallback_min_prefix_support", 0.10)
        ),
        "search_fallback_min_stability": float(
            getattr(args, "search_fallback_min_stability", 0.70)
        ),
        "search_support_confidence_z": float(
            getattr(args, "search_support_confidence_z", 1.96)
        ),
        "audit_fraction": float(getattr(args, "audit_fraction", 0.25)),
        "has_dfa": dfa_active,
        "dfa_available": dfa_available,
        "audit_dfa_available": audit_dfa_available,
        "has_multi_dfa": bool(var_cfg.get("use_dfa", True) and multi_dfa_clusters is not None and len(multi_dfa_clusters) > 0),
        "search_dfa_compile_coverage": search_dfa_compile_coverage,
        "search_dfa_compile_complete": search_dfa_compile_complete,
        "search_dfa_compile_relaxed_count": len(compile_relaxed_rule_ids),
        "search_dfa_compile_relaxed_rule_ids": ";".join(
            compile_relaxed_rule_ids
        ),
        "search_dfa_compile_omitted_count": len(compile_omitted_rule_ids),
        "search_dfa_compile_omitted_rule_ids": ";".join(compile_omitted_rule_ids),
        "n_static_cat": cfg.n_static_cat, "n_static_num": cfg.n_static_num,
        "n_dynamic_cat": cfg.n_dynamic_cat, "n_dynamic_num": cfg.n_dynamic_num,
        "bp17_anti_leak": attr_dims.get("bp17_anti_leak", False),
        "train_only_preprocessing": attr_dims.get("train_only_preprocessing", False),
        "cf_generator": getattr(cfg, "cf_generator", "default"),
        "cf_search_strategy": getattr(
            cfg, "cf_search_strategy", "minimal_constraint_relaxation"
        ),
        "cf_search_protocol_version": int(cfg.cf_search_protocol_version),
        "cf_minimal_relaxation": bool(cfg.cf_minimal_relaxation),
        "cf_max_relaxed_constraints": cfg.cf_max_relaxed_constraints,
        "cf_non_relaxable_rule_ids": ";".join(cfg.cf_non_relaxable_rule_ids),
        "search_constraint_semantics": search_constraint_semantics,
        "training_objective_protocol_version": 2,
        "cf_per_position_limit": cfg.cf_per_position_limit,
        "cf_combination_budget": cfg.cf_combination_budget,
        "cf_evaluation_sample_count": len(test_indices),
        "cf_evaluation_manifest": evaluation_manifest_path,
        "cf_topk": cfg.cf_topk,
        "cf_intervention_points": cfg.cf_intervention_points,
        "cf_ranking": cfg.cf_ranking,
        "cf_success_weight": cfg.cf_success_weight,
        "cf_proximity_weight": cfg.cf_proximity_weight,
        "cf_sparsity_weight": cfg.cf_sparsity_weight,
        "cf_diversity_weight": cfg.cf_diversity_weight,
        "cf_soft_constraint_weight": cfg.cf_soft_constraint_weight,
        "cf_backbone_rerank": bool(cfg.cf_backbone_rerank),
        "cf_prob_shift_weight": cfg.cf_prob_shift_weight,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "next_act_loss_weight": cfg.next_act_loss_weight,
        "dfa_next_act_aux_protocol_version": int(
            cfg.dfa_next_act_aux_protocol_version
        ),
        "dfa_next_act_aux_requested_weight": requested_dfa_aux_weight,
        "main_model_profile": (
            "with_dfa_aux_ablation"
            if dfa_aux_variant_enabled else "aux_free_search_dfa"
        ),
        "dfa_next_act_aux_variant_enabled": dfa_aux_variant_enabled,
        "dfa_next_act_aux_configured_weight": configured_dfa_aux_weight,
        "dfa_next_act_aux_effective_weight": cfg.dfa_next_act_loss_weight,
        "dfa_next_act_aux_active": bool(cfg.dfa_next_act_loss_weight > 0.0),
        "dfa_next_act_aux_require_target_valid": bool(
            cfg.dfa_next_act_require_target_valid
        ),
        "dfa_next_act_aux_source": dfa_aux_diagnostics["source"],
        "dfa_next_act_aux_mode": dfa_aux_diagnostics["mode"],
        "dfa_next_act_aux_selection_uses_outcome_label": bool(
            dfa_aux_diagnostics["selection_uses_outcome_label"]
        ),
        "dfa_next_act_aux_uses_target_safety_gate": bool(
            dfa_aux_diagnostics[
                "uses_next_activity_target_for_safety_gate"
            ]
        ),
        "dfa_next_act_aux_state_applicable_count": int(
            dfa_aux_diagnostics["state_applicable_count"]
        ),
        "dfa_next_act_aux_no_progress_action_count": int(
            dfa_aux_diagnostics["no_progress_action_count"]
        ),
        "dfa_next_act_aux_informative_mask_count": int(
            dfa_aux_diagnostics["informative_mask_count"]
        ),
        "dfa_next_act_aux_target_compatible_count": int(
            dfa_aux_diagnostics["target_compatible_count"]
        ),
        "dfa_next_act_aux_target_incompatible_count": int(
            dfa_aux_diagnostics["target_incompatible_count"]
        ),
        "dfa_next_act_aux_applicable_count": int(
            dfa_aux_diagnostics["aux_applicable_count"]
        ),
        "dfa_next_act_aux_applicable_fraction": float(
            dfa_aux_diagnostics["aux_applicable_fraction"]
        ),
        "dfa_next_act_aux_mean_progress_action_fraction": float(
            dfa_aux_diagnostics["mean_progress_action_fraction"]
        ),
        # Protocol-v1 column retained as a compatibility alias.
        "dfa_next_act_aux_mean_valid_action_fraction": float(
            dfa_aux_diagnostics["mean_valid_action_fraction"]
        ),
        "dfa_next_act_aux_final_train_loss": (
            trainer.train_history[-1].get("loss_dfa_next_act", float("nan"))
            if trainer.train_history else float("nan")
        ),
        "dfa_next_act_aux_final_allowed_mass": (
            trainer.train_history[-1].get("dfa_aux_allowed_mass", float("nan"))
            if trainer.train_history else float("nan")
        ),
        "time_delta_loss_weight": cfg.time_delta_loss_weight,
        "time_tail_weight": cfg.time_tail_weight,
        "time_tail_quantile": cfg.time_tail_quantile,
        "time_std_loss_weight": cfg.time_std_loss_weight,
        "early_stop_time_weight": cfg.early_stop_time_weight,
        "next_act_label_smoothing": cfg.next_act_label_smoothing,
        "next_act_class_weighted": bool(cfg.next_act_class_weights is not None),
        "next_act_transition_prior_weight": cfg.next_act_transition_prior_weight,
        "next_act_transition_prior_used": bool(cfg.next_act_transition_log_prior is not None),
        "outcome_loss_weight": cfg.outcome_loss_weight,
        "outcome_label_smoothing": cfg.outcome_label_smoothing,
        "uwl_log_var_min": cfg.uwl_log_var_min,
        "uwl_log_var_max": cfg.uwl_log_var_max,
        "balanced_outcome_sampler": bool(getattr(args, "balanced_outcome_sampler", False)),
        "outcome_class_weight_0": (
            cfg.outcome_class_weights[0] if cfg.outcome_class_weights is not None else float("nan")
        ),
        "outcome_class_weight_1": (
            cfg.outcome_class_weights[1] if cfg.outcome_class_weights is not None else float("nan")
        ),
        "epochs": args.epochs,
        "best_val_acc": best_val_acc,
        "best_val_auc": best_val_auc,
        "best_val_monitor": trainer.best_val_score,
        "best_epoch": trainer.best_epoch,
        "best_learning_rate": getattr(trainer, "best_learning_rate", None),
        **{f"best_task_log_var_{i}": float(s) for i, s in enumerate(trainer.task_log_vars.detach().cpu())},
        "stopped_early": stopped_early,
        "actual_epochs": len(trainer.train_history),
        **{f"train_{k}": v for k, v in train_fit_logs.items()},
        "fit_outcome_acc_gap": (
            train_fit_logs.get("outcome_acc", float("nan")) - test_logs["outcome_acc"]
            if train_fit_logs else float("nan")
        ),
        "fit_next_act_acc_gap": (
            train_fit_logs.get("next_act_acc", float("nan")) - test_logs["next_act_acc"]
            if train_fit_logs else float("nan")
        ),
        **{f"test_{k}": v for k, v in test_logs.items()},
        **{f"cf_{k}": v for k, v in cf_metrics.items()},
    }

    # ---- Save ----
    os.makedirs(args.output_dir, exist_ok=True)

    # Training log CSV
    history = getattr(trainer, "train_history", None)
    if history:
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(
            os.path.join(
                args.output_dir,
                f"train_{dataset_name}_{variant}_seed{seed}.csv",
            ),
            index=False,
        )

    # Model checkpoint
    model_path = os.path.join(
        args.output_dir, f"model_{dataset_name}_{variant}_seed{seed}.pt"
    )
    from cf_replay import make_evaluation_context
    evaluation_context = make_evaluation_context(
        test_ds, test_indices, ref_matrix, act_encoder, pattern_encoder,
        pattern_dict, constraint_dict, _soft_constraints_to_use, audit_rules,
        dfa, multi_dfa_clusters, _dfa_enforced, result,
    )
    trainer.save_checkpoint(model_path, evaluation_context=evaluation_context)
    print(f"  Model saved:  {model_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Dataset names to run. Defaults to the seven thesis datasets.")
    parser.add_argument("--variants", type=str, nargs="*",
                        default=[
                            "full", "no_pattern", "no_uwl", "no_dfa", "with_dfa_aux",
                            "soft_constraint_cf", "fixed_constraint_cf",
                            "single_point_cf", "multi_point_cf",
                            "score_ranking_cf", "pareto_ranking_cf",
                            "model_score_cf", "random_baseline",
                        ],
                        help="Experiment variants to run.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 101, 2026, 4096],
                        help="Training seeds. All variants share the same split and evaluation manifest.")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Fixed train/validation/test case split seed.")
    parser.add_argument("--audit_seed", type=int, default=2026,
                        help="Fixed case-level seed for discovery/calibration/audit symbolic folds.")
    parser.add_argument("--audit_fraction", type=float, default=0.25,
                        help="Fraction of training cases reserved only for independent complete-trace audit-rule discovery.")
    parser.add_argument("--constraint_calibration_fraction", type=float, default=0.15,
                        help="Fraction of training cases reserved for label-free Search-Pattern calibration.")
    parser.add_argument("--constraint_policy", choices=["calibrated", "fixed"],
                        default="calibrated",
                        help="Calibrated hard/soft assignment or fixed-support ablation.")
    parser.add_argument("--prediction_pattern_top_k", type=int, default=32,
                        help="Maximum next-activity-supervised Prediction-Patterns.")
    parser.add_argument("--search_pattern_pool_size", type=int, default=64,
                        help="Maximum label-free Search-Patterns before calibration.")
    parser.add_argument("--search_min_calibration_support", type=float, default=0.05,
                        help="Minimum calibration-fold support for a hard Search-DFA rule.")
    parser.add_argument("--search_max_calibration_support", type=float, default=1.00,
                        help="Maximum calibration-fold support for a hard Search-DFA rule.")
    parser.add_argument("--search_min_stability", type=float, default=0.85,
                        help="Minimum discovery/calibration support stability for hard constraints.")
    parser.add_argument("--search_soft_constraints", type=int, default=20,
                        help="Maximum non-hard Search-Patterns used only for CF ranking.")
    parser.add_argument("--search_min_hard_constraints", type=int, default=3,
                        help="Target minimum hard-rule count, subject to safe fallback eligibility.")
    parser.add_argument("--search_fallback_min_case_support", type=float, default=0.50,
                        help="Minimum robust complete-case support for every calibrated hard rule, including fallback.")
    parser.add_argument("--search_fallback_min_prefix_support", type=float, default=0.10,
                        help="Minimum calibration-prefix support for stable-anchor fallback.")
    parser.add_argument("--search_fallback_min_stability", type=float, default=0.70,
                        help="Minimum uncertainty-adjusted stability for safe fallback rules.")
    parser.add_argument("--search_support_confidence_z", type=float, default=1.96,
                        help="Normal critical value for support uncertainty and Wilson bounds.")
    parser.add_argument("--audit_rule_source", choices=["heldout", "external", "hybrid"],
                        default="heldout",
                        help="Independent audit source: held-out complete cases, pre-registered external rules, or both.")
    parser.add_argument("--audit_rule_file", type=str, default=None,
                        help="JSON/CSV pre-registered audit rules; required for external or hybrid source.")
    parser.add_argument("--audit_templates", nargs="+",
                        choices=["response", "precedence", "chain_response", "not_coexistence"],
                        default=["response", "precedence", "chain_response", "not_coexistence"],
                        help="Pre-specified global Declare families used only by Audit-DFA.")
    parser.add_argument("--audit_min_activation_cases", type=int, default=10,
                        help="Minimum number of held-out cases activating an audit rule.")
    parser.add_argument("--audit_min_activation_fraction", type=float, default=0.02,
                        help="Minimum fraction of held-out cases activating an audit rule.")
    parser.add_argument("--audit_min_confidence", type=float, default=0.80,
                        help="Minimum activated-case confidence for data-derived audit rules.")
    parser.add_argument("--audit_max_activities", type=int, default=40,
                        help="Maximum frequent activities considered in pairwise audit discovery.")
    parser.add_argument("--audit_max_constraints", type=int, default=20,
                        help="Maximum separately compiled independent audit rules.")
    parser.add_argument("--audit_dfa_max_states", type=int, default=500,
                        help="Per-rule Audit-DFA state limit.")
    parser.add_argument("--audit_manifest_dir", type=str, default="./audit_manifests",
                        help="Directory for rule-level Audit-DFA provenance manifests.")
    parser.add_argument(
        "--allow_missing_independent_audit", dest="require_independent_audit",
        action="store_false", default=True,
        help="Exploratory-only override: continue when independent Audit-DFA construction fails.",
    )
    parser.add_argument("--evaluation_seed", type=int, default=2026,
                        help="Local seed for the variant-independent CF evaluation sample manifest.")
    parser.add_argument("--cf_eval_samples", type=int, default=0,
                        help="Number of fixed test prefixes used for CF evaluation; 0 evaluates all.")
    parser.add_argument("--evaluation_manifest_dir", type=str,
                        default="./evaluation_manifests",
                        help="Persistent directory for paired CF evaluation sample manifests.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--lstm_hidden_dim", type=int, default=128,
                        help="Hidden size of the task-aligned lstm_mtl baseline.")
    parser.add_argument("--lstm_num_layers", type=int, default=2,
                        help="Number of unidirectional LSTM layers for lstm_mtl.")
    parser.add_argument("--lstm_dropout", type=float, default=0.1)
    parser.add_argument("--lstm_patience", type=int, default=5,
                        help="Validation-based early stopping patience for lstm_mtl.")
    parser.add_argument("--lstm_use_attributes", action=argparse.BooleanOptionalAction,
                        default=True, help="Use the same case/event attributes as the proposed model.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader worker processes. Use 0 on Windows if multiprocessing is unstable.")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Enable DataLoader pin_memory when training on CUDA.")
    parser.add_argument("--persistent_workers", action="store_true",
                        help="Keep DataLoader workers alive across epochs. Disabled by default to avoid file-descriptor leaks in long batch runs.")
    parser.add_argument("--cf_topk", type=int, default=5, help="Top-k counterfactuals per sample.")
    parser.add_argument("--cf_intervention_points", type=int, default=3,
                        help="Number of candidate intervention positions for CF generation.")
    parser.add_argument("--cf_per_position_limit", type=int, default=5,
                        help="Top local actions retained per intervention position before combination.")
    parser.add_argument("--cf_combination_budget", type=int, default=100,
                        help="Maximum multi-position combinations evaluated per sample.")
    parser.add_argument(
        "--cf_minimal_relaxation", action=argparse.BooleanOptionalAction,
        default=True,
        help="Use strict-first, minimum-cardinality hard-constraint relaxation.",
    )
    parser.add_argument("--cf_max_relaxed_constraints", type=int, default=None,
                        help="Maximum runtime exemptions within the selected Search cluster; 0 forbids exemptions.")
    parser.add_argument("--cf_non_relaxable_rule_ids", nargs="*", default=[],
                        help="User-designated global obligations. Must compile without weakening.")
    parser.add_argument("--cf_ranking", choices=["score", "pareto", "dpp"], default="pareto",
                        help="Counterfactual ranking strategy. score restores the old scalar ranking.")
    parser.add_argument("--cf_success_weight", type=float, default=1.0,
                        help="Weight for the original CF model score in multi-objective ranking.")
    parser.add_argument("--cf_proximity_weight", type=float, default=0.15,
                        help="Weight for action/time proximity in multi-objective ranking.")
    parser.add_argument("--cf_sparsity_weight", type=float, default=0.10,
                        help="Weight for sparse action edits in multi-objective ranking.")
    parser.add_argument("--cf_diversity_weight", type=float, default=0.10,
                        help="Weight for diversity among selected Top-K counterfactuals.")
    parser.add_argument("--cf_backbone_rerank", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Re-score generated CF candidates with the main backbone outcome probability.")
    parser.add_argument("--cf_prob_shift_weight", type=float, default=0.5,
                        help="Extra weight for positive probability shift during backbone CF reranking.")
    parser.add_argument("--cf_soft_constraint_weight", type=float, default=0.10,
                        help="Maximum score reward contributed by calibrated soft Search-Patterns.")
    parser.add_argument("--prefix_mode", choices=["all", "last"], default="all",
                        help="Prefix expansion mode: all prefixes or only the final old-style prefix.")
    parser.add_argument("--max_prefixes_per_case", type=int, default=None,
                        help="Optional deterministic cap on generated prefixes per case.")
    parser.add_argument("--fit_diagnostics", action="store_true",
                        help="Evaluate train split after training and report train-test fit gaps.")
    parser.add_argument("--next_act_label_smoothing", type=float, default=0.02,
                        help="Label smoothing for next-activity cross-entropy.")
    parser.add_argument("--next_act_class_weights", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Enable inverse-frequency class weights for next-activity loss.")
    parser.add_argument("--next_act_class_weight_clip", type=float, default=5.0,
                        help="Clip range for next-activity class weights.")
    parser.add_argument("--next_act_loss_weight", type=float, default=1.0,
                        help="Multiplier for the next-activity loss before UWL weighting.")
    parser.add_argument("--dfa_next_act_loss_weight", type=float, default=0.0,
                        help=(
                            "Experimental Search-DFA next-activity auxiliary "
                            "weight. The main model keeps it disabled."
                        ))
    parser.add_argument("--dfa_next_act_require_target_valid",
                        action=argparse.BooleanOptionalAction, default=True,
                        help=(
                            "Apply DFA guidance only when the observed next "
                            "activity makes strict Search-DFA acceptance progress."
                        ))
    parser.add_argument("--next_act_transition_prior_weight", type=float, default=0.35,
                        help="Weight of train-only bigram transition log-prior added to next-activity logits; 0 disables it.")
    parser.add_argument("--next_act_transition_smoothing", type=float, default=1.0,
                        help="Additive smoothing for the next-activity transition prior.")
    parser.add_argument("--outcome_loss_weight", type=float, default=2.0,
                        help="Multiplier for the main outcome classification loss.")
    parser.add_argument("--outcome_label_smoothing", type=float, default=0.02,
                        help="Label smoothing for outcome cross-entropy; helps probability calibration.")
    parser.add_argument("--outcome_class_weight_clip", type=float, default=5.0,
                        help="Clip range for outcome class weights when the sampler is disabled.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay for regularization.")
    parser.add_argument("--early_stop_time_weight", type=float, default=0.0,
                        help="Weight of time quality in the early-stop monitor; 0 means outcome-only.")
    parser.add_argument("--time_tail_weight", type=float, default=0.0,
                        help="Extra SmoothL1 weight for high true next-time targets; useful for long-tail Sepsis time prediction.")
    parser.add_argument("--time_tail_quantile", type=float, default=0.80,
                        help="Train-batch quantile used to identify high-time targets for --time_tail_weight.")
    parser.add_argument("--time_std_loss_weight", type=float, default=0.0,
                        help="Weight for matching predicted and true next-time standard deviation within a batch.")
    parser.add_argument("--uwl_log_var_min", type=float, default=-1.0,
                        help="Lower bound for learnable UWL log variances.")
    parser.add_argument("--uwl_log_var_max", type=float, default=3.0,
                        help="Upper bound for learnable UWL log variances.")
    parser.add_argument("--balanced_outcome_sampler", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Balance training batches by outcome label. Disabled by default; class-weight loss is safer on small imbalanced datasets.")
    parser.add_argument("--prediction_detail_limit", type=int, default=0,
                        help="Max prediction detail rows saved per run; use 0 to save all test prefixes.")
    parser.add_argument("--prediction_detail_batch_size", type=int, default=256,
                        help="Batch size used only when saving prediction detail CSV files.")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for result files")
    parser.add_argument("--incremental_results_dir", type=str, default="./all_results",
                        help="Directory for per-dataset all_results CSV files saved after each completed variant.")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Write results directly into output_dir without a timestamp suffix")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--no-pattern", action="store_true",
                        help="Disable pattern discovery globally")
    parser.add_argument("--prefer-csv", action="store_true",
                        help="Prefer CSV files when both CSV and XES are available")
    parser.add_argument(
        "--bp17-anti-leak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable BPIC17 anti-leak mode that removes Accepted/Selected",
    )
    parser.add_argument("--data_path", type=str, default=None,
                        help="Optional explicit dataset file path.")
    args = parser.parse_args()
    if not 0.0 < args.audit_fraction < 1.0:
        parser.error("--audit_fraction must be strictly between 0 and 1")
    if not 0.0 <= args.constraint_calibration_fraction < 1.0:
        parser.error("--constraint_calibration_fraction must be in [0, 1)")
    if args.audit_fraction + args.constraint_calibration_fraction >= 1.0:
        parser.error(
            "--audit_fraction + --constraint_calibration_fraction must be below 1"
        )
    if not (
        0.0 <= args.search_min_calibration_support
        <= args.search_max_calibration_support <= 1.0
    ):
        parser.error("Search calibration support bounds must satisfy 0 <= min <= max <= 1")
    if not 0.0 <= args.search_min_stability <= 1.0:
        parser.error("--search_min_stability must be in [0, 1]")
    if not 0.0 <= args.search_fallback_min_case_support <= 1.0:
        parser.error("--search_fallback_min_case_support must be in [0, 1]")
    if not 0.0 <= args.search_fallback_min_prefix_support <= 1.0:
        parser.error("--search_fallback_min_prefix_support must be in [0, 1]")
    if not 0.0 <= args.search_fallback_min_stability <= 1.0:
        parser.error("--search_fallback_min_stability must be in [0, 1]")
    if args.search_support_confidence_z <= 0.0:
        parser.error("--search_support_confidence_z must be positive")
    if (
        args.prediction_pattern_top_k < 1
        or args.search_pattern_pool_size < 1
        or args.search_soft_constraints < 0
        or args.search_min_hard_constraints < 0
    ):
        parser.error(
            "Pattern pool/count arguments must be positive "
            "(soft/min-hard may be zero)"
        )
    if args.cf_soft_constraint_weight < 0.0:
        parser.error("--cf_soft_constraint_weight must be non-negative")
    if args.cf_max_relaxed_constraints is not None and args.cf_max_relaxed_constraints < 0:
        parser.error("--cf_max_relaxed_constraints must be non-negative")
    if args.dfa_next_act_loss_weight < 0.0:
        parser.error("--dfa_next_act_loss_weight must be non-negative")
    if not 0.0 <= args.audit_min_activation_fraction <= 1.0:
        parser.error("--audit_min_activation_fraction must be in [0, 1]")
    if not 0.0 <= args.audit_min_confidence <= 1.0:
        parser.error("--audit_min_confidence must be in [0, 1]")
    if args.audit_min_activation_cases < 1 or args.audit_max_constraints < 1:
        parser.error("audit activation and constraint counts must be positive")
    if args.audit_rule_source in ("external", "hybrid") and not args.audit_rule_file:
        parser.error("--audit_rule_file is required for external or hybrid audit source")

    # Add timestamp to avoid overwriting
    if not args.no_timestamp:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"{args.output_dir}_{ts}"

    datasets = args.datasets or PAPER_DATASETS
    variants = args.variants

    # Create the run directory before experiments start so that failure
    # diagnostics can still be persisted when every run fails.
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    failures = []

    for seed in args.seeds:
        for ds in datasets:
            for var in variants:
                try:
                    res = run_experiment(ds, var, args, seed=seed)
                    all_results.append(res)
                    save_incremental_result(res, args)
                    print(f"  OK {ds} [{var}] seed={seed} done")
                except Exception as e:
                    print(f"  FAIL {ds} [{var}] seed={seed}: {e}")
                    traceback.print_exc()
                    failures.append({
                        "dataset": ds,
                        "variant": var,
                        "seed": seed,
                        "error": str(e),
                    })

    # ---- Summary table ----
    print(f"\n{'='*80}")
    print(f"  Experiment summary: {len(all_results)} succeeded, {len(failures)} failed")
    print(f"{'='*80}")

    if all_results:
        os.makedirs(args.output_dir, exist_ok=True)

        df = pd.DataFrame(all_results)
        all_csv = os.path.join(args.output_dir, "all_results.csv")
        df.to_csv(all_csv, index=False)
        print(f"\nResults saved:  {all_csv} ({len(df)}  rows")

        # Filter numeric columns for aggregation (exclude string, boolean columns)
        numeric_cols = [c for c in df.columns
                        if c not in ("dataset", "variant", "has_dfa")
                        and pd.api.types.is_numeric_dtype(df[c])]

        if numeric_cols:
            try:
                summary = df.groupby("variant")[numeric_cols].mean()
                print("\nSummary by variant (mean):")
                print(summary.to_string())
                sv = os.path.join(args.output_dir, "summary_by_variant.csv")
                summary.to_csv(sv)
                print(f"Saved:  {sv}")

                ds_summary = df.groupby("dataset")[numeric_cols].mean()
                print("\nSummary by dataset (mean):")
                print(ds_summary.to_string())
                sd = os.path.join(args.output_dir, "summary_by_dataset.csv")
                ds_summary.to_csv(sd)
                print(f"Saved {sd}")
            except Exception as e:
                print(f"Aggregation failed without affecting raw outputs: {e}")
        else:
            print("No numeric columns available for aggregation; skipping summaries.")

    if failures:
        os.makedirs(args.output_dir, exist_ok=True)
        pd.DataFrame(failures).to_csv(
            os.path.join(args.output_dir, "failures.csv"), index=False
        )

    print(f"\nResults saved to:  {args.output_dir}/")


if __name__ == "__main__":
    main()
