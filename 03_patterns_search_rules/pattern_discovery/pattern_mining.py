# pattern_discovery/pattern_mining.py
# 基于 pattern_miner.py 的高效模式挖掘 (跳跃子序列, 无 VF2/pm4py)
import json
import math
import os, hashlib
import pickle, glob, networkx as nx
import pandas as pd
from explaination.visualizations.utils import normalize_activity_name
from pattern_miner import (
    PatternMiningConfig, TransformerPatternMiner,
    compute_pattern_attributes as _compute_attrs,
)


def _pat_id(tokens):
    head = "_".join(str(t)[:8] for t in tokens[:3])
    digest = hashlib.sha1("||".join(str(t) for t in tokens).encode()).hexdigest()[:8]
    return f"P{len(tokens)}_{head}_{digest}"


def _pat_graph(tokens, act_encoder=None, max_gap=None):
    g = nx.DiGraph()
    for i, tok in enumerate(tokens):
        name = normalize_activity_name(tok, act_encoder=act_encoder)
        g.add_node(i + 1, value=name, parallel=False, color="#888")
        if i > 0:
            edge_attrs = {"eventually": True}
            if max_gap is not None:
                edge_attrs["max_gap"] = int(max_gap)
            g.add_edge(i, i + 1, **edge_attrs)
    return g


def run_pattern_discovery(train_samples, act_encoder=None, output_path='./patterns',
                           return_dict=True, transform_samples=None,
                           return_transform=False,
                           return_miner=False,
                           target_col="outcome",
                           selection_mode="supervised_binary",
                           top_k=32,
                           min_feature_coverage=0.03,
                           max_feature_coverage=0.90,
                           include_unigrams=False,
                           min_abs_hit_rate_diff=0.02):
    """Discover task-aligned or label-free ordered trace patterns."""
    os.makedirs(output_path, exist_ok=True)
    mode = str(selection_mode).strip().lower()
    mode_map = {
        "supervised_binary": ("supervised", "binary"),
        "supervised_categorical": ("supervised", "categorical"),
        "label_free": ("label_free", "binary"),
    }
    if mode not in mode_map:
        raise ValueError(f"Unsupported pattern selection mode: {selection_mode}")
    miner_selection, outcome_type = mode_map[mode]

    cfg = PatternMiningConfig(
        max_trace_len=30,
        sample_size=400,
        min_support=0.03,
        max_pattern_len=4,
        max_gap=5,
        top_k=int(top_k),
        random_state=42,
        outcome_type=outcome_type,
        selection_mode=miner_selection,
        factual_outcome=1,
        include_unigrams=bool(include_unigrams),
        use_pareto=False,
        min_feature_coverage=float(min_feature_coverage),
        max_feature_coverage=float(max_feature_coverage),
        min_abs_hit_rate_diff=float(min_abs_hit_rate_diff),    )
    # Normalized [0, 1] times are incompatible with the former hour buckets.
    miner = TransformerPatternMiner(cfg, include_time_buckets=False)
    miner.fit(train_samples, target_col=target_col)
    train_encoded = miner.transform(train_samples)
    encoded_samples = (
        miner.transform(transform_samples)
        if transform_samples is not None else train_encoded
    )
    attr_df = miner.get_pattern_table()

    if len(train_encoded) == 0 or len(attr_df) == 0:
        print("模式发现: 0 模式")
        if return_dict:
            if return_transform:
                if return_miner:
                    return None, None, None, None
                return None, None, None
            if return_miner:
                return None, None, None
            return None, None
        return None

    # 构建 pattern_dict (属性列表 → DiGraph)
    pattern_dict = {}
    for pat_tuple in miner.selected_patterns_:
        pid = miner.pattern_to_col_.get(pat_tuple, _pat_id(pat_tuple))
        g = _pat_graph(pat_tuple, act_encoder=act_encoder, max_gap=cfg.max_gap)
        pattern_dict[pid] = g
        try:
            pickle.dump(g, open(os.path.join(output_path, f"{pid}.pickle"), "wb"))
        except Exception:
            pass
    attr_df.to_csv(os.path.join(output_path, "pattern_attributes.csv"), index=False)

    print(f"模式发现完成: {train_encoded.shape[1]} 模式特征, {len(pattern_dict)} 模式图")
    if return_dict:
        if return_transform:
            if return_miner:
                return train_encoded, pattern_dict, encoded_samples, miner
            return train_encoded, pattern_dict, encoded_samples
        if return_miner:
            return train_encoded, pattern_dict, miner
        return train_encoded, pattern_dict
    return train_encoded


def compute_pattern_attributes(pattern_dict, patient_data, outcome_col='label', factual_outcome=1):
    df = _compute_attrs(
        pattern_dict,
        patient_data,
        factual_outcome=factual_outcome,
        target_col=outcome_col,
    )
    if 'mi' in df.columns and 'outcome_interest' not in df.columns:
        df['outcome_interest'] = df['mi']
    return df


def filter_high_quality_patterns(pattern_dict, attr_df, min_interest=0.01, min_hit_rate_diff=0.0):
    if len(attr_df) == 0:
        return {}, attr_df
    mi_col = 'mi' if 'mi' in attr_df.columns else 'outcome_interest'
    hq = attr_df[(attr_df[mi_col] > min_interest) &
                 (attr_df['hit_rate_diff'] > min_hit_rate_diff)]
    filtered = {pid: pattern_dict[pid] for pid in hq['pattern_id'] if pid in pattern_dict}
    return filtered, hq

def _pattern_case_key(index_value):
    """Recover the complete-case identifier from a prefix feature index."""
    value = str(index_value)
    return value.rsplit("::p", 1)[0] if "::p" in value else value


def _pattern_support(encoded, pattern_id, *, case_level=False):
    """Return binary support and the independent unit count."""
    if (
        encoded is None
        or len(encoded) == 0
        or pattern_id not in encoded.columns
    ):
        return float("nan"), 0
    values = (
        pd.to_numeric(encoded[pattern_id], errors="coerce")
        .fillna(0.0)
        .gt(0.0)
        .astype(float)
    )
    if case_level:
        case_frame = pd.DataFrame({
            "case_id": [_pattern_case_key(index) for index in values.index],
            "hit": values.to_numpy(dtype=float),
        })
        values = case_frame.groupby("case_id", sort=True)["hit"].max()
    return float(values.mean()), int(len(values))


def _two_sample_support_margin(p_discovery, n_discovery, p_calibration,
                               n_calibration, z_value=1.96):
    """Approximate sampling margin for a difference between two supports."""
    if n_discovery <= 0 or n_calibration <= 0:
        return 0.0
    pooled = (
        p_discovery * n_discovery + p_calibration * n_calibration
    ) / float(n_discovery + n_calibration)
    variance = max(0.0, pooled * (1.0 - pooled))
    standard_error = math.sqrt(
        variance * (1.0 / n_discovery + 1.0 / n_calibration)
    )
    return min(1.0, max(0.0, float(z_value) * standard_error))


def _wilson_lower_bound(support, sample_count, z_value=1.96):
    """One-sided conservative support estimate used by the safe fallback."""
    if sample_count <= 0 or not math.isfinite(float(support)):
        return 0.0
    support = min(1.0, max(0.0, float(support)))
    z_sq = float(z_value) ** 2
    denominator = 1.0 + z_sq / sample_count
    centre = support + z_sq / (2.0 * sample_count)
    radius = float(z_value) * math.sqrt(
        support * (1.0 - support) / sample_count
        + z_sq / (4.0 * sample_count ** 2)
    )
    return max(0.0, (centre - radius) / denominator)


def calibrate_search_patterns(
    pattern_dict,
    discovery_attributes,
    calibration_encoded,
    *,
    discovery_encoded=None,
    policy="calibrated",
    min_calibration_support=0.05,
    max_calibration_support=1.00,
    min_stability=0.85,
    max_hard_constraints=10,
    max_soft_constraints=20,
    min_hard_constraints=3,
    fallback_min_case_support=0.50,
    fallback_min_prefix_support=0.10,
    fallback_min_stability=0.70,
    support_confidence_z=1.96,
):
    """Assign label-free patterns to hard-DFA or soft-ranking roles.

    Supports are aggregated at complete-case granularity so multiple prefixes
    from one case do not masquerade as independent observations. Strict hard
    constraints require both uncertainty-adjusted discovery/calibration
    stability and the robust complete-case support floor. If a dataset has no
    strict rule, a stable-anchor fallback may select up to
    min_hard_constraints rules under the same support floor; it never uses an
    outcome or next-activity label.
    """
    policy = str(policy).strip().lower()
    if policy not in {"calibrated", "fixed"}:
        raise ValueError(f"Unsupported constraint policy: {policy}")
    if not 0.0 <= min_calibration_support <= max_calibration_support <= 1.0:
        raise ValueError("Support bounds must satisfy 0 <= min <= max <= 1")
    for name, value in (
        ("min_stability", min_stability),
        ("fallback_min_case_support", fallback_min_case_support),
        ("fallback_min_prefix_support", fallback_min_prefix_support),
        ("fallback_min_stability", fallback_min_stability),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if float(support_confidence_z) < 0.0:
        raise ValueError("support_confidence_z must be non-negative")
    if int(max_hard_constraints) < 0 or int(max_soft_constraints) < 0:
        raise ValueError("Constraint limits must be non-negative")
    if int(min_hard_constraints) < 0:
        raise ValueError("min_hard_constraints must be non-negative")

    pattern_dict = pattern_dict or {}
    pattern_by_id = {
        str(pattern_id): graph
        for pattern_id, graph in pattern_dict.items()
    }
    attributes = (
        discovery_attributes.copy()
        if discovery_attributes is not None else pd.DataFrame()
    )
    attr_by_pid = {}
    if not attributes.empty and "pattern_id" in attributes:
        attr_by_pid = {
            str(row["pattern_id"]): row
            for _, row in attributes.iterrows()
        }

    records = []
    for pattern_id, graph in pattern_by_id.items():
        row = attr_by_pid.get(pattern_id)
        attribute_prefix_support = (
            float(row.get("coverage", 0.0)) if row is not None else 0.0
        )
        discovery_prefix_support, discovery_prefix_count = _pattern_support(
            discovery_encoded, pattern_id, case_level=False
        )
        if not math.isfinite(discovery_prefix_support):
            discovery_prefix_support = attribute_prefix_support

        discovery_case_support, discovery_case_count = _pattern_support(
            discovery_encoded, pattern_id, case_level=True
        )
        if not math.isfinite(discovery_case_support):
            discovery_case_support = discovery_prefix_support

        calibration_prefix_support, calibration_prefix_count = _pattern_support(
            calibration_encoded, pattern_id, case_level=False
        )
        calibration_case_support, calibration_case_count = _pattern_support(
            calibration_encoded, pattern_id, case_level=True
        )
        calibration_available = bool(
            calibration_case_count > 0
            and math.isfinite(calibration_case_support)
        )

        support_drift = (
            abs(discovery_case_support - calibration_case_support)
            if calibration_available else float("nan")
        )
        sampling_margin = (
            _two_sample_support_margin(
                discovery_case_support,
                discovery_case_count,
                calibration_case_support,
                calibration_case_count,
                z_value=support_confidence_z,
            )
            if calibration_available else 0.0
        )
        excess_drift = (
            max(0.0, support_drift - sampling_margin)
            if calibration_available else float("nan")
        )
        adjusted_stability = (
            max(0.0, 1.0 - excess_drift)
            if calibration_available else float("nan")
        )
        raw_stability = (
            max(0.0, 1.0 - support_drift)
            if calibration_available else float("nan")
        )
        discovery_lower = _wilson_lower_bound(
            discovery_case_support,
            discovery_case_count,
            z_value=support_confidence_z,
        )
        calibration_lower = _wilson_lower_bound(
            calibration_case_support,
            calibration_case_count,
            z_value=support_confidence_z,
        )

        robust_support = (
            min(discovery_case_support, calibration_case_support)
            if calibration_available else discovery_case_support
        )
        pattern_length = (
            int(row.get("length", len(graph.nodes)))
            if row is not None else int(len(graph.nodes))
        )
        length_penalty = 0.02 * max(0, pattern_length - 1)
        calibration_score = (
            0.45 * robust_support
            + 0.25 * (adjusted_stability if calibration_available else 0.0)
            + 0.15 * (
                calibration_prefix_support
                if math.isfinite(calibration_prefix_support) else 0.0
            )
            + 0.15 * discovery_prefix_support
            - length_penalty
        )
        fixed_support_score = (
            0.70 * discovery_case_support
            + 0.30 * discovery_prefix_support
            - length_penalty
        )

        within_case_support_bounds = bool(
            calibration_available
            and min_calibration_support
            <= calibration_case_support
            <= max_calibration_support
            and min_calibration_support
            <= discovery_case_support
            <= max_calibration_support
        )
        strict_calibrated_eligible = bool(
            within_case_support_bounds
            # A hard rule must be a genuine complete-case anchor. The
            # ordinary calibration lower bound controls candidate quality,
            # but is intentionally too permissive to authorize a hard DFA
            # constraint on its own.
            and robust_support >= fallback_min_case_support
            and math.isfinite(calibration_prefix_support)
            and calibration_prefix_support >= min_calibration_support
            and adjusted_stability >= min_stability
        )
        fixed_eligible = bool(
            min_calibration_support
            <= discovery_case_support
            <= max_calibration_support
        )
        fallback_eligible = bool(
            policy == "calibrated"
            and within_case_support_bounds
            and robust_support >= fallback_min_case_support
            and math.isfinite(calibration_prefix_support)
            and calibration_prefix_support >= fallback_min_prefix_support
            and adjusted_stability >= fallback_min_stability
            and min(discovery_lower, calibration_lower)
            >= min_calibration_support
        )
        strict_eligible = (
            strict_calibrated_eligible
            if policy == "calibrated" else fixed_eligible
        )

        records.append({
            "pattern_id": pattern_id,
            "pattern_length": pattern_length,
            "discovery_prefix_support": discovery_prefix_support,
            "calibration_prefix_support": calibration_prefix_support,
            "discovery_prefix_count": discovery_prefix_count,
            "calibration_prefix_count": calibration_prefix_count,
            "discovery_support": discovery_case_support,
            "calibration_support": calibration_case_support,
            "discovery_case_count": discovery_case_count,
            "calibration_case_count": calibration_case_count,
            "support_stability": raw_stability,
            "support_sampling_margin": sampling_margin,
            "uncertainty_adjusted_stability": adjusted_stability,
            "discovery_support_lower_bound": discovery_lower,
            "calibration_support_lower_bound": calibration_lower,
            "calibration_score": calibration_score,
            "fixed_support_score": fixed_support_score,
            "constraint_selection_score": (
                calibration_score
                if policy == "calibrated" else fixed_support_score
            ),
            "calibration_available": calibration_available,
            "strict_hard_eligible": strict_eligible,
            "fallback_hard_eligible": fallback_eligible,
            "hard_eligible": strict_eligible,
            "selection_uses_outcome_label": False,
            "support_granularity": "complete_case_any_prefix",
        })

    columns = [
        "pattern_id", "pattern_length",
        "discovery_prefix_support", "calibration_prefix_support",
        "discovery_prefix_count", "calibration_prefix_count",
        "discovery_support", "calibration_support",
        "discovery_case_count", "calibration_case_count",
        "support_stability", "support_sampling_margin",
        "uncertainty_adjusted_stability",
        "discovery_support_lower_bound",
        "calibration_support_lower_bound",
        "calibration_score", "fixed_support_score",
        "constraint_selection_score", "calibration_available",
        "strict_hard_eligible", "fallback_hard_eligible",
        "hard_eligible", "selection_uses_outcome_label",
        "support_granularity", "hard_selection_reason",
        "hard_selected_after_calibration", "constraint_role",
    ]
    if not records:
        return {}, {}, pd.DataFrame(columns=columns)

    frame = pd.DataFrame(records)
    ranked = frame.sort_values(
        [
            "strict_hard_eligible",
            "constraint_selection_score",
            "uncertainty_adjusted_stability",
            "pattern_id",
        ],
        ascending=[False, False, False, True],
        na_position="last",
    )
    max_hard = max(0, int(max_hard_constraints))
    hard_target = min(max_hard, max(0, int(min_hard_constraints)))
    strict_ids = ranked.loc[
        ranked["strict_hard_eligible"], "pattern_id"
    ].head(max_hard).tolist()
    hard_ids = list(strict_ids)
    fallback_selected = []
    if policy == "calibrated" and len(hard_ids) < hard_target:
        needed = hard_target - len(hard_ids)
        fallback_selected = ranked.loc[
            ranked["fallback_hard_eligible"]
            & ~ranked["pattern_id"].isin(hard_ids),
            "pattern_id",
        ].head(needed).tolist()
        hard_ids.extend(fallback_selected)

    hard_set = set(hard_ids)
    fallback_set = set(fallback_selected)
    soft_ids = ranked.loc[
        ~ranked["pattern_id"].isin(hard_set), "pattern_id"
    ].head(max(0, int(max_soft_constraints))).tolist()
    soft_set = set(soft_ids)

    frame["hard_selection_reason"] = frame["pattern_id"].map(
        lambda pattern_id: (
            "strict"
            if pattern_id in hard_set and pattern_id not in fallback_set
            else "stable_anchor_fallback"
            if pattern_id in fallback_set
            else "not_selected"
        )
    )
    frame["hard_selected_after_calibration"] = frame[
        "pattern_id"
    ].isin(hard_set)
    frame["constraint_role"] = frame["pattern_id"].map(
        lambda pattern_id: (
            "hard" if pattern_id in hard_set
            else "soft" if pattern_id in soft_set
            else "unused"
        )
    )

    metadata = frame.set_index("pattern_id").to_dict(orient="index")
    for pattern_id, graph in pattern_by_id.items():
        graph.graph.update(metadata.get(pattern_id, {}))

    hard = {
        pattern_id: pattern_by_id[pattern_id]
        for pattern_id in hard_ids
        if pattern_id in pattern_by_id
    }
    soft = {
        pattern_id: pattern_by_id[pattern_id]
        for pattern_id in soft_ids
        if pattern_id in pattern_by_id
    }
    frame = frame.sort_values(
        ["constraint_role", "constraint_selection_score", "pattern_id"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return hard, soft, frame

def persist_constraint_manifest(output_path, calibration_frame, metadata=None):
    """Persist role assignment and provenance for reproducible experiments."""
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, "search_constraint_calibration.csv")
    json_path = os.path.join(output_path, "search_constraint_manifest.json")
    frame = (
        calibration_frame.copy()
        if calibration_frame is not None else pd.DataFrame()
    )
    frame.to_csv(csv_path, index=False)
    payload = {
        "protocol_version": 2,
        "selection_uses_outcome_label": False,
        "metadata": dict(metadata or {}),
        "patterns": json.loads(frame.to_json(orient="records")),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return {"csv": csv_path, "json": json_path}


def _feature_sample_key(sample, fallback_index=0):
    explicit = sample.get("pattern_sample_id")
    if explicit is not None:
        return str(explicit)
    source_case = sample.get("source_case_id", sample.get("case_id"))
    prefix_len = sample.get("prefix_len")
    if source_case is None:
        source_case = f"Case_{fallback_index}"
    if prefix_len is not None:
        return f"{source_case}::p{int(prefix_len)}"
    return str(source_case)


def encode_pattern_features(samples, train_encoded, act_encoder=None, strict=True):
    """Inject one uniquely indexed pattern row into every PPM prefix."""
    if train_encoded is None:
        if strict and samples:
            raise ValueError("Pattern feature matrix is unavailable")
        for sample in samples:
            sample['pattern_features'] = []
        return samples

    if not train_encoded.index.is_unique:
        raise ValueError("Pattern feature index must be unique per prefix")

    missing = []
    for idx, sample in enumerate(samples):
        sample_key = _feature_sample_key(sample, idx)
        if sample_key not in train_encoded.index:
            sample['pattern_features'] = []
            missing.append(sample_key)
            continue
        row = train_encoded.loc[sample_key]
        sample['pattern_features'] = row.to_numpy(dtype='float32').tolist()

    if missing and strict:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Pattern feature mapping incomplete: {len(missing)}/{len(samples)} "
            f"prefixes missing ({preview})"
        )
    print(f"模式特征映射覆盖率: {len(samples) - len(missing)}/{len(samples)}")
    return samples


def load_pattern_dict(output_path):
    files = sorted(glob.glob(os.path.join(output_path, '*.pickle')))
    pattern_dict = {}
    for f in files:
        pid = os.path.basename(f).replace('.pickle', '')
        try:
            g = pickle.load(open(f, 'rb'))
            if isinstance(g, nx.DiGraph):
                pattern_dict[pid] = g
        except Exception:
            pass
    return pattern_dict if pattern_dict else None
