from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover
    Parallel = None
    delayed = None

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
except Exception:  # pragma: no cover
    mutual_info_classif = None
    mutual_info_regression = None

try:
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover
    train_test_split = None

try:
    from paretoset import paretoset
except Exception:  # pragma: no cover
    paretoset = None


def _pareto_mask_fallback(frame: pd.DataFrame, sense: List[str]) -> np.ndarray:
    """Return a deterministic non-dominated mask when paretoset is unavailable."""
    values = frame.to_numpy(dtype=float, copy=True)
    directions = np.asarray(
        [1.0 if direction == "max" else -1.0 for direction in sense],
        dtype=float,
    )
    values *= directions
    keep = np.ones(len(values), dtype=bool)
    for idx, value in enumerate(values):
        dominated = np.all(values >= value, axis=1) & np.any(values > value, axis=1)
        dominated[idx] = False
        if bool(np.any(dominated)):
            keep[idx] = False
    return keep


@dataclass
class PatternMiningConfig:
    max_trace_len: int = 30
    sample_size: int = 400
    min_support: float = 0.03
    max_pattern_len: int = 4
    max_gap: int = 5
    min_pattern_len: int = 1
    top_k: int = 200
    random_state: int = 42
    testing_percentage: float = 0.2
    outcome_type: str = "binary"
    # supervised selects patterns against an explicitly supplied target;
    # label_free uses only trace support/stability evidence.
    selection_mode: str = "supervised"
    factual_outcome: Any = 1

    score_w_mi: float = 0.55
    score_w_coverage: float = 0.20
    score_w_hit_rate: float = 0.15
    score_w_freq_interest: float = 0.10
    score_len_penalty: float = 0.02

    include_unigrams: bool = True
    use_pareto: bool = True

    # Discard patterns that are almost absent or almost constant.  Thresholds
    # are estimated on the symbolic-discovery split only.
    min_feature_coverage: float = 0.03
    max_feature_coverage: float = 0.90
    min_abs_hit_rate_diff: float = 0.02

    # speed-up knobs
    n_jobs: int = -1
    candidate_cap: int = 5000
    feature_binary: bool = True
    prune_by_frequency: bool = True
    prune_by_length_first: bool = True


def _safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def _sanitize_token(token: str) -> str:
    token = _safe_str(token)
    token = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fa5]+", "_", token)
    token = token.strip("_")
    return token or "X"


def _pattern_id(pattern: Tuple[str, ...]) -> str:
    base = "||".join(pattern)
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    head = "_".join(_sanitize_token(x) for x in pattern[:3])
    return f"P{len(pattern)}_{head}_{digest}"


def _label_mask(series: pd.Series, factual_outcome: Any) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        try:
            return series.astype(float).fillna(0) == float(factual_outcome)
        except Exception:
            return series.astype(str) == str(factual_outcome)
    return series.astype(str) == str(factual_outcome)


def _safe_mutual_info(
    feature: pd.Series,
    target: pd.Series,
    target_type: str = "binary",
) -> float:
    x = feature.fillna(0).astype(float).values.reshape(-1, 1)
    y = target.fillna(0)

    try:
        if target_type == "numerical":
            if mutual_info_regression is not None and pd.api.types.is_numeric_dtype(y):
                mi = mutual_info_regression(
                    x,
                    y.astype(float).values,
                    random_state=42,
                )
                return float(mi[0]) if len(mi) else 0.0

            y_bin = pd.qcut(
                y.rank(method="average"),
                q=min(10, max(2, y.nunique())),
                duplicates="drop",
                labels=False,
            )
            if mutual_info_classif is not None:
                mi = mutual_info_classif(
                    x,
                    y_bin.fillna(0).astype(int).values,
                    discrete_features=True,
                    random_state=42,
                )
                return float(mi[0]) if len(mi) else 0.0
            return 0.0

        if mutual_info_classif is not None:
            y_enc = pd.factorize(y.astype(str).fillna("NA"))[0]
            mi = mutual_info_classif(
                x,
                y_enc,
                discrete_features=True,
                random_state=42,
            )
            return float(mi[0]) if len(mi) else 0.0
        return 0.0
    except Exception:
        return 0.0


def _time_bucket_from_value(v: Any) -> str:
    if pd.isna(v):
        return "NA"
    try:
        x = float(v)
    except Exception:
        return "NA"

    if x < 1:
        return "B0"
    if x < 3:
        return "B1"
    if x < 6:
        return "B2"
    if x < 12:
        return "B3"
    if x < 24:
        return "B4"
    return "B5"


def _tokenize_trace(
    acts: List[Any],
    times: Optional[List[Any]] = None,
    include_time_buckets: bool = False,
) -> List[str]:
    acts = [_safe_str(a) for a in acts]
    if not include_time_buckets or times is None or len(times) != len(acts):
        return acts

    return [f"{a}__TB{_time_bucket_from_value(t)}" for a, t in zip(acts, times)]


def _truncate_sample(sample: Dict[str, Any], max_len: int) -> Dict[str, Any]:
    if sample.get("length", 0) <= max_len:
        return sample
    start = sample["length"] - max_len
    sample = dict(sample)
    sample["activities"] = sample["activities"][start:]
    if "times" in sample and isinstance(sample["times"], (list, tuple, np.ndarray)):
        sample["times"] = sample["times"][start:]
    sample["length"] = max_len
    return sample


def _source_case_key(sample: Dict[str, Any], fallback_index: int = 0) -> str:
    for key in ("source_case_id", "case_id", "Case ID", "case:concept:name"):
        value = sample.get(key)
        if value is not None:
            return _safe_str(value)
    return f"Case_{fallback_index}"


def _pattern_sample_key(sample: Dict[str, Any], fallback_index: int = 0) -> str:
    explicit = sample.get("pattern_sample_id")
    if explicit is not None:
        return _safe_str(explicit)
    source_case = _source_case_key(sample, fallback_index)
    prefix_len = sample.get("prefix_len")
    if prefix_len is not None:
        return f"{source_case}::p{int(prefix_len)}"
    return source_case


def _prefix_priority_indices(size: int) -> List[int]:
    """Return deterministic middle/early/late prefix indices for one case."""
    if size <= 0:
        return []
    fractions = [0.50, 0.0, 1.0, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875]
    ordered = []
    for fraction in fractions:
        idx = int(round((size - 1) * fraction))
        if idx not in ordered:
            ordered.append(idx)
    ordered.extend(idx for idx in range(size) if idx not in ordered)
    return ordered


def _case_balanced_sample(
    samples: List[Dict[str, Any]],
    sample_size: int,
    random_state: int,
) -> List[Dict[str, Any]]:
    """Sample prefixes round-robin by source case instead of by raw prefix."""
    if sample_size <= 0 or len(samples) <= sample_size:
        return samples

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for idx, sample in enumerate(samples):
        grouped.setdefault(_source_case_key(sample, idx), []).append(sample)
    for group in grouped.values():
        group.sort(key=lambda s: int(s.get("prefix_len", s.get("length", 0))))

    case_keys = sorted(grouped)
    random.Random(random_state).shuffle(case_keys)
    priorities = {
        key: _prefix_priority_indices(len(grouped[key]))
        for key in case_keys
    }
    selected: List[Dict[str, Any]] = []
    round_idx = 0
    while len(selected) < sample_size:
        added = False
        for key in case_keys:
            order = priorities[key]
            if round_idx < len(order):
                selected.append(grouped[key][order[round_idx]])
                added = True
                if len(selected) >= sample_size:
                    break
        if not added:
            break
        round_idx += 1
    return selected


def _sample_normalize(
    samples: List[Dict[str, Any]],
    config: PatternMiningConfig,
    sample_for_fit: bool = False,
) -> List[Dict[str, Any]]:
    normalized = []

    for idx, sample in enumerate(samples):
        s = dict(sample)
        source_case = _source_case_key(s, idx)
        s["case_id"] = source_case
        s["source_case_id"] = source_case
        s["pattern_sample_id"] = _pattern_sample_key(s, idx)
        s["activities"] = list(s.get("activities", []))
        s["times"] = list(s.get("times", [])) if "times" in s else []
        s["length"] = int(s.get("length", len(s["activities"])))

        if len(s["activities"]) != s["length"]:
            s["length"] = min(len(s["activities"]), s["length"])
            s["activities"] = s["activities"][: s["length"]]
            if s["times"]:
                s["times"] = s["times"][: s["length"]]

        # length includes the held-out next event in the PPM dataset. A
        # pattern feature may only inspect the actually observed prefix.
        visible_len = int(s.get("prefix_len", s["length"]))
        visible_len = max(0, min(visible_len, len(s["activities"])))
        s["activities"] = s["activities"][:visible_len]
        if s["times"]:
            s["times"] = s["times"][:visible_len]
        s["length"] = visible_len
        normalized.append(_truncate_sample(s, config.max_trace_len))

    if sample_for_fit and len(normalized) > config.sample_size:
        normalized = _case_balanced_sample(
            normalized,
            sample_size=int(config.sample_size),
            random_state=int(config.random_state),
        )

    return normalized


def _enumerate_trace_patterns(
    acts: List[Any],
    times: Optional[List[Any]] = None,
    max_len: int = 4,
    max_gap: int = 5,
    min_len: int = 1,
    include_time_buckets: bool = False,
) -> set:
    """
    Enumerate ordered subsequence patterns with bounded gaps.
    """
    tokens = _tokenize_trace(
        acts,
        times,
        include_time_buckets=include_time_buckets,
    )
    n = len(tokens)
    if n == 0:
        return set()

    patterns = set()
    min_len = max(1, int(min_len))

    def dfs(last_pos: int, current: List[str]) -> None:
        if len(current) >= min_len:
            patterns.add(tuple(current))
        if len(current) >= max_len:
            return

        start = 0 if last_pos < 0 else last_pos + 1
        end = n if last_pos < 0 else min(n, last_pos + max_gap + 2)

        for pos in range(start, end):
            current.append(tokens[pos])
            dfs(pos, current)
            current.pop()

    for i in range(n):
        dfs(i, [tokens[i]])

    return patterns


def _count_pattern_occurrences(
    acts: List[Any],
    pattern: Tuple[str, ...],
    max_gap: int = 5,
) -> int:
    seq = [_safe_str(a) for a in acts]
    pat = tuple(pattern)
    n = len(seq)
    m = len(pat)

    if m == 0 or n == 0:
        return 0

    @lru_cache(maxsize=None)
    def rec(prev_pos: int, p_idx: int) -> int:
        if p_idx == m:
            return 1

        lo = prev_pos + 1
        hi = n if prev_pos < 0 else min(n, prev_pos + max_gap + 2)
        total = 0

        for pos in range(lo, hi):
            if seq[pos] == pat[p_idx]:
                total += rec(pos, p_idx + 1)

        return total

    return rec(-1, 0)


def _pattern_to_column_name(pattern: Tuple[str, ...]) -> str:
    return _pattern_id(pattern)


def _prepare_samples(
    samples: List[Dict[str, Any]],
    config: PatternMiningConfig,
    sample_for_fit: bool = False,
) -> List[Dict[str, Any]]:
    samples = _sample_normalize(samples, config, sample_for_fit=sample_for_fit)
    prepared = []

    for sample in samples:
        s = dict(sample)
        s["case_id"] = _safe_str(s.get("case_id"))
        s["pattern_sample_id"] = _safe_str(
            s.get("pattern_sample_id", s.get("case_id"))
        )
        s["activities"] = [_safe_str(a) for a in s.get("activities", [])]
        s["times"] = list(s.get("times", [])) if s.get("times", None) is not None else []
        s["length"] = len(s["activities"])
        prepared.append(s)

    return prepared


def _compute_feature_for_pattern(
    pattern: Tuple[str, ...],
    traces_patterns: List[set],
    traces_tokens: List[List[str]],
    case_ids: List[str],
    y_eval: pd.Series,
    config: PatternMiningConfig,
) -> Dict[str, Any]:
    if config.feature_binary:
        feat = pd.Series(
            [1.0 if pattern in pats else 0.0 for pats in traces_patterns],
            index=pd.Index(case_ids, name="case_id"),
            name=_pattern_to_column_name(pattern),
        )
    else:
        feat = pd.Series(
            [
                _count_pattern_occurrences(tokens, pattern, max_gap=config.max_gap)
                for tokens in traces_tokens
            ],
            index=pd.Index(case_ids, name="case_id"),
            name=_pattern_to_column_name(pattern),
            dtype=float,
        )

    coverage = float((feat > 0).mean())
    support_count = int((feat > 0).sum())
    label_free = str(config.selection_mode).strip().lower() == "label_free"
    mi = 0.0 if label_free else _safe_mutual_info(
        feat, y_eval, target_type=config.outcome_type
    )

    if label_free:
        hit_rate_diff = 0.0
    elif config.outcome_type == "binary":
        pos_mask = y_eval.astype(bool)
        pos_rate = float(feat[pos_mask].mean()) if bool(pos_mask.any()) else 0.0
        neg_rate = float(feat[~pos_mask].mean()) if bool((~pos_mask).any()) else 0.0
        hit_rate_diff = pos_rate - neg_rate
    elif config.outcome_type == "categorical":
        class_rates = feat.groupby(y_eval.astype(str)).mean()
        hit_rate_diff = (
            float(class_rates.max() - class_rates.min())
            if len(class_rates) > 1 else 0.0
        )
    else:
        y_num = pd.to_numeric(y_eval, errors="coerce").fillna(0.0).astype(float)
        if feat.nunique(dropna=False) > 1 and y_num.nunique(dropna=False) > 1:
            corr = float(np.corrcoef(feat.values.astype(float), y_num.values.astype(float))[0, 1])
            hit_rate_diff = 0.0 if np.isnan(corr) else corr
        else:
            hit_rate_diff = 0.0

    freq_interest = coverage * (1.0 - coverage)
    len_penalty = config.score_len_penalty * max(0, len(pattern) - 1)
    balanced_coverage = 4.0 * freq_interest
    if label_free:
        # Search knowledge should favour prevalent process anchors. Balanced
        # coverage remains a secondary diversity term; no task label is used.
        score = (
            0.75 * coverage + 0.25 * balanced_coverage - len_penalty
        )
    else:
        score = (
            config.score_w_mi * mi
            + config.score_w_coverage * coverage
            + config.score_w_hit_rate * abs(hit_rate_diff)
            + config.score_w_freq_interest * balanced_coverage
            - len_penalty
        )

    return {
        "pattern": pattern,
        "pattern_id": _pattern_to_column_name(pattern),
        "length": len(pattern),
        "support_count": support_count,
        "coverage": coverage,
        "mi": mi,
        "hit_rate_diff": hit_rate_diff,
        "freq_interest": freq_interest,
        "score": score,
    }


class TransformerPatternMiner:
    def __init__(
        self,
        config: Optional[PatternMiningConfig] = None,
        include_time_buckets: bool = False,
        binary_features: bool = True,
    ):
        self.config = config if config is not None else PatternMiningConfig()
        self.include_time_buckets = include_time_buckets
        self.binary_features = binary_features

        self.selected_patterns_: List[Tuple[str, ...]] = []
        self.pattern_to_col_: Dict[Tuple[str, ...], str] = {}
        self.pattern_attributes_: pd.DataFrame = pd.DataFrame()
        self.train_features_: pd.DataFrame = pd.DataFrame()
        self.case_ids_: List[str] = []
        self.target_: Optional[pd.Series] = None

    def _trace_patterns(self, sample: Dict[str, Any]) -> set:
        return _enumerate_trace_patterns(
            sample.get("activities", []),
            sample.get("times", []),
            max_len=self.config.max_pattern_len,
            max_gap=self.config.max_gap,
            min_len=self.config.min_pattern_len,
            include_time_buckets=self.include_time_buckets,
        )

    def _trace_tokens(self, sample: Dict[str, Any]) -> List[str]:
        return _tokenize_trace(
            sample.get("activities", []),
            sample.get("times", []),
            include_time_buckets=self.include_time_buckets,
        )

    def fit(
        self,
        samples: List[Dict[str, Any]],
        target_col: str = "outcome",
    ) -> "TransformerPatternMiner":
        samples = _prepare_samples(samples, self.config, sample_for_fit=True)

        if not samples:
            self.selected_patterns_ = []
            self.pattern_to_col_ = {}
            self.pattern_attributes_ = pd.DataFrame()
            self.train_features_ = pd.DataFrame()
            self.case_ids_ = []
            self.target_ = None
            return self

        case_ids = [s["pattern_sample_id"] for s in samples]
        traces_patterns = []
        traces_tokens = []
        targets = []

        label_free = str(self.config.selection_mode).strip().lower() == "label_free"
        if not label_free:
            missing_target = [
                s.get("pattern_sample_id", s.get("case_id", "?"))
                for s in samples if target_col not in s
            ]
            if missing_target:
                raise ValueError(
                    f"Supervised pattern discovery requires explicit target "
                    f"'{target_col}'; missing for {len(missing_target)} samples"
                )

        for sample in samples:
            traces_patterns.append(self._trace_patterns(sample))
            traces_tokens.append(self._trace_tokens(sample))
            targets.append(0 if label_free else sample[target_col])

        self.case_ids_ = case_ids
        y = pd.Series(targets, index=pd.Index(case_ids, name="case_id"), name=target_col)
        self.target_ = y

        if label_free:
            y_eval = pd.Series(0, index=y.index, dtype=int)
        elif self.config.outcome_type == "binary":
            y_eval = _label_mask(y, self.config.factual_outcome).astype(int)
        elif self.config.outcome_type == "categorical":
            y_eval = y.astype(str).fillna("NA")
        else:
            y_eval = pd.to_numeric(y, errors="coerce")

        pattern_case_counts = Counter()
        for pats in traces_patterns:
            pattern_case_counts.update(pats)

        n_cases = len(samples)
        min_support_count = max(1, int(math.ceil(self.config.min_support * n_cases)))

        candidate_patterns = [
            pat
            for pat, cnt in pattern_case_counts.items()
            if cnt >= min_support_count
            and (self.config.include_unigrams or len(pat) > 1)
            and self.config.min_pattern_len <= len(pat) <= self.config.max_pattern_len
        ]

        if self.config.prune_by_length_first:
            candidate_patterns.sort(key=lambda p: (len(p), -pattern_case_counts[p]))
            candidate_patterns = candidate_patterns[: self.config.candidate_cap]
        elif self.config.prune_by_frequency:
            candidate_patterns.sort(key=lambda p: (-pattern_case_counts[p], len(p)))
            candidate_patterns = candidate_patterns[: self.config.candidate_cap]

        if not candidate_patterns:
            self.selected_patterns_ = []
            self.pattern_to_col_ = {}
            self.pattern_attributes_ = pd.DataFrame()
            self.train_features_ = pd.DataFrame(index=pd.Index(case_ids, name="case_id"))
            return self

        if Parallel is not None and delayed is not None and len(candidate_patterns) > 1:
            rows = Parallel(n_jobs=self.config.n_jobs, prefer="threads")(
                delayed(_compute_feature_for_pattern)(
                    pattern,
                    traces_patterns,
                    traces_tokens,
                    case_ids,
                    y_eval,
                    self.config,
                )
                for pattern in candidate_patterns
            )
        else:
            rows = [
                _compute_feature_for_pattern(
                    pattern,
                    traces_patterns,
                    traces_tokens,
                    case_ids,
                    y_eval,
                    self.config,
                )
                for pattern in candidate_patterns
            ]

        attr_df = pd.DataFrame(rows)
        if attr_df.empty:
            self.selected_patterns_ = []
            self.pattern_to_col_ = {}
            self.pattern_attributes_ = attr_df
            self.train_features_ = pd.DataFrame(
                index=pd.Index(case_ids, name="pattern_sample_id")
            )
            return self

        selection_mask = (
            (attr_df["coverage"] >= float(self.config.min_feature_coverage))
            & (attr_df["coverage"] <= float(self.config.max_feature_coverage))
        )
        if not label_free:
            selection_mask &= (
                attr_df["hit_rate_diff"].abs()
                >= float(self.config.min_abs_hit_rate_diff)
            )
        attr_df = attr_df[selection_mask].copy()
        if attr_df.empty:
            self.selected_patterns_ = []
            self.pattern_to_col_ = {}
            self.pattern_attributes_ = attr_df
            self.train_features_ = pd.DataFrame(
                index=pd.Index(case_ids, name="pattern_sample_id")
            )
            return self

        if self.config.use_pareto and len(attr_df) > 1:
            try:
                pareto_objectives = attr_df[
                    ["coverage", "mi", "length"]
                ].assign(
                    hit_rate_diff_abs=attr_df["hit_rate_diff"].abs()
                )
                pareto_sense = ["max", "max", "min", "max"]
                if callable(paretoset):
                    pareto_mask = paretoset(
                        pareto_objectives,
                        sense=pareto_sense,
                    )
                else:
                    pareto_mask = _pareto_mask_fallback(
                        pareto_objectives,
                        pareto_sense,
                    )
                attr_df = attr_df.loc[pareto_mask].copy()
            except Exception:
                pass

        attr_df = attr_df.sort_values(
            by=["score", "coverage", "mi", "hit_rate_diff", "length"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)

        if self.config.top_k is not None and self.config.top_k > 0:
            attr_df = attr_df.head(self.config.top_k).copy()

        self.pattern_attributes_ = attr_df
        self.selected_patterns_ = list(attr_df["pattern"].values)
        self.pattern_to_col_ = {
            pat: _pattern_to_column_name(pat)
            for pat in self.selected_patterns_
        }

        self.train_features_ = self.transform(samples, include_target=False)
        return self

    def transform(
        self,
        samples: List[Dict[str, Any]],
        include_target: bool = False,
        target_col: str = "outcome",
    ) -> pd.DataFrame:
        samples = _prepare_samples(samples, self.config, sample_for_fit=False)

        if not samples:
            return pd.DataFrame()

        case_ids = [s["pattern_sample_id"] for s in samples]
        index = pd.Index(case_ids, name="pattern_sample_id")
        out = pd.DataFrame(index=index)

        if not self.selected_patterns_:
            if include_target:
                out[target_col] = [s.get(target_col, s.get("outcome", 0)) for s in samples]
            return out

        traces_patterns = [self._trace_patterns(s) for s in samples]
        traces_tokens = [self._trace_tokens(s) for s in samples]

        if Parallel is not None and delayed is not None and len(self.selected_patterns_) > 1:
            cols = Parallel(n_jobs=self.config.n_jobs, prefer="threads")(
                delayed(self._build_single_feature_column)(
                    pat,
                    traces_patterns,
                    traces_tokens,
                )
                for pat in self.selected_patterns_
            )
            for col_name, values in cols:
                out[col_name] = values
        else:
            for pat in self.selected_patterns_:
                col_name, values = self._build_single_feature_column(
                    pat,
                    traces_patterns,
                    traces_tokens,
                )
                out[col_name] = values

        if include_target:
            out[target_col] = [s.get(target_col, s.get("outcome", 0)) for s in samples]

        return out

    def _build_single_feature_column(
        self,
        pattern: Tuple[str, ...],
        traces_patterns: List[set],
        traces_tokens: List[List[str]],
    ) -> Tuple[str, List[float]]:
        col = self.pattern_to_col_[pattern]
        if self.binary_features:
            values = [1.0 if pattern in pats else 0.0 for pats in traces_patterns]
        else:
            values = [
                _count_pattern_occurrences(tokens, pattern, max_gap=self.config.max_gap)
                for tokens in traces_tokens
            ]
        return col, values

    def fit_transform(
        self,
        samples: List[Dict[str, Any]],
        target_col: str = "outcome",
        include_target: bool = True,
    ) -> pd.DataFrame:
        self.fit(samples, target_col=target_col)
        return self.transform(
            samples,
            include_target=include_target,
            target_col=target_col,
        )

    def get_pattern_table(self) -> pd.DataFrame:
        return self.pattern_attributes_.copy()

    def get_selected_patterns(self) -> List[Tuple[str, ...]]:
        return list(self.selected_patterns_)


def compute_pattern_attributes(
    pattern_dict: Any,
    patient_data: pd.DataFrame,
    factual_outcome: Any = 1,
    target_col: str = "label",
    outcome_type: str = "binary",
) -> pd.DataFrame:
    if patient_data is None or patient_data.empty:
        return pd.DataFrame()

    df = patient_data.copy()
    df.index = df.index.astype(str)

    if target_col in df.columns:
        y = df[target_col]
    else:
        y = None

    resolved = []
    if isinstance(pattern_dict, pd.DataFrame):
        if "pattern_id" in pattern_dict.columns:
            for _, row in pattern_dict.iterrows():
                col = _safe_str(row["pattern_id"])
                if col in df.columns:
                    resolved.append((row.get("pattern", col), col))
        else:
            for col in pattern_dict.columns:
                if col in df.columns:
                    resolved.append((col, col))
    elif isinstance(pattern_dict, dict):
        for k, v in pattern_dict.items():
            pat = k
            if isinstance(v, dict):
                col = v.get("column", v.get("pattern_id", None))
                pat = v.get("pattern", k)
            elif isinstance(v, str):
                col = v
            elif hasattr(v, "nodes"):
                # The active pipeline passes ``pattern_id -> nx.DiGraph``.
                # Its encoded feature column is the pattern ID, not the graph's
                # string representation.  Recover node values for length and
                # scoring while preserving the stable ID-to-column alignment.
                col = _safe_str(k)
                try:
                    ordered_nodes = sorted(v.nodes())
                    pat = tuple(
                        _safe_str(v.nodes[node].get("value", node))
                        for node in ordered_nodes
                    )
                except Exception:
                    pat = k
            elif isinstance(k, tuple):
                col = _pattern_to_column_name(k)
            else:
                col = _safe_str(k)

            if col in df.columns:
                resolved.append((pat, col))
    elif isinstance(pattern_dict, (list, tuple)):
        for item in pattern_dict:
            if isinstance(item, tuple):
                col = _pattern_to_column_name(item)
                if col in df.columns:
                    resolved.append((item, col))
            elif isinstance(item, str) and item in df.columns:
                resolved.append((item, item))

    rows = []
    for pat, col in resolved:
        feat = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
        coverage = float((feat > 0).mean())
        support_count = int((feat > 0).sum())
        length = len(pat) if isinstance(pat, tuple) else 1

        if y is not None:
            if outcome_type == "binary":
                mask = _label_mask(y, factual_outcome)
                pos_rate = float(feat[mask].mean()) if bool(mask.any()) else 0.0
                neg_rate = float(feat[~mask].mean()) if bool((~mask).any()) else 0.0
                hit_rate_diff = pos_rate - neg_rate
                mi = _safe_mutual_info(feat, mask.astype(int), target_type="binary")
            else:
                y_num = pd.to_numeric(y, errors="coerce")
                if feat.nunique(dropna=False) > 1 and y_num.nunique(dropna=False) > 1:
                    corr = float(np.corrcoef(feat.values.astype(float), y_num.fillna(0.0).values.astype(float))[0, 1])
                    hit_rate_diff = 0.0 if np.isnan(corr) else corr
                else:
                    hit_rate_diff = 0.0
                mi = _safe_mutual_info(feat, y_num, target_type="numerical")
        else:
            hit_rate_diff = 0.0
            mi = 0.0

        freq_interest = coverage * (1.0 - coverage)
        balanced_coverage = 4.0 * freq_interest
        score = (
            0.55 * mi
            + 0.15 * abs(hit_rate_diff)
            + 0.10 * balanced_coverage
            - 0.02 * max(0, length - 1)
        )

        rows.append(
            {
                "pattern": pat,
                "pattern_id": col,
                "length": length,
                "support_count": support_count,
                "coverage": coverage,
                "mi": mi,
                "hit_rate_diff": hit_rate_diff,
                "freq_interest": freq_interest,
                "score": score,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            by=["score", "coverage", "mi", "hit_rate_diff", "length"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
    return out


def rank_patterns_for_transformer(
    pattern_dict: Any,
    patient_data: pd.DataFrame,
    factual_outcome: Any = 1,
    target_col: str = "label",
    outcome_type: str = "binary",
    top_k: int = 200,
) -> pd.DataFrame:
    attrs = compute_pattern_attributes(
        pattern_dict=pattern_dict,
        patient_data=patient_data,
        factual_outcome=factual_outcome,
        target_col=target_col,
        outcome_type=outcome_type,
    )
    if attrs.empty:
        return attrs
    return attrs.head(top_k).copy()


def encode_pattern_features(
    samples: List[Dict[str, Any]],
    config: Optional[PatternMiningConfig] = None,
    target_col: str = "outcome",
    include_target: bool = True,
    include_time_buckets: bool = False,
    binary_features: bool = True,
) -> Tuple[pd.DataFrame, TransformerPatternMiner]:
    miner = TransformerPatternMiner(
        config=config if config is not None else PatternMiningConfig(),
        include_time_buckets=include_time_buckets,
        binary_features=binary_features,
    )
    features = miner.fit_transform(
        samples,
        target_col=target_col,
        include_target=include_target,
    )
    return features, miner


def build_transformer_ready_dataset(
    samples: List[Dict[str, Any]],
    config: Optional[PatternMiningConfig] = None,
    target_col: str = "outcome",
    test_size: Optional[float] = None,
    include_time_buckets: bool = False,
    binary_features: bool = True,
    random_state: Optional[int] = None,
):
    cfg = config if config is not None else PatternMiningConfig()
    if random_state is not None:
        cfg = PatternMiningConfig(**{**cfg.__dict__, "random_state": int(random_state)})

    miner = TransformerPatternMiner(
        config=cfg,
        include_time_buckets=include_time_buckets,
        binary_features=binary_features,
    )
    full_df = miner.fit_transform(
        samples,
        target_col=target_col,
        include_target=True,
    )

    if test_size is None or train_test_split is None or full_df.empty:
        return full_df, miner

    train_df, test_df = train_test_split(
        full_df,
        test_size=float(test_size),
        random_state=cfg.random_state,
        shuffle=True,
    )
    return train_df.sort_index(), test_df.sort_index(), miner


def samples_to_dataframe(
    samples: List[Dict[str, Any]],
    act_encoder=None,
) -> pd.DataFrame:
    rows = []
    for idx, sample in enumerate(samples):
        case_id = _safe_str(sample.get("case_id", f"Case_{idx}"))
        act_ids = list(sample.get("activities", []))
        times = list(sample.get("times", []))
        outcome = sample.get("outcome", 0)

        if act_encoder is not None:
            act_names = [
                act_encoder.inverse_transform([a - 1])[0]
                if isinstance(a, (int, np.integer)) and a > 0
                else "PAD"
                for a in act_ids
            ]
        else:
            act_names = [_safe_str(a) for a in act_ids]

        base_time = pd.Timestamp("2016-01-01 00:00:00")
        for i, (act, t) in enumerate(zip(act_names, times)):
            try:
                current_time = base_time + pd.Timedelta(hours=float(t))
            except Exception:
                current_time = base_time + pd.Timedelta(hours=float(i))

            rows.append(
                {
                    "Case ID": case_id,
                    "Activity": act,
                    "time:timestamp": current_time,
                    "label": outcome,
                    "likelihood": 0.5,
                }
            )

    return pd.DataFrame(rows)


def df_to_samples(
    df: pd.DataFrame,
    case_id: str,
    activity: str,
    timestamp: str,
    outcome: str,
) -> List[Dict[str, Any]]:
    samples = []
    df = df.copy()
    df[timestamp] = pd.to_datetime(df[timestamp], errors="coerce")

    for cid, group in df.groupby(case_id, sort=False):
        group = group.sort_values(timestamp)
        activities = group[activity].astype(str).tolist()
        t0 = group[timestamp].iloc[0]

        if pd.isna(t0):
            times = list(range(len(activities)))
        else:
            times = [
                (ts - t0).total_seconds() / 3600.0 if pd.notna(ts) else i
                for i, ts in enumerate(group[timestamp])
            ]

        samples.append(
            {
                "case_id": _safe_str(cid),
                "activities": activities,
                "times": times,
                "outcome": group[outcome].iloc[0],
                "length": len(activities),
            }
        )

    return samples


def save_pattern_miner(miner: TransformerPatternMiner, path: str) -> None:
    import pickle

    with open(path, "wb") as f:
        pickle.dump(miner, f)


def load_pattern_miner(path: str) -> TransformerPatternMiner:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)
