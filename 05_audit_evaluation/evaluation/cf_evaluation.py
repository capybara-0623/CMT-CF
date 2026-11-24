"""Counterfactual recommendation metrics and DFA compliance checks."""

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import median_abs_deviation

from explaination.visualizations.utils import normalize_activity_name, trace_accepts_dfa
from trace_alignment import does_trace_contain_pattern_advanced


def _as_index_list(idx):
    return list(idx) if idx is not None else []


def _aggregate(values, agg=None):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    if agg is None or agg == "mean":
        return float(np.mean(values))
    if agg == "max":
        return float(np.max(values))
    if agg == "min":
        return float(np.min(values))
    raise ValueError(f"Unsupported aggregation: {agg}")


def mad_cityblock(u, v, mad):
    """MAD-weighted Manhattan distance."""
    return np.sum(np.abs(u - v) / mad)


def continuous_distance(query, cf_list, cont_feature_idx, metric="euclidean", X=None, agg=None):
    cont_feature_idx = _as_index_list(cont_feature_idx)
    if len(cont_feature_idx) == 0:
        return 0.0
    if metric == "mad" and X is not None:
        mad = median_abs_deviation(X[:, cont_feature_idx], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])
        dist = cdist(
            query[:, cont_feature_idx],
            cf_list[:, cont_feature_idx],
            metric=lambda u, v: mad_cityblock(u, v, mad),
        )
    else:
        dist = cdist(query[:, cont_feature_idx], cf_list[:, cont_feature_idx], metric=metric)
    return _aggregate(dist, agg)


def _categorical_sequence_distance_matrix(left, right):
    """Position-wise mismatch distance for integer activity IDs.

    scipy's Jaccard treats all non-zero integers as True, so activity 3 and
    activity 8 can look identical.  Process traces need ID-level comparison.
    Padding positions are ignored unless only one side is padded.
    """
    left = np.asarray(left)
    right = np.asarray(right)
    if left.ndim == 1:
        left = left.reshape(1, -1)
    if right.ndim == 1:
        right = right.reshape(1, -1)

    mismatch = left[:, None, :] != right[None, :, :]
    valid = (left[:, None, :] != 0) | (right[None, :, :] != 0)
    denom = valid.sum(axis=2)
    numer = (mismatch & valid).sum(axis=2)
    return np.divide(
        numer,
        denom,
        out=np.zeros_like(numer, dtype=float),
        where=denom > 0,
    )


def categorical_distance(query, cf_list, cat_feature_idx, metric="jaccard", agg=None):
    cat_feature_idx = _as_index_list(cat_feature_idx)
    if len(cat_feature_idx) == 0:
        return 0.0
    query_cat = query[:, cat_feature_idx]
    cf_cat = cf_list[:, cat_feature_idx]
    if metric in ("jaccard", "hamming", "sequence"):
        dist = _categorical_sequence_distance_matrix(query_cat, cf_cat)
    else:
        dist = cdist(query_cat, cf_cat, metric=metric)
    return _aggregate(dist, agg)


def continuous_diversity(cf_list, cont_feature_idx, metric="euclidean", X=None, agg=None):
    cont_feature_idx = _as_index_list(cont_feature_idx)
    if len(cont_feature_idx) == 0 or len(cf_list) < 2:
        return 0.0
    if metric == "mad" and X is not None:
        mad = median_abs_deviation(X[:, cont_feature_idx], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])
        dist = pdist(cf_list[:, cont_feature_idx], metric=lambda u, v: mad_cityblock(u, v, mad))
    else:
        dist = pdist(cf_list[:, cont_feature_idx], metric=metric)
    return _aggregate(dist, agg)


def categorical_diversity(cf_list, cat_feature_idx, metric="jaccard", agg=None):
    cat_feature_idx = _as_index_list(cat_feature_idx)
    if len(cat_feature_idx) == 0 or len(cf_list) < 2:
        return 0.0
    cf_cat = cf_list[:, cat_feature_idx]
    if metric in ("jaccard", "hamming", "sequence"):
        dist_mat = _categorical_sequence_distance_matrix(cf_cat, cf_cat)
        dist = dist_mat[np.triu_indices(len(cf_cat), k=1)]
    else:
        dist = pdist(cf_cat, metric=metric)
    return _aggregate(dist, agg)


def _ratio_cont(cont_feat_idx, cat_feat_idx, ratio_cont=None):
    if ratio_cont is not None:
        return ratio_cont
    denom = len(cont_feat_idx) + len(cat_feat_idx)
    return len(cont_feat_idx) / denom if denom > 0 else 0.0


def distance_l2j(query, cf_list, cont_feat_idx, cat_feat_idx, ratio_cont=None, agg=None):
    cont_feat_idx = _as_index_list(cont_feat_idx)
    cat_feat_idx = _as_index_list(cat_feat_idx)
    d_cont = continuous_distance(query, cf_list, cont_feat_idx, metric="euclidean", X=None, agg=agg)
    d_cat = categorical_distance(query, cf_list, cat_feat_idx, metric="jaccard", agg=agg)
    r = _ratio_cont(cont_feat_idx, cat_feat_idx, ratio_cont)
    return r * d_cont + (1 - r) * d_cat


def distance_mh(query, cf_list, cont_feat_idx, cat_feat_idx, X, ratio_cont=None, agg=None):
    cont_feat_idx = _as_index_list(cont_feat_idx)
    cat_feat_idx = _as_index_list(cat_feat_idx)
    d_cont = continuous_distance(query, cf_list, cont_feat_idx, metric="mad", X=X, agg=agg)
    d_cat = categorical_distance(query, cf_list, cat_feat_idx, metric="hamming", agg=agg)
    r = _ratio_cont(cont_feat_idx, cat_feat_idx, ratio_cont)
    return r * d_cont + (1 - r) * d_cat


def diversity_l2j(cf_list, cont_feat_idx, cat_feat_idx, ratio_cont=None, agg=None):
    cont_feat_idx = _as_index_list(cont_feat_idx)
    cat_feat_idx = _as_index_list(cat_feat_idx)
    d_cont = continuous_diversity(cf_list, cont_feat_idx, metric="euclidean", X=None, agg=agg)
    d_cat = categorical_diversity(cf_list, cat_feat_idx, metric="jaccard", agg=agg)
    r = _ratio_cont(cont_feat_idx, cat_feat_idx, ratio_cont)
    return r * d_cont + (1 - r) * d_cat


def diversity_mh(cf_list, cont_feat_idx, cat_feat_idx, X, ratio_cont=None, agg=None):
    cont_feat_idx = _as_index_list(cont_feat_idx)
    cat_feat_idx = _as_index_list(cat_feat_idx)
    d_cont = continuous_diversity(cf_list, cont_feat_idx, metric="mad", X=X, agg=agg)
    d_cat = categorical_diversity(cf_list, cat_feat_idx, metric="hamming", agg=agg)
    r = _ratio_cont(cont_feat_idx, cat_feat_idx, ratio_cont)
    return r * d_cont + (1 - r) * d_cat


def _change_stats(query_instance, cf_list, cat_feat_idx):
    cat_feat_idx = _as_index_list(cat_feat_idx)
    if len(cf_list) == 0 or len(cat_feat_idx) == 0:
        return np.nan, np.nan, 0

    query_acts = np.asarray(query_instance[0, cat_feat_idx])
    valid = np.where(query_acts != 0)[0]
    effective_len = int(valid[-1] + 1) if len(valid) > 0 else len(cat_feat_idx)
    if effective_len <= 0:
        return np.nan, np.nan, 0

    changed = 0.0
    for cf in cf_list:
        cf_acts = np.asarray(cf[cat_feat_idx[:effective_len]])
        changed += float(np.sum(cf_acts != query_acts[:effective_len]))

    avg_changed_positions = changed / len(cf_list)
    sparsity_rate = changed / (len(cf_list) * effective_len)
    return sparsity_rate, avg_changed_positions, effective_len


def avg_nbr_changes(query_instance, cf_list, cont_feat_idx, cat_feat_idx):
    """Normalized fraction of changed activity positions."""
    sparsity_rate, _, _ = _change_stats(query_instance, cf_list, cat_feat_idx)
    return sparsity_rate


def implausibility(query_instance, cf_list, cont_feat_idx, cat_feat_idx, reference_population, ratio_cont=None):
    """Average distance from each CF to its nearest training reference trace."""
    cont_feat_idx = _as_index_list(cont_feat_idx)
    cat_feat_idx = _as_index_list(cat_feat_idx)
    if len(cf_list) == 0 or reference_population is None or len(reference_population) == 0:
        return np.nan

    feat_idx = cont_feat_idx + cat_feat_idx
    all_data = reference_population[:, feat_idx]
    cont_local = list(range(len(cont_feat_idx)))
    cat_local = list(range(len(cont_feat_idx), len(cont_feat_idx) + len(cat_feat_idx)))

    sum_dist = 0.0
    for cf in cf_list:
        cf_vec = cf.reshape(1, -1)[:, feat_idx]
        dist = distance_mh(
            cf_vec,
            all_data,
            cont_feat_idx=cont_local,
            cat_feat_idx=cat_local,
            X=all_data,
            ratio_cont=ratio_cont,
            agg="min",
        )
        sum_dist += dist
    return sum_dist / len(cf_list)


def compliance_score(cf_traces, original_activities, dfa, alphabet_map=None):
    """Return DFA compliance rate and per-trace flags."""
    if not cf_traces:
        return 0.0, []
    if alphabet_map is None:
        alphabet_map = {i: str(i) for i in set.union(*[set(t) for t in cf_traces])}

    compliant = []
    for trace in cf_traces:
        word = [alphabet_map[act] for act in trace]
        try:
            accepted = dfa.accepts(word)
        except AttributeError:
            accepted = _manual_dfa_accepts(dfa, word)
        compliant.append(accepted)
    return float(np.mean(compliant)) if compliant else 0.0, compliant


def _manual_dfa_accepts(dfa, word):
    state = dfa.initial_state
    for letter in word:
        trans = dfa.transition_function.get(state, {})
        if letter in trans:
            state = trans[letter]
        elif "*" in trans:
            state = trans["*"]
        else:
            return False
    return state in dfa.accepting_states


def evaluate_counterfactuals(
    query_activities,
    query_times,
    recommended_sequences,
    cont_feat_idx,
    cat_feat_idx,
    reference_data,
    dfa=None,
    activity_to_letter=None,
    ratio_cont=None,
    pattern_dict=None,
    act_encoder=None,
    pattern_delta_time=3600,
    comp_rate=None,
):
    """Evaluate generated counterfactual activity-time sequences."""
    query_feat = np.hstack([query_activities, query_times]).reshape(1, -1).astype(float)
    cf_list = []
    for act_seq, time_seq in recommended_sequences:
        cf_list.append(np.hstack([act_seq, time_seq]).astype(float))
    cf_list = np.array(cf_list)

    if cf_list.size == 0:
        return {
            "nbr_cfs": 0,
            "implausibility": np.nan,
            "distance_l2j": np.nan,
            "distance_mh": np.nan,
            "diversity_l2j": np.nan,
            "diversity_mh": np.nan,
            "avg_nbr_changes": np.nan,
            "avg_n_changed_positions": np.nan,
            "compliance_rate": np.nan,
            "pattern_hit_rate": np.nan,
        }

    cont_feat_idx = _as_index_list(cont_feat_idx)
    cat_feat_idx = _as_index_list(cat_feat_idx)

    dist_l2j = distance_l2j(query_feat, cf_list, cont_feat_idx, cat_feat_idx, ratio_cont=ratio_cont, agg="mean")
    dist_mh = distance_mh(query_feat, cf_list, cont_feat_idx, cat_feat_idx, X=reference_data,
                          ratio_cont=ratio_cont, agg="mean")
    div_l2j = diversity_l2j(cf_list, cont_feat_idx, cat_feat_idx, ratio_cont=ratio_cont, agg="mean")
    div_mh = diversity_mh(cf_list, cont_feat_idx, cat_feat_idx, X=reference_data,
                          ratio_cont=ratio_cont, agg="mean")

    sparsity, avg_changed_positions, _ = _change_stats(query_feat, cf_list, cat_feat_idx)
    implaus = implausibility(query_feat, cf_list, cont_feat_idx, cat_feat_idx, reference_data, ratio_cont=ratio_cont)

    if comp_rate is not None:
        compliance_rate = float(comp_rate)
    elif dfa is not None and act_encoder is not None:
        compliant = 0
        for act_seq, _ in recommended_sequences:
            valid_acts = [a for a in act_seq if a != 0]
            names = [act_encoder.inverse_transform([a - 1])[0] if a > 0 else None for a in valid_acts]
            names = [normalize_activity_name(n) for n in names if n is not None]
            if trace_accepts_dfa(names, dfa):
                compliant += 1
        compliance_rate = compliant / len(recommended_sequences)
    else:
        compliance_rate = np.nan

    if pattern_dict is not None and act_encoder is not None:
        hits = 0
        for act_seq, time_seq in recommended_sequences:
            valid_indices = [i for i, a in enumerate(act_seq) if a != 0]
            valid_acts = [act_seq[i] for i in valid_indices]
            valid_times = [time_seq[i] for i in valid_indices]
            names = [act_encoder.inverse_transform([a - 1])[0] if a > 0 else None for a in valid_acts]
            names = [n for n in names if n is not None]
            if any(
                does_trace_contain_pattern_advanced(
                    names,
                    valid_times,
                    pattern_dict[pid],
                    delta_time_seconds=pattern_delta_time,
                )
                for pid in pattern_dict
            ):
                hits += 1
        pattern_hit_rate = hits / len(recommended_sequences)
    else:
        pattern_hit_rate = np.nan

    return {
        "nbr_cfs": len(cf_list),
        "implausibility": implaus,
        "distance_l2j": dist_l2j,
        "distance_mh": dist_mh,
        "diversity_l2j": div_l2j,
        "diversity_mh": div_mh,
        "avg_nbr_changes": sparsity,
        "avg_n_changed_positions": avg_changed_positions,
        "compliance_rate": compliance_rate,
        "pattern_hit_rate": pattern_hit_rate,
    }
