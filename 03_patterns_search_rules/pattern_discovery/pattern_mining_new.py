# pattern_mining_new.py — 高效版: 有界跳跃子序列 + 互信息筛选, 无 VF2/无 pm4py
import networkx as nx
import pandas as pd
import numpy as np
import os, pickle, glob, hashlib
from collections import Counter

# =========================================================================
# 核心: 有界跳跃子序列枚举 O(n·gap·len), 替代 VF2
# =========================================================================

def _enumerate_trace_patterns(tokens, max_len=4, max_gap=5):
    """DFS 枚举单条轨迹中所有有界跳跃子序列"""
    counts = Counter()
    n = len(tokens)
    if n == 0:
        return counts
    for i in range(n):
        base = (tokens[i],)
        counts[base] += 1
        stack = [(base, i)]
        while stack:
            pat, pos = stack.pop()
            if len(pat) >= max_len:
                continue
            for j in range(pos + 1, min(n, pos + max_gap + 2)):
                new_pat = pat + (tokens[j],)
                counts[new_pat] += 1
                stack.append((new_pat, j))
    return counts


def _build_pattern_graph(tokens, color="#888888"):
    g = nx.DiGraph()
    for i, tok in enumerate(tokens):
        g.add_node(i + 1, value=str(tok), parallel=False, color=color)
        if i > 0:
            g.add_edge(i, i + 1, eventually=True)
    return g


def _pattern_id(tokens):
    head = "_".join(str(t)[:8] for t in tokens[:3])
    digest = hashlib.sha1("||".join(str(t) for t in tokens).encode()).hexdigest()[:8]
    return f"P{len(tokens)}_{head}_{digest}"


# =========================================================================
# 主函数: 替代原来的 run_pattern_discovery (完全兼容接口)
# =========================================================================

def run_pattern_discovery(train_samples, act_encoder=None, output_path='./patterns',
                           return_dict=True, max_len=4, max_gap=5, top_k=200,
                           min_support=0.03):
    """高效模式发现: 无 VF2, 无 pm4py, 完全兼容原接口"""
    os.makedirs(output_path, exist_ok=True)

    # 1. 截断 + 采样
    for s in train_samples:
        if s["length"] > 30:
            s["activities"] = s["activities"][s["length"] - 30:]
            s["times"] = s["times"][s["length"] - 30:]
            s["length"] = 30
    import random as _r
    _r.seed(42)
    if len(train_samples) > 400:
        train_samples = _r.sample(train_samples, 400)

    n_total = len(train_samples)
    min_sup = max(1, int(min_support * n_total))

    # 2. 活动ID → 名称
    token_seqs, case_ids, labels = [], [], []
    for s in train_samples:
        case_ids.append(str(s.get("case_id", "")))
        labels.append(int(s.get("outcome", 0)))
        tokens = []
        for a in s["activities"]:
            if a > 0 and act_encoder:
                try:
                    tokens.append(act_encoder.inverse_transform([a - 1])[0])
                except Exception:
                    tokens.append(str(a))
            else:
                tokens.append(str(a))
        token_seqs.append(tokens)

    # 3. 全局枚举 + 支持度过滤
    global_sup = Counter()
    for tokens in token_seqs:
        counts = _enumerate_trace_patterns(tokens, max_len=max_len, max_gap=max_gap)
        for pat in counts:
            global_sup[pat] += 1
    candidates = [p for p, c in global_sup.items()
                  if c >= min_sup and 1 <= len(p) <= max_len]
    if len(candidates) > 2000:
        candidates.sort(key=lambda p: global_sup[p], reverse=True)
        candidates = candidates[:2000]

    # 4. 编码矩阵
    pids = [_pattern_id(p) for p in candidates]
    mat = np.zeros((n_total, len(candidates)), dtype=np.float32)
    for i, tokens in enumerate(token_seqs):
        counts = _enumerate_trace_patterns(tokens, max_len=max_len, max_gap=max_gap)
        for j, pat in enumerate(candidates):
            mat[i, j] = float(counts.get(pat, 0))

    patient_df = pd.DataFrame(mat, index=case_ids, columns=pids)
    patient_df["label"] = labels

    # 5. 属性计算 (互信息 + 命中率差异)
    from sklearn.feature_selection import mutual_info_classif
    label_arr = np.array(labels)
    rows = []
    for j, pat in enumerate(candidates):
        pid = pids[j]
        col = patient_df[pid].fillna(0)
        freq = float(col.sum())
        sup = int((col > 0).sum())
        cov = sup / n_total
        pos = col[np.array(labels) == 1]
        neg = col[np.array(labels) != 1]
        pa = float(pos.mean()) if len(pos) > 0 else 0.0
        na = float(neg.mean()) if len(neg) > 0 else 0.0
        fi = abs(na - pa) / max(abs(na), abs(pa), 1e-8)
        phr = float((pos > 0).mean()) if len(pos) > 0 else 0.0
        nhr = float((neg > 0).mean()) if len(neg) > 0 else 0.0
        try:
            mi = float(mutual_info_classif(col.values.reshape(-1, 1), label_arr,
                                           discrete_features=True)[0])
        except Exception:
            mi = 0.0
        rows.append({"pattern_id": pid, "frequency": freq, "support": sup,
                      "coverage": cov, "pos_avg_freq": pa, "neg_avg_freq": na,
                      "freq_interest": fi, "pos_hit_rate": phr, "neg_hit_rate": nhr,
                      "hit_rate_diff": nhr - phr, "outcome_interest": mi})

    attr_df = pd.DataFrame(rows).sort_values("outcome_interest", ascending=False)

    # 6. 保存模式图 (pickle) + 属性表 (csv)
    pattern_dict = {}
    for _, row in attr_df.iterrows():
        pid = row["pattern_id"]
        tokens = candidates[pids.index(pid)]
        g = _build_pattern_graph(tokens)
        pattern_dict[pid] = g
        try:
            pickle.dump(g, open(os.path.join(output_path, f"{pid}.pickle"), "wb"))
        except Exception:
            pass
    attr_df.to_csv(os.path.join(output_path, "pattern_attributes.csv"), index=False)

    print(f"模式发现完成: {len(candidates)} 模式特征, {len(pattern_dict)} 模式图")

    train_encoded = patient_df.drop(columns=["label"])
    if return_dict:
        return train_encoded, pattern_dict
    return train_encoded


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


def encode_pattern_features(samples, train_encoded, act_encoder=None):
    missing_count = 0
    for s in samples:
        cid = s.get('case_id')
        if cid is not None and cid in train_encoded.index:
            s['pattern_features'] = train_encoded.loc[cid].values.astype(np.float32).tolist()
        else:
            s['pattern_features'] = []
            missing_count += 1
    if missing_count > 0:
        print(f"模式特征注入: {missing_count}/{len(samples)} 个样本未找到")
    return samples


def compute_pattern_attributes(pattern_dict, patient_data, outcome_col='label', factual_outcome=1):
    from sklearn.feature_selection import mutual_info_classif
    rows = []
    pattern_ids = list(pattern_dict.keys())
    for pid in pattern_ids:
        if pid not in patient_data.columns:
            continue
        col = patient_data[pid].fillna(0)
        freq = float(col.sum())
        sup = int((col > 0).sum())
        cov = sup / len(patient_data)
        pos = col[patient_data[outcome_col] == factual_outcome]
        neg = col[patient_data[outcome_col] != factual_outcome]
        pa = float(pos.mean()) if len(pos) > 0 else 0.0
        na = float(neg.mean()) if len(neg) > 0 else 0.0
        fi = abs(na - pa) / max(abs(na), abs(pa), 1e-8)
        phr = float((pos > 0).mean()) if len(pos) > 0 else 0.0
        nhr = float((neg > 0).mean()) if len(neg) > 0 else 0.0
        try:
            mi = float(mutual_info_classif(col.values.reshape(-1, 1),
                                           patient_data[outcome_col].fillna(0).values,
                                           discrete_features=True)[0])
        except Exception:
            mi = 0.0
        rows.append({"pattern_id": pid, "frequency": freq, "support": sup, "coverage": cov,
                      "pos_avg_freq": pa, "neg_avg_freq": na, "freq_interest": fi,
                      "pos_hit_rate": phr, "neg_hit_rate": nhr, "hit_rate_diff": nhr - phr,
                      "outcome_interest": mi})
    attr_df = pd.DataFrame(rows)
    if len(attr_df) > 0:
        attr_df = attr_df.sort_values("outcome_interest", ascending=False)
    return attr_df


def filter_high_quality_patterns(pattern_dict, attr_df, min_interest=0.01, min_hit_rate_diff=0.0):
    if len(attr_df) == 0:
        return {}, attr_df
    hq = attr_df[(attr_df['outcome_interest'] > min_interest) &
                 (attr_df['hit_rate_diff'] > min_hit_rate_diff)]
    filtered = {pid: pattern_dict[pid] for pid in hq['pattern_id'] if pid in pattern_dict}
    return filtered, hq


def samples_to_dataframe(samples, act_encoder=None):
    rows = []
    for idx, s in enumerate(samples):
        case_id = str(s.get('case_id', f'Case_{idx}'))
        act_ids = list(s.get('activities', []))
        times = list(s.get('times', []))
        outcome = s.get('outcome', 0)
        if act_encoder:
            names = []
            for a in act_ids:
                if isinstance(a, (int, np.integer)) and a > 0:
                    try:
                        names.append(act_encoder.inverse_transform([int(a) - 1])[0])
                    except Exception:
                        names.append(str(a))
                else:
                    names.append(str(a))
        else:
            names = [str(a) for a in act_ids]
        base = pd.Timestamp('2016-01-01 00:00:00')
        for i, (act, t) in enumerate(zip(names, times)):
            try:
                ts = base + pd.Timedelta(hours=float(t))
            except Exception:
                ts = base + pd.Timedelta(hours=float(i))
            rows.append({'Case ID': case_id, 'Activity': act, 'time:timestamp': ts,
                          'label': outcome, 'likelihood': 0.5})
    return pd.DataFrame(rows)
