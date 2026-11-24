# pattern_mining_minimal.py — 最简 IMPRESSED 调用, 无任何优化
import pandas as pd
import numpy as np
import os, pickle, glob
from pattern_discovery.wrappers.impressed_wrapper import impressed_wrapper


def samples_to_dataframe(samples, act_encoder=None):
    rows = []
    for idx, s in enumerate(samples):
        case_id = s.get('case_id', f'Case_{idx}')
        act_ids = s['activities']
        times = s['times']
        outcome = s['outcome']
        if act_encoder is not None:
            act_names = [act_encoder.inverse_transform([a - 1])[0] if a > 0 else 'PAD' for a in act_ids]
        else:
            act_names = [str(a) for a in act_ids]
        base_time = pd.Timestamp('2016-01-01 00:00:00')
        for i, (act, t) in enumerate(zip(act_names, times)):
            current_time = base_time + pd.Timedelta(hours=t)
            rows.append({
                'Case ID': case_id,
                'Activity': act,
                'time:timestamp': current_time,
                'label': outcome,
                'likelihood': 0.5,  # 中性似然度
            })
    return pd.DataFrame(rows)


def load_pattern_dict(output_path):
    files = sorted(glob.glob(os.path.join(output_path, '*.pickle')))
    pattern_dict = {}
    import networkx as nx
    for f in files:
        pid = os.path.basename(f).replace('.pickle', '')
        try:
            g = pickle.load(open(f, 'rb'))
            if isinstance(g, nx.DiGraph):
                pattern_dict[pid] = g
        except:
            pass
    return pattern_dict if pattern_dict else None


def run_pattern_discovery(train_samples, act_encoder=None, output_path='./patterns', return_dict=True):
    """最简版: 仅加截断+采样(有数学保证), 不加补丁/缓存/去重"""
    # 1. 轨迹截断: 长轨迹只保留最后 30 步
    for s in train_samples:
        if s["length"] > 30:
            s["activities"] = s["activities"][s["length"] - 30:]
            s["times"] = s["times"][s["length"] - 30:]
            s["length"] = 30

    # 2. 轨迹采样: >400 条时随机保留 400
    import random as _r
    _r.seed(42)
    if len(train_samples) > 400:
        train_samples = _r.sample(train_samples, 400)

    train_df = samples_to_dataframe(train_samples, act_encoder)
    train_df['label'] = train_df['label'].apply(lambda x: int(x) if x >= 0 else 0)

    try:
        train_encoded, _, _ = impressed_wrapper(
            df=train_df,
            output_path=output_path,
            discovery_type='auto',
            case_id='Case ID',
            activity='Activity',
            timestamp='time:timestamp',
            outcome='label',
            outcome_type='binary',
            delta_time=3600,
            max_gap=5,
            max_extension_step=2,
            factual_outcome=1,
            likelihood='likelihood',
            encoding=True,
            testing_percentage=0.2,
            pareto_only=False,
        )
        print(f"模式发现完成，生成 {train_encoded.shape[1]} 个模式特征")
        pattern_dict = load_pattern_dict(output_path) if return_dict else None
        if return_dict:
            return train_encoded, pattern_dict
        return train_encoded
    except Exception as e:
        print(f"模式发现失败: {e}")
        if return_dict:
            return None, None
        return None


# ---- 以下三个函数与之前完全相同 ----
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
        print(f"模式特征注入: {missing_count}/{len(samples)} 个样本未找到对应模式特征（使用零向量）")
    return samples


def compute_pattern_attributes(pattern_dict, patient_data, outcome_col='label', factual_outcome=1):
    rows = []
    pattern_ids = list(pattern_dict.keys())
    n_matched, n_skipped = 0, 0
    for pid in pattern_ids:
        if pid not in patient_data.columns:
            n_skipped += 1
            continue
        n_matched += 1
        freq = patient_data[pid].sum()
        support = (patient_data[pid] > 0).sum()
        coverage = support / len(patient_data)
        pos_data = patient_data[patient_data[outcome_col] == factual_outcome][pid]
        neg_data = patient_data[patient_data[outcome_col] != factual_outcome][pid]
        pos_freq = pos_data.mean() if len(pos_data) > 0 else 0
        neg_freq = neg_data.mean() if len(neg_data) > 0 else 0
        freq_interest = abs(neg_freq - pos_freq) / (max(neg_freq, pos_freq) + 1e-8)
        pos_hit_rate = (pos_data > 0).mean() if len(pos_data) > 0 else 0
        neg_hit_rate = (neg_data > 0).mean() if len(neg_data) > 0 else 0
        hit_rate_diff = neg_hit_rate - pos_hit_rate
        from sklearn.feature_selection import mutual_info_classif
        try:
            mi_arr = mutual_info_classif(
                patient_data[[pid]].fillna(0),
                patient_data[outcome_col].fillna(0),
                discrete_features=True,
            )
            mi = float(mi_arr[0]) if len(mi_arr) > 0 else 0.0
        except Exception:
            mi = 0.0
        rows.append({
            'pattern_id': pid, 'frequency': freq, 'support': support,
            'coverage': coverage, 'pos_avg_freq': pos_freq, 'neg_avg_freq': neg_freq,
            'freq_interest': freq_interest, 'pos_hit_rate': pos_hit_rate,
            'neg_hit_rate': neg_hit_rate, 'hit_rate_diff': hit_rate_diff, 'outcome_interest': mi,
        })
    attr_df = pd.DataFrame(rows)
    attr_df = attr_df.sort_values('outcome_interest', ascending=False).reset_index(drop=True)
    return attr_df


def filter_high_quality_patterns(pattern_dict, attr_df, min_interest=0.01, min_hit_rate_diff=0.0):
    high_quality = attr_df[(attr_df['outcome_interest'] > min_interest) &
                           (attr_df['hit_rate_diff'] > min_hit_rate_diff)]
    filtered_dict = {pid: pattern_dict[pid] for pid in high_quality['pattern_id'] if pid in pattern_dict}
    return filtered_dict, high_quality
