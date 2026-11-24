from typing import Dict, List, Any
import torch
from torch.utils.data import Dataset

class ProcessDataset(Dataset):
    def __init__(self, samples, max_seq_len=50, pattern_feat_dim=0,
                 n_static_cat=0, n_static_num=0, n_dynamic_cat=0, n_dynamic_num=0,
                 dfa_action_mask_dim=0):
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.pattern_feat_dim = pattern_feat_dim
        self.n_static_cat = n_static_cat
        self.n_static_num = n_static_num
        self.n_dynamic_cat = n_dynamic_cat
        self.n_dynamic_num = n_dynamic_num
        self.dfa_action_mask_dim = max(0, int(dfa_action_mask_dim))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        L = min(item.get("length", len(item["activities"])), self.max_seq_len)

        activities = item["activities"][:L]
        times = item["times"][:L]

        pattern_feat = item.get("pattern_features", [])
        if self.pattern_feat_dim > 0:
            if pattern_feat is None or len(pattern_feat) == 0:
                raise ValueError(
                    "Missing pattern features while pattern_feat_dim > 0"
                )
            raw_pattern = torch.as_tensor(pattern_feat, dtype=torch.float)
            if raw_pattern.ndim > 1:
                raw_pattern = raw_pattern[0]
            raw_pattern = raw_pattern.flatten()
            if raw_pattern.numel() != self.pattern_feat_dim:
                raise ValueError(
                    "Pattern feature dimension mismatch: "
                    f"{raw_pattern.numel()} != {self.pattern_feat_dim}"
                )
            pattern_tensor = raw_pattern
        else:
            pattern_tensor = torch.zeros(0, dtype=torch.float)

        # 属性: 空则填零
        sc = item.get("static_cat", [])
        sn = item.get("static_num", [])
        dc = item.get("dynamic_cat", [])
        dn = item.get("dynamic_num", [])

        static_cat_t = torch.tensor(sc[:self.n_static_cat] if sc else [],
                                    dtype=torch.long)
        static_num_t = torch.tensor(sn[:self.n_static_num] if sn else [],
                                    dtype=torch.float)
        # dynamic: 截断到 L
        dyn_cat_t = torch.zeros(L, max(self.n_dynamic_cat, 1), dtype=torch.long)
        dyn_num_t = torch.zeros(L, max(self.n_dynamic_num, 1), dtype=torch.float)
        if self.n_dynamic_cat > 0 and dc:
            for t in range(min(L, len(dc))):
                row = dc[t]
                for j in range(min(self.n_dynamic_cat, len(row))):
                    dyn_cat_t[t, j] = row[j]
        if self.n_dynamic_num > 0 and dn:
            for t in range(min(L, len(dn))):
                row = dn[t]
                for j in range(min(self.n_dynamic_num, len(row))):
                    dyn_num_t[t, j] = row[j]

        dfa_action_mask = torch.zeros(
            self.dfa_action_mask_dim, dtype=torch.bool
        )
        if self.dfa_action_mask_dim > 0:
            auxiliary_action_ids = item.get(
                "dfa_aux_action_ids",
                # Compatibility with protocol-v1 cached samples.
                item.get("dfa_valid_action_ids", ()),
            )
            for raw_aid in auxiliary_action_ids:
                aid = int(raw_aid)
                if not 1 <= aid <= self.dfa_action_mask_dim:
                    raise ValueError(
                        "DFA auxiliary action id is outside the non-padding "
                        f"activity range: {aid} not in "
                        f"[1, {self.dfa_action_mask_dim}]"
                    )
                dfa_action_mask[aid - 1] = True
        dfa_aux_applicable = bool(item.get("dfa_aux_applicable", False))
        if dfa_aux_applicable and not bool(dfa_action_mask.any()):
            raise ValueError(
                "DFA auxiliary supervision is marked applicable but has no "
                "progress-supported next activity"
            )

        return {
            "activities": torch.tensor(activities, dtype=torch.long),
            "times": torch.tensor(times, dtype=torch.float),
            "outcome": torch.tensor(item["outcome"], dtype=torch.long),
            "length": torch.tensor(L, dtype=torch.long),
            "pattern_features": pattern_tensor,
            "static_cat": static_cat_t,
            "static_num": static_num_t,
            "dynamic_cat": dyn_cat_t,
            "dynamic_num": dyn_num_t,
            "dfa_next_act_mask": dfa_action_mask,
            "dfa_aux_applicable": torch.tensor(
                dfa_aux_applicable, dtype=torch.bool
            ),
        }

def collate_fn(batch):
    pattern_feats = torch.stack([b["pattern_features"] for b in batch])
    lengths = torch.stack([b["length"] for b in batch])
    max_len = int(lengths.max().item())
    B = len(batch)

    dfa_action_masks = torch.stack([b["dfa_next_act_mask"] for b in batch])
    dfa_aux_applicable = torch.stack([b["dfa_aux_applicable"] for b in batch])
    activities = torch.zeros(B, max_len, dtype=torch.long)
    times = torch.zeros(B, max_len, dtype=torch.float)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    outcomes = torch.stack([b["outcome"] for b in batch])

    # 属性
    static_cat = torch.stack([b["static_cat"] for b in batch])
    static_num = torch.stack([b["static_num"] for b in batch])
    dyn_cat = torch.zeros(B, max_len, batch[0]["dynamic_cat"].shape[1], dtype=torch.long)
    dyn_num = torch.zeros(B, max_len, batch[0]["dynamic_num"].shape[1], dtype=torch.float)

    for i, b in enumerate(batch):
        L = int(b["length"].item())
        activities[i, :L] = b["activities"][:L]
        times[i, :L] = b["times"][:L]
        mask[i, :L] = True
        dc = b["dynamic_cat"]
        if dc.shape[1] > 0:
            dyn_cat[i, :min(L, dc.shape[0])] = dc[:min(L, dc.shape[0])]
        dn = b["dynamic_num"]
        if dn.shape[1] > 0:
            dyn_num[i, :min(L, dn.shape[0])] = dn[:min(L, dn.shape[0])]

    return {
        "pattern_features": pattern_feats,
        "activities": activities,
        "times": times,
        "mask": mask,
        "lengths": lengths,
        "outcome": outcomes,
        "dfa_next_act_mask": dfa_action_masks,
        "dfa_aux_applicable": dfa_aux_applicable,
        "static_cat": static_cat,
        "static_num": static_num,
        "dynamic_cat": dyn_cat,
        "dynamic_num": dyn_num,
    }

import numpy as np

def build_reference_matrix(dataset: ProcessDataset, max_seq_len: int) -> np.ndarray:
    """
    将 ProcessDataset 中的所有样本转换为展平的特征矩阵。
    每一行为一个样本，特征顺序：前 max_seq_len 列是活动 ID（填充0），
    后 max_seq_len 列是对应的时间（填充0.0）。
    """
    features = []
    for i in range(len(dataset)):
        sample = dataset[i]
        acts = sample["activities"].tolist()
        times = sample["times"].tolist()
        L = sample["length"].item()
        # 截断/填充到 max_seq_len
        pad_acts = acts[:max_seq_len] + [0] * (max_seq_len - len(acts[:max_seq_len]))
        pad_times = times[:max_seq_len] + [0.0] * (max_seq_len - len(times[:max_seq_len]))
        feat = pad_acts + pad_times
        features.append(feat)
    return np.array(features, dtype=np.float32)
