"""
外部 Baseline: XGBoost (PPM 经典) + LSTM (深度学习基线)

用法:
    from baselines import XGBoostBaseline, LSTMBaseline
    xgb = XGBoostBaseline()
    xgb.fit(train_samples, val_samples, cfg)
    preds = xgb.predict(test_samples, cfg)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from torch.utils.data import DataLoader
from dataset import ProcessDataset, collate_fn
from models import MultiTaskProcessModel
from sklearn.metrics import f1_score, roc_auc_score
import copy


# ===========================================================================
# 特征工程: 序列 → 固定维度向量 (XGBoost 用)
# ===========================================================================

def _flatten_sequence(samples, max_seq_len, n_static_cat, n_static_num,
                      n_dynamic_cat, n_dynamic_num, num_activities,
                      last_k=5):
    """将序列样本展平为固定维度特征向量"""
    X, y_out, y_act, y_time = [], [], [], []

    for s in samples:
        acts = s["activities"][:max_seq_len]
        times = s["times"][:max_seq_len]
        L = min(s["length"], max_seq_len)
        obs_L = max(1, L - 1) if L >= 2 else L

        feat = []

        # 1. 最后 K 个活动的 one-hot (K × num_activities)
        recent_acts = acts[max(0, obs_L - last_k):obs_L]
        for a in recent_acts:
            onehot = np.zeros(num_activities)
            if 0 <= a < num_activities:
                onehot[a] = 1
            feat.extend(onehot)
        # 补齐不足 K 个的
        for _ in range(last_k - len(recent_acts)):
            feat.extend(np.zeros(num_activities))

        # 2. 时间统计量 (6 个)
        t_arr = np.array(times[:obs_L]) if obs_L > 0 else np.array([0.0])
        feat.extend([
            np.mean(t_arr), np.std(t_arr) if len(t_arr) > 1 else 0,
            np.min(t_arr), np.max(t_arr),
            t_arr[-1] if len(t_arr) > 0 else 0,
            t_arr[-1] - t_arr[0] if len(t_arr) > 1 else 0,
        ])

        # 3. 静态类别 (已编码)
        sc = s.get("static_cat", [])
        feat.extend(list(sc[:n_static_cat]) + [0] * max(0, n_static_cat - len(sc)))

        # 4. 静态数值
        sn = s.get("static_num", [])
        feat.extend(list(sn[:n_static_num]) + [0.0] * max(0, n_static_num - len(sn)))

        # 5. 动态类别 (众数 per column)
        dc = s.get("dynamic_cat", [])
        for j in range(n_dynamic_cat):
            col_vals = [dc[t][j] for t in range(min(obs_L, len(dc))) if j < len(dc[t])]
            feat.append(max(set(col_vals), key=col_vals.count) if col_vals else 0)

        # 6. 动态数值 (mean, std per column)
        dn = s.get("dynamic_num", [])
        for j in range(n_dynamic_num):
            col_vals = [dn[t][j] for t in range(min(obs_L, len(dn))) if j < len(dn[t])]
            arr = np.array(col_vals) if col_vals else np.array([0])
            feat.extend([arr.mean(), arr.std() if len(arr) > 1 else 0])

        X.append(feat)

        # 目标
        if L >= 2:
            y_out.append(s["outcome"])
            y_act.append(acts[L - 1])
            y_time.append(times[L - 1])
        else:
            y_out.append(s["outcome"])
            y_act.append(0)
            y_time.append(0)

    return (np.array(X, dtype=np.float32),
            np.array(y_out), np.array(y_act), np.array(y_time))


# ===========================================================================
# XGBoost Baseline
# ===========================================================================

class XGBoostBaseline:
    """XGBoost + 特征工程 (PPM 经典 baseline)"""

    def __init__(self, n_estimators=100, max_depth=6):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.outcome_model = None
        self.act_model = None
        self.time_model = None

    def fit(self, train_samples, val_samples, cfg):
        X_train, y_out, y_act, y_time = _flatten_sequence(
            train_samples, cfg.max_seq_len,
            cfg.n_static_cat, cfg.n_static_num,
            cfg.n_dynamic_cat, cfg.n_dynamic_num,
            cfg.num_activities,
        )
        X_val, y_out_val, y_act_val, y_time_val = _flatten_sequence(
            val_samples, cfg.max_seq_len,
            cfg.n_static_cat, cfg.n_static_num,
            cfg.n_dynamic_cat, cfg.n_dynamic_num,
            cfg.num_activities,
        )

        self.outcome_model = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            eval_metric="logloss", random_state=42,
        )
        self.outcome_model.fit(X_train, y_out,
                               eval_set=[(X_val, y_out_val)], verbose=False)

        self.act_model = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            eval_metric="mlogloss", random_state=42,
        )
        self.act_model.fit(X_train, y_act,
                           eval_set=[(X_val, y_act_val)], verbose=False)

        self.time_model = XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=42,
        )
        self.time_model.fit(X_train, y_time,
                            eval_set=[(X_val, y_time_val)], verbose=False)
        return self

    def predict(self, test_samples, cfg):
        X_test, y_out, y_act, y_time = _flatten_sequence(
            test_samples, cfg.max_seq_len,
            cfg.n_static_cat, cfg.n_static_num,
            cfg.n_dynamic_cat, cfg.n_dynamic_num,
            cfg.num_activities,
        )
        pred_out = self.outcome_model.predict(X_test)
        pred_act = self.act_model.predict(X_test)
        pred_time = self.time_model.predict(X_test)
        try:
            act_probs = self.act_model.predict_proba(X_test)
            act_classes = np.asarray(self.act_model.classes_, dtype=int)
        except Exception:
            act_probs = None
            act_classes = None

        valid = np.array([min(s["length"], cfg.max_seq_len) >= 2 for s in test_samples])
        top3_correct = 0
        top5_correct = 0
        valid_count = int(valid.sum())
        if act_probs is not None and act_classes is not None and valid_count > 0:
            for row_probs, true_act, is_valid in zip(act_probs, y_act, valid):
                if not is_valid:
                    continue
                order = np.argsort(row_probs)[::-1]
                top3 = set(int(x) for x in act_classes[order[:min(3, len(order))]])
                top5 = set(int(x) for x in act_classes[order[:min(5, len(order))]])
                top3_correct += int(int(true_act) in top3)
                top5_correct += int(int(true_act) in top5)

        return {
            "outcome_acc": float(np.mean(pred_out == y_out)),
            "next_act_acc": float(np.mean((pred_act[valid] == y_act[valid]) if valid.sum() > 0 else 0)),
            "next_act_top3_acc": float(top3_correct / max(valid_count, 1)) if act_probs is not None else float("nan"),
            "next_act_top5_acc": float(top5_correct / max(valid_count, 1)) if act_probs is not None else float("nan"),
            "next_time_mae": float(np.mean(np.abs(pred_time[valid] - y_time[valid])) if valid.sum() > 0 else 0),
            "outcome_f1": float("nan"),
            "outcome_auc": float("nan"),
        }


# ===========================================================================
# LSTM Baseline
# ===========================================================================

class LSTMModel(nn.Module):
    def __init__(self, num_activities, hidden_dim=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.act_emb = nn.Embedding(num_activities, 64, padding_idx=0)
        self.time_fc = nn.Linear(1, 16)

        in_dim = 64 + 16
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.next_act_head = nn.Linear(hidden_dim, num_activities)
        self.next_time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, activities, times, mask):
        B, T = activities.shape
        act_x = self.act_emb(activities)
        time_x = self.time_fc(times.unsqueeze(-1))
        x = torch.cat([act_x, time_x], dim=-1)

        lengths = mask.sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        # h_n: [num_layers, B, hidden]
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=T
        )
        lengths_for_next = lengths.clamp(min=2)
        prev_pos = lengths_for_next - 2
        prev_idx = prev_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, lstm_out.size(-1))
        prev_h = lstm_out.gather(1, prev_idx).squeeze(1)
        last_h = prev_h

        prev_h = last_h  # LSTM 的隐藏态天然是因果的 (前向)

        return {
            "outcome_logits": self.outcome_head(prev_h),
            "next_act_logits": self.next_act_head(prev_h),
            "next_time_pred": self.next_time_head(prev_h).squeeze(-1),
        }


class LSTMBaseline:
    """LSTM + 前缀编码 (深度学习 baseline)"""

    def __init__(self, epochs=15, lr=1e-3, device="cpu"):
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.model = None

    def fit(self, train_samples, val_samples, cfg):
        self.device = cfg.device
        ds_kwargs = dict(max_seq_len=cfg.max_seq_len, pattern_feat_dim=0,
                         n_static_cat=0, n_static_num=0,
                         n_dynamic_cat=0, n_dynamic_num=0)

        train_ds = ProcessDataset(train_samples, **ds_kwargs)
        val_ds = ProcessDataset(val_samples, **ds_kwargs)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

        model = LSTMModel(cfg.num_activities).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)

        best_acc = 0
        best_state = None

        for epoch in range(self.epochs):
            model.train()
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optim.zero_grad()
                out = model(batch["activities"], batch["times"], batch["mask"])

                B = batch["activities"].size(0)
                lengths = batch["mask"].sum(dim=1)

                # 标签
                act_target = torch.zeros(B, dtype=torch.long, device=self.device)
                time_target = torch.zeros(B, device=self.device)
                valid = torch.zeros(B, dtype=torch.bool, device=self.device)
                for i in range(B):
                    L = int(lengths[i].item())
                    if L >= 2:
                        act_target[i] = batch["activities"][i, L - 1]
                        time_target[i] = batch["times"][i, L - 1]
                        valid[i] = True

                loss_act = F.cross_entropy(out["next_act_logits"], act_target, reduction="none")
                loss_act = (loss_act * valid.float()).sum() / valid.float().sum().clamp(min=1)

                loss_time = F.mse_loss(out["next_time_pred"], time_target, reduction="none")
                loss_time = (loss_time * valid.float()).sum() / valid.float().sum().clamp(min=1)

                loss_out = F.cross_entropy(out["outcome_logits"], batch["outcome"])

                loss = loss_act + loss_time + loss_out
                loss.backward()
                optim.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    out = model(batch["activities"], batch["times"], batch["mask"])
                    pred = out["outcome_logits"].argmax(dim=-1)
                    correct += (pred == batch["outcome"]).sum().item()
                    total += batch["outcome"].numel()

            acc = correct / max(total, 1)
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())

        if best_state:
            model.load_state_dict(best_state)
        self.model = model
        return self

    @torch.no_grad()
    def predict(self, test_samples, cfg):
        ds_kwargs = dict(max_seq_len=cfg.max_seq_len, pattern_feat_dim=0,
                         n_static_cat=0, n_static_num=0,
                         n_dynamic_cat=0, n_dynamic_num=0)
        test_ds = ProcessDataset(test_samples, **ds_kwargs)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

        self.model.eval()
        correct = 0; total = 0
        mae_sum = 0.0; mae_cnt = 0
        act_correct = 0; act_total = 0
        act_top3_correct = 0
        act_top5_correct = 0

        for batch in test_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(batch["activities"], batch["times"], batch["mask"])

            pred_out = out["outcome_logits"].argmax(dim=-1)
            correct += (pred_out == batch["outcome"]).sum().item()
            total += batch["outcome"].numel()

            lengths = batch["mask"].sum(dim=1)
            activities = batch["activities"]
            times = batch["times"]
            for i in range(batch["activities"].size(0)):
                L = int(lengths[i].item())
                if L >= 2:
                    mae_sum += abs(out["next_time_pred"][i].item() - times[i, L - 1].item())
                    mae_cnt += 1
                    act_logits = out["next_act_logits"][i]
                    pred_act = act_logits[1:].argmax().item() + 1 if act_logits.numel() > 1 else 0
                    true_act = activities[i, L - 1].item()
                    if pred_act == true_act:
                        act_correct += 1
                    if act_logits.numel() > 1:
                        k = min(3, act_logits.numel() - 1)
                        topk = act_logits[1:].topk(k).indices.cpu().numpy() + 1
                        act_top3_correct += int(true_act in set(int(x) for x in topk))
                        k = min(5, act_logits.numel() - 1)
                        topk = act_logits[1:].topk(k).indices.cpu().numpy() + 1
                        act_top5_correct += int(true_act in set(int(x) for x in topk))
                    act_total += 1

        return {
            "outcome_acc": correct / max(total, 1),
            "next_act_acc": act_correct / max(act_total, 1),
            "next_act_top3_acc": act_top3_correct / max(act_total, 1),
            "next_act_top5_acc": act_top5_correct / max(act_total, 1),
            "next_time_mae": mae_sum / max(mae_cnt, 1),
            "outcome_f1": float("nan"),
            "outcome_auc": float("nan"),
        }


# ===========================================================================
# Declare 模板 Baseline — 5 种标准模板 → LTLf → DFA
# ===========================================================================

# ===========================================================================
# ProcessTransformer Baseline
# ===========================================================================

class ProcessTransformerBaseline:
    """Causal Transformer baseline for predictive process monitoring."""

    def __init__(self, epochs=15, lr=1e-3, device="cpu", batch_size=32, patience=5):
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.batch_size = batch_size
        self.patience = patience
        self.model = None
        self.best_val_acc = 0.0
        self.best_val_auc = float("nan")
        self.actual_epochs = 0
        self.stopped_early = False

    def _baseline_cfg(self, cfg):
        base_cfg = copy.deepcopy(cfg)
        base_cfg.pattern_feat_dim = 0
        base_cfg.n_static_cat = 0
        base_cfg.n_static_num = 0
        base_cfg.n_dynamic_cat = 0
        base_cfg.n_dynamic_num = 0
        base_cfg.device = self.device
        return base_cfg

    def _make_loader(self, samples, cfg, shuffle):
        ds_kwargs = dict(max_seq_len=cfg.max_seq_len, pattern_feat_dim=0,
                         n_static_cat=0, n_static_num=0,
                         n_dynamic_cat=0, n_dynamic_num=0)
        ds = ProcessDataset(samples, **ds_kwargs)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          collate_fn=collate_fn)

    def _loss(self, out, batch):
        B = batch["activities"].size(0)
        lengths = batch["mask"].sum(dim=1)
        act_target = torch.zeros(B, dtype=torch.long, device=self.device)
        time_target = torch.zeros(B, device=self.device)
        valid = torch.zeros(B, dtype=torch.bool, device=self.device)

        for i in range(B):
            L = int(lengths[i].item())
            if L >= 2:
                act_target[i] = batch["activities"][i, L - 1]
                time_target[i] = batch["times"][i, L - 1]
                valid[i] = True

        loss_act = F.cross_entropy(out["next_act_logits"], act_target, reduction="none")
        loss_act = (loss_act * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
        loss_time = F.smooth_l1_loss(out["next_time_pred"], time_target, reduction="none")
        loss_time = (loss_time * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
        loss_out = F.cross_entropy(out["outcome_logits"], batch["outcome"])
        return loss_act + loss_time + loss_out

    @torch.no_grad()
    def _evaluate_loader(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        mae_sum = 0.0
        mae_cnt = 0
        act_correct = 0
        act_total = 0
        act_top3_correct = 0
        act_top5_correct = 0
        out_true = []
        out_prob = []

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(batch["activities"], batch["times"], batch["mask"])
            probs = torch.softmax(out["outcome_logits"], dim=-1)[:, 1]
            pred_out = out["outcome_logits"].argmax(dim=-1)
            correct += (pred_out == batch["outcome"]).sum().item()
            total += batch["outcome"].numel()
            out_true.extend(batch["outcome"].cpu().tolist())
            out_prob.extend(probs.cpu().tolist())

            lengths = batch["mask"].sum(dim=1)
            activities = batch["activities"]
            times = batch["times"]
            for i in range(batch["activities"].size(0)):
                L = int(lengths[i].item())
                if L >= 2:
                    mae_sum += abs(out["next_time_pred"][i].item() - times[i, L - 1].item())
                    mae_cnt += 1
                    act_logits = out["next_act_logits"][i]
                    pred_act = act_logits[1:].argmax().item() + 1 if act_logits.numel() > 1 else 0
                    true_act = activities[i, L - 1].item()
                    act_correct += int(pred_act == true_act)
                    if act_logits.numel() > 1:
                        k = min(3, act_logits.numel() - 1)
                        topk = act_logits[1:].topk(k).indices.cpu().numpy() + 1
                        act_top3_correct += int(true_act in set(int(x) for x in topk))
                        k = min(5, act_logits.numel() - 1)
                        topk = act_logits[1:].topk(k).indices.cpu().numpy() + 1
                        act_top5_correct += int(true_act in set(int(x) for x in topk))
                    act_total += 1

        pred_labels = (np.array(out_prob) >= 0.5).astype(int)
        try:
            f1 = f1_score(out_true, pred_labels, average="macro", zero_division=0)
        except Exception:
            f1 = float("nan")
        try:
            auc = roc_auc_score(out_true, out_prob) if len(set(out_true)) >= 2 else float("nan")
        except Exception:
            auc = float("nan")

        return {
            "outcome_acc": correct / max(total, 1),
            "next_act_acc": act_correct / max(act_total, 1),
            "next_act_top3_acc": act_top3_correct / max(act_total, 1),
            "next_act_top5_acc": act_top5_correct / max(act_total, 1),
            "next_time_mae": mae_sum / max(mae_cnt, 1),
            "outcome_f1": float(f1),
            "outcome_auc": float(auc),
        }

    @staticmethod
    def _monitor(logs):
        for key in ("outcome_auc", "outcome_f1", "outcome_acc"):
            value = float(logs.get(key, float("nan")))
            if np.isfinite(value):
                return value
        return 0.0

    def fit(self, train_samples, val_samples, cfg):
        self.device = cfg.device
        base_cfg = self._baseline_cfg(cfg)
        train_loader = self._make_loader(train_samples, base_cfg, shuffle=True)
        val_loader = self._make_loader(val_samples, base_cfg, shuffle=False)

        model = MultiTaskProcessModel(base_cfg).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="max", factor=0.5, patience=2, min_lr=1e-5,
        )
        best_score = -float("inf")
        best_state = None
        stale_epochs = 0

        for epoch in range(1, self.epochs + 1):
            self.actual_epochs = epoch
            model.train()
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optim.zero_grad()
                out = model(batch["activities"], batch["times"], batch["mask"])
                loss = self._loss(out, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            self.model = model
            val_logs = self._evaluate_loader(val_loader)
            score = self._monitor(val_logs)
            scheduler.step(score)
            self.best_val_acc = max(self.best_val_acc, val_logs["outcome_acc"])
            if np.isfinite(val_logs["outcome_auc"]):
                self.best_val_auc = (
                    val_logs["outcome_auc"]
                    if not np.isfinite(self.best_val_auc)
                    else max(self.best_val_auc, val_logs["outcome_auc"])
                )

            if score > best_score + 1e-4:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    self.stopped_early = True
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        return self

    def predict(self, test_samples, cfg):
        base_cfg = self._baseline_cfg(cfg)
        test_loader = self._make_loader(test_samples, base_cfg, shuffle=False)
        return self._evaluate_loader(test_loader)


DECLARE_TEMPLATES = {
    "Response":        ("G({a} -> F {b})", ""),      # G(a→Fb)
    "ChainResponse":   ("G({a} -> X {b})", ""),      # G(a→Xb)
    "NotSuccession":   ("! (F {a} & F {b})", ""),    # ¬(Fa ∧ Fb)
    "Absence":         ("! (F {a})", ""),            # ¬Fa (a 永远不出现)
    "Existence":       ("F {a}", ""),                # Fa (a 最终出现)
}

def _sanitize(name: str) -> str:
    """flloat 只接受 [a-z][a-z0-9_]* 的原子命题"""
    s = name.replace(" ", "_").replace("-", "_").lower()
    if s and s[0].isdigit():
        s = "a_" + s
    return s


def _check_template(act_seq, a, b, template_name):
    """检查单条轨迹是否满足 Declare 模板"""
    acts = [x for x in act_seq if x is not None]
    n = len(acts)
    if n < 1:
        return True
    if template_name == "Response":
        try: return b in acts[acts.index(a) + 1:]
        except ValueError: return True
    if template_name == "Precedence":
        try: return a in acts[:acts.index(b)]
        except ValueError: return True
    if template_name == "ChainResponse":
        for i in range(n - 1):
            if acts[i] == a and acts[i + 1] != b: return False
        return True
    if template_name == "ChainPrecedence":
        for i in range(1, n):
            if acts[i] == b and acts[i - 1] != a: return False
        return True
    if template_name == "NotSuccession":
        return not (a in acts and b in acts)
    if template_name == "Absence":
        return a not in acts
    if template_name == "Existence":
        return a in acts
    return True


def build_declare_dfa(train_samples, act_encoder, dfa_max_states=500,
                      min_support=0.50, max_pairs=60):
    """从训练数据构建 Declare 模板 DFA, 返回 (dfa, clusters, n_formulas)"""
    from explaination.visualizations.utils import (
        parse, to_dfa, symbolic_to_singleton, alphabet,
    )

    # ---- 创建消毒后的 act_encoder (flloat 不支持空格) ----
    class _SanitizedEncoder:
        """包装 LabelEncoder, inverse_transform 返回下划线替空格的活动名"""
        def __init__(self, orig):
            self.orig = orig
            self._map = {}
            for name in orig.classes_:
                self._map[_sanitize(name)] = name
        def inverse_transform(self, ids):
            raw = self.orig.inverse_transform(ids)
            return [_sanitize(r) for r in raw]
        @property
        def classes_(self):
            return [_sanitize(c) for c in self.orig.classes_]

    sanitized_enc = _SanitizedEncoder(act_encoder)

    n_total = len(train_samples)
    train_end = int(0.7 * n_total)
    train_s = train_samples[:train_end]

    all_traces = []
    for s in train_s:
        names = []
        for a in s["activities"]:
            if a > 0:
                try:
                    names.append(sanitized_enc.inverse_transform([a - 1])[0])
                except (IndexError, ValueError): pass
        all_traces.append(names)

    from collections import Counter
    pair_counts = Counter()
    for names in all_traces:
        seen = set(names)
        for a in seen:
            for b in seen:
                if a != b: pair_counts[(a, b)] += 1

    formulas, processed = {}, 0
    for (a, b), _ in pair_counts.most_common(max_pairs * 2):
        if processed >= max_pairs: break
        best_tpl, best_sup = None, 0.0
        for tpl_name, (tpl_fmt, _) in DECLARE_TEMPLATES.items():
            sat = sum(1 for t in all_traces if _check_template(t, a, b, tpl_name))
            sup = sat / max(len(all_traces), 1)
            if sup > best_sup: best_sup, best_tpl = sup, tpl_name
        if best_tpl and best_sup >= min_support:
            formulas[f"{best_tpl}({a},{b})"] = DECLARE_TEMPLATES[best_tpl][0].format(a=a, b=b)
            processed += 1

    print(f"  Declare: {len(formulas)} formulas from {processed} pairs (sup>={min_support:.0%})")
    if len(formulas) == 0:
        return None, None, 0, None

    # 逐公式构建独立 DFA (每个 DFA 只有 3-5 状态, 构建快)
    # 合成 multi-DFA clusters (析取验证)
    clusters = []
    for fid, formula in formulas.items():
        try:
            parsed = parse(formula)
            sym_dfa = to_dfa(parsed)
            if 0 < len(sym_dfa.states) <= dfa_max_states:
                letters = alphabet(parsed)
                letters.add('*')
                dfa_i = symbolic_to_singleton(sym_dfa, letters)
                clusters.append({"pids": [fid], "formula": formula, "dfa": dfa_i})
        except Exception:
            pass

    if len(clusters) == 0:
        return None, None, 0, sanitized_enc

    print(f"  Declare DFA: {len(clusters)} individual DFAs "
          f"(avg states: {sum(len(c['dfa'].states) for c in clusters)/len(clusters):.0f})")
    if len(clusters) == 1:
        return clusters[0]["dfa"], None, len(formulas), sanitized_enc
    return None, clusters, len(formulas), sanitized_enc
