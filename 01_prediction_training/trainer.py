import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import Config
from models import MultiTaskProcessModel
from counterfactual import CounterfactualRecommender
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score

class Trainer:
    def __init__(self, cfg: Config, model: MultiTaskProcessModel, cf_model: CounterfactualRecommender):
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.cf_model = cf_model.to(cfg.device)

        # 初始化: [act, time, out, cf_out, cf_time]
        # 给较大值压低高噪声任务(time)的初始权重, 避免初始梯度崩溃
        self.task_log_vars = nn.Parameter(torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0], device=cfg.device
        ))

        trainable_params = []
        seen_params = set()
        for module in (self.model, self.cf_model):
            for p in module.parameters():
                if not p.requires_grad or id(p) in seen_params:
                    continue
                trainable_params.append(p)
                seen_params.add(id(p))
        trainable_params.append(self.task_log_vars)
        self.optim = torch.optim.AdamW(
            trainable_params,
            lr=float(getattr(cfg, "learning_rate", 1e-3)),
            weight_decay=float(getattr(cfg, "weight_decay", 1e-4)),
        )

        # LR 调度: 验证 AUC 平台期时自动降学习率
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode="max", factor=0.5, patience=3,
            min_lr=1e-5,
        )

        self.train_history: list = []
        self.val_history: list = []

        # 早停状态
        self.best_val_score = -float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_state = None

    def _outcome_weight_tensor(self):
        weights = getattr(self.cfg, "outcome_class_weights", None)
        if weights is None:
            return None
        return torch.tensor(weights, dtype=torch.float, device=self.cfg.device)

    def _next_act_weight_tensor(self):
        weights = getattr(self.cfg, "next_act_class_weights", None)
        if weights is None:
            return None
        return torch.tensor(weights, dtype=torch.float, device=self.cfg.device)

    def _outcome_ce_loss(self, logits, target):
        return F.cross_entropy(
            logits,
            target,
            weight=self._outcome_weight_tensor(),
            label_smoothing=float(getattr(self.cfg, "outcome_label_smoothing", 0.0)),
        )

    def save_checkpoint(self, path: str, evaluation_context=None):
        torch.save({
            "model_state": self.model.state_dict(),
            "cf_model_state": self.cf_model.state_dict(),
            "optim_state": self.optim.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "task_log_vars": self.task_log_vars,
            "best_val_score": self.best_val_score,
            "best_epoch": self.best_epoch,
            "best_learning_rate": getattr(self, "best_learning_rate", None),
            "evaluation_context": evaluation_context,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "cfg": self.cfg,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.cf_model.load_state_dict(ckpt["cf_model_state"])
        self.optim.load_state_dict(ckpt["optim_state"])
        # Keep the Parameter object registered with the optimizer.
        with torch.no_grad():
            self.task_log_vars.copy_(ckpt["task_log_vars"].to(self.cfg.device))
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.best_val_score = ckpt.get("best_val_score", -float("inf"))
        self.best_epoch = ckpt.get("best_epoch", 0)
        self.best_learning_rate = ckpt.get("best_learning_rate")
        self.train_history = ckpt.get("train_history", [])
        self.val_history = ckpt.get("val_history", [])

    def _time_tail_weights(self, values, valid_mask):
        """Up-weight high-time targets in a batch while keeping the mean scale stable."""
        valid = valid_mask.float()
        tail_weight = float(getattr(self.cfg, "time_tail_weight", 0.0))
        if tail_weight <= 0.0 or valid_mask.sum().item() < 2:
            return valid

        q = float(getattr(self.cfg, "time_tail_quantile", 0.80))
        q = min(max(q, 0.0), 0.99)
        valid_values = values[valid_mask].detach()
        if valid_values.numel() < 2:
            return valid

        threshold = torch.quantile(valid_values, q)
        tail = ((values >= threshold) & valid_mask).float()
        weights = valid + tail_weight * tail
        mean_weight = weights[valid_mask].mean().clamp(min=1e-8)
        return weights / mean_weight

    @staticmethod
    def _masked_weighted_mean(losses, weights):
        denom = weights.sum().clamp(min=1e-8)
        return (losses * weights).sum() / denom

    def _compute_base_losses(self, outputs, batch):
        mask = batch["mask"]
        activities = batch["activities"]
        times = batch["times"]
        outcome = batch["outcome"]
        B, T = activities.shape
        lengths = mask.sum(dim=1)

        next_act_targets = torch.zeros(B, dtype=torch.long, device=activities.device)
        next_time_targets = torch.zeros(B, dtype=torch.float, device=activities.device)
        next_time_delta_targets = torch.zeros(B, dtype=torch.float, device=activities.device)
        next_act_valid = torch.zeros(B, dtype=torch.bool, device=activities.device)

        for i in range(B):
            L = int(lengths[i].item())
            if L >= 2:
                next_act_targets[i] = activities[i, L - 1]
                next_time_targets[i] = times[i, L - 1]
                next_time_delta_targets[i] = (times[i, L - 1] - times[i, L - 2]).clamp(min=0.0)
                next_act_valid[i] = True

        if next_act_valid.any():
            act_logits = outputs["next_act_logits"][next_act_valid, 1:]
            act_targets = next_act_targets[next_act_valid] - 1
            loss_next_act = F.cross_entropy(
                act_logits,
                act_targets,
                weight=self._next_act_weight_tensor(),
                label_smoothing=float(getattr(self.cfg, "next_act_label_smoothing", 0.0)),
            )
        else:
            loss_next_act = outputs["next_act_logits"].sum() * 0.0

        time_weights = self._time_tail_weights(next_time_targets, next_act_valid)
        delta_weights = self._time_tail_weights(next_time_delta_targets, next_act_valid)

        time_abs_loss = F.smooth_l1_loss(outputs["next_time_pred"], next_time_targets, reduction="none")
        time_abs_loss = self._masked_weighted_mean(time_abs_loss, time_weights)
        if "next_time_delta_pred" in outputs:
            time_delta_loss = F.smooth_l1_loss(
                outputs["next_time_delta_pred"], next_time_delta_targets, reduction="none"
            )
            time_delta_loss = self._masked_weighted_mean(time_delta_loss, delta_weights)
            loss_next_time = time_abs_loss + self.cfg.time_delta_loss_weight * time_delta_loss
        else:
            loss_next_time = time_abs_loss

        std_weight = float(getattr(self.cfg, "time_std_loss_weight", 0.0))
        if std_weight > 0.0 and next_act_valid.sum().item() >= 2:
            pred_valid = outputs["next_time_pred"][next_act_valid]
            target_valid = next_time_targets[next_act_valid]
            pred_std = torch.std(pred_valid, unbiased=False)
            target_std = torch.std(target_valid.detach(), unbiased=False)
            std_loss = F.smooth_l1_loss(pred_std, target_std)
            loss_next_time = loss_next_time + std_weight * std_loss

        loss_outcome = self._outcome_ce_loss(outputs["outcome_logits"], outcome)

        return loss_next_act, loss_next_time, loss_outcome

    def _compute_dfa_next_act_loss(self, outputs, batch):
        """Penalize probability mass outside strict Search-DFA progress actions."""
        logits = outputs["next_act_logits"][:, 1:]
        zero = logits.sum() * 0.0
        if logits.shape[1] == 0:
            return zero, 0, 0.0
        configured_weight = max(
            0.0, float(getattr(self.cfg, "dfa_next_act_loss_weight", 0.0))
        )
        action_mask = batch.get("dfa_next_act_mask")
        applicable = batch.get("dfa_aux_applicable")
        if configured_weight <= 0.0 or action_mask is None or applicable is None:
            return zero, 0, 0.0

        if action_mask.ndim != 2 or action_mask.shape != logits.shape:
            raise ValueError(
                "DFA next-activity mask/logit shape mismatch: "
                f"{tuple(action_mask.shape)} != {tuple(logits.shape)}"
            )
        if applicable.ndim != 1 or applicable.shape[0] != logits.shape[0]:
            raise ValueError(
                "DFA auxiliary applicability shape mismatch: "
                f"{tuple(applicable.shape)} for batch {logits.shape[0]}"
            )

        action_mask = action_mask.bool()
        eligible = applicable.bool() & action_mask.any(dim=1)

        # Recheck target compatibility so callers cannot accidentally combine
        # a progress mask with a contradictory supervised label.
        if bool(getattr(self.cfg, "dfa_next_act_require_target_valid", True)):
            activities = batch["activities"]
            lengths = batch["mask"].sum(dim=1).long()
            target_ids = torch.zeros_like(lengths)
            has_target = lengths >= 2
            rows = torch.arange(lengths.shape[0], device=lengths.device)
            if has_target.any():
                target_ids[has_target] = activities[
                    rows[has_target], lengths[has_target] - 1
                ]
            target_cols = (target_ids - 1).clamp(
                min=0, max=max(logits.shape[1] - 1, 0)
            )
            target_in_range = (
                (target_ids >= 1) & (target_ids <= logits.shape[1])
            )
            target_allowed = action_mask.gather(
                1, target_cols.unsqueeze(1)
            ).squeeze(1)
            eligible = eligible & has_target & target_in_range & target_allowed

        if not eligible.any():
            return zero, 0, 0.0

        selected_logits = logits[eligible]
        selected_mask = action_mask[eligible]
        log_total_mass = torch.logsumexp(selected_logits, dim=1)
        log_allowed_mass = torch.logsumexp(
            selected_logits.masked_fill(~selected_mask, -torch.inf), dim=1
        )
        per_sample_loss = (log_total_mass - log_allowed_mass).clamp_min(0.0)
        loss = per_sample_loss.mean()
        eligible_count = int(eligible.sum().detach().cpu())
        allowed_mass_sum = float(torch.exp(-per_sample_loss).sum().detach().cpu())
        return loss, eligible_count, allowed_mass_sum

    def _uwl_weighted(self, loss, log_var_idx):
        """Stable non-negative UWL term.

        The original Kendall-style ``+ log_var`` term can become negative when
        a near-zero auxiliary loss drives log_var far below zero.  Using
        softplus(log_var) preserves a learnable uncertainty penalty while
        keeping the reported objective non-negative and numerically bounded.
        """
        s = torch.clamp(
            self.task_log_vars[log_var_idx],
            min=float(getattr(self.cfg, "uwl_log_var_min", -1.0)),
            max=float(getattr(self.cfg, "uwl_log_var_max", 3.0)),
        )
        prec = torch.exp(-s)
        return 2.0 * prec * loss + F.softplus(s)

    def _clamp_task_log_vars_(self):
        with torch.no_grad():
            self.task_log_vars.clamp_(
                min=float(getattr(self.cfg, "uwl_log_var_min", -1.0)),
                max=float(getattr(self.cfg, "uwl_log_var_max", 3.0)),
            )

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        self.cf_model.train()

        total_logs = {
            "loss_total": 0.0,
            "loss_base_total": 0.0,
            "loss_next_act": 0.0,
            "loss_dfa_next_act": 0.0,
            "loss_next_act_objective": 0.0,
            "dfa_aux_applicable_fraction": 0.0,
            "dfa_aux_allowed_mass": 0.0,
            "loss_next_time": 0.0,
            "loss_outcome": 0.0,
            "loss_cf_outcome": 0.0,
            "loss_cf_time": 0.0,
            "uwl_next_act": 0.0,
            "uwl_next_time": 0.0,
            "uwl_outcome": 0.0,
            "uwl_cf_outcome": 0.0,
            "uwl_cf_time": 0.0,
        }
        for k in range(5):
            total_logs[f"w_{k}"] = 0.0
        n = 0
        dfa_aux_eligible_total = 0
        dfa_aux_sample_total = 0
        dfa_aux_allowed_mass_total = 0.0

        for batch in loader:
            batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
            self.optim.zero_grad()

            outputs = self.model(
                batch["activities"], batch["times"], batch["mask"],
                pattern_features=batch.get("pattern_features", None),
                static_cat=batch.get("static_cat"),
                static_num=batch.get("static_num"),
                dynamic_cat=batch.get("dynamic_cat"),
                dynamic_num=batch.get("dynamic_num"),
            )

            # --- 基础任务损失 ---
            l_act, l_time, l_out = self._compute_base_losses(outputs, batch)
            l_dfa_act, dfa_aux_eligible, dfa_allowed_mass_sum = (
                self._compute_dfa_next_act_loss(outputs, batch)
            )
            dfa_aux_eligible_total += dfa_aux_eligible
            dfa_aux_sample_total += int(batch["activities"].shape[0])
            dfa_aux_allowed_mass_total += dfa_allowed_mass_sum

            # --- 反事实训练（对比学习） ---
            # 策略: 对每个样本，选真实下一活动作为正例，
            #       随机采样一个不同活动作为负例，用对比损失训练
            B = batch["activities"].size(0)
            lengths = batch["mask"].sum(dim=1)

            # 获取真实下一活动
            pos_actions = torch.zeros(B, dtype=torch.long, device=self.cfg.device)
            valid = torch.zeros(B, dtype=torch.bool, device=self.cfg.device)
            for i in range(B):
                L = int(lengths[i].item())
                if L >= 2:
                    pos_actions[i] = batch["activities"][i, L - 1]
                    valid[i] = True

            # 正例 CF 预测（复用已计算的 backbone outputs）
            pos_cf = self.cf_model.cf_forward(
                pos_actions, base_outputs=outputs
            )

            # 负例 CF 预测（随机采样不等于正例的动作）
            neg_actions = torch.randint(1, self.cfg.num_activities, (B,), device=self.cfg.device)
            # 确保负例不等于正例
            collision_mask = (neg_actions == pos_actions) & valid
            neg_actions[collision_mask] = (neg_actions[collision_mask] + 1) % (self.cfg.num_activities - 1) + 1

            neg_cf = self.cf_model.cf_forward(
                neg_actions, base_outputs=outputs
            )

            # CF outcome 对比损失:
            #   正例 → 应该匹配真实 outcome
            #   负例 → 应推离真实 outcome（通过 margin ranking loss）
            true_outcome = batch["outcome"]
            pos_outcome_loss = self._outcome_ce_loss(
                pos_cf["cf_outcome_logits"], true_outcome
            )

            # 负例的 outcome 预测应与真实 outcome 有 margin 差距 (仅有效样本)
            pos_prob = F.softmax(pos_cf["cf_outcome_logits"], dim=-1)[range(B), true_outcome]
            neg_prob = F.softmax(neg_cf["cf_outcome_logits"], dim=-1)[range(B), true_outcome]
            contrastive_per_sample = F.relu(
                neg_prob - pos_prob + self.cfg.cf_contrastive_margin
            )
            contrastive_loss = (contrastive_per_sample * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

            # CF time 损失: 只对有效正例 (L>=2) 预测 time，无效样本不参与
            pos_time_targets = torch.zeros(B, device=self.cfg.device)
            for i in range(B):
                L = int(lengths[i].item())
                if L >= 2:
                    pos_time_targets[i] = batch["times"][i, L - 1]
            # 用 valid mask 排除 L<2 的样本
            time_loss_per_sample = F.smooth_l1_loss(
                pos_cf["cf_time_pred"], pos_time_targets, reduction='none'
            )
            pos_time_loss = (time_loss_per_sample * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

            l_cf_out = pos_outcome_loss + contrastive_loss
            l_cf_time = pos_time_loss

            # --- UWL 加权所有损失 ---
            next_act_weight = max(0.0, float(getattr(self.cfg, "next_act_loss_weight", 1.0)))
            dfa_next_act_weight = max(
                0.0, float(getattr(self.cfg, "dfa_next_act_loss_weight", 0.0))
            )
            l_act_objective = l_act + dfa_next_act_weight * l_dfa_act
            uwl_act = self._uwl_weighted(next_act_weight * l_act_objective, 0)
            uwl_time = self._uwl_weighted(l_time, 1)
            outcome_weight = max(0.0, float(getattr(self.cfg, "outcome_loss_weight", 1.0)))
            uwl_out = self._uwl_weighted(outcome_weight * l_out, 2)
            uwl_cf_out = self._uwl_weighted(l_cf_out, 3)
            uwl_cf_time = self._uwl_weighted(l_cf_time, 4)
            total_loss = (
                uwl_act +
                uwl_time +
                uwl_out +
                uwl_cf_out +
                uwl_cf_time
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.optim.param_groups[0]["params"], 1.0)
            self.optim.step()
            self._clamp_task_log_vars_()

            # 记录日志
            total_logs["loss_total"] += total_loss.item()
            total_logs["loss_base_total"] += (
                l_act_objective.item() + l_time.item() + l_out.item()
                + l_cf_out.item() + l_cf_time.item()
            )
            total_logs["loss_next_act"] += l_act.item()
            total_logs["loss_dfa_next_act"] += l_dfa_act.item()
            total_logs["loss_next_act_objective"] += l_act_objective.item()
            total_logs["loss_next_time"] += l_time.item()
            total_logs["loss_outcome"] += l_out.item()
            total_logs["loss_cf_outcome"] += l_cf_out.item()
            total_logs["loss_cf_time"] += l_cf_time.item()
            total_logs["uwl_next_act"] += uwl_act.item()
            total_logs["uwl_next_time"] += uwl_time.item()
            total_logs["uwl_outcome"] += uwl_out.item()
            total_logs["uwl_cf_outcome"] += uwl_cf_out.item()
            total_logs["uwl_cf_time"] += uwl_cf_time.item()
            for k in range(5):
                total_logs[f"w_{k}"] += self.task_log_vars[k].item()
            n += 1

        for k in total_logs:
            total_logs[k] /= max(n, 1)
        total_logs["dfa_aux_applicable_fraction"] = (
            dfa_aux_eligible_total / max(dfa_aux_sample_total, 1)
        )
        total_logs["dfa_aux_allowed_mass"] = (
            dfa_aux_allowed_mass_total / dfa_aux_eligible_total
            if dfa_aux_eligible_total > 0 else float("nan")
        )
        self.train_history.append(total_logs)
        return total_logs

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, record_history: bool = True):
        self.model.eval()
        self.cf_model.eval()

        correct = 0
        total = 0
        mae_time_sum = 0.0
        mae_time_count = 0
        time_true = []
        time_pred = []
        act_correct = 0
        act_total = 0
        act_top3_correct = 0
        act_top5_correct = 0

        # F1 / AUC 收集
        all_out_true, all_out_prob, all_out_pred = [], [], []
        all_act_true, all_act_pred = [], []

        for batch in loader:
            batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
            outputs = self.model(
                batch["activities"], batch["times"], batch["mask"],
                pattern_features=batch.get("pattern_features", None),
                static_cat=batch.get("static_cat"),
                static_num=batch.get("static_num"),
                dynamic_cat=batch.get("dynamic_cat"),
                dynamic_num=batch.get("dynamic_num"),
            )

            probs = torch.softmax(outputs["outcome_logits"], dim=-1)

            # F1/AUC
            all_out_true.extend(batch["outcome"].cpu().tolist())
            all_out_prob.extend(probs[:, 1].cpu().tolist())

            # Outcome accuracy
            pred_outcome = outputs["outcome_logits"].argmax(dim=-1)
            correct += (pred_outcome == batch["outcome"]).sum().item()
            total += batch["outcome"].numel()
            all_out_pred.extend(pred_outcome.cpu().tolist())

            # Next time MAE
            lengths = batch["mask"].sum(dim=1)
            times = batch["times"]
            for i in range(batch["activities"].size(0)):
                L = int(lengths[i].item())
                if L >= 2:
                    pred_t = outputs["next_time_pred"][i].item()
                    true_t = times[i, L - 1].item()
                    mae_time_sum += abs(pred_t - true_t)
                    time_true.append(true_t)
                    time_pred.append(pred_t)
                    mae_time_count += 1

            # Next activity accuracy
            activities = batch["activities"]
            for i in range(batch["activities"].size(0)):
                L = int(lengths[i].item())
                if L >= 2:
                    act_logits = outputs["next_act_logits"][i]
                    pred_act = act_logits[1:].argmax().item() + 1 if act_logits.numel() > 1 else 0
                    true_act = activities[i, L - 1].item()
                    if pred_act == true_act:
                        act_correct += 1
                    if act_logits.numel() > 1:
                        k = min(3, act_logits.numel() - 1)
                        topk = act_logits[1:].topk(k).indices.cpu().numpy() + 1
                        if true_act in set(int(x) for x in topk):
                            act_top3_correct += 1
                        k = min(5, act_logits.numel() - 1)
                        topk = act_logits[1:].topk(k).indices.cpu().numpy() + 1
                        if true_act in set(int(x) for x in topk):
                            act_top5_correct += 1
                    act_total += 1
                    all_act_true.append(true_act)
                    all_act_pred.append(pred_act)

        # 计算 F1 / AUC (用 argmax 而非 rounded prob, 与 accuracy 一致)
        try:
            pred_labels = np.asarray(all_out_pred, dtype=int)
            f1 = f1_score(all_out_true, pred_labels, average="macro", zero_division=0)
        except Exception:
            f1 = float("nan")
        try:
            true_arr = np.asarray(all_out_true, dtype=int)
            pred_arr = np.asarray(all_out_pred, dtype=int)
            recalls = []
            for cls in np.unique(true_arr):
                cls_mask = true_arr == cls
                if cls_mask.any():
                    recalls.append(float(np.mean(pred_arr[cls_mask] == cls)))
            balanced_acc = float(np.mean(recalls)) if recalls else float("nan")
            pred_pos_rate = float(np.mean(pred_arr == 1)) if len(pred_arr) else float("nan")
            true_pos_rate = float(np.mean(true_arr == 1)) if len(true_arr) else float("nan")
        except Exception:
            balanced_acc = float("nan")
            pred_pos_rate = float("nan")
            true_pos_rate = float("nan")
        try:
            if len(set(all_out_true)) >= 2:
                auc = roc_auc_score(all_out_true, all_out_prob)
            else:
                auc = float("nan")
        except Exception:
            auc = float("nan")
        auc_inverted = (
            1.0 - auc
            if np.isfinite(auc) else float("nan")
        )

        try:
            prob_arr = np.asarray(all_out_prob, dtype=float)
            true_prob_arr = np.asarray(all_out_true, dtype=float)
            outcome_brier = (
                float(np.mean((prob_arr - true_prob_arr) ** 2))
                if len(prob_arr) else float("nan")
            )
            ece = 0.0
            mce = 0.0
            n_prob = len(prob_arr)
            for b in range(10):
                lo = b / 10.0
                hi = (b + 1) / 10.0
                if b == 9:
                    bin_mask = (prob_arr >= lo) & (prob_arr <= hi)
                else:
                    bin_mask = (prob_arr >= lo) & (prob_arr < hi)
                if not bin_mask.any():
                    continue
                mean_conf = float(np.mean(prob_arr[bin_mask]))
                observed = float(np.mean(true_prob_arr[bin_mask]))
                gap = abs(mean_conf - observed)
                ece += (float(bin_mask.sum()) / max(n_prob, 1)) * gap
                mce = max(mce, gap)
            outcome_ece = float(ece) if n_prob else float("nan")
            outcome_mce = float(mce) if n_prob else float("nan")
        except Exception:
            outcome_brier = float("nan")
            outcome_ece = float("nan")
            outcome_mce = float("nan")

        try:
            next_act_f1 = (
                f1_score(all_act_true, all_act_pred, average="macro", zero_division=0)
                if all_act_true else float("nan")
            )
            next_act_padding_pred_rate = (
                float(np.mean(np.asarray(all_act_pred, dtype=int) == 0))
                if all_act_pred else float("nan")
            )
        except Exception:
            next_act_f1 = float("nan")
            next_act_padding_pred_rate = float("nan")

        if len(time_true) >= 2 and np.var(time_true) > 0:
            t_true = np.asarray(time_true, dtype=float)
            t_pred = np.asarray(time_pred, dtype=float)
            residual = t_pred - t_true
            abs_residual = np.abs(residual)
            ss_res = float(np.sum((t_true - t_pred) ** 2))
            ss_tot = float(np.sum((t_true - t_true.mean()) ** 2))
            next_time_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            next_time_corr = (
                float(np.corrcoef(t_true, t_pred)[0, 1])
                if np.var(t_pred) > 0 else float("nan")
            )
            next_time_rmse = float(np.sqrt(np.mean(residual ** 2)))
            next_time_bias = float(np.mean(residual))
            next_time_true_std = float(np.std(t_true))
            next_time_pred_std = float(np.std(t_pred))
            next_time_std_ratio = (
                next_time_pred_std / next_time_true_std
                if next_time_true_std > 0 else float("nan")
            )
            next_time_mae_p50 = float(np.percentile(abs_residual, 50))
            next_time_mae_p90 = float(np.percentile(abs_residual, 90))
            next_time_mae_p95 = float(np.percentile(abs_residual, 95))
            tail_threshold = float(np.percentile(t_true, 90))
            tail_mask = t_true >= tail_threshold
            next_time_tail_mae = (
                float(np.mean(abs_residual[tail_mask]))
                if tail_mask.any() else float("nan")
            )
        else:
            next_time_r2 = float("nan")
            next_time_corr = float("nan")
            next_time_rmse = float("nan")
            next_time_bias = float("nan")
            next_time_true_std = float("nan")
            next_time_pred_std = float("nan")
            next_time_std_ratio = float("nan")
            next_time_mae_p50 = float("nan")
            next_time_mae_p90 = float("nan")
            next_time_mae_p95 = float("nan")
            next_time_tail_mae = float("nan")

        result = {
            "outcome_acc": correct / max(total, 1),
            "outcome_f1": f1,
            "outcome_balanced_acc": balanced_acc,
            "outcome_auc": auc,
            "outcome_auc_inverted": auc_inverted,
            "outcome_brier": outcome_brier,
            "outcome_ece": outcome_ece,
            "outcome_mce": outcome_mce,
            "outcome_true_pos_rate": true_pos_rate,
            "outcome_pred_pos_rate": pred_pos_rate,
            "next_time_mae": mae_time_sum / max(mae_time_count, 1),
            "next_time_rmse": next_time_rmse,
            "next_time_r2": next_time_r2,
            "next_time_corr": next_time_corr,
            "next_time_bias": next_time_bias,
            "next_time_true_std": next_time_true_std,
            "next_time_pred_std": next_time_pred_std,
            "next_time_std_ratio": next_time_std_ratio,
            "next_time_mae_p50": next_time_mae_p50,
            "next_time_mae_p90": next_time_mae_p90,
            "next_time_mae_p95": next_time_mae_p95,
            "next_time_tail_mae": next_time_tail_mae,
            "next_act_acc": act_correct / max(act_total, 1),
            "next_act_top3_acc": act_top3_correct / max(act_total, 1),
            "next_act_top5_acc": act_top5_correct / max(act_total, 1),
            "next_act_f1": next_act_f1,
            "next_act_padding_pred_rate": next_act_padding_pred_rate,
        }
        if record_history:
            self.val_history.append(result)
        return result

    def check_early_stop(self, val_score: float) -> bool:
        """返回 True 表示应该停止训练。val_score 越大越好。"""
        if not np.isfinite(val_score):
            return False
        if val_score > self.best_val_score + self.cfg.early_stop_min_delta:
            self.best_val_score = val_score
            self.best_epoch = len(self.train_history)
            self.best_learning_rate = self.optim.param_groups[0]["lr"]
            self.patience_counter = 0
            self.best_state = {
                "model": copy.deepcopy(self.model.state_dict()),
                "cf_model": copy.deepcopy(self.cf_model.state_dict()),
                "task_log_vars": self.task_log_vars.detach().clone(),
            }
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.cfg.early_stop_patience

    def restore_best(self):
        """恢复到最佳验证集状态"""
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state["model"])
            self.cf_model.load_state_dict(self.best_state["cf_model"])
            with torch.no_grad():
                self.task_log_vars.copy_(self.best_state["task_log_vars"])
