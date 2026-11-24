import torch
import torch.nn as nn

from config import Config


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, embed_dim)

    def forward(self, activities: torch.Tensor):
        seq_len = activities.size(1)
        positions = torch.arange(seq_len, device=activities.device).unsqueeze(0)
        return self.pos_emb(positions)


class TimeEncoder(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, times: torch.Tensor):
        # The data loader already normalizes times globally to [0, 1].
        # Per-trace normalization would erase the absolute elapsed-time scale.
        x = times.clamp(min=0.0, max=1.0).unsqueeze(-1)
        return self.net(x)


class AttributeEncoder(nn.Module):
    def __init__(self, n_static_cat, n_static_num, n_dynamic_cat, n_dynamic_num,
                 out_dim=32, cat_cardinality=100):
        super().__init__()
        self.n_static_cat = n_static_cat
        self.n_static_num = n_static_num
        self.n_dynamic_cat = n_dynamic_cat
        self.n_dynamic_num = n_dynamic_num

        self.has_static = (n_static_cat + n_static_num) > 0
        self.has_dynamic = (n_dynamic_cat + n_dynamic_num) > 0

        if n_static_cat > 0:
            self.static_cat_embs = nn.ModuleList([
                nn.Embedding(cat_cardinality, 8) for _ in range(n_static_cat)
            ])
        else:
            self.static_cat_embs = nn.ModuleList()

        if n_dynamic_cat > 0:
            self.dynamic_cat_embs = nn.ModuleList([
                nn.Embedding(cat_cardinality, 8) for _ in range(n_dynamic_cat)
            ])
        else:
            self.dynamic_cat_embs = nn.ModuleList()

        static_in = n_static_cat * 8 + n_static_num
        dynamic_in = n_dynamic_cat * 8 + n_dynamic_num
        self.static_proj = nn.Linear(static_in, out_dim) if self.has_static else None
        self.dynamic_proj = nn.Linear(dynamic_in, out_dim) if self.has_dynamic else None

    def forward(self, static_cat, static_num, dynamic_cat, dynamic_num):
        dev = None
        for tensor in (static_cat, static_num, dynamic_cat, dynamic_num):
            if tensor is not None:
                dev = tensor.device
                break
        if dev is None:
            dev = torch.device("cpu")
        if static_cat is None:
            static_cat = torch.zeros(1, max(self.n_static_cat, 1), dtype=torch.long, device=dev)
        if static_num is None:
            static_num = torch.zeros(1, max(self.n_static_num, 1), device=dev)
        if dynamic_cat is None:
            dynamic_cat = torch.zeros(1, 1, max(self.n_dynamic_cat, 1), dtype=torch.long, device=dev)
        if dynamic_num is None:
            dynamic_num = torch.zeros(1, 1, max(self.n_dynamic_num, 1), device=dev)

        B = static_num.shape[0]
        T = dynamic_num.shape[1]

        static_parts = []
        if self.n_static_cat > 0:
            for j in range(self.n_static_cat):
                ids = static_cat[:, j].clamp(0, 99)
                static_parts.append(self.static_cat_embs[j](ids))
        if self.n_static_num > 0:
            static_parts.append(static_num)
        static_vec = (
            torch.cat(static_parts, dim=-1)
            if static_parts else torch.zeros(B, 0, device=static_num.device)
        )
        static_out = (
            self.static_proj(static_vec)
            if self.has_static else torch.zeros(B, 0, device=static_num.device)
        )

        dyn_parts = []
        if self.n_dynamic_cat > 0:
            for j in range(self.n_dynamic_cat):
                ids = dynamic_cat[:, :, j].clamp(0, 99)
                dyn_parts.append(self.dynamic_cat_embs[j](ids))
        if self.n_dynamic_num > 0:
            dyn_parts.append(dynamic_num)
        dyn_vec = (
            torch.cat(dyn_parts, dim=-1)
            if dyn_parts else torch.zeros(B, T, 0, device=dynamic_num.device)
        )
        dyn_out = (
            self.dynamic_proj(dyn_vec)
            if self.has_dynamic else torch.zeros(B, T, 0, device=dynamic_num.device)
        )
        return static_out, dyn_out


class MultiTaskProcessModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.act_emb = nn.Embedding(cfg.num_activities, cfg.activity_emb_dim, padding_idx=0)
        self.pos_emb = PositionalEmbedding(cfg.max_seq_len, cfg.activity_emb_dim)
        self.time_enc = TimeEncoder(cfg.time_emb_dim)

        self.attr_enc = AttributeEncoder(
            cfg.n_static_cat, cfg.n_static_num,
            cfg.n_dynamic_cat, cfg.n_dynamic_num,
            out_dim=cfg.attr_emb_dim,
        )

        in_dim = (
            cfg.activity_emb_dim
            + cfg.time_emb_dim
            + (cfg.attr_emb_dim if cfg.n_static_cat + cfg.n_static_num > 0 else 0)
            + (cfg.attr_emb_dim if cfg.n_dynamic_cat + cfg.n_dynamic_num > 0 else 0)
        )
        self.input_proj = nn.Linear(in_dim, cfg.hidden_dim)

        self.pattern_proj = (
            nn.Sequential(
                nn.Linear(cfg.pattern_feat_dim, cfg.hidden_dim, bias=False),
                nn.ReLU(),
                nn.LayerNorm(cfg.hidden_dim),
                nn.Dropout(getattr(cfg, "pattern_dropout", cfg.dropout)),
            )
            if cfg.pattern_feat_dim > 0 else None
        )
        self.next_act_pattern_gate = (
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)
            if cfg.pattern_feat_dim > 0 else None
        )
        if cfg.pattern_feat_dim > 0 and getattr(
            cfg, "pattern_fusion_target", "next_activity"
        ) != "next_activity":
            raise ValueError(
                "pattern_fusion_target must be 'next_activity' when pattern features are enabled"
            )
        if self.next_act_pattern_gate is not None:
            # Start conservatively: the backbone dominates until validation
            # evidence teaches the model to open the pattern branch.
            nn.init.zeros_(self.next_act_pattern_gate.weight)
            nn.init.constant_(
                self.next_act_pattern_gate.bias,
                getattr(cfg, "pattern_gate_init_bias", -2.0),
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=4,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head_dropout = nn.Dropout(cfg.dropout)

        self.next_act_head = nn.Linear(cfg.hidden_dim, cfg.num_activities)
        transition_prior = getattr(cfg, "next_act_transition_log_prior", None)
        if transition_prior is not None:
            transition_prior = torch.as_tensor(transition_prior, dtype=torch.float)
            if tuple(transition_prior.shape) != (cfg.num_activities, cfg.num_activities):
                transition_prior = torch.empty(0)
        else:
            transition_prior = torch.empty(0)
        self.register_buffer("next_act_transition_log_prior", transition_prior)
        self.next_act_transition_prior_weight = float(
            getattr(cfg, "next_act_transition_prior_weight", 0.0)
        )
        self.next_time_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        nn.init.constant_(self.next_time_head[-1].bias, -2.0)
        self.outcome_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 2),
        )

    def encode(self, activities: torch.Tensor, times: torch.Tensor, mask: torch.Tensor,
               static_cat=None, static_num=None, dynamic_cat=None, dynamic_num=None):
        act_x = self.act_emb(activities)
        pos_x = self.pos_emb(activities)
        act_x = act_x + pos_x

        time_x = self.time_enc(times)
        static_attr, dynamic_attr = self.attr_enc(
            static_cat, static_num, dynamic_cat, dynamic_num
        )

        T = act_x.size(1)
        parts = [act_x, time_x]
        if self.attr_enc.has_static:
            parts.append(static_attr.unsqueeze(1).expand(-1, T, -1))
        if self.attr_enc.has_dynamic:
            if dynamic_attr.size(1) != T:
                dynamic_attr = dynamic_attr.expand(-1, T, -1)
            parts.append(dynamic_attr)

        x = torch.cat(parts, dim=-1)
        x = self.input_proj(x)


        x = self.dropout(x)
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        x = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=~mask,
        )
        return x

    def forward_prefix_outcome(self, activities, times, mask,
                               static_cat=None, static_num=None,
                               dynamic_cat=None, dynamic_num=None):
        """Score a pure observed prefix using the checkpoint's outcome head.

        Unlike legacy forward, every retained event is observed: no future tail
        or placeholder is required. Causal encoding makes this equivalent to
        legacy outcome scoring of the same prefix followed by a held-out event.
        """
        if (mask.shape != activities.shape or times.shape != activities.shape
                or activities.size(1) > self.cfg.max_seq_len):
            raise ValueError("Invalid prefix shapes or sequence length")
        lengths = mask.sum(dim=1)
        expected = torch.arange(mask.size(1), device=mask.device)[None, :] < lengths[:, None]
        if (lengths < 1).any() or not torch.equal(mask, expected):
            raise ValueError("Prefix masks must be nonempty and right-padded")
        x = self.encode(activities, times, mask, static_cat, static_num, dynamic_cat, dynamic_num)
        idx = (lengths - 1)[:, None, None].expand(-1, 1, x.size(-1))
        last_h = x.gather(1, idx).squeeze(1)
        weights = mask.unsqueeze(-1).float()
        mean_h = (x * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        context = torch.cat([self.head_dropout(last_h), self.head_dropout(mean_h)], dim=-1)
        return self.outcome_head(context)

    def forward(self, activities, times, mask, pattern_features=None,
                static_cat=None, static_num=None, dynamic_cat=None, dynamic_num=None):
        x = self.encode(
            activities, times, mask,
            static_cat, static_num, dynamic_cat, dynamic_num,
        )

        lengths = mask.sum(dim=1).clamp(min=1)
        last_idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, x.size(-1))
        last_h = x.gather(1, last_idx).squeeze(1)

        lengths_for_next = mask.sum(dim=1).clamp(min=2)
        prev_pos = lengths_for_next - 2
        prev_idx = prev_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, x.size(-1))
        prev_h = x.gather(1, prev_idx).squeeze(1)

        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        prefix_mask = (positions <= prev_pos.unsqueeze(1)) & mask
        prefix_weight = prefix_mask.unsqueeze(-1).float()
        prefix_mean_h = (x * prefix_weight).sum(dim=1) / prefix_weight.sum(dim=1).clamp(min=1.0)

        head_prev_h = self.head_dropout(prev_h)
        next_act_hidden = head_prev_h
        next_act_pattern_gate = None
        if self.pattern_proj is not None:
            if pattern_features is None:
                raise ValueError(
                    "pattern_features are required when pattern_feat_dim > 0"
                )
            if (
                pattern_features.dim() != 2
                or pattern_features.size(0) != activities.size(0)
                or pattern_features.size(1) != self.cfg.pattern_feat_dim
            ):
                raise ValueError(
                    "pattern_features must have shape "
                    f"[batch, {self.cfg.pattern_feat_dim}]"
                )
            pattern_emb = self.pattern_proj(pattern_features)
            next_act_pattern_gate = torch.sigmoid(
                self.next_act_pattern_gate(
                    torch.cat([head_prev_h, pattern_emb], dim=-1)
                )
            )
            # Task-specific residual fusion: pattern features directly affect
            # only the next-activity head, not the shared backbone or the
            # outcome/next-time heads.
            next_act_hidden = (
                head_prev_h + next_act_pattern_gate * pattern_emb
            )

        next_act_logits = self.next_act_head(next_act_hidden)
        if (
            self.next_act_transition_log_prior.numel() > 0
            and self.next_act_transition_prior_weight > 0.0
        ):
            prev_act_ids = activities.gather(1, prev_pos.unsqueeze(-1)).squeeze(-1)
            prev_act_ids = prev_act_ids.clamp(0, self.cfg.num_activities - 1)
            transition_bias = self.next_act_transition_log_prior[prev_act_ids]
            next_act_logits = (
                next_act_logits
                + self.next_act_transition_prior_weight * transition_bias
            )
        next_time_raw = self.next_time_head(head_prev_h).squeeze(-1)
        prev_time = times.gather(1, prev_pos.unsqueeze(-1)).squeeze(-1)
        remaining_time = (1.0 - prev_time).clamp(min=0.0)
        next_time_delta_pred = remaining_time * torch.sigmoid(next_time_raw)
        next_time_pred = prev_time + next_time_delta_pred
        # Use the prefix state before the held-out next event.  This avoids
        # learning outcome labels from terminal activities that act as proxies.
        base_outcome_context = torch.cat(
            [self.head_dropout(prev_h), self.head_dropout(prefix_mean_h)],
            dim=-1,
        )
        outcome_context = base_outcome_context
        outcome_logits = self.outcome_head(outcome_context)

        return {
            "hidden": x,
            "last_hidden": last_h,
            "prev_hidden": prev_h,
            "next_act_hidden": next_act_hidden,
            "outcome_hidden": outcome_context,
            "next_act_logits": next_act_logits,
            "next_time_delta_pred": next_time_delta_pred,
            "next_time_pred": next_time_pred,
            "outcome_logits": outcome_logits,
            "next_act_pattern_gate": next_act_pattern_gate,
            # Backward-compatible alias for existing diagnostics.
            "pattern_gate": next_act_pattern_gate,
        }
