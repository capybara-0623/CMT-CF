from dataclasses import dataclass, field
from typing import Any, Dict

import torch


@dataclass
class Config:
    num_activities: int
    num_roles: int = 0
    num_resources: int = 0

    activity_emb_dim: int = 64
    time_emb_dim: int = 16
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    pattern_feat_dim: int = 0
    pattern_dropout: float = 0.1
    pattern_gate_init_bias: float = -2.0
    pattern_fusion_target: str = "next_activity"
    pattern_fusion_protocol_version: int = 5

    max_seq_len: int = 50

    # Attribute encoding dimensions.
    n_static_cat: int = 0
    n_static_num: int = 0
    n_dynamic_cat: int = 0
    n_dynamic_num: int = 0
    attr_emb_dim: int = 32

    # Multitask training.
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    w_cf: float = 1.0
    w_temporal: float = 0.5
    time_delta_loss_weight: float = 0.5
    time_tail_weight: float = 0.0
    time_tail_quantile: float = 0.80
    time_std_loss_weight: float = 0.0
    next_act_loss_weight: float = 1.0
    next_act_class_weights: Any = None
    next_act_label_smoothing: float = 0.02
    next_act_transition_log_prior: Any = None
    next_act_transition_prior_weight: float = 0.35
    # Soft training-time guidance from explicit Search-DFA transitions that
    # strictly reduce the shortest distance to acceptance. The term is folded
    # into the next-activity objective before the existing UWL weight so old
    # five-task checkpoints remain structurally compatible.
    dfa_next_act_loss_weight: float = 0.0
    dfa_next_act_require_target_valid: bool = True
    dfa_next_act_aux_protocol_version: int = 2
    outcome_class_weights: Any = None
    outcome_loss_weight: float = 2.0
    outcome_label_smoothing: float = 0.02
    uwl_log_var_min: float = -1.0
    uwl_log_var_max: float = 3.0

    # Counterfactual recommendation.
    cf_topk: int = 5
    cf_intervention_points: int = 3
    cf_lambda_action: float = 1.0
    cf_lambda_time: float = 0.2
    cf_ranking: str = "pareto"
    cf_success_weight: float = 1.0
    cf_proximity_weight: float = 0.15
    cf_sparsity_weight: float = 0.10
    cf_diversity_weight: float = 0.10
    cf_generator: str = "default"
    cf_search_strategy: str = "minimal_constraint_relaxation"
    cf_search_protocol_version: int = 3
    cf_minimal_relaxation: bool = True
    # None preserves the previous policy. Protected IDs apply across clusters.
    cf_max_relaxed_constraints: Any = None
    cf_non_relaxable_rule_ids: Any = ()
    cf_per_position_limit: int = 5
    cf_combination_budget: int = 100
    cf_backbone_rerank: bool = True
    cf_prob_shift_weight: float = 0.5
    # Label-free Search-Patterns are ranking evidence only.  They never replace
    # Search-DFA or the independently constructed Audit-DFA.
    cf_soft_constraint_weight: float = 0.10

    # Training control.
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.001
    early_stop_time_weight: float = 0.0

    # DFA constraints.
    dfa_max_states: int = 500
    dfa_max_constraints: int = 10

    # CF training.
    cf_contrastive_margin: float = 0.5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    pattern_dict: Dict[str, Any] = field(default_factory=dict)
    pattern_delta_time: int = 3600
