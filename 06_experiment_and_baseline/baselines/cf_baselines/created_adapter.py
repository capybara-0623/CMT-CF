"""Explicit unavailable adapter, not a CREATED reimplementation or fake baseline."""
from evaluation.cf_types import SearchResult
from evaluation.experiment_protocol import stable_hash

REASON = ("CREATED not reproducibly integrated: paper-cited commit lacks the full evolutionary "
          "implementation; later GPL-3.0 implementation requires a provenance-checked train-only "
          "DataDistribution/Markov+attribute artifact and isolated dependency integration. "
          "No such artifact is frozen; this run is forbidden to fit/train one.")


def generate(query, oracle, search_policy, budget, **kwargs):
    if any(oracle.statistics().values()):
        raise ValueError("Requires a fresh oracle")
    return SearchResult([], "method_unavailable", dict(oracle=oracle.statistics(),
        attempted_nodes=0, walltime_seconds=0., cf_head_calls=0, time_policy="fixed",
        scored_pool=[], scored_pool_size=0, scored_pool_sha256=stable_hash([]),
        reason=REASON, external_algorithm_calls=0, usable_external_baseline=False))
