import math
import multiprocessing as mp
import queue
from collections import deque
from itertools import combinations, product
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from pythomata import SimpleDFA
from config import Config
from explaination.visualizations.utils import normalize_activity_name, trace_accepts_dfa
from models import MultiTaskProcessModel
from utils.pattern_utils import trace_contains_pattern
from trace_alignment import does_trace_contain_pattern_advanced


def _compile_ltlf_dfa_worker(formula, dfa_max_states):
    """Compile one LTLf formula in a child process so callers can time out safely."""
    from explaination.visualizations.utils import parse, to_dfa, symbolic_to_singleton, alphabet

    parsed = parse(formula)
    sym_dfa = to_dfa(parsed)
    if len(sym_dfa.states) == 0 or len(sym_dfa.states) > dfa_max_states:
        return None, len(sym_dfa.states)
    letters = alphabet(parsed)
    letters.add('*')
    dfa = symbolic_to_singleton(sym_dfa, letters)
    return dfa, len(sym_dfa.states)


def _compile_ltlf_dfa_queue_worker(formula, dfa_max_states, result_queue):
    try:
        dfa, n_states = _compile_ltlf_dfa_worker(formula, dfa_max_states)
        result_queue.put(("ok", dfa, n_states))
    except Exception as exc:
        result_queue.put(("err", type(exc).__name__, str(exc)))


def flatten_cluster_rule_dfas(clusters: Optional[List[Dict]]) -> List[Dict]:
    """Return unique atomic rules retained by conflict-aware DFA clusters."""
    rules = []
    seen = set()
    for cluster in clusters or []:
        for rule in cluster.get("rule_dfas", ()):
            rule_id = str(rule.get("rule_id", rule.get("pid", "")))
            if not rule_id or rule_id in seen or rule.get("dfa") is None:
                continue
            seen.add(rule_id)
            rules.append(rule)
    return rules


def evaluate_minimal_constraint_relaxation(
    names: List[str], clusters: Optional[List[Dict]],
    max_relaxed_constraints=None, non_relaxable_rule_ids=(), *, exact_cost_order=False,
) -> Optional[Dict]:
    """Find the least hard-rule relaxation needed by one candidate trace.

    Each conflict-aware cluster remains an alternative conjunction, matching
    the existing multi-DFA semantics. Within a cluster, every atomic rule is
    checked independently. The best cluster is chosen lexicographically by
    relaxed-rule cardinality, calibration-derived cost, stable rule ids, and
    cluster id. Audit rules are never passed to this function.
    """
    filtered = [name for name in names if name is not None]
    protected = set(map(str, non_relaxable_rule_ids))
    atomic = {str(r.get("rule_id", r.get("pid", ""))): r
              for r in flatten_cluster_rule_dfas(clusters)}
    if protected - set(atomic):
        raise ValueError("Protected rules are not compiled: " + ", ".join(sorted(protected - set(atomic))))
    if any(atomic[r].get("compile_mode", "strict") != "strict" for r in protected):
        raise ValueError("Protected rules must retain their original compilation semantics")
    if max_relaxed_constraints is not None and max_relaxed_constraints < 0:
        raise ValueError("Maximum relaxed constraint count must be non-negative")
    if any(not trace_accepts_dfa(filtered, atomic[r]["dfa"]) for r in protected):
        return None
    best = None
    for cluster_index, cluster in enumerate(clusters or []):
        rules = list(cluster.get("rule_dfas", ()))
        if not rules:
            continue
        relaxed = []
        cost = 0.0
        exact_weights = []
        for rule in rules:
            if trace_accepts_dfa(filtered, rule["dfa"]):
                continue
            rule_id = str(rule.get("rule_id", rule.get("pid", "")))
            try:
                weight = float(rule.get("relaxation_weight", 1.0))
            except (TypeError, ValueError):
                if exact_cost_order:
                    raise ValueError("Exact relaxation requires finite positive rule weights")
                weight = 1.0
            if exact_cost_order and (not math.isfinite(weight) or weight <= 0):
                raise ValueError("Exact relaxation requires finite positive rule weights")
            if not math.isfinite(weight):
                weight = 1.0
            relaxed.append(rule_id)
            cost += max(weight, 1e-6)
            exact_weights.append(weight)

        relaxed_ids = tuple(sorted(relaxed))
        if exact_cost_order:
            cost = math.fsum(exact_weights)
        if max_relaxed_constraints is not None and len(relaxed_ids) > max_relaxed_constraints:
            continue
        cluster_id = str(cluster.get("cluster_id", f"cluster_{cluster_index:03d}"))
        key = (
            len(relaxed_ids),
            float(cost) if exact_cost_order else round(float(cost), 12),
            relaxed_ids,
            cluster_id,
        )
        result = {
            "strict": len(relaxed_ids) == 0,
            "relaxed_constraint_ids": relaxed_ids,
            "relaxation_count": len(relaxed_ids),
            "relaxation_cost": float(cost),
            "cluster_id": cluster_id,
            "cluster_rule_count": len(rules),
            "satisfied_constraint_count": len(rules) - len(relaxed_ids),
            "compile_relaxed_constraint_ids": tuple(sorted(
                str(rule.get("rule_id", rule.get("pid", "")))
                for rule in rules
                if rule.get("compile_mode", "strict") != "strict"
            )),
            "compile_omitted_constraint_ids": tuple(sorted(
                str(pid) for pid in cluster.get("compile_omitted_pids", ())
            )),
            "semantics": "disjunction_of_conjunctive_clusters",
            "selection_key": key,
        }
        if best is None or key < best[0]:
            best = (key, result)
    return best[1] if best is not None else None


def verify_retained_constraints(names, clusters, cluster_id, relaxed_ids):
    """Recheck the declared retained rules; unavailable evidence is undefined."""
    matches = [c for i, c in enumerate(clusters or [])
               if str(c.get("cluster_id", f"cluster_{i:03d}")) == str(cluster_id)]
    if len(matches) != 1 or not matches[0].get("rule_dfas"):
        return float("nan")
    rules = matches[0]["rule_dfas"]
    ids = {str(r.get("rule_id", r.get("pid", ""))) for r in rules}
    relaxed = set(map(str, relaxed_ids))
    if not relaxed <= ids or len(ids) != len(rules):
        return float("nan")
    retained = [r for r in rules if str(r.get("rule_id", r.get("pid", ""))) not in relaxed]
    # An empty retained set is logically vacuous, not positive compliance evidence.
    if not retained:
        return float("nan")
    if any(r.get("dfa") is None for r in retained):
        return float("nan")
    return int(all(trace_accepts_dfa(names, r["dfa"]) for r in retained))


# =========================================================================
# 冲突感知多 DFA 管理器 (贪婪兼容聚类 + 析取验证)
# =========================================================================

class ConflictAwareDFAManager:
    """管理多个模式 LTLf 公式的兼容性聚类，生成多 DFA"""

    def __init__(self, pattern_dict, act_encoder, parse_fn, to_dfa_fn,
                 sym_to_singleton_fn, alphabet_fn, dfa_max_states=500,
                 compilation_modes=None):
        """
        pattern_dict: {pid: networkx.DiGraph}
        其他参数用于 LTLf → DFA 转换
        """
        self.pattern_dict = pattern_dict
        self.act_encoder = act_encoder
        self.dfa_max_states = dfa_max_states
        self.max_cluster_size = 4
        self.max_combined_formula_len = 768
        self.direct_compile_formula_len = 48
        self.single_compile_timeout = 3.0
        self.relaxed_compile_timeout = 5.0

        # 预计算每个模式的 LTLf 公式和单体 DFA
        from pattern_to_ltlf import pattern_to_ltlf
        self._pattern_formulas: Dict[str, str] = {}
        self._pattern_dfas: Dict[str, any] = {}
        self._pattern_compile_modes: Dict[str, str] = {}
        self._compile_failures: Dict[str, str] = {}
        # Replay uses the saved compilation policy, never this machine's
        # timeout-dependent choice of strict/weak/omitted rules.
        if compilation_modes is not None:
            if set(compilation_modes) != set(pattern_dict) or any(
                    mode not in {"strict", "weakened_max_gap", "omitted"}
                    for mode in compilation_modes.values()):
                raise ValueError("Compilation manifest must cover every rule with a known mode")

        n_patterns = len(pattern_dict)
        print(f"  DFA compilation: compiling {n_patterns} pattern formulas", flush=True)
        for idx, (pid, graph) in enumerate(pattern_dict.items(), 1):
            saved_mode = compilation_modes[pid] if compilation_modes is not None else None
            if saved_mode == "omitted":
                self._compile_failures[str(pid)] = "omitted in source experiment"
                continue
            formula = self._relaxed_formula(graph) if saved_mode == "weakened_max_gap" else pattern_to_ltlf(graph)
            if saved_mode is not None and not formula:
                raise ValueError("Saved rule has no restorable formula: " + str(pid))
            if formula:
                self._pattern_formulas[pid] = formula
                try:
                    print(
                        f"    [{idx}/{n_patterns}] single DFA {pid} "
                        f"(formula_len={len(formula)})",
                        flush=True,
                    )
                    dfa = self._build_single_dfa(
                        formula, parse_fn, to_dfa_fn,
                        sym_to_singleton_fn, alphabet_fn,
                        timeout_s=30.0 if saved_mode is not None else self.single_compile_timeout,
                    )
                    if dfa is not None and len(dfa.states) > 0:
                        self._pattern_dfas[pid] = dfa
                        self._pattern_compile_modes[pid] = saved_mode or "strict"
                        print(f"        OK states={len(dfa.states)}", flush=True)
                    else:
                        if saved_mode is not None:
                            raise ValueError("Saved rule returned no usable DFA: " + str(pid))
                        self._compile_failures[str(pid)] = (
                            "strict compilation returned no usable DFA"
                        )
                        print("        skipped: empty or too many states", flush=True)
                except Exception as exc:
                    if saved_mode is not None:
                        raise ValueError(f"Cannot restore saved compilation mode for {pid}: {saved_mode}") from exc
                    print(
                        f"        strong DFA failed: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    relaxed_formula = self._relaxed_formula(graph)
                    if relaxed_formula and relaxed_formula != formula:
                        try:
                            print(
                                f"        retry relaxed DFA "
                                f"(formula_len={len(relaxed_formula)})",
                                flush=True,
                            )
                            dfa = self._build_single_dfa(
                                relaxed_formula, parse_fn, to_dfa_fn,
                                sym_to_singleton_fn, alphabet_fn,
                                timeout_s=self.relaxed_compile_timeout,
                            )
                            if dfa is not None and len(dfa.states) > 0:
                                self._pattern_formulas[pid] = relaxed_formula
                                self._pattern_dfas[pid] = dfa
                                self._pattern_compile_modes[pid] = (
                                    "weakened_max_gap"
                                )
                                print(
                                    f"        relaxed OK states={len(dfa.states)}",
                                    flush=True,
                                )
                            else:
                                print("        relaxed skipped: empty or too many states", flush=True)
                        except Exception as relaxed_exc:
                            self._compile_failures[str(pid)] = (
                                f"{type(relaxed_exc).__name__}: {relaxed_exc}"
                            )
                            print(
                                f"        relaxed failed: {type(relaxed_exc).__name__}: "
                                f"{relaxed_exc}",
                                flush=True,
                            )
                    else:
                        self._compile_failures[str(pid)] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        print("        no relaxed fallback available", flush=True)

        # 贪婪聚类
        self.clusters: List[Dict] = self._greedy_cluster(
            parse_fn, to_dfa_fn, sym_to_singleton_fn, alphabet_fn
        )

    def _build_single_dfa(self, formula, parse_fn, to_dfa_fn,
                          sym_to_singleton_fn, alphabet_fn, timeout_s=None):
        """将单个 LTLf 公式构建为 DFA"""
        if (
            timeout_s is not None
            and timeout_s > 0
            and len(formula) > self.direct_compile_formula_len
        ):
            return self._build_single_dfa_with_timeout(formula, timeout_s)

        parsed = parse_fn(formula)
        sym_dfa = to_dfa_fn(parsed)
        if len(sym_dfa.states) == 0:
            return None
        if len(sym_dfa.states) > self.dfa_max_states:
            return None
        letters = alphabet_fn(parsed)
        letters.add('*')
        return sym_to_singleton_fn(sym_dfa, letters)

    def _build_single_dfa_with_timeout(self, formula, timeout_s):
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_compile_ltlf_dfa_queue_worker,
            args=(formula, self.dfa_max_states, result_queue),
        )
        proc.start()
        proc.join(timeout_s)
        if proc.is_alive():
            proc.terminate()
            proc.join(2.0)
            if proc.is_alive() and hasattr(proc, "kill"):
                proc.kill()
                proc.join(2.0)
            result_queue.close()
            result_queue.join_thread()
            raise TimeoutError(
                f"LTLf->DFA compile timeout after {timeout_s:.0f}s "
                f"(formula_len={len(formula)})"
            )

        try:
            status, payload, details = result_queue.get_nowait()
        except queue.Empty:
            result_queue.close()
            result_queue.join_thread()
            if proc.exitcode and proc.exitcode != 0:
                raise RuntimeError(f"DFA worker exited with code {proc.exitcode}")
            return None

        result_queue.close()
        result_queue.join_thread()
        if status == "ok":
            return payload
        raise RuntimeError(f"{payload}: {details}")

    def _relaxed_formula(self, graph):
        """Fallback formula: keep activity order but remove bounded-X expansion."""
        from pattern_to_ltlf import pattern_to_ltlf

        relaxed = graph.copy()
        for _, _, data in relaxed.edges(data=True):
            data.pop("max_gap", None)
        return pattern_to_ltlf(relaxed)

    @staticmethod
    def _next_state_with_wildcard(dfa, state, letter):
        trans = dfa.transition_function.get(state, {})
        if letter in trans:
            return trans[letter]
        if "*" in trans:
            return trans["*"]
        return None

    def _intersect_dfas(self, dfa_a, dfa_b):
        """Build the product DFA for conjunction without recompiling LTLf."""
        alphabet_a = getattr(dfa_a, "alphabet", getattr(dfa_a, "_alphabet", set()))
        alphabet_b = getattr(dfa_b, "alphabet", getattr(dfa_b, "_alphabet", set()))
        alphabet = set(alphabet_a) | set(alphabet_b)
        if not alphabet:
            alphabet = set()
        alphabet.add("*")

        initial = (dfa_a.initial_state, dfa_b.initial_state)
        states = {initial}
        accepting = set()
        transitions = {}
        q = deque([initial])

        while q:
            state = q.popleft()
            sa, sb = state
            if sa in dfa_a.accepting_states and sb in dfa_b.accepting_states:
                accepting.add(state)
            state_transitions = {}
            for letter in alphabet:
                na = self._next_state_with_wildcard(dfa_a, sa, letter)
                nb = self._next_state_with_wildcard(dfa_b, sb, letter)
                if na is None or nb is None:
                    continue
                nxt = (na, nb)
                state_transitions[letter] = nxt
                if nxt not in states:
                    states.add(nxt)
                    if len(states) > self.dfa_max_states:
                        return None
                    q.append(nxt)
            transitions[state] = state_transitions

        if not accepting:
            return None
        return SimpleDFA(states, alphabet, initial, accepting, transitions).complete()

    def _check_compatibility(self, formula_a, formula_b,
                             parse_fn, to_dfa_fn):
        """检查两个公式是否兼容 (AND 后 DFA 非空)"""
        combined = f"({formula_a}) & ({formula_b})"
        try:
            parsed = parse_fn(combined)
            sym_dfa = to_dfa_fn(parsed)
            return len(sym_dfa.states) > 0
        except Exception:
            return False

    def _greedy_cluster(self, parse_fn, to_dfa_fn,
                        sym_to_singleton_fn, alphabet_fn):
        """Greedily merge compatible formulas into small, stronger DFA clusters."""
        pids = sorted(
            [p for p in self._pattern_formulas if p in self._pattern_dfas],
            key=lambda p: (-len(self._pattern_formulas[p]), str(p)),
        )
        clusters: List[Dict] = []
        combined_cache: Dict[Tuple[str, str], Optional[object]] = {}
        print(f"  DFA clustering: {len(pids)} compiled formulas available", flush=True)

        def atomic_rule(pid, formula, dfa):
            graph = self.pattern_dict.get(pid)
            metadata = getattr(graph, "graph", {}) if graph is not None else {}
            raw_weight = metadata.get(
                "constraint_selection_score",
                metadata.get("calibration_score", 1.0),
            )
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                weight = 1.0
            if not math.isfinite(weight):
                weight = 1.0
            return {
                "rule_id": str(pid),
                "pid": str(pid),
                "formula": formula,
                "dfa": dfa,
                "relaxation_weight": max(weight, 1e-6),
                "compile_mode": self._pattern_compile_modes.get(
                    pid, "strict"
                ),
            }

        for pid in pids:
            formula = self._pattern_formulas[pid]
            single_dfa = self._pattern_dfas.get(pid)
            if single_dfa is None or len(single_dfa.states) == 0:
                continue

            placed = False
            clusters.sort(
                key=lambda c: (len(c["pids"]), len(c["formula"])),
                reverse=True,
            )
            for cluster in clusters:
                if len(cluster["pids"]) >= self.max_cluster_size:
                    continue

                combined_formula = f"({cluster['formula']}) & ({formula})"
                if len(combined_formula) > self.max_combined_formula_len:
                    continue

                cache_key = (cluster["formula"], formula)
                if cache_key not in combined_cache:
                    try:
                        print(
                            f"    merge try: cluster_size={len(cluster['pids'])} + {pid} "
                            f"(formula_len={len(combined_formula)})",
                            flush=True,
                        )
                        combined_cache[cache_key] = self._intersect_dfas(
                            cluster["dfa"], single_dfa
                        )
                        if combined_cache[cache_key] is not None:
                            print(
                                f"    merge OK states={len(combined_cache[cache_key].states)}",
                                flush=True,
                            )
                    except Exception as exc:
                        print(
                            f"    merge skipped: {cluster['pids']} + {pid} "
                            f"({type(exc).__name__}: {exc})",
                            flush=True,
                        )
                        combined_cache[cache_key] = None
                merged_dfa = combined_cache[cache_key]
                if merged_dfa is None or len(merged_dfa.states) == 0:
                    continue

                cluster["pids"].append(pid)
                cluster["formula"] = combined_formula
                cluster["dfa"] = merged_dfa
                cluster["rule_dfas"].append(atomic_rule(pid, formula, single_dfa))
                placed = True
                break

            if not placed:
                clusters.append({
                    "pids": [pid],
                    "formula": formula,
                    "dfa": single_dfa,
                    "rule_dfas": [atomic_rule(pid, formula, single_dfa)],
                    "constraint_semantics": (
                        "disjunction_of_conjunctive_clusters"
                    ),
                })

        compiled = [
            c for c in clusters
            if c["dfa"] is not None and len(c["dfa"].states) > 0
        ]
        compiled.sort(key=lambda c: tuple(sorted(str(pid) for pid in c["pids"])))
        compiled_pids = {
            str(pid) for cluster in compiled for pid in cluster["pids"]
        }
        omitted = sorted(
            str(pid) for pid in self.pattern_dict if str(pid) not in compiled_pids
        )
        for index, cluster in enumerate(compiled):
            cluster["cluster_id"] = f"cluster_{index:03d}"
            cluster["compile_omitted_pids"] = list(omitted)
            cluster["compile_coverage"] = (
                len(compiled_pids) / max(len(self.pattern_dict), 1)
            )
            cluster["compile_failures"] = {
                pid: self._compile_failures.get(pid, "not compiled")
                for pid in omitted
            }
        return compiled

    def num_clusters(self) -> int:
        return len(self.clusters)

    def num_patterns_covered(self) -> int:
        return sum(len(c["pids"]) for c in self.clusters)


# =========================================================================
# 多 DFA 候选动作剪枝器
# =========================================================================

class MultiDFAActionPruner:
    """按多个 DFA cluster 的可达性并集剪枝候选动作。"""

    def __init__(self, clusters: List[Dict], act_encoder, num_activities: int):
        self.clusters = clusters
        self.act_encoder = act_encoder
        self.num_activities = num_activities
        self.supports_minimal_relaxation = bool(flatten_cluster_rule_dfas(clusters))
        self._num_dfas = len(clusters)

        # 每个 cluster 预计算可达态
        self._reachable_sets = []
        self._acceptance_distances = []
        self._dfas = []
        for c in clusters:
            dfa = c["dfa"]
            self._dfas.append(dfa)
            self._reachable_sets.append(
                self._compute_backwards_reachability(dfa)
            )
            self._acceptance_distances.append(
                self._compute_acceptance_distances(dfa)
            )

        # 动作 ID → 名称映射
        self._action_id_to_letter: Dict[int, Optional[str]] = {}
        for aid in range(1, num_activities):
            try:
                name = act_encoder.inverse_transform([aid - 1])[0]
            except (IndexError, ValueError):
                name = str(aid)
            self._action_id_to_letter[aid] = normalize_activity_name(name)

        # 统计 (先xN再汇总不便, 只记总)
        self.total_pruned_before = 0
        self.total_passed_actions = 0
        self.total_explicit_passes = 0
        self.total_wildcard_passes = 0
        self.total_live_clusters = 0

    def _compute_backwards_reachability(self, dfa) -> Set:
        reachable = set(dfa.accepting_states)
        queue = deque(dfa.accepting_states)
        reverse_adj = {}
        for s, trans in dfa.transition_function.items():
            for letter, ns in trans.items():
                if ns not in reverse_adj:
                    reverse_adj[ns] = []
                reverse_adj[ns].append((s, letter))
        while queue:
            current = queue.popleft()
            for prev_s, _ in reverse_adj.get(current, []):
                if prev_s not in reachable:
                    reachable.add(prev_s)
                    queue.append(prev_s)
        return reachable

    @staticmethod
    def _compute_acceptance_distances(dfa) -> Dict:
        """Return the shortest transition distance from each state to acceptance."""
        distances = {state: 0 for state in dfa.accepting_states}
        queue = deque(dfa.accepting_states)
        reverse_adj = {}
        for state, transitions in dfa.transition_function.items():
            for next_state in transitions.values():
                reverse_adj.setdefault(next_state, set()).add(state)

        while queue:
            current = queue.popleft()
            next_distance = distances[current] + 1
            for previous in reverse_adj.get(current, ()):
                if previous in distances:
                    continue
                distances[previous] = next_distance
                queue.append(previous)
        return distances

    def trace_all_prefixes(self, names: List[str]) -> List:
        """返回每个 DFA 的 (q_prefix, dead_end) 列表"""
        results = []
        for dfa_idx, dfa in enumerate(self._dfas):
            state = dfa.initial_state
            dead = False
            for name in names:
                if name is None:
                    continue
                letter = normalize_activity_name(name)
                trans = dfa.transition_function.get(state, {})
                if letter in trans:
                    state = trans[letter]
                elif '*' in trans:
                    state = trans['*']
                else:
                    dead = True
                    break
            results.append((state, dead))
        return results

    def get_union_candidates(
        self,
        prefix_states: List,
        action_ids: List[int],
        track_stats: bool = True,
    ) -> List[int]:
        """对多个 DFA 求候选的并集 (去重). 剪枝统计用实际检查数。"""
        valid_set = set()
        actual_checks = 0
        explicit_passes = 0
        wildcard_passes = 0
        live_cluster_checks = 0

        for aid in action_ids:
            letter = self._action_id_to_letter.get(aid)
            if letter is None:
                continue
            checked_this_action = False
            pass_mode = None
            for dfa_idx, (q_prefix, dead) in enumerate(prefix_states):
                if dead or q_prefix is None:
                    continue
                checked_this_action = True
                live_cluster_checks += 1
                trans = self._dfas[dfa_idx].transition_function.get(q_prefix, {})
                trans_letter = letter if letter in trans else '*'
                if trans_letter in trans:
                    next_state = trans[trans_letter]
                    if next_state in self._reachable_sets[dfa_idx]:
                        valid_set.add(aid)
                        pass_mode = "explicit" if trans_letter == letter else "wildcard"
                        break
            if checked_this_action:
                actual_checks += 1
            if pass_mode == "explicit":
                explicit_passes += 1
            elif pass_mode == "wildcard":
                wildcard_passes += 1

        if track_stats:
            self.total_pruned_before += actual_checks
            self.total_passed_actions += len(valid_set)
            self.total_explicit_passes += explicit_passes
            self.total_wildcard_passes += wildcard_passes
            self.total_live_clusters += live_cluster_checks
        return sorted(valid_set)

    def has_accepting_continuation(self, prefix_states: List) -> bool:
        """Return whether any live Search-DFA can still reach acceptance."""
        return any(
            not dead
            and state is not None
            and state in self._acceptance_distances[dfa_idx]
            for dfa_idx, (state, dead) in enumerate(prefix_states)
        )

    def get_strict_progress_candidates(
        self,
        prefix_states: List,
        action_ids: List[int],
    ) -> List[int]:
        """Return explicit actions that shorten the distance to acceptance.

        This training-only method is deliberately separate from
        ``get_union_candidates``. Hard counterfactual search continues to
        accept wildcard transitions and any accepting-reachable cluster.
        Mere future satisfiability is not positive evidence for a specific
        next activity, so wildcard fallback never creates supervision here.
        """
        progress_actions = set()
        for action_id in action_ids:
            letter = self._action_id_to_letter.get(action_id)
            if letter is None:
                continue
            for dfa_idx, (state, dead) in enumerate(prefix_states):
                if dead or state is None:
                    continue
                distances = self._acceptance_distances[dfa_idx]
                current_distance = distances.get(state)
                if current_distance is None or current_distance <= 0:
                    continue

                transitions = self._dfas[dfa_idx].transition_function.get(
                    state, {}
                )
                if letter not in transitions:
                    continue
                next_state = transitions[letter]
                next_distance = distances.get(next_state)
                if (
                    next_distance is not None
                    and next_distance < current_distance
                ):
                    progress_actions.add(action_id)
                    break
        return sorted(progress_actions)

    def decode_action_ids(self, action_ids: List[int]) -> List[Optional[str]]:
        """Map encoded non-padding activity ids to normalized DFA letters."""
        return [
            self._action_id_to_letter.get(int(aid))
            for aid in action_ids
        ]

    def minimum_relaxation(self, names: List[str]) -> Optional[Dict]:
        """Return the deterministic minimum relaxation for one full trace."""
        return evaluate_minimal_constraint_relaxation(names, self.clusters)

    def check_full_trace(self, names: List[str]) -> bool:
        """CF 被任意 DFA 接受? (析取)"""
        filtered = [n for n in names if n is not None]
        if len(filtered) == 0:
            return False
        for dfa in self._dfas:
            if trace_accepts_dfa(filtered, dfa):
                return True
        return False

    def prune_stats(self) -> Tuple[int, int, float]:
        total = self.total_pruned_before
        passed = self.total_passed_actions
        rate = 1.0 - (passed / total) if total > 0 else 0.0
        return total, passed, rate

    def prune_summary(self) -> str:
        return (f" [clusters={self.num_dfas}, explicit={self.total_explicit_passes}, "
                f"wildcard={self.total_wildcard_passes}, "
                f"live-cluster-checks={self.total_live_clusters}]")

    @property
    def num_dfas(self) -> int:
        return self._num_dfas


def annotate_dfa_next_activity_supervision(
    samples: List[Dict],
    clusters: List[Dict],
    act_encoder,
    num_activities: int,
    require_target_valid: bool = True,
) -> Dict:
    """Attach train-only Search-DFA progress masks to prefix samples.

    The last encoded activity in each expanded sample is the observed
    next-activity target; all preceding activities form the model input
    prefix. Mere acceptance reachability is intentionally insufficient:
    existential and subsequence constraints often allow every immediate
    action because the rule may still be satisfied later. A candidate action
    is auxiliary evidence only when its explicit transition strictly shortens
    the shortest distance to an accepting Search-DFA state. By default, the
    observed target must be one of those progress actions. Audit-DFA rules are
    deliberately not accepted by this interface.
    """
    non_padding_actions = max(0, int(num_activities) - 1)
    diagnostics = {
        "source": "search_dfa",
        "mode": "strict_acceptance_progress",
        "selection_uses_outcome_label": False,
        "uses_next_activity_target_for_safety_gate": bool(require_target_valid),
        "prefix_count": len(samples),
        "state_applicable_count": 0,
        "no_progress_action_count": 0,
        "informative_mask_count": 0,
        "target_compatible_count": 0,
        "target_incompatible_count": 0,
        "aux_applicable_count": 0,
        "aux_applicable_fraction": 0.0,
        "mean_progress_action_fraction": float("nan"),
        # Backward-compatible alias retained for existing analysis scripts.
        "mean_valid_action_fraction": float("nan"),
    }

    for sample in samples:
        sample.pop("dfa_aux_action_ids", None)
        sample.pop("dfa_valid_action_ids", None)
        sample["dfa_aux_applicable"] = False

    if not samples or not clusters or non_padding_actions <= 0:
        return diagnostics

    pruner = MultiDFAActionPruner(clusters, act_encoder, num_activities)
    all_action_ids = list(range(1, num_activities))
    progress_action_fractions = []

    for sample in samples:
        activities = sample.get("activities", ())
        length = min(
            int(sample.get("length", len(activities))), len(activities)
        )
        if length < 2:
            continue

        context_ids = [int(aid) for aid in activities[:length - 1]]
        target_id = int(activities[length - 1])
        if not 1 <= target_id < num_activities:
            continue

        context_names = pruner.decode_action_ids(context_ids)
        prefix_states = pruner.trace_all_prefixes(context_names)
        if not pruner.has_accepting_continuation(prefix_states):
            continue
        diagnostics["state_applicable_count"] += 1

        progress_action_ids = pruner.get_strict_progress_candidates(
            prefix_states,
            all_action_ids,
        )
        if not progress_action_ids:
            diagnostics["no_progress_action_count"] += 1
            continue

        progress_action_fractions.append(
            len(progress_action_ids) / float(non_padding_actions)
        )
        sample["dfa_aux_action_ids"] = progress_action_ids

        informative = len(progress_action_ids) < non_padding_actions
        if not informative:
            continue
        diagnostics["informative_mask_count"] += 1

        target_supported = target_id in progress_action_ids
        if target_supported:
            diagnostics["target_compatible_count"] += 1
        else:
            diagnostics["target_incompatible_count"] += 1

        if target_supported or not require_target_valid:
            sample["dfa_aux_applicable"] = True
            diagnostics["aux_applicable_count"] += 1

    diagnostics["aux_applicable_fraction"] = (
        diagnostics["aux_applicable_count"] / float(len(samples))
    )
    if progress_action_fractions:
        mean_fraction = (
            sum(progress_action_fractions) / len(progress_action_fractions)
        )
        diagnostics["mean_progress_action_fraction"] = mean_fraction
        diagnostics["mean_valid_action_fraction"] = mean_fraction
    return diagnostics


# =========================================================================
# 单 DFA 候选动作剪枝器（回退用，接口兼容）
# =========================================================================

class DFAActionPruner:
    """封装单个 DFA 的反向可达性信息，供候选动作快速剪枝。"""

    def __init__(self, dfa, act_encoder, num_activities: int):
        self.dfa = dfa
        self.act_encoder = act_encoder
        self.num_activities = num_activities

        self._action_id_to_letter: Dict[int, Optional[str]] = {}
        for aid in range(1, num_activities):
            try:
                name = act_encoder.inverse_transform([aid - 1])[0]
            except (IndexError, ValueError):
                name = str(aid)
            self._action_id_to_letter[aid] = normalize_activity_name(name)

        self._reachable_to_accept: Set = self._compute_backwards_reachability()
        self.total_pruned_before = 0
        self.total_passed_actions = 0
        self.total_explicit_passes = 0
        self.total_wildcard_passes = 0

    def _compute_backwards_reachability(self) -> Set:
        reachable = set(self.dfa.accepting_states)
        queue = deque(self.dfa.accepting_states)
        reverse_adj = {}
        for s, trans in self.dfa.transition_function.items():
            for letter, ns in trans.items():
                if ns not in reverse_adj:
                    reverse_adj[ns] = []
                reverse_adj[ns].append((s, letter))
        while queue:
            current = queue.popleft()
            for prev_s, _ in reverse_adj.get(current, []):
                if prev_s not in reachable:
                    reachable.add(prev_s)
                    queue.append(prev_s)
        return reachable

    def trace_prefix_state(self, names: List[str]):
        state = self.dfa.initial_state
        for name in names:
            if name is None:
                continue
            letter = normalize_activity_name(name)
            trans = self.dfa.transition_function.get(state, {})
            if letter in trans:
                state = trans[letter]
            elif '*' in trans:
                state = trans['*']
            else:
                return None, True
        return state, False

    def get_valid_candidates(
        self, q_prefix, action_ids: List[int]
    ) -> List[int]:
        if q_prefix is None:
            return []
        trans = self.dfa.transition_function.get(q_prefix, {})
        valid = []
        explicit_passes = 0
        wildcard_passes = 0
        for aid in action_ids:
            letter = self._action_id_to_letter.get(aid)
            if letter is None:
                continue
            trans_letter = letter if letter in trans else '*'
            if trans_letter in trans:
                next_state = trans[trans_letter]
                if next_state in self._reachable_to_accept:
                    valid.append(aid)
                    if trans_letter == letter:
                        explicit_passes += 1
                    else:
                        wildcard_passes += 1
        self.total_pruned_before += len(action_ids)
        self.total_passed_actions += len(valid)
        self.total_explicit_passes += explicit_passes
        self.total_wildcard_passes += wildcard_passes
        return valid

    def check_full_trace(self, names: List[str]) -> bool:
        return trace_accepts_dfa(
            [n for n in names if n is not None], self.dfa
        )

    def prune_stats(self) -> Tuple[int, int, float]:
        total = self.total_pruned_before
        passed = self.total_passed_actions
        rate = 1.0 - (passed / total) if total > 0 else 0.0
        return total, passed, rate


# =========================================================================
# CounterfactualRecommender
# =========================================================================

class CounterfactualRecommender(nn.Module):
    def __init__(self, cfg: Config, backbone: MultiTaskProcessModel):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone

        self.action_embed = nn.Embedding(cfg.num_activities, cfg.hidden_dim)

        self.cf_outcome_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 2),
        )
        self.cf_time_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
            nn.Sigmoid(),
        )

        # DFA 动作剪枝器缓存：统一支持单 DFA 与冲突感知多 DFA。
        self._dfa_pruner_cache: Dict[int, any] = {}
        self.pattern_encoder = None

    def set_pattern_encoder(self, pattern_encoder):
        """Attach the train-fitted encoder used for factual and CF prefixes."""
        selected = len(getattr(pattern_encoder, "selected_patterns_", []))
        expected = int(self.cfg.pattern_feat_dim)
        if selected != expected:
            raise ValueError(
                "Pattern encoder/model dimension mismatch: "
                f"{selected} != {expected}"
            )
        self.pattern_encoder = pattern_encoder

    def encode_pattern_batch(
        self,
        action_sequences,
        time_sequences,
        observed_lengths,
        device=None,
    ):
        if self.pattern_encoder is None or int(self.cfg.pattern_feat_dim) <= 0:
            return None
        if isinstance(observed_lengths, int):
            observed_lengths = [observed_lengths] * len(action_sequences)
        if not (
            len(action_sequences) == len(time_sequences) == len(observed_lengths)
        ):
            raise ValueError(
                "Candidate activity, time, and observed-length batches must align"
            )
        samples = []
        for idx, (actions, times, observed_len) in enumerate(
            zip(action_sequences, time_sequences, observed_lengths)
        ):
            samples.append({
                "case_id": f"cf_{idx}",
                "source_case_id": f"cf_{idx}",
                "pattern_sample_id": f"cf_{idx}::p{int(observed_len)}",
                "activities": list(actions),
                "times": list(times),
                "length": len(actions),
                "prefix_len": int(observed_len),
            })
        encoded = self.pattern_encoder.transform(samples)
        if len(encoded) != len(samples):
            raise ValueError(
                f"Candidate pattern encoding incomplete: {len(encoded)}/{len(samples)}"
            )
        if encoded.shape[1] != int(self.cfg.pattern_feat_dim):
            raise ValueError(
                "Candidate pattern feature dimension mismatch: "
                f"{encoded.shape[1]} != {self.cfg.pattern_feat_dim}"
            )
        target_device = device if device is not None else self.action_embed.weight.device
        return torch.as_tensor(
            encoded.to_numpy(dtype="float32"),
            dtype=torch.float,
            device=target_device,
        )

    def _compute_cf_scores(self, hidden, action_ids):
        if len(action_ids) == 0:
            return (
                torch.empty(0, device=hidden.device),
                torch.empty(0, device=hidden.device),
                torch.empty(0, device=hidden.device),
            )
        a_emb = self.action_embed(action_ids)
        cf_h = torch.cat([hidden.unsqueeze(0).expand(a_emb.size(0), -1), a_emb], dim=-1)
        cf_out = self.cf_outcome_head(cf_h)
        cf_time = self.cf_time_head(cf_h).squeeze(-1)
        prob_good = F.softmax(cf_out, dim=-1)[:, 1]
        score = prob_good - self.cfg.cf_lambda_time * F.relu(cf_time)
        return prob_good, cf_time, score

    def set_multi_dfa(self, clusters: List[Dict], act_encoder):
        """设置多 DFA 模式 (冲突感知聚类结果)"""
        if clusters is None or len(clusters) == 0:
            self._dfa_pruner_cache.pop("multi", None)
            return
        ctx = MultiDFAActionPruner(clusters, act_encoder, self.cfg.num_activities)
        self._dfa_pruner_cache["multi"] = ctx
        print(f"  多 DFA 模式: {ctx.num_dfas} clusters, "
              f"覆盖 {sum(len(c['pids']) for c in clusters)} 模式")

    def _get_dfa_pruner(self, pattern_dfa, act_encoder):
        """获取动作剪枝器（优先多 DFA，其次单 DFA）。"""
        multi = self._dfa_pruner_cache.get("multi")
        if multi is not None:
            return multi

        if pattern_dfa is None or act_encoder is None:
            return None
        key = id(pattern_dfa)
        if key not in self._dfa_pruner_cache:
            self._dfa_pruner_cache[key] = DFAActionPruner(
                pattern_dfa, act_encoder, self.cfg.num_activities
            )
        return self._dfa_pruner_cache[key]

    def _dfa_filter(self, candidate_names, pattern_dfa):
        """对完整候选轨迹执行 DFA 接受检查。"""
        if pattern_dfa is None:
            return True
        word = [n for n in candidate_names if n is not None]
        if len(word) == 0:
            return False
        return trace_accepts_dfa(word, pattern_dfa)

    def _legacy_pattern_soft_penalty(self, names, times, pattern_dict):
        """模式包含软惩罚：未命中任何已知高质量模式则扣分"""
        if pattern_dict is None:
            return 0.0
        for pid in pattern_dict:
            if does_trace_contain_pattern_advanced(
                [n for n in names if n is not None],
                times,
                pattern_dict[pid],
                delta_time_seconds=self.cfg.pattern_delta_time
            ):
                return 0.0
        return -0.1

    def soft_constraint_score(self, names, times, pattern_dict):
        """Return a bounded support-weighted reward for soft Search-Patterns."""
        if not pattern_dict:
            return 0.0
        total_weight = 0.0
        hit_weight = 0.0
        for graph in pattern_dict.values():
            weight = max(
                0.0,
                float(graph.graph.get("calibration_score", 1.0)),
            )
            total_weight += weight
            if does_trace_contain_pattern_advanced(
                [n for n in names if n is not None],
                times,
                graph,
                delta_time_seconds=self.cfg.pattern_delta_time,
            ):
                hit_weight += weight
        if total_weight <= 0.0:
            return 0.0
        reward_cap = max(
            0.0, float(getattr(self.cfg, "cf_soft_constraint_weight", 0.10))
        )
        return reward_cap * (hit_weight / total_weight)

    def _pattern_soft_penalty(self, names, times, pattern_dict):
        """Backward-compatible alias for the calibrated soft ranking reward."""
        return self.soft_constraint_score(names, times, pattern_dict)

    @staticmethod
    def _sequence_distance(seq_a, seq_b):
        max_len = max(len(seq_a), len(seq_b), 1)
        diff = 0
        for k in range(max_len):
            a = seq_a[k] if k < len(seq_a) else None
            b = seq_b[k] if k < len(seq_b) else None
            if a != b:
                diff += 1
        return diff / max_len

    @staticmethod
    def _normalise(value, lo, hi):
        if hi <= lo:
            return 0.5
        return (value - lo) / (hi - lo)

    def _candidate_objectives(self, orig_acts, orig_times, candidate):
        act_seq, time_seq, base_score = candidate
        denom = max(len(orig_acts), 1)
        changed = sum(
            1 for a, b in zip(orig_acts, act_seq)
            if a != b
        )
        change_rate = changed / denom

        time_diffs = [
            abs(float(a) - float(b))
            for a, b in zip(orig_times, time_seq)
        ]
        mean_time_diff = sum(time_diffs) / max(len(time_diffs), 1)

        sparsity = 1.0 - min(change_rate, 1.0)
        temporal_proximity = 1.0 - min(mean_time_diff, 1.0)
        proximity = 1.0 - min((change_rate + mean_time_diff) / 2.0, 1.0)

        return {
            "base_score": float(base_score),
            "proximity": proximity,
            "sparsity": sparsity,
            "temporal_proximity": temporal_proximity,
            "changed": changed,
        }

    @staticmethod
    def _pareto_front_indices(objectives):
        front = []
        for i, obj_i in enumerate(objectives):
            dominated = False
            values_i = (obj_i["base_score"], obj_i["proximity"], obj_i["sparsity"])
            for j, obj_j in enumerate(objectives):
                if i == j:
                    continue
                values_j = (obj_j["base_score"], obj_j["proximity"], obj_j["sparsity"])
                if all(b >= a for a, b in zip(values_i, values_j)) and any(
                    b > a for a, b in zip(values_i, values_j)
                ):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        return front

    def _pareto_order(self, candidates, objectives):
        remaining = list(range(len(candidates)))
        ordered = []
        while remaining:
            local_objectives = [objectives[idx] for idx in remaining]
            local_front = self._pareto_front_indices(local_objectives)
            if not local_front:
                ordered.extend(sorted(remaining, key=lambda idx: candidates[idx][2], reverse=True))
                break

            front = [remaining[pos] for pos in local_front]
            front.sort(key=lambda idx: candidates[idx][2], reverse=True)
            ordered.extend(front)

            front_set = set(front)
            remaining = [idx for idx in remaining if idx not in front_set]
        return ordered

    def _rank_counterfactuals(self, orig_acts, orig_times, candidates, topk):
        if not candidates:
            return []

        strategy = str(getattr(self.cfg, "cf_ranking", "score")).strip().lower()
        if strategy in ("score", "legacy", "scalar"):
            return sorted(candidates, key=lambda x: x[2], reverse=True)[:topk]

        if strategy not in ("pareto", "pareto_diverse", "dpp"):
            strategy = "pareto"

        objectives = [
            self._candidate_objectives(orig_acts, orig_times, cand)
            for cand in candidates
        ]
        ordered = self._pareto_order(candidates, objectives)

        base_scores = [obj["base_score"] for obj in objectives]
        lo, hi = min(base_scores), max(base_scores)
        w_success = max(0.0, float(getattr(self.cfg, "cf_success_weight", 1.0)))
        w_prox = max(0.0, float(getattr(self.cfg, "cf_proximity_weight", 0.15)))
        w_sparse = max(0.0, float(getattr(self.cfg, "cf_sparsity_weight", 0.10)))
        w_diverse = max(0.0, float(getattr(self.cfg, "cf_diversity_weight", 0.10)))
        w_rank = 0.05
        denom = max(w_success + w_prox + w_sparse + w_diverse + w_rank, 1e-8)

        selected = []
        selected_set = set()
        selected_scores = {}
        rank_den = max(len(ordered) - 1, 1)

        while len(selected) < min(topk, len(ordered)):
            best_idx = None
            best_key = None

            for rank_pos, idx in enumerate(ordered):
                if idx in selected_set:
                    continue

                cand = candidates[idx]
                obj = objectives[idx]
                norm_success = self._normalise(obj["base_score"], lo, hi)
                rank_bonus = 1.0 - (rank_pos / rank_den)
                if not selected:
                    novelty = 1.0
                else:
                    novelty = min(
                        self._sequence_distance(cand[0], candidates[s_idx][0])
                        for s_idx in selected
                    )

                composite = (
                    w_success * norm_success
                    + w_prox * obj["proximity"]
                    + w_sparse * obj["sparsity"]
                    + w_diverse * novelty
                    + w_rank * rank_bonus
                ) / denom

                key = (composite, obj["base_score"], -rank_pos)
                if best_key is None or key > best_key:
                    best_idx = idx
                    best_key = key

            if best_idx is None:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            selected_scores[best_idx] = float(best_key[0])

        ranked = []
        for idx in selected:
            cand = candidates[idx]
            rank_score = selected_scores.get(idx, float(cand[2]))
            ranked.append((cand[0], cand[1], float(rank_score)))

        return ranked

    def _candidate_model_kwargs(self, index, length, repeat,
                                pattern_features=None, static_cat=None,
                                static_num=None, dynamic_cat=None,
                                dynamic_num=None):
        kwargs = {}

        def select_and_repeat(tensor, dims):
            if tensor is None:
                return None
            if tensor.dim() == 0:
                return None
            if dims == 2:
                out = tensor[index:index + 1]
            else:
                out = tensor[index:index + 1, :length]
            if repeat > 1:
                reps = [repeat] + [1] * (out.dim() - 1)
                out = out.repeat(*reps)
            return out

        t = select_and_repeat(pattern_features, 2)
        if t is not None:
            kwargs["pattern_features"] = t
        t = select_and_repeat(static_cat, 2)
        if t is not None:
            kwargs["static_cat"] = t
        t = select_and_repeat(static_num, 2)
        if t is not None:
            kwargs["static_num"] = t
        t = select_and_repeat(dynamic_cat, 3)
        if t is not None:
            kwargs["dynamic_cat"] = t
        t = select_and_repeat(dynamic_num, 3)
        if t is not None:
            kwargs["dynamic_num"] = t
        return kwargs

    def _rerank_with_backbone(self, sample_idx, orig_acts, orig_times, candidates,
                              observed_length, pattern_features=None, static_cat=None,
                              static_num=None, dynamic_cat=None,
                              dynamic_num=None):
        if not candidates or not bool(getattr(self.cfg, "cf_backbone_rerank", True)):
            return candidates

        device = self.action_embed.weight.device
        length = len(orig_acts)
        if length == 0:
            return candidates

        try:
            orig_act_t = torch.tensor([orig_acts], dtype=torch.long, device=device)
            orig_time_t = torch.tensor([orig_times], dtype=torch.float, device=device)
            orig_mask = torch.ones_like(orig_act_t, dtype=torch.bool)
            orig_kwargs = self._candidate_model_kwargs(
                sample_idx, length, 1,
                pattern_features, static_cat, static_num, dynamic_cat, dynamic_num,
            )
            encoded_patterns = self.encode_pattern_batch(
                [orig_acts], [orig_times], [observed_length], device=device
            )
            if encoded_patterns is not None:
                orig_kwargs["pattern_features"] = encoded_patterns
            orig_logits = self.backbone(
                orig_act_t, orig_time_t, orig_mask, **orig_kwargs
            )["outcome_logits"]
            orig_prob = torch.softmax(orig_logits, dim=-1)[0, 1].item()
        except Exception:
            if self.pattern_encoder is not None:
                raise
            orig_prob = 0.0

        reranked = []
        batch_size = 64
        shift_weight = max(0.0, float(getattr(self.cfg, "cf_prob_shift_weight", 0.5)))

        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start:start + batch_size]
            try:
                act_batch = torch.tensor(
                    [c[0][:length] for c in chunk],
                    dtype=torch.long,
                    device=device,
                )
                time_batch = torch.tensor(
                    [c[1][:length] for c in chunk],
                    dtype=torch.float,
                    device=device,
                )
                mask_batch = torch.ones_like(act_batch, dtype=torch.bool)
                kwargs = self._candidate_model_kwargs(
                    sample_idx, length, len(chunk),
                    pattern_features, static_cat, static_num, dynamic_cat, dynamic_num,
                )
                encoded_patterns = self.encode_pattern_batch(
                    [c[0][:length] for c in chunk],
                    [c[1][:length] for c in chunk],
                    [observed_length] * len(chunk),
                    device=device,
                )
                if encoded_patterns is not None:
                    kwargs["pattern_features"] = encoded_patterns
                logits = self.backbone(
                    act_batch, time_batch, mask_batch, **kwargs
                )["outcome_logits"]
                probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
            except Exception:
                if self.pattern_encoder is not None:
                    raise
                probs = [float(c[2]) for c in chunk]

            for (act_seq, time_seq, old_score), prob in zip(chunk, probs):
                prob = float(prob)
                prob_shift = prob - float(orig_prob)
                score = prob + shift_weight * prob_shift
                reranked.append((act_seq, time_seq, score))

        return reranked

    def generate_counterfactual_traces(self, activities, times, mask, topk=5,
                                       pattern_dict=None, act_encoder=None,
                                       pattern_dfa=None, intervention_positions=None,
                                       pattern_features=None, static_cat=None,
                                       static_num=None, dynamic_cat=None,
                                       dynamic_num=None):
        """
        生成 Top-K 反事实轨迹，采用 DFA 动作剪枝和有界多点组合。

        约束优先级:
          1. DFA 动作剪枝：只评分仍可到达接受状态的候选动作
          2. DFA 完整验证：单点与组合候选均需通过全轨迹检查
          3. 校准软模式：仅以有界权重影响候选排序
          4. 无约束: 纯模型得分排序 (回退)
        """
        self.eval()
        if intervention_positions is None:
            intervention_positions = [-1]

        # 初始化 DFA 动作剪枝器。
        dfa_pruner = self._get_dfa_pruner(pattern_dfa, act_encoder)
        full_checked = 0
        full_rejected = 0
        minimal_relaxation_enabled = bool(
            getattr(self.cfg, "cf_minimal_relaxation", True)
            and isinstance(dfa_pruner, MultiDFAActionPruner)
            and dfa_pruner.supports_minimal_relaxation
        )
        protected_ids = tuple(getattr(self.cfg, "cf_non_relaxable_rule_ids", ()))
        max_relaxed = getattr(self.cfg, "cf_max_relaxed_constraints", None)
        if protected_ids or max_relaxed is not None:
            if not isinstance(dfa_pruner, MultiDFAActionPruner) or not dfa_pruner.supports_minimal_relaxation:
                raise ValueError("Relaxation policy requires compiled atomic Search-DFA rules")
            evaluate_minimal_constraint_relaxation(
                [], dfa_pruner.clusters, max_relaxed, protected_ids
            )

        def names_for_actions(action_seq, observed_length):
            if act_encoder is None:
                return []
            raw_names = [
                act_encoder.inverse_transform([x - 1])[0] if x > 0 else None
                for x in action_seq[:observed_length]
            ]
            return [normalize_activity_name(n) if n else None for n in raw_names]

        def evaluate_full_dfa(action_seq, observed_length):
            nonlocal full_checked, full_rejected
            relaxation = None
            names = names_for_actions(action_seq, observed_length)
            if act_encoder is None:
                return True, names, relaxation
            if dfa_pruner is not None:
                full_checked += 1
                if minimal_relaxation_enabled:
                    relaxation = dfa_pruner.minimum_relaxation(names)
                    ok = bool(relaxation and relaxation["strict"])
                else:
                    ok = dfa_pruner.check_full_trace(names)
            elif pattern_dfa is not None:
                full_checked += 1
                ok = self._dfa_filter(names, pattern_dfa)
            else:
                ok = True
            if not ok:
                full_rejected += 1
            return ok, names, relaxation

        def passes_full_dfa(action_seq, observed_length):
            ok, names, _ = evaluate_full_dfa(
                action_seq, observed_length
            )
            return ok, names

        with torch.no_grad():
            full_output = self.backbone(
                activities, times, mask,
                pattern_features=pattern_features,
                static_cat=static_cat,
                static_num=static_num,
                dynamic_cat=dynamic_cat,
                dynamic_num=dynamic_num,
            )
            hidden_states = full_output["hidden"]  # [B, T, H]
            last_h = full_output["last_hidden"]

            B = last_h.size(0)
            all_traces = []

            for i in range(B):
                raw_len = mask[i].sum().int().item()
                if raw_len < 2:
                    all_traces.append([])
                    continue
                observed_len = raw_len - 1

                orig_acts = activities[i, :raw_len].tolist()
                orig_times = times[i, :raw_len].tolist()
                hi = last_h[i]

                # 收集所有干预位置的候选
                all_candidates = []
                position_options = {}
                candidate_relaxations = {}

                for pos in intervention_positions:
                    if pos < 0:
                        abs_pos = observed_len + pos
                    else:
                        abs_pos = pos
                    if abs_pos < 0 or abs_pos >= observed_len:
                        continue

                    if abs_pos > 0:
                        pos_hidden = hidden_states[i, abs_pos - 1, :]
                    else:
                        continue

                    # -------------------------------------------------
                    # DFA 可达性剪枝：生成每个位置的候选动作
                    # -------------------------------------------------
                    action_range = [
                        aid for aid in range(1, self.cfg.num_activities)
                        if aid != orig_acts[abs_pos]
                    ]

                    if dfa_pruner is not None and act_encoder is not None:
                        # 通过原始编码器获取活动名, 再消毒 (空格→_→小写→数字前缀)
                        raw_names = [
                            act_encoder.inverse_transform([x - 1])[0] if x > 0 else None
                            for x in orig_acts[:abs_pos]
                        ]
                        prefix_names = [
                            normalize_activity_name(n) if n else None
                            for n in raw_names
                        ]

                        if isinstance(dfa_pruner, MultiDFAActionPruner):
                            prefix_states = dfa_pruner.trace_all_prefixes(prefix_names)
                            all_dead = all(dead for _, dead in prefix_states)
                            n_before = len(action_range)
                            strict_candidates = (
                                [] if all_dead else dfa_pruner.get_union_candidates(
                                    prefix_states, action_range
                                )
                            )
                            if minimal_relaxation_enabled:
                                dfa_candidates = sorted(action_range)
                            elif all_dead:
                                continue
                            else:
                                dfa_candidates = strict_candidates
                            if not hasattr(self, '_dfa_action_prune_log'):
                                self._dfa_action_prune_log = []
                            self._dfa_action_prune_log.append((n_before, len(strict_candidates)))
                            # 仅打印首个位置的动作剪枝诊断。
                            if len(self._dfa_action_prune_log) == 1:
                                print(f"  [DFA-PRUNE] MultiDFA: checked {n_before} actions -> "
                                      f"{len(strict_candidates)} strict-reachable "
                                      f"({1.0-len(strict_candidates)/max(n_before, 1):.1%} strict-pruned)"
                                      f"{dfa_pruner.prune_summary()}")
                        else:
                            # 单 DFA: 原始路径
                            q_prefix, dead_end = dfa_pruner.trace_prefix_state(prefix_names)
                            if dead_end:
                                continue
                            n_before = len(action_range)
                            dfa_candidates = dfa_pruner.get_valid_candidates(
                                q_prefix, action_range
                            )
                            if not hasattr(self, '_dfa_action_prune_log'):
                                self._dfa_action_prune_log = []
                            self._dfa_action_prune_log.append((n_before, len(dfa_candidates)))
                            if len(self._dfa_action_prune_log) == 1:
                                print(f"  [DFA-PRUNE] SingleDFA: checked {n_before} actions -> "
                                      f"{len(dfa_candidates)} passed "
                                      f"({1.0-len(dfa_candidates)/n_before:.1%} pruned)"
                                      f"{dfa_pruner.prune_summary()}")
                    else:
                        # 回退: 无 DFA → 全量枚举
                        dfa_candidates = action_range
                        if not hasattr(self, '_fallback_warned'):
                            self._fallback_warned = True
                            print(f"  [DFA-PRUNE] Fallback: brute force ({len(action_range)} actions, no pruning)")

                    if len(dfa_candidates) == 0:
                        continue

                    # 批量计算 CF scores
                    prechecked_full_names = {}
                    prechecked_relaxations = {}
                    if dfa_pruner is not None and act_encoder is not None:
                        n_before_full = len(dfa_candidates)
                        full_valid_candidates = []
                        for a in dfa_candidates:
                            cf_acts = orig_acts[:abs_pos] + [a] + orig_acts[abs_pos + 1:]
                            ok, names, relaxation = evaluate_full_dfa(
                                cf_acts, observed_len
                            )
                            if ok:
                                full_valid_candidates.append(a)
                            if ok or minimal_relaxation_enabled:
                                prechecked_full_names[a] = names
                                prechecked_relaxations[a] = relaxation
                        if not hasattr(self, '_dfa_prefilter_log'):
                            self._dfa_prefilter_log = []
                        self._dfa_prefilter_log.append((n_before_full, len(full_valid_candidates)))
                        if not minimal_relaxation_enabled:
                            dfa_candidates = full_valid_candidates

                    if len(dfa_candidates) == 0:
                        continue

                    candidate_tensor = torch.tensor(dfa_candidates, device=hi.device, dtype=torch.long)
                    probs, times_pred, scores = self._compute_cf_scores(pos_hidden, candidate_tensor)

                    for j, a in enumerate(dfa_candidates):

                        cf_acts = orig_acts[:abs_pos] + [a] + orig_acts[abs_pos + 1:]
                        cf_times = orig_times[:abs_pos] + [times_pred[j].item()] + orig_times[abs_pos + 1:]

                        score = scores[j].item()

                        # --- DFA 完整验证 ---
                        names = prechecked_full_names.get(a)
                        if names is None:
                            ok, names = passes_full_dfa(cf_acts, observed_len)
                            if not ok:
                                continue
                        if minimal_relaxation_enabled:
                            candidate_relaxations[tuple(cf_acts)] = (
                                prechecked_relaxations.get(a)
                            )

                        if act_encoder is not None and pattern_dict is not None:
                            score += self._pattern_soft_penalty(
                                names, cf_times[:observed_len], pattern_dict
                            )

                        all_candidates.append((cf_acts, cf_times, score))
                        position_options.setdefault(abs_pos, []).append(
                            (a, times_pred[j].item(), score)
                        )

                # 去重并按分数排序
                combo_count = 0
                combo_budget = None
                combination_space = 0
                truncated_option_count = 0
                relaxation_diagnostic = None
                if len(position_options) > 1:
                    option_limit = max(
                        1,
                        int(getattr(self.cfg, "cf_per_position_limit", 5)),
                    )
                    ordered_positions = sorted(position_options)
                    def option_sort_key(position, option):
                        if minimal_relaxation_enabled:
                            option_actions = list(orig_acts)
                            option_actions[position] = option[0]
                            info = candidate_relaxations.get(tuple(option_actions))
                            if info is not None:
                                return (*info["selection_key"][:2], -option[2], option[0])
                        return (float("inf"), -option[2], option[0])

                    trimmed_options = {
                        p: sorted(opts, key=lambda x, p=p: option_sort_key(p, x))[:option_limit]
                        for p, opts in position_options.items()
                    }
                    truncated_option_count = sum(
                        len(position_options[p]) - len(trimmed_options[p])
                        for p in ordered_positions
                    )
                    max_edit_size = min(
                        len(ordered_positions),
                        max(1, int(getattr(self.cfg, "cf_intervention_points", 1))),
                    )
                    combo_budget = max(
                        1,
                        int(getattr(
                            self.cfg,
                            "cf_combination_budget",
                            max(topk * 20, 20),
                        )),
                    )
                    combo_count = 0
                    for size in range(2, max_edit_size + 1):
                        for positions in combinations(ordered_positions, size):
                            count = 1
                            for p in positions:
                                count *= len(trimmed_options[p])
                            combination_space += count

                    for edit_size in range(2, max_edit_size + 1):
                        for pos_tuple in combinations(ordered_positions, edit_size):
                            option_lists = [trimmed_options[p] for p in pos_tuple]
                            for edits in product(*option_lists):
                                cf_acts = list(orig_acts)
                                cf_times = list(orig_times)
                                edit_scores = []
                                for p, (a, t_pred, s) in zip(pos_tuple, edits):
                                    cf_acts[p] = a
                                    cf_times[p] = t_pred
                                    edit_scores.append(s)

                                # The configured budget limits combinations
                                # checked, including those rejected by the DFA.
                                combo_count += 1
                                ok, names, relaxation = evaluate_full_dfa(
                                    cf_acts, observed_len
                                )
                                if ok or minimal_relaxation_enabled:
                                    score = float(
                                        sum(edit_scores) / max(len(edit_scores), 1)
                                    )
                                    score -= (
                                        self.cfg.cf_lambda_action
                                        * 0.01 * (edit_size - 1)
                                    )
                                    all_candidates.append(
                                        (cf_acts, cf_times, score)
                                    )
                                    if minimal_relaxation_enabled:
                                        candidate_relaxations[tuple(cf_acts)] = (
                                            relaxation
                                        )
                                if combo_count >= combo_budget:
                                    break
                            if combo_count >= combo_budget:
                                break
                        if combo_count >= combo_budget:
                            break

                seen = set()
                unique_candidates = []
                for act_seq, time_seq, score in all_candidates:
                    key = tuple(act_seq)
                    if key not in seen:
                        seen.add(key)
                        unique_candidates.append((act_seq, time_seq, score))

                if protected_ids or max_relaxed is not None:
                    allowed = []
                    for candidate in unique_candidates:
                        info = evaluate_minimal_constraint_relaxation(
                            names_for_actions(candidate[0], observed_len),
                            dfa_pruner.clusters, max_relaxed, protected_ids,
                        )
                        if info is not None:
                            allowed.append(candidate)
                            candidate_relaxations[tuple(candidate[0])] = info
                    unique_candidates = allowed

                if minimal_relaxation_enabled:
                    expanded_pool_count = len(unique_candidates)
                    strict_candidates = [
                        candidate
                        for candidate in unique_candidates
                        if (
                            candidate_relaxations.get(tuple(candidate[0]))
                            and candidate_relaxations[tuple(candidate[0])]["strict"]
                        )
                    ]
                    selected_info = None
                    if strict_candidates:
                        unique_candidates = strict_candidates
                        search_mode = "strict"
                        selected_info = candidate_relaxations[
                            tuple(strict_candidates[0][0])
                        ]
                    else:
                        valid_candidates = [
                            candidate
                            for candidate in unique_candidates
                            if candidate_relaxations.get(tuple(candidate[0]))
                        ]
                        if valid_candidates:
                            best_key = min(
                                candidate_relaxations[tuple(candidate[0])][
                                    "selection_key"
                                ][:2]
                                for candidate in valid_candidates
                            )
                            unique_candidates = [
                                candidate for candidate in valid_candidates
                                if candidate_relaxations[tuple(candidate[0])][
                                    "selection_key"
                                ][:2] == best_key
                            ]
                            selected_info = candidate_relaxations[
                                tuple(unique_candidates[0][0])
                            ]
                            search_mode = "minimal_relaxation"
                        else:
                            unique_candidates = []
                            search_mode = "no_candidate"
                    relaxation_diagnostic = {
                        "search_mode": search_mode,
                        "expanded_pool_count": expanded_pool_count,
                        "strict_pool_count": len(strict_candidates),
                        "selected_pool_count": len(unique_candidates),
                        "combination_count": int(combo_count),
                        "combination_space": combination_space,
                        "truncated_option_count": truncated_option_count,
                        "combination_budget_exhausted": bool(
                            combo_count < combination_space
                        ),
                        "combination_budget_reached": bool(
                            combo_budget is not None and combo_count >= combo_budget
                        ),
                        **(selected_info or {}),
                    }

                if relaxation_diagnostic is None:
                    relaxation_diagnostic = {
                        "search_mode": "strict" if dfa_pruner is not None else "no_dfa",
                        "combination_count": combo_count,
                        "combination_space": combination_space,
                        "truncated_option_count": truncated_option_count,
                        "combination_budget_exhausted": combo_count < combination_space,
                        "combination_budget_reached": bool(
                            combo_budget is not None and combo_count >= combo_budget
                        ),
                    }

                unique_candidates = self._rerank_with_backbone(
                    i, orig_acts, orig_times, unique_candidates, observed_len,
                    pattern_features, static_cat, static_num,
                    dynamic_cat, dynamic_num,
                )
                # Backbone reranking replaces the generator score.  Reapply the
                # calibrated soft reward once so it remains effective.
                if (
                    bool(getattr(self.cfg, "cf_backbone_rerank", True))
                    and act_encoder is not None
                    and pattern_dict
                ):
                    rescored = []
                    for act_seq, time_seq, score in unique_candidates:
                        names = names_for_actions(act_seq, observed_len)
                        score += self.soft_constraint_score(
                            names, time_seq[:observed_len], pattern_dict
                        )
                        rescored.append((act_seq, time_seq, score))
                    unique_candidates = rescored

                ranked_candidates = self._rank_counterfactuals(
                    orig_acts, orig_times, unique_candidates, topk
                )
                all_traces.append(ranked_candidates)
                if relaxation_diagnostic is not None:
                    relaxation_diagnostic = dict(relaxation_diagnostic)
                    relaxation_diagnostic["returned_count"] = len(
                        ranked_candidates
                    )
                    if not hasattr(self, "_constraint_relaxation_log"):
                        self._constraint_relaxation_log = []
                    self._constraint_relaxation_log.append(relaxation_diagnostic)

        # 打印剪枝统计 (从本轮生成中直接计算)
        if dfa_pruner is not None and hasattr(self, '_dfa_action_prune_log'):
            total_all = sum(x[0] for x in self._dfa_action_prune_log)
            passed_all = sum(x[1] for x in self._dfa_action_prune_log)
            if total_all > 0:
                rate = 1.0 - passed_all / total_all
                print(f"  DFA candidate filtering: checked {total_all} actions -> {passed_all} passed "
                      f"({rate:.1%} pruned)"
                      f"{dfa_pruner.prune_summary() if hasattr(dfa_pruner, 'prune_summary') else ''}")
        elif dfa_pruner is None:
            print(f"  [DFA ctx is None — pruning not active]")

        if full_checked > 0:
            if not hasattr(self, "_dfa_full_log"):
                self._dfa_full_log = []
            self._dfa_full_log.append((full_checked, full_rejected))

        return all_traces

    def cf_forward(self, cf_action, base_outputs=None,
                   activities=None, times=None, mask=None):
        """
        反事实前向传播：给定替换动作，预测替换后的结果和时间。
        """
        if base_outputs is not None:
            base_h = base_outputs.get("prev_hidden", base_outputs["last_hidden"])
        else:
            base_outputs = self.backbone(activities, times, mask)
            base_h = base_outputs.get("prev_hidden", base_outputs["last_hidden"])

        a_emb = self.action_embed(cf_action)
        cf_h = torch.cat([base_h, a_emb], dim=-1)

        cf_outcome_logits = self.cf_outcome_head(cf_h)
        cf_time_pred = self.cf_time_head(cf_h).squeeze(-1)

        return {
            "cf_outcome_logits": cf_outcome_logits,
            "cf_time_pred": cf_time_pred,
        }
