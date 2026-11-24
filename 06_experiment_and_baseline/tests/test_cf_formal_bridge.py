"""Synthetic-only bridge tests. Never restore a real predictor/query dataset."""
import copy
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from pythomata import SimpleDFA
from audit_constraints import rule_formula

from config import Config
from models import MultiTaskProcessModel
from counterfactual import CounterfactualRecommender
from cf_search import SearchBudget, SearchPolicy
from cf_search.proprietary import FrozenCFHeads
from cf_search.suite import METHODS, GROUPS, SCOPES, seal, checked_json
from cf_search.formal_bridge import generate_input, call_algorithm, physical_generation, restore_policy, execution_id
from evaluation.cf_types import QueryContext, SearchResult, OracleBudget
from evaluation.outcome_oracle import OutcomeOracle
from evaluation.cf_bridge_protocol import validate, authorize, selected_cohort, preflight, PURPOSE, SAFETY
from evaluation.cf_bridge_results import validate_saved, save_generation, load_generation, audit, summarize, export_statistical_input
from evaluation.cf_paired_statistics import indicators
from evaluation.cf_formal_protocol import DATASETS
from evaluation.cf_experiment_results import write_json, write_jsonl, read_jsonl
from evaluation.experiment_protocol import artifact, stable_hash, canonical_dfa_snapshot
from tools.verify_p6b_bridge import independent_search, independent_audit_checks, independent_selection, independent_costs, independent_indicators, independent_intervention, verify_saved
from run_cf_formal_bridge import search, main
from run_cf_protocol import load_formal_saved


def config():
    return json.loads(Path("configs/cf_p6b_bridge.json").read_text(encoding="utf-8"))


class SyntheticFixture:
    def __init__(self, root, count=1, datasets=("synthetic_only",)):
        self.root = Path(root)
        torch.set_num_threads(1)
        torch.manual_seed(31)
        cfg = Config(num_activities=4, hidden_dim=8, num_layers=1, activity_emb_dim=4, time_emb_dim=4,
                     max_seq_len=5, dropout=0., n_static_cat=0, n_static_num=0, n_dynamic_cat=0, n_dynamic_num=0)
        model = MultiTaskProcessModel(cfg).eval().requires_grad_(False)
        legacy = CounterfactualRecommender(cfg, model).eval()
        self.checkpoint = dict(cfg=cfg, cf_model_state=copy.deepcopy(legacy.state_dict()))
        base = OutcomeOracle(model, {"Failure": 0, "Success": 1}, "synthetic_unit_test_only", budget=OracleBudget(500))
        dfa = SimpleDFA({0}, {"a", "b", "c"}, 0, {0}, {0: {"a": 0, "b": 0, "c": 0}})
        rule = dict(rule_id="r", formula="G(a | b | c)", compile_mode="strict", relaxation_weight=1.)
        policy = SearchPolicy([dict(cluster_id="c", rule_dfas=[dict(rule, dfa=dfa)])], {1: "a", 2: "b", 3: "c"})
        self.item = dict(base=base, heads={m: FrozenCFHeads(cfg, legacy.state_dict(), m == "O2") for m in ("O1", "O2")},
            policy=policy, combined=None, reference=dict(rows=[dict(case_id="train_a", activities=[1, 2, 2]),
                dict(case_id="train_b", activities=[1, 3, 3])]), activity_ids=(1, 2, 3), budget=SearchBudget(500, 10000, 60),
            model_identity="synthetic_unit_test_only", preparation=dict(operation="synthetic_only", walltime_seconds=0., model_evaluations=0))
        self.originals, self.cohort, self.sources = {}, [], []
        for dataset in datasets:
            original_path = self.root/(dataset+"_source.json")
            original = dict(dataset=dataset, model=dict(activity_vocabulary=["a", "b", "c"], checkpoint=dict(sha256="synthetic_unit_test_only")),
                search=dict(rules=[rule], clusters=[dict(cluster_id="c", rule_ids=["r"], formula=rule["formula"])],
                            protected_rule_ids=[], omitted_rule_ids=[]))
            write_json(original_path, original)
            rules = [dict(rule_id="audit_r", template="precedence", activation="a", target="b", formula=rule_formula("precedence", "a", "b"))]
            policy_path = self.root/(dataset+"_audit_policy.json")
            write_json(policy_path, dict(rules=rules, rule_set_sha256=stable_hash(rules)))
            reference_path = self.root/(dataset+"_reference.json")
            write_json(reference_path, seal(dict(rows=self.item["reference"]["rows"])))
            source = dict(dataset=dataset, protocol=artifact(original_path), budget=dict(model_evaluations=500, candidate_attempts=10000, walltime_seconds=60),
                search=original["search"], atomic_snapshots={"r": canonical_dfa_snapshot(dfa)},
                reference=dict(allowed_case_ids=["train_a", "train_b"], historical_build_walltime_seconds=.01, artifact=artifact(reference_path)),
                audit_policy=artifact(policy_path), model=artifact(original_path),
                cf_head_inventory={m: dict(available=True) for m in ("O1", "O2")})
            self.originals[str(original_path)] = original
            self.sources.append(source)
            for offset in range(count):
                query = QueryContext(dataset, "SYNTHETIC_ONLY/case/"+str(offset), offset, (1, 2, 2), (0., .2, .4), (2, 1), 3)
                probability = float(base.predict_proba([query.activities], query)[0, 1])
                self.cohort.append(dict(dataset=dataset, offset=offset, input_sha256=stable_hash(asdict(query)), context=json.loads(json.dumps(asdict(query))),
                    original_probability=probability, original_class="Success" if probability >= .5 else "Failure",
                    common_buliga_applicable=True, factual_search_strict=True))
        self.protocol = dict(sources=self.sources, cohort=self.cohort)
        self.protocol_path = self.root/"p6b_protocol.json"
        write_json(self.protocol_path, seal(self.protocol))
        self.preflight_dir = self.root/"preflight"
        self.preflight_dir.mkdir()
        self.bridge = dict(purpose=PURPOSE, protocol=artifact(self.protocol_path), prerequisite_preparation_costs={}, code=[])
        write_json(self.preflight_dir/"bridge_preflight.json", seal(self.bridge))

    def loader(self, path):
        return self.originals[str(Path(path))]

    def records(self, actual=True):
        run = self.root/"run"
        run.mkdir()
        queries, candidates, resources, units = [], [], [], []
        def empty_algorithm(query, oracle, item, method):
            return SearchResult([], "completed", dict(oracle=oracle.statistics(), scored_pool=[], scored_pool_size=0,
                scored_pool_sha256=stable_hash([]), context_encodings=0, cf_outcome_evaluations=0, cf_time_evaluations=0,
                total_backbone_evaluations=0, time_policy="normalized_elapsed_coordinate_neighbor_projection_v1" if method == "O2" else "fixed",
                original_probability=self.cohort[0]["original_probability"], committed_population=[], committed_population_fully_scored=True,
                usable_external_baseline=True, projected_reference=[dict(case_id="train_a", activities=[1, 2, 2])],
                projected_reference_sha256=stable_hash([dict(case_id="train_a", activities=[1, 2, 2])])))
        patches = [] if actual else [patch("cf_search.formal_bridge.call_algorithm", side_effect=empty_algorithm)]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            for frozen in self.cohort:
                qs, cs, physical = generate_input(frozen, self.item, run)
                queries.extend(qs)
                candidates.extend(cs)
                resources.extend(physical)
                path = run/(frozen["input_sha256"]+".json")
                write_json(path, seal(dict(input_sha256=frozen["input_sha256"], queries=qs, candidates=cs, physical_artifacts=physical)))
                units.append(artifact(path))
        return run, queries, candidates, resources, units

    def bind_generation(self, run, qs, cs, resources, units, scope="fixed_six_query_smoke"):
        receipt_path = self.root/"authorization.json"
        receipt = dict(purpose="P6_B_explicit_generation_authorization_v1", authorized=True, scope=scope,
            bridge_preflight=artifact(self.preflight_dir/"bridge_preflight.json"), run_dir=str(run), user_instruction="SYNTHETIC UNIT TEST ONLY")
        write_json(receipt_path, receipt)
        bridge = dict(self.bridge, preflight_path=str(self.preflight_dir/"bridge_preflight.json"))
        with patch("evaluation.cf_bridge_results.load_protocol", side_effect=self.loader):
            save_generation(run, bridge, self.protocol, self.cohort, qs, cs, resources, units,
                dict(receipt, artifact=artifact(receipt_path)), [])


class BridgeConfigTests(unittest.TestCase):
    def test_read_only_live_preflight_never_restores_or_forwards_predictor(self):
        with tempfile.TemporaryDirectory() as d, \
                patch("evaluation.freeze_protocol.load_frozen_oracle", side_effect=AssertionError("No real predictor restore")), \
                patch("torch.load", side_effect=AssertionError("No real checkpoint deserialization")), \
                patch("cf_search.formal_bridge.call_algorithm", side_effect=AssertionError("No real generation")), \
                patch("evaluation.cf_bridge_results.compile_audit_rules", side_effect=AssertionError("No real Audit compile")):
            report = preflight(config(), d)
            self.assertEqual(report["query_method_count"], 3850)
            self.assertEqual(report["model_evaluations"], 0)
            self.assertEqual(report["generated_candidate_count"], 0)
            self.assertFalse(report["execution_authorized"])

    def test_preflight_rejects_nonempty_before_resource_restore(self):
        with tempfile.TemporaryDirectory() as d, patch("evaluation.cf_bridge_protocol.load_inputs", side_effect=AssertionError("No restore")):
            write_json(Path(d)/"user_file.json", dict(preserve=True))
            with self.assertRaises(ValueError): preflight(config(), d)
            self.assertTrue((Path(d)/"user_file.json").exists())

    def test_default_config_locked(self):
        self.assertTrue(validate(config()))

    def test_all_execution_or_expansion_flags_rejected(self):
        for field in SAFETY:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(dict(config(), **{field: True}))

    def test_integer_zero_not_false(self):
        with self.assertRaises(ValueError): validate(dict(config(), candidate_generation_allowed=0))

    def test_unknown_audit_budget_seed_or_subset_rejected(self):
        for field in ("audit_grid", "budget", "search_seed", "query_offsets", "methods"):
            with self.subTest(field=field), self.assertRaises(ValueError): validate(dict(config(), **{field: []}))

    def test_other_prerequisite_root_rejected(self):
        with self.assertRaises(ValueError): validate(dict(config(), prerequisite_dir="outputs/p6b_prerequisites_20260918_v1"))

    def test_search_default_fails_before_any_model_or_output(self):
        with tempfile.TemporaryDirectory() as d, patch("cf_search.formal_bridge.runtime", side_effect=AssertionError("No model")):
            run = Path(d)/"run"
            with self.assertRaises(PermissionError): search(Path(d)/"missing", run)
            self.assertFalse(run.exists())

    def test_authorization_requires_literal_true_and_exact_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_json(root/"bridge_preflight.json", dict(fake="synthetic"))
            base = dict(purpose="P6_B_explicit_generation_authorization_v1", authorized=True, scope="seven_task_matrix",
                bridge_preflight=artifact(root/"bridge_preflight.json"), run_dir=str(root/"run"), user_instruction="synthetic only")
            for index, change in enumerate((dict(authorized=1), dict(scope="pick_best_queries"), dict(run_dir=str(root/"other")),
                                           dict(bridge_preflight=dict(path="wrong", sha256="wrong")), dict(user_instruction=""), dict(extra=True))):
                path = root/(str(index)+".json")
                write_json(path, dict(base, **change))
                with self.subTest(change=change), self.assertRaises(PermissionError): authorize(root, root/"run", path)
            write_json(root/"good.json", base)
            self.assertEqual(authorize(root, root/"run", root/"good.json")[0], base)

    def test_scope_full_and_predeclared_six_only(self):
        rows = [dict(dataset=ds, offset=i) for ds in DATASETS for i in range(50)]
        p = dict(cohort=rows)
        self.assertEqual(len(selected_cohort(p, "seven_task_matrix")), 350)
        self.assertEqual(len(selected_cohort(p, "fixed_six_query_smoke")), 6)
        with self.assertRaises(ValueError): selected_cohort(p, "failure_only")

    def test_cli_validate_never_searches(self):
        with patch("run_cf_formal_bridge.search", side_effect=AssertionError("No generation")):
            main(["validate"])

    def test_cli_no_smoke_or_training_shortcut(self):
        for mode in ("smoke", "train", "complete", "recover"):
            with self.subTest(mode=mode), self.assertRaises(SystemExit): main([mode])

    def test_generator_signature_has_no_audit_or_label(self):
        for f in (call_algorithm, generate_input, physical_generation):
            self.assertFalse(any("audit" in name or "label" in name for name in inspect.signature(f).parameters))

    def test_authorized_synthetic_failure_is_persisted_not_filled(self):
        with tempfile.TemporaryDirectory() as d:
            f = SyntheticFixture(d)
            run = Path(d)/"new_run"
            receipt = Path(d)/"synthetic_authorization.json"
            write_json(receipt, dict(purpose="P6_B_explicit_generation_authorization_v1", authorized=True,
                scope="seven_task_matrix", bridge_preflight=artifact(f.preflight_dir/"bridge_preflight.json"),
                run_dir=str(run), user_instruction="SYNTHETIC TEST ONLY"))
            with patch("run_cf_formal_bridge.guard", return_value=(f.bridge, f.protocol)), \
                    patch("cf_search.formal_bridge.runtime", side_effect=FileNotFoundError("synthetic missing artifact")):
                with self.assertRaises(FileNotFoundError): search(f.preflight_dir, run, receipt)
            failure = checked_json(run/"generation_failure.json")
            self.assertEqual(failure["completed_query_method_rows"], 0)
            self.assertEqual(failure["audit_compilation_count"], 0)
            self.assertFalse((run/"generation_manifest.json").exists())
            self.assertFalse((run/"candidates.jsonl").exists())
            self.assertFalse((run/"audit_manifest.json").exists())


class BridgeSyntheticExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.fixture = SyntheticFixture(cls.temp.name)
        cls.run_root, cls.qs, cls.cs, cls.resources, cls.units = cls.fixture.records(actual=True)
        cls.physical = [checked_json(p["path"]) for p in cls.resources]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def validate(self, qs=None, cs=None, physical=None):
        with patch("evaluation.cf_bridge_results.load_protocol", side_effect=self.fixture.loader):
            return validate_saved(self.fixture.protocol, self.fixture.cohort, self.qs if qs is None else qs,
                self.cs if cs is None else cs, self.physical if physical is None else physical)

    def test_actual_nine_routes_eleven_records_and_shared_pool(self):
        self.assertEqual(len(self.qs), 11)
        self.assertEqual(len(self.physical), 9)
        self.assertTrue(self.validate())
        r1, r2 = [q for q in self.qs if q["method"] in ("R1_expanded_cap0", "R2_expanded_cap1")]
        self.assertEqual(r1["execution_id"], r2["execution_id"])
        self.assertEqual(r1["statistics"]["scored_pool_sha256"], r2["statistics"]["scored_pool_sha256"])
        self.assertTrue(all(p["saved_before_policy_selection"] for p in self.physical if p["family"] == "relaxation"))

    def test_observed_calls_and_no_double_count(self):
        counts = independent_costs(self.fixture.cohort, self.qs, self.physical)
        self.assertEqual(counts["physical_generation_count"], 9)
        self.assertGreater(counts["outcome_sequences"], 0)
        self.assertGreater(counts["cf_outcome_evaluations"], 0)
        self.assertGreater(counts["cf_time_evaluations"], 0)

    def test_missing_or_duplicate_query_is_not_no_candidate(self):
        for qs in (self.qs[:-1], self.qs+[self.qs[0]]):
            with self.assertRaises(ValueError): self.validate(qs=qs)

    def test_drift_group_or_original_prediction_rejected(self):
        for change in (dict(original_probability=.99), dict(comparison_groups=["time_intervention_auxiliary"]),
                       dict(common_buliga_applicable=False), dict(time_policy="anything")):
            qs = copy.deepcopy(self.qs)
            qs[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): self.validate(qs=qs)

    def test_physical_profile_tampering_rejected(self):
        ps = copy.deepcopy(self.physical)
        ps[0]["costs"]["outcome_sequences"] += 1
        with self.assertRaises(ValueError): self.validate(physical=ps)

    def test_physical_audit_input_or_unpersisted_relaxation_rejected(self):
        for index, field, value in ((0, "audit_input_present", True), (3, "saved_before_policy_selection", False)):
            ps = copy.deepcopy(self.physical)
            ps[index][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.validate(physical=ps)

    def test_candidates_cross_method_or_edit_violation_rejected(self):
        self.assertTrue(self.cs)
        for change in (dict(method="O2"), dict(edit_count=99), dict(times=[0., .99, .99])):
            cs = copy.deepcopy(self.cs)
            cs[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): self.validate(cs=cs)

    def test_unavailable_created_never_becomes_measured_zero(self):
        qs = copy.deepcopy(self.qs)
        qs[-1]["status"] = "completed"
        with self.assertRaises(ValueError): self.validate(qs=qs)

    def test_recorded_terminal_status_cannot_be_rewritten(self):
        qs = copy.deepcopy(self.qs)
        qs[0]["status"] = "attempt_budget_exhausted" if qs[0]["status"] != "attempt_budget_exhausted" else "completed"
        with self.assertRaises(ValueError): self.validate(qs=qs)

    def test_duplicate_or_orphan_physical_execution_rejected(self):
        with self.assertRaises(ValueError): self.validate(physical=self.physical+[self.physical[0]])

    def test_old_unconstrained_postfilter_pools_still_match(self):
        pairs = [p["pool_sha256"] for p in self.physical if p["method"] in ("enum_outcome", "postfilter_outcome")]
        self.assertEqual(len(set(pairs)), 1)

    def test_raw_algorithm_return_tampering_rejected(self):
        ps = copy.deepcopy(self.physical)
        ps[0]["returned"] = []
        with self.assertRaises(ValueError): self.validate(physical=ps)

    def test_budget_or_forbidden_head_calls_fail_acceptance(self):
        for change in (dict(model_evaluations=501), dict(model_evaluations=-1)):
            qs = copy.deepcopy(self.qs)
            qs[0]["statistics"]["oracle"].update(change)
            with self.assertRaises(ValueError): independent_costs(self.fixture.cohort, qs, self.physical)

    def test_independent_full_pool_selection_all_families(self):
        source = self.fixture.sources[0]
        for q in self.qs[:-1]:
            p = next(p for p in self.physical if p["execution_id"] == q["execution_id"])
            cs = sorted((c for c in self.cs if c["query_id"] == q["query_id"]), key=lambda c: c["rank"])
            independent_selection(q, p, source, ["a", "b", "c"], cs)


class BridgeReadOnlyAndAuditTests(unittest.TestCase):
    def test_independent_strict_nonimproving_layer_still_blocks_relaxation(self):
        graph = dict(initial_state=0, accepting_states=[1], transitions=[[["a", 1], ["b", 0]], [["a", 1], ["b", 1]]])
        source = dict(atomic_snapshots={"r": graph}, search=dict(rules=[dict(rule_id="r", relaxation_weight=1.)],
            clusters=[dict(cluster_id="c", rule_ids=["r"])], protected_rule_ids=[]))
        strict = dict(activities=[1], edit_count=1, probability=.2, search=dict(strict=True, cluster_id="c", relaxation_count=0,
            relaxation_cost=0., relaxed_constraint_ids=[]))
        violated = dict(activities=[2], edit_count=1, probability=.8, search=dict(strict=False, cluster_id="c", relaxation_count=1,
            relaxation_cost=1., relaxed_constraint_ids=["r"]))
        pool = [strict, violated]
        q = dict(method="R2_expanded_cap1", family="relaxation", selection=dict(selected_layer=[0, 0.], strict_pool_size=1,
            pool_sha256=stable_hash(pool), pool_size=2, globally_protected_rejected=0, minimum_layer_size=1))
        physical = dict(rows=pool, pool_sha256=stable_hash(pool))
        independent_selection(q, physical, source, ["a", "b"], [strict])
        with self.assertRaises(ValueError): independent_selection(q, physical, source, ["a", "b"], [violated])

    def test_independent_o2_projection_without_production_projector(self):
        context = dict(activities=[1, 2, 2, 3], times=[0., .2, .4, .6], allowed_positions=[1, 2], max_changed_positions=3)
        row = dict(activities=[1, 3, 3, 3], times=[0., .6, .6, .6], edit_positions=[1, 2], edit_count=2, cf_time_raw={"1": .9, "2": .1})
        independent_intervention(context, row, "O2", ["a", "b", "c"])
        for change in (dict(times=[0., .5, .6, .6]), dict(cf_time_raw={"1": .9}), dict(edit_count=1)):
            with self.subTest(change=change), self.assertRaises(ValueError): independent_intervention(context, dict(row, **change), "O2", ["a", "b", "c"])

    def test_independent_unknown_replacement_forbidden(self):
        context = dict(activities=[1, 2], times=[0., .2], allowed_positions=[1], max_changed_positions=3)
        row = dict(activities=[1, 3], times=[0., .2], edit_positions=[1], edit_count=1)
        with self.assertRaises(ValueError): independent_intervention(context, row, "enum_outcome", ["a", "b", "__UNK__"])

    def test_actual_toy_save_audit_metrics_independent_acceptance_and_pilot_export_guard(self):
        with tempfile.TemporaryDirectory() as d:
            f = SyntheticFixture(d)
            run, qs, cs, resources, units = f.records(actual=True)
            f.bind_generation(run, qs, cs, resources, units)
            contexts = [QueryContext(**r["context"]) for r in f.cohort]
            with patch("evaluation.cf_bridge_protocol.guard", return_value=(f.bridge, f.protocol)), \
                    patch("evaluation.cf_bridge_protocol.selected_cohort", return_value=f.cohort), \
                    patch("evaluation.cf_bridge_results.load_protocol", side_effect=f.loader), \
                    patch("tools.verify_p6b_bridge.load_protocol", side_effect=f.loader), \
                    patch("evaluation.freeze_protocol.load_frozen_oracle", return_value=(f.item["base"], contexts)), \
                    patch("torch.load", return_value=f.checkpoint):
                audit(run)
                report = summarize(run)
                accepted = verify_saved(run)
                self.assertEqual(accepted["status"], "passed")
                self.assertEqual(report["query_method_count"], 11)
                self.assertGreater(accepted["independently_rechecked_physical_pool_rows"], 0)
                self.assertFalse(accepted["new_candidate_generation"])
                with self.assertRaises(ValueError): export_statistical_input(run)
                with self.assertRaises(ValueError): audit(run)
                with self.assertRaises(ValueError): verify_saved(run)
                self.assertFalse((run/"statistical_input_manifest.json").exists())

    def test_snapshot_restore_no_compiler_or_product_build(self):
        with tempfile.TemporaryDirectory() as d:
            f = SyntheticFixture(d)
            source = dict(f.sources[0], combined_snapshot=dict(initial_state=0, accepting_states=[0], transitions=[[["eventa", 0], ["eventb", 0], ["eventc", 0]]]),
                          combined_build_statistics=dict(walltime_seconds=0.))
            with patch("cf_search.formal_bridge.PolicyDFA.__init__", side_effect=AssertionError("No compile")):
                policy, combined = restore_policy(source, f.originals[str(Path(d)/"synthetic_only_source.json")])
            self.assertTrue(policy.evidence([1, 2, 3])["strict"])
            self.assertEqual(combined.get_successors(1, {"eventb": True}), {1})
            self.assertEqual(combined.snapshot, source["combined_snapshot"])

    def test_independent_audit_activation_and_last_event_response(self):
        rule = dict(rule_id="r", template="response", activation="b", target="c")
        self.assertEqual(independent_audit_checks(["a"], [rule])[0]["satisfied"], None)
        self.assertFalse(independent_audit_checks(["b"], [rule])[0]["satisfied"])
        self.assertTrue(independent_audit_checks(["b", "c"], [rule])[0]["satisfied"])

    def test_independent_precedence_chain_and_not_coexistence(self):
        for template, trace, ok in (("precedence", ["a", "b"], True), ("precedence", ["b", "a"], False),
                                   ("chain_response", ["a", "c", "b"], False), ("not_coexistence", ["a", "b"], False)):
            r = dict(rule_id="r", template=template, activation="a", target="b")
            with self.subTest(template=template): self.assertEqual(independent_audit_checks(trace, [r])[0]["satisfied"], ok)

    def test_unsupported_audit_template_fail_closed(self):
        with self.assertRaises(ValueError): independent_audit_checks(["a"], [dict(rule_id="r", template="invented")])

    def test_incomplete_save_cannot_be_audited(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError): audit(d)
            self.assertFalse((Path(d)/"audit_started.json").exists())

    def test_strict_search_is_complete_finite_acceptance(self):
        source = dict(atomic_snapshots={"r": dict(initial_state=0, accepting_states=[1], transitions=[[["a", 1], ["b", 0]], [["a", 1], ["b", 1]]])},
            search=dict(rules=[dict(rule_id="r", relaxation_weight=1.)], protected_rule_ids=[], clusters=[dict(cluster_id="c", rule_ids=["r"])]))
        self.assertFalse(independent_search(source, ["a", "b"], [2])["strict"])
        self.assertTrue(independent_search(source, ["a", "b"], [2, 1])["strict"])

    def test_independent_cross_cluster_protection(self):
        graph = dict(initial_state=0, accepting_states=[1], transitions=[[["a", 1], ["b", 0]], [["a", 1], ["b", 1]]])
        source = dict(atomic_snapshots={"a": graph, "b": dict(initial_state=0, accepting_states=[0], transitions=[[["a", 0], ["b", 0]]])},
            search=dict(rules=[dict(rule_id=r, relaxation_weight=1.) for r in ("a", "b")], protected_rule_ids=["a"],
                        clusters=[dict(cluster_id="c1", rule_ids=["a"]), dict(cluster_id="c2", rule_ids=["b"])]))
        self.assertFalse(independent_search(source, ["a", "b"], [2])["protected"])
        self.assertFalse(independent_search(source, ["a", "b"], [2])["strict"])


class BridgeStatisticsContractTests(unittest.TestCase):
    def test_joint_requires_one_candidate_and_nonvacuous_audit(self):
        r = dict(dataset="synthetic_only", input_sha256="k", original_probability=.2, original_class="Failure",
                 common_buliga_applicable=True, context=dict(case_id="SYNTHETIC", times=[0., .2]))
        qid = stable_hash(dict(input_sha256="k", method="enum_outcome"))
        q = dict(query_id=qid, input_sha256="k", method="enum_outcome", dataset="synthetic_only", original_probability=.2,
                 original_class="Failure", common_buliga_applicable=True, candidate_count=2, status="completed", error=None)
        cs = [dict(query_id=qid, input_sha256="k", method="enum_outcome", dataset="synthetic_only", candidate_id=str(i), rank=i,
                   probability=p, search=dict(strict=strict, relaxation_count=0)) for i, p, strict in ((1, .8, False), (2, .4, True))]
        audits = {str(i): dict(candidate_id=str(i), query_id=qid, status="audit_pass", applicable_rule_count=1, violated_rule_ids=[],
                    error=None, rule_checks=[dict(rule_id="r", applicable=True, satisfied=True)]) for i in (1, 2)}
        result = indicators([r], [q], cs, audits, methods=["enum_outcome"])[0]
        self.assertEqual(result["improvement_success_at_k"], 1)
        self.assertEqual(result["audit_joint_improvement_success_at_k"], 0)
        cs[0]["search"]["strict"] = True
        audits["1"].update(status="not_applicable", applicable_rule_count=0, rule_checks=[])
        self.assertEqual(indicators([r], [q], cs, audits, methods=["enum_outcome"])[0]["audit_joint_improvement_success_at_k"], 0)
        audits["1"]["status"] = "audit_pass"
        with self.assertRaises(ValueError): indicators([r], [q], cs, audits, methods=["enum_outcome"])

    def test_synthetic_full350_empty_result_contract_and_old_consumer(self):
        # Mechanical 350x11 interface fixture, NOT any real query or search run.
        with tempfile.TemporaryDirectory() as d:
            f = SyntheticFixture(d, count=50, datasets=DATASETS)
            # Create empty evidence directly; invoking real search 350 times is prohibited here.
            run, qs, cs, resources, units = f.records(actual=False)
            f.bind_generation(run, qs, cs, resources, units, scope="seven_task_matrix")
            with patch("evaluation.cf_bridge_protocol.guard", return_value=(f.bridge, f.protocol)), patch("evaluation.cf_bridge_results.load_protocol", side_effect=f.loader):
                loaded = load_generation(run)
            self.assertEqual(len(loaded[4]), 3850)
            self.assertEqual(len(loaded[6]), 3150)
            write_jsonl(run/"audit_results.jsonl", [])
            g = loaded[3]
            write_json(run/"audit_manifest.json", seal(dict(generation_manifest=artifact(run/"generation_manifest.json"),
                candidate_file=g["candidates"], audits=artifact(run/"audit_results.jsonl"), candidate_count=0,
                all_candidates_saved_before_audit=True, audit_used_in_generation=False, audit_used_in_selection=False,
                audit_used_in_tuning=False, model_evaluations=0, training_started=False)))
            saved = dict(queries=g["queries"], candidates=g["candidates"], audits=artifact(run/"audit_results.jsonl"),
                generation_manifest=artifact(run/"generation_manifest.json"), audit_manifest=artifact(run/"audit_manifest.json"))
            metrics_path = run/"metrics.json"
            write_json(metrics_path, seal(dict(synthetic_only=True)))
            write_json(run/"independent_acceptance.json", seal(dict(purpose="P6_formal_generation_independent_acceptance_v1", status="passed",
                scope="seven_task_matrix", protocol=f.bridge["protocol"], saved_files=saved, bridge_code=[], metrics=artifact(metrics_path),
                complete_denominators_and_same_candidate_joint_verified=True, budget_and_intervention_verified=True,
                search_and_audit_evidence_independently_verified=True, verification_code=artifact(__file__))))
            with patch("evaluation.cf_bridge_results.load_generation", return_value=loaded):
                export_statistical_input(run)
                export_statistical_input(run)  # Exact idempotent export, not overwrite.
            cohort, oqs, ocs, audits, _ = load_formal_saved(run, f.protocol, Path(d))
            binary = indicators(cohort, oqs, ocs, audits)
            self.assertEqual(len(binary), 3850)
            self.assertEqual(len(independent_indicators(cohort, oqs, ocs, audits)), 3850)
            self.assertTrue(all(r["unavailable"] for r in binary if r["method"] == "created_unavailable"))


if __name__ == "__main__":
    unittest.main()
