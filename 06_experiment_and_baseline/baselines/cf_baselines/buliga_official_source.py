"""Load licensed adapted author methods without training/import side effects."""
import ast
import copy
import itertools
from pathlib import Path
from types import SimpleNamespace
import timeit
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist,_validate_vector
from scipy.stats import median_abs_deviation
from sklearn.neighbors import KDTree
from cf_baselines.source import SOURCE,BULIGA_COMMIT,load_author_methods,HASHES
from evaluation.experiment_protocol import file_sha256

VENDOR = "cf_baselines/vendor/buliga/dice_ltlf_genetic_protocol.py"
BASE = "local_packages/DiCE/dice_ml/explainer_interfaces/explainer_base.py"
BASE_SHA = "8c568272f10b9b58230ac07cf545b300b983f13beff4771430ebdd8fb4bae33d"
METHODS = ("do_random_init","do_KD_init","do_cf_initializations","compute_plausibility","distance_mh",
    "continuous_distance","categorical_distance","compute_yloss","compute_sparsity_loss","compute_loss",
    "mate_2","find_counterfactuals","_predict_fn_custom")


def load_methods(root=".runtime/p5_sources/buliga"):
    namespace = load_author_methods(root)
    base = Path(root)/BASE
    if file_sha256(base) != BASE_SHA:
        raise ValueError("Pinned official base helper changed")
    namespace.update(copy=copy,timeit=timeit,cdist=cdist,median_abs_deviation=median_abs_deviation,
        _validate_vector=_validate_vector,KDTree=KDTree,np=np,pd=pd,itertools=itertools,
        ModelTypes=SimpleNamespace(Classifier="classifier",Regressor="regressor"))
    tree = ast.parse(Path(VENDOR).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == "DiceGeneticLTLf")
    nodes = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in METHODS]
    if len(nodes) != len(METHODS):
        raise ValueError("Missing official main-flow method")
    nodes += [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "mad_cityblock"]
    base_tree = ast.parse(base.read_text(encoding="utf-8"))
    base_cls = next(n for n in base_tree.body if isinstance(n,ast.ClassDef) and n.name == "ExplainerBase")
    nodes += [n for n in base_cls.body if isinstance(n,ast.FunctionDef) and n.name in
              ("build_KD_tree","is_cf_valid","round_to_precision")]
    # Budget exit must select from last committed population, using original tail.
    main = next(n for n in nodes if n.name == "find_counterfactuals")
    start = next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
                 and ast.unparse(n.targets[0]) == "self.cfs_preds")
    finalizer = ast.parse("def finalize(self, population, query_instance):\n    pass").body[0]
    finalizer.body = copy.deepcopy(main.body[start:])
    finalizer.body.insert(0,ast.parse("verbose = False").body[0])
    finalizer.body.insert(0,ast.parse("iterations = self.committed_iterations").body[0])
    nodes.append(finalizer)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),VENDOR,"exec"),namespace)
    # The combined frozen policy is already a singleton-letter DFA facade.
    namespace.update(parse=lambda x:x,alphabet=lambda x:set(),symbolic_to_singleton=lambda d,letters:d)
    return namespace
