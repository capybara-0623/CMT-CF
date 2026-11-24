"""Hash-pinned, unmodified author methods, without importing training pipelines."""
import ast
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from evaluation.experiment_protocol import file_sha256

BULIGA_COMMIT = "213acbab59bd2952194378ca4ef984d567e67e7f"
SOURCE = "local_packages/DiCE/dice_ml/explainer_interfaces/dice_ltlf_genetic.py"
UTILS = "local_packages/Declare4Py/Declare4Py/Utils/utils.py"
HASHES = {
    SOURCE: "5356c9266b300b2454db019eb1f310c1892abeccc3fcd17832e52b0e93b0fe53",
    UTILS: "720698f8d879ad2bba81652e0f22a03742d7962aae8e71eb9d220bbaa5366d04",
    "LICENSE.md": "d97360644c9d0116ba8d8ada9c7d2a43804bcee9230b9beb8c163cdcb346f087",
    "local_packages/DiCE/LICENSE": "27ebda9d51f0a56b7e281ccd8230a27236dcb51c05f64b07869ecf6e965d68b0",
}
AUTHOR_METHODS = ("mate_2", "compute_yloss", "compute_sparsity_loss", "compute_loss")


def verify_source_lock(lock="cf_baselines/sources.lock.json"):
    records = json.loads(Path(lock).read_text(encoding="utf-8"))
    for source in records.values():
        for record in source["files"]:
            if file_sha256(record["path"]) != record["sha256"]:
                raise ValueError("External evidence/source lock checksum mismatch")
    return records


def load_author_methods(root=".runtime/p5_sources/buliga"):
    root = Path(root)
    for relative, expected in HASHES.items():
        if file_sha256(root/relative) != expected:
            raise ValueError("Pinned author source/license checksum mismatch: "+relative)
    namespace = dict(np=np, pd=pd, itertools=itertools,
                     ModelTypes=SimpleNamespace(Classifier="classifier", Regressor="regressor"),
                     print=lambda *args, **kwargs: None)
    # AST selection keeps bodies exactly as published. No imports, base constructors,
    # module side effects, training, installers, or original experiment entry points.
    tree = ast.parse((root/UTILS).read_text(encoding="utf-8"))
    utility = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Utils")
    import re
    namespace["re"] = re
    exec(compile(ast.Module(body=[utility], type_ignores=[]), str(root/UTILS), "exec"), namespace)
    tree = ast.parse((root/SOURCE).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DiceGeneticLTLf")
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in AUTHOR_METHODS]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check_state"]
    if {n.name for n in nodes} != set(AUTHOR_METHODS)|{"check_state"}:
        raise ValueError("Missing pinned author method")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root/SOURCE), "exec"), namespace)
    return namespace
