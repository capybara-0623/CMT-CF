"""Pack the frozen comparison dependencies and relocate copies for Linux.

No search, training, parameter changes or changes to the source freeze.
Run pack on Windows; run relocate after extracting the archive on AutoDL.
"""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SEED_DIRS = ("outputs/p6b_bridge_preflight_20260918_v1",
             "outputs/p6b_prerequisites_20260918_v2", "outputs/p6b_protocol_20260918_v2")
SOURCE_DIRS = ("evaluation", "cf_search", "cf_baselines", "tools", "configs", "encoding",
               "pattern_discovery", "utils", "explaination", "labeling", "tests")
MANIFEST = "autodl_bundle_manifest.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def origin_relative(value, origin):
    path, base = PureWindowsPath(value), PureWindowsPath(origin)
    if not path.is_absolute():
        return None
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return None


def resolve_record(value, owner, root, origin):
    relative = origin_relative(value, origin)
    if relative is not None:
        return root / relative
    path = Path(value.replace("\\", "/"))
    if path.is_absolute():
        return path
    near = (owner.parent/path).resolve()
    if near.is_relative_to(root) and near.is_file():
        return near
    direct = (root/path).resolve()
    if direct.is_relative_to(root) and direct.is_file():
        return direct
    # Some saved cache manifests embed a path relative to the source
    # protocol rather than their own cache directory.
    parts = list(path.parts)
    while parts and parts[0] == "..":
        parts.pop(0)
    workspace = root.joinpath(*parts)
    return workspace if workspace.is_file() else near


def records(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for item in value.values():
            yield from records(item)


def record_entries(value, owner, root, origin):
    """Embedded model records retain their original protocol-relative base."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value, owner
        protocol = value.get("protocol")
        protocol_path = protocol.get("path") if isinstance(protocol, dict) else protocol
        child_owner = resolve_record(protocol_path, owner, root, origin) if isinstance(protocol_path, str) else owner
        for key,item in value.items():
            yield from record_entries(item, owner if key == "protocol" else child_owner, root, origin)
    elif isinstance(value, list):
        for item in value:
            yield from record_entries(item, owner, root, origin)
    elif isinstance(value, list):
        for item in value:
            yield from records(item)


def pack(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Archive exists; use a new name")
    origin = str(ROOT)
    files = set(ROOT.glob("*.py"))
    files.add(ROOT / "configs/autodl_requirements.txt")
    files.add(ROOT / "docs/AUTODL_RUN.md")
    for folder in SOURCE_DIRS:
        for path in (ROOT/folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and (
                    path.suffix in (".py", ".json", ".ps1", ".sh") or "LICENSE" in path.name.upper()):
                files.add(path)
    files.update(p for p in (ROOT/".runtime/p5_sources").rglob("*") if p.is_file())
    for folder in SEED_DIRS:
        files.update(p for p in (ROOT/folder).rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl"))
    queue, scanned, hashes = list(files), set(), {}
    while queue:
        owner = queue.pop()
        if owner in scanned:
            continue
        scanned.add(owner)
        if owner.suffix != ".json":
            continue
        payload = json.loads(owner.read_text(encoding="utf-8-sig"))
        # Original P0 freeze code hashes describe that historical freeze, not
        # the later accepted runner. Its runtime validator intentionally does
        # not require those historical code versions to be installed.
        referenced = {k:v for k,v in payload.items() if k != "code"} if payload.get("protocol_id") == "cf_comparison_v1" else payload
        for record, record_owner in record_entries(referenced, owner, ROOT, origin):
            path = resolve_record(record["path"], record_owner, ROOT, origin).resolve()
            path.relative_to(ROOT)
            if path not in hashes:
                hashes[path] = sha(path)
            if hashes[path] != record["sha256"]:
                raise ValueError("Referenced frozen artifact changed: " + str(path))
            if path not in files:
                files.add(path)
                queue.append(path)
    listing = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                    sha256=hashes.get(p) or sha(p)) for p in sorted(files)]
    manifest = dict(purpose="P6_B_AutoDL_frozen_dependency_bundle_v1", origin_root=origin,
        files=listing, file_count=len(listing), total_bytes=sum(r["bytes"] for r in listing),
        model_training=False, new_candidates=False, notes="Includes referenced historical evidence, not live formal outputs or Windows installed libraries")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz", compresslevel=1) as archive:
        for path in sorted(files):
            archive.add(path, arcname="MMOE_UWL_cf/"+path.relative_to(ROOT).as_posix(), recursive=False)
        data = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        info = tarfile.TarInfo("MMOE_UWL_cf/"+MANIFEST)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    result = dict(archive=str(output), archive_bytes=output.stat().st_size,
        file_count=manifest["file_count"], uncompressed_bytes=manifest["total_bytes"])
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def relocate(destination):
    manifest = json.loads((ROOT/MANIFEST).read_text(encoding="utf-8"))
    out = Path(destination).resolve()
    if out == ROOT or (out.exists() and any(out.iterdir())):
        raise ValueError("Relocate to a new empty project directory")
    origin = manifest["origin_root"]
    entries = {r["path"]:r for r in manifest["files"]}
    done, active = {}, set()
    out.mkdir(parents=True, exist_ok=True)

    def translate(value):
        relative = origin_relative(value, origin)
        return str(out/relative) if relative is not None else value

    def migrate(relative):
        if relative in done:
            return done[relative]
        if relative in active:
            raise ValueError("Cyclic artifact references: " + relative)
        active.add(relative)
        source, target = ROOT/relative, out/relative
        if sha(source) != entries[relative]["sha256"]:
            raise ValueError("Transfer changed a file: " + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(source.read_text(encoding="utf-8-sig")) if source.suffix == ".json" else None

        def walk(value, trail=(), owner=source):
            if trail[-2:] == ("model", "config"):
                # Must equal the cfg stored in the unchanged torch checkpoint.
                return copy.deepcopy(value)
            if old.get("protocol_id") == "cf_comparison_v1" and trail == ("code",):
                return copy.deepcopy(value)
            if isinstance(value, dict):
                protocol = value.get("protocol")
                protocol_path = protocol.get("path") if isinstance(protocol, dict) else protocol
                child_owner = resolve_record(protocol_path, owner, ROOT, origin) if isinstance(protocol_path, str) else owner
                new = {key:walk(item, trail+(key,), owner if key == "protocol" else child_owner) for key,item in value.items()}
                if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                    linked = resolve_record(value["path"], owner, ROOT, origin)
                    linked_relative = linked.resolve().relative_to(ROOT).as_posix()
                    new["sha256"] = migrate(linked_relative)
                if "config_sha256" in new and "config" in new:
                    new["config_sha256"] = stable(new["config"])
                if "content_sha256" in new:
                    new["content_sha256"] = stable({k:v for k,v in new.items() if k != "content_sha256"})
                return new
            if isinstance(value, list):
                return [walk(item, trail, owner) for item in value]
            return translate(value) if isinstance(value, str) else value

        new = walk(old) if old is not None else None
        if new != old:
            with target.open("x", encoding="utf-8") as handle:
                json.dump(new, handle, ensure_ascii=False, indent=2, allow_nan=False)
        else:
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        done[relative] = sha(target)
        active.remove(relative)
        return done[relative]

    for relative in entries:
        migrate(relative)
    base = out/"outputs/p6b_bridge_preflight_20260918_v1"
    authorization = dict(purpose="P6_B_explicit_generation_authorization_v1", authorized=True,
        scope="seven_task_matrix", bridge_preflight=dict(path=str(base/"bridge_preflight.json"), sha256=sha(base/"bridge_preflight.json")),
        run_dir=str(out/"outputs/p6b_formal_autodl_v1"),
        user_instruction="完成正式矩阵、保存后 Audit、最终验收和统计制表；原查询、模型、规则、预算和比较分组保持；CREATED 不处理。服务器执行前由操作者启动命令。")
    with (out/"outputs/autodl_generation_authorization.json").open("x", encoding="utf-8") as handle:
        json.dump(authorization, handle, ensure_ascii=False, indent=2)
    receipt = dict(purpose="P6_B_path_only_relocation_v1", origin_root=origin, source_root=str(ROOT), destination=str(out),
        source_bundle=sha(ROOT/MANIFEST), changed_json_files=sum(done[k]!=entries[k]["sha256"] for k in entries),
        unchanged_binary_files=True, checkpoint_cfg_preserved=True, new_candidates=False, model_training=False)
    with (out/"outputs/autodl_relocation.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    subs.add_parser("pack").add_argument("--output", required=True)
    subs.add_parser("relocate").add_argument("--destination", required=True)
    args = parser.parse_args()
    pack(args.output) if args.mode == "pack" else relocate(args.destination)


if __name__ == "__main__":
    main()
