"""P6-B authorized five-reference completion; no candidate or formal matrix mode."""
import argparse
import json
from pathlib import Path
from evaluation.cf_prerequisites import validate, recover, prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("validate", "recover", "prepare", "verify", "complete"))
    p.add_argument("--config", default="configs/cf_p6b_prerequisites.json")
    p.add_argument("--output-dir")
    args = p.parse_args()
    if args.mode in ("validate", "recover", "complete"):
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        validate(config)
        if args.mode == "validate":
            print("Five isolated references authorized; search/train/CREATED/Audit tuning forbidden")
            return
    if not args.output_dir:
        p.error("--output-dir required")
    if args.mode in ("recover", "complete"):
        recover(config, args.output_dir)
    if args.mode in ("prepare", "complete"):
        prepare(args.output_dir)
    if args.mode in ("verify", "complete"):
        from tools.verify_p6b_prerequisites import verify
        verify(args.output_dir)


if __name__ == "__main__": main()
