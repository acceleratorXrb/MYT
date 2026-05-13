#!/usr/bin/env python3
"""List saved model variants and print their recorded training commands."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
VARIANT_DIR = ROOT / "model_variants"


def load_variants() -> dict[str, dict]:
    variants: dict[str, dict] = {}
    for path in sorted(VARIANT_DIR.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        variant_id = data.get("id") or path.stem
        data["_path"] = str(path.relative_to(ROOT))
        variants[variant_id] = data
    return variants


def require_variant(variants: dict[str, dict], variant_id: str) -> dict:
    if variant_id not in variants:
        available = ", ".join(sorted(variants)) or "<none>"
        raise SystemExit(f"Unknown variant: {variant_id}\nAvailable variants: {available}")
    return variants[variant_id]


def cmd_to_shell(command: list[object]) -> str:
    if not command:
        return ""
    lines = []
    for i, part in enumerate(command):
        text = shlex.quote(str(part))
        if i == 0:
            lines.append(text)
        else:
            lines.append(f"  {text}")
    return " \\\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="List saved model variants.")

    show = sub.add_parser("show", help="Print one variant YAML record.")
    show.add_argument("variant_id")

    train = sub.add_parser("train-command", help="Print the recorded training command.")
    train.add_argument("variant_id")
    train.add_argument("--name", default=None, help="Override the recorded --name value in the printed command.")

    args = parser.parse_args()
    variants = load_variants()

    if args.cmd == "list":
        for variant_id, data in variants.items():
            print(
                f"{variant_id}\t{data.get('date', '')}\t{data.get('status', '')}\t"
                f"{data.get('name', '')}\t{data.get('_path', '')}"
            )
        return

    data = require_variant(variants, args.variant_id)
    if args.cmd == "show":
        print(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
        return

    command = list(data.get("training_command") or [])
    if args.name is not None:
        for i, part in enumerate(command[:-1]):
            if part == "--name":
                command[i + 1] = args.name
                break
    print(cmd_to_shell(command))


if __name__ == "__main__":
    main()

