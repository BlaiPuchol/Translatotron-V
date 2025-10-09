#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import torch


def find_checkpoints(root: str, model_file_name: str) -> List[dict]:
    checkpoints = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")

    for entry in os.scandir(root):
        if entry.is_dir():
            model_path = os.path.join(entry.path, model_file_name)
            if os.path.isfile(model_path):
                basename = os.path.basename(entry.path)
                nums = [int(n) for n in re.findall(r"\d+", basename)]
                num = nums[-1] if nums else None
                mtime = os.path.getmtime(model_path)
                checkpoints.append(
                    {"dir": entry.path, "file": model_path, "num": num, "mtime": mtime, "name": basename}
                )
    return checkpoints


def select_last(checkpoints: List[dict], n_last: int, sort_by: str) -> List[dict]:
    if not checkpoints:
        return []
    if sort_by == "number":
        # Sort by numeric tokens in dirname; if absent, treat as -1 so they go last
        key_fn = lambda c: (c["num"] if c["num"] is not None else -1)
    else:
        key_fn = lambda c: c["mtime"]
    sorted_cps = sorted(checkpoints, key=key_fn, reverse=True)
    if n_last > 0:
        return sorted_cps[:n_last]
    return sorted_cps


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
        if sd and all(isinstance(v, torch.Tensor) for v in sd.values()):
            return sd
    raise ValueError(f"Unsupported checkpoint format: {path}")


def average_state_dicts(paths: List[str]) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    state_dicts = [load_state_dict(p) for p in paths]
    common_keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        common_keys &= set(sd.keys())

    mismatched = []
    for key in list(common_keys):
        ref = state_dicts[0][key]
        if not isinstance(ref, torch.Tensor):
            common_keys.discard(key)
            continue
        for sd in state_dicts[1:]:
            v = sd[key]
            if not isinstance(v, torch.Tensor) or v.shape != ref.shape or v.dtype != ref.dtype:
                common_keys.discard(key)
                mismatched.append(key)
                break

    averaged: Dict[str, torch.Tensor] = {}
    skipped_non_float: List[str] = []

    count = len(state_dicts)
    for key in common_keys:
        ref = state_dicts[0][key]
        if torch.is_floating_point(ref):
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for sd in state_dicts:
                acc.add_(sd[key].to(torch.float32))
            avg = (acc / count).to(ref.dtype)
            averaged[key] = avg
        else:
            # Copy non-floating values from the most recent checkpoint
            averaged[key] = state_dicts[-1][key]
            skipped_non_float.append(key)

    # Include keys not in common from the latest checkpoint to keep completeness
    last_sd = state_dicts[-1]
    for key, val in last_sd.items():
        if key not in averaged:
            averaged[key] = val

    return averaged, skipped_non_float, mismatched


def main():
    parser = argparse.ArgumentParser(
        description="Average model weights from the last N checkpoints into a single .bin file."
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Directory containing checkpoint subfolders.",
    )
    parser.add_argument(
        "--n_last",
        type=int,
        default=5,
        help="Number of most recent checkpoints to average. Use <=0 to average all.",
    )
    parser.add_argument(
        "--model_file_name",
        type=str,
        default="pytorch_model.bin",
        help="Model weight filename inside each checkpoint folder.",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        choices=["number", "mtime"],
        default="number",
        help="Sort checkpoints by numeric token in folder name or by modification time.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the averaged model (e.g., /path/to/average_pytorch_model.bin). Defaults to checkpoints_dir/average_pytorch_model.bin",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output.",
    )
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.checkpoints_dir, args.model_file_name)
    if not checkpoints:
        print(f"No checkpoints with '{args.model_file_name}' found in: {args.checkpoints_dir}", file=sys.stderr)
        sys.exit(1)

    selected = select_last(checkpoints, args.n_last, args.sort_by)
    if not selected:
        print("No checkpoints selected for averaging.", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print("Selected checkpoints:")
        for c in selected:
            tag = f"(num={c['num']})" if c["num"] is not None else "(num=NA)"
            print(f" - {c['file']} {tag}")

    paths = [c["file"] for c in selected]
    averaged_sd, non_float_keys, mismatched_keys = average_state_dicts(paths)

    output_path = args.output or os.path.join(args.checkpoints_dir, "average_pytorch_model.bin")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(averaged_sd, output_path)

    if not args.quiet:
        print(f"Averaged {len(selected)} checkpoints into: {output_path}")
        if non_float_keys:
            print(f"Non-floating keys copied from latest checkpoint: {len(non_float_keys)}")
        if mismatched_keys:
            print(f"Keys skipped due to mismatch across checkpoints: {len(mismatched_keys)}")


if __name__ == "__main__":
    main()