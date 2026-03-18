#!/usr/bin/env python3
"""
Run GeoWizard inference on a dataset directory and save outputs under the dataset.

Example:
  python scripts/run_geowizard.py \
    --geowizard_repo /path/to/GeoWizard \
    --entry run_infer_v2.py \
    --dataset_dir /path/to/datasets/TUM_RGBD/freiburg3_sitting_rpy/rgb \
    --out_dir /path/to/datasets/TUM_RGBD/freiburg3_sitting_rpy/normal \
    --conda_env geowizard \
    -- --ensemble_size 3 --denoise_steps 10 --seed 0 --domain indoor

Batch over sequences (e.g. BONN/sequence1/rgb -> BONN/sequence1/normal):
  python scripts/run_geowizard.py \
    --geowizard_repo /path/to/GeoWizard \
    --entry run_infer_v2.py \
    --dataset_root /path/to/BONN \
    --input_subdir rgb \
    --output_subdir normal \
    --conda_env geowizard \
    -- --ensemble_size 3 --denoise_steps 10 --seed 0 --domain indoor
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def _resolve_entry(geowizard_repo: Path, entry: str) -> tuple[Path, Path]:
    """
    Returns (workdir, entry_py_path).

    GeoWizard's README suggests running from `geowizard/` subdir.
    """
    repo = geowizard_repo.expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"--geowizard_repo not found: {repo}")

    entry_path = Path(entry)
    if not entry_path.is_absolute():
        # allow: "run_infer.py" (assume in geowizard/), or "geowizard/run_infer.py"
        candidates = [
            repo / "geowizard" / entry_path,
            repo / entry_path,
        ]
        for c in candidates:
            if c.exists():
                entry_path = c
                break
        else:
            raise FileNotFoundError(
                "Cannot find GeoWizard entry script. Tried: "
                + ", ".join(str(c) for c in candidates)
            )
    else:
        entry_path = entry_path.resolve()
        if not entry_path.exists():
            raise FileNotFoundError(f"--entry not found: {entry_path}")

    # Prefer running with cwd at repo/geowizard if it exists (GeoWizard uses relative imports / assets)
    workdir = (repo / "geowizard") if (repo / "geowizard").is_dir() else repo
    return workdir, entry_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run GeoWizard on a dataset directory; save outputs under that dataset directory."
    )
    parser.add_argument(
        "--geowizard_repo",
        type=Path,
        required=True,
        help="Path to the cloned GeoWizard repo, e.g. /path/to/GeoWizard",
    )
    parser.add_argument(
        "--entry",
        type=str,
        default="run_infer_v2.py",
        help="GeoWizard entry .py to run (relative to repo or absolute). Default: run_infer_v2.py",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        help="Single images directory (passed to GeoWizard --input_dir), e.g. .../sequence1/rgb",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Single output directory (passed to GeoWizard --output_dir), e.g. .../sequence1/normal",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        help="Dataset root for batch mode. The script will search for subfolders named --input_subdir under this root.",
    )
    parser.add_argument(
        "--input_subdir",
        type=str,
        default="rgb",
        help="In batch mode: name of per-sequence input folder. Default: rgb",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="normal",
        help="In batch mode: name of per-sequence output folder. Default: normal",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="In batch mode: if output_subdir already exists and is non-empty, skip this sequence.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="In batch mode: continue to next sequence if GeoWizard fails on one sequence.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to run GeoWizard with (default: current python).",
    )
    parser.add_argument(
        "--conda_env",
        type=str,
        default=None,
        help="If set, run GeoWizard via `conda run -n <env>` (e.g. geowizard). "
        "This avoids needing to `conda activate` manually.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="If set, export CUDA_VISIBLE_DEVICES for GeoWizard process (e.g. '0' or '0,1'). "
        "Note: CUDA usage still depends on your GeoWizard env having CUDA-enabled PyTorch.",
    )
    parser.add_argument(
        "--require_outputs",
        action="store_true",
        help="After each run, fail if output_dir stays empty (helps catch silent failures).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the command without executing.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments after `--` are forwarded to GeoWizard entry script.",
    )
    args = parser.parse_args()

    workdir, entry_py = _resolve_entry(args.geowizard_repo, args.entry)

    forwarded = args.extra_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    env = os.environ.copy()
    # Make GeoWizard import resolution a bit more robust when running from another repo.
    env.setdefault("PYTHONPATH", "")
    env["PYTHONPATH"] = str(args.geowizard_repo.expanduser().resolve()) + (
        (os.pathsep + env["PYTHONPATH"]) if env["PYTHONPATH"] else ""
    )
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    def run_one(input_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "geowizard.log"
        run_header = (
            f"===== GeoWizard run @ {datetime.now().isoformat(timespec='seconds')} =====\n"
            f"cwd: {workdir}\n"
            f"input_dir: {input_dir}\n"
            f"output_dir: {output_dir}\n"
            f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', '')}\n"
            f"PYTHONPATH: {env.get('PYTHONPATH', '')}\n"
        )
        if args.conda_env:
            cmd = [
                "conda",
                "run",
                "-n",
                args.conda_env,
                "python",
                str(entry_py),
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(output_dir),
                *forwarded,
            ]
        else:
            cmd = [
                args.python,
                str(entry_py),
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(output_dir),
                *forwarded,
            ]
        if args.dry_run:
            print(" ".join(subprocess.list2cmdline([c]) if " " in c else c for c in cmd))
            print(f"[dry_run] cwd={workdir}")
            print(f"[dry_run] input_dir={input_dir}")
            print(f"[dry_run] output_dir={output_dir}")
            print(f"[dry_run] log_path={log_path}")
            return
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(run_header)
            f.write("cmd: " + " ".join(cmd) + "\n\n")
            f.flush()

            # Tee stdout/stderr to BOTH terminal and log file (no silent runs).
            proc = subprocess.Popen(
                cmd,
                cwd=str(workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
            ret = proc.wait()
            if ret != 0:
                raise subprocess.CalledProcessError(ret, cmd)

        # Basic sanity: did it write anything?
        try:
            non_log_files = [p for p in output_dir.iterdir() if p.name != "geowizard.log"]
        except FileNotFoundError:
            non_log_files = []
        if args.require_outputs and len(non_log_files) == 0:
            raise RuntimeError(
                f"GeoWizard finished but output_dir is empty: {output_dir}. "
                f"Check log: {log_path}"
            )

        print(f"[OK] GeoWizard outputs saved to: {output_dir} (log: {log_path})")

    # ===== Batch mode: dataset_root/<seq>/<input_subdir> -> dataset_root/<seq>/<output_subdir> =====
    if args.dataset_root is not None:
        root = args.dataset_root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"--dataset_root not found: {root}")

        input_name = args.input_subdir
        output_name = args.output_subdir
        input_dirs = sorted([p for p in root.rglob(input_name) if p.is_dir()])
        if len(input_dirs) == 0:
            raise FileNotFoundError(
                f"No '{input_name}' directories found under dataset_root={root}. "
                f"Check --input_subdir or your dataset layout."
            )

        failures = 0
        for in_dir in input_dirs:
            seq_dir = in_dir.parent
            out_dir = seq_dir / output_name

            if args.skip_existing and out_dir.exists():
                try:
                    if any(out_dir.iterdir()):
                        print(f"[SKIP] output exists and non-empty: {out_dir}")
                        continue
                except PermissionError:
                    print(f"[SKIP] cannot inspect output dir (permission): {out_dir}")
                    continue

            try:
                run_one(in_dir, out_dir)
            except subprocess.CalledProcessError as e:
                failures += 1
                print(f"[ERR] GeoWizard failed for input_dir={in_dir} (exit={e.returncode})")
                if not args.continue_on_error:
                    raise

        if failures > 0:
            print(f"[DONE] Completed with failures: {failures}/{len(input_dirs)}")
            return 2
        print(f"[DONE] Completed: {len(input_dirs)} sequences")
        return 0

    # ===== Single mode =====
    if args.dataset_dir is None or args.out_dir is None:
        raise SystemExit(
            "Must provide either batch mode (--dataset_root) OR single mode (--dataset_dir and --out_dir)."
        )

    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"--dataset_dir not found: {dataset_dir}")
    out_dir = args.out_dir.expanduser().resolve()
    run_one(dataset_dir, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

