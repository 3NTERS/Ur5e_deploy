#!/usr/bin/env python3
"""Evaluate unique checkpoints found below train_dir best* directories."""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


PLAYER_RE = re.compile(
    r"^av reward: ([0-9.eE+-]+) av steps: ([0-9.eE+-]+)$"
)
OBJECTIVE_RE = re.compile(r"_best_obj_([0-9.]+)_")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(train_dir):
    groups = defaultdict(list)
    for path in sorted(train_dir.rglob("*.pth")):
        relative = path.relative_to(train_dir)
        if not any(part.startswith("best") for part in relative.parts[:-1]):
            continue
        groups[sha256_file(path)].append(path.resolve())
    return sorted(groups.items(), key=lambda item: str(item[1][0]))


def parse_log(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    player = [PLAYER_RE.match(line) for line in lines]
    player = [match for match in player if match]
    if not player:
        return None
    average_reward, average_steps = map(float, player[-1].groups())
    totals = [float(line) for line in lines if re.fullmatch(r"[0-9.-]+", line)]
    maxima = [
        float(line.rsplit(" ", 1)[-1])
        for line in lines
        if line.startswith("Max num successes:")
    ]
    consecutive = [
        float(line.rsplit(" ", 1)[-1])
        for line in lines
        if line.startswith("Average consecutive successes:")
    ]
    if not totals or not maxima or not consecutive:
        return None
    completed_games = ""
    if abs(average_reward) > 1e-12:
        completed_games = int(round(totals[-1] / average_reward))
    text = "\n".join(lines)
    return {
        "completed_games": completed_games,
        "max_successes": max(maxima),
        "max_reported_mean_consecutive_successes": max(consecutive),
        "average_reward": average_reward,
        "average_steps": average_steps,
        "traceback": "Traceback" in text,
        "cuda_error": bool(
            re.search(r"CUDA error|illegal memory|out of memory", text, re.I)
        ),
        "fbx_warning": "FBX library failed" in text,
    }


def write_summary(path, rows):
    fields = [
        "sha256",
        "source_group",
        "best_directory",
        "checkpoint",
        "checkpoint_objective",
        "task_kind",
        "evaluation_task",
        "alias_count",
        "completed_games",
        "max_successes",
        "max_reported_mean_consecutive_successes",
        "average_reward",
        "average_steps",
        "status",
        "returncode",
        "log",
        "aliases",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def checkpoint_metadata(train_dir, digest, aliases):
    checkpoint = aliases[0]
    relative = checkpoint.relative_to(train_dir.resolve())
    best_directory = next(
        part for part in relative.parts[:-1] if part.startswith("best")
    )
    objective = OBJECTIVE_RE.search(checkpoint.name)
    task_kind = "regrasp" if "regrasp" in str(relative).lower() else "throw"
    return {
        "sha256": digest,
        "source_group": relative.parts[0],
        "best_directory": best_directory,
        "checkpoint": str(checkpoint),
        "checkpoint_objective": float(objective.group(1)) if objective else "",
        "task_kind": task_kind,
        "evaluation_task": (
            "AllegroRokaeLSTM_throw_transfer"
            if task_kind == "regrasp"
            else "AllegroRokaeLSTM_throw"
        ),
        "alias_count": len(aliases),
        "aliases": "|".join(str(path) for path in aliases),
    }


def command(checkpoint):
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "rlgpu",
        "python",
        "-m",
        "isaacgymenvs.train",
        "task=AllegroRokaeLSTM",
        "train=AllegroRokaeLSTMPPO",
        "checkpoint={}".format(checkpoint),
        "test=True",
        "headless=True",
        "force_render=False",
        "num_envs=16",
        "seed=0",
        "torch_deterministic=True",
        "task.env.enableDebugVis=False",
        "task.env.evalStats=True",
        "train.params.config.player.games_num=16",
        "train.params.config.player.deterministic=True",
        "train.params.config.player.print_stats=True",
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--isaac-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reuse-log", action="append", default=[])
    parser.add_argument("--max-models", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    groups = discover(args.train_dir)
    if args.max_models is not None:
        groups = groups[: args.max_models]
    reuse = {}
    for item in args.reuse_log:
        digest, value = item.split("=", 1)
        reuse[digest] = Path(value).resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "isaacgym_seed0_summary.csv"
    rows = []
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(args.isaac_root.resolve()) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )

    total = len(groups)
    for index, (digest, aliases) in enumerate(groups, 1):
        metadata = checkpoint_metadata(args.train_dir, digest, aliases)
        result_dir = args.output_dir / digest[:12]
        result_dir.mkdir(parents=True, exist_ok=True)
        log_path = result_dir / "isaacgym_seed0.log"
        manifest_path = result_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        reused = False
        if not log_path.exists() and digest in reuse:
            shutil.copyfile(str(reuse[digest]), str(log_path))
            reused = True

        metrics = parse_log(log_path) if log_path.exists() else None
        returncode = 0
        if metrics and not metrics["traceback"] and not metrics["cuda_error"]:
            status = "reused" if reused else "resumed"
            print(
                "[{}/{}] {} {} reward={:.6f} max_success={:.0f}".format(
                    index,
                    total,
                    status.upper(),
                    digest[:12],
                    metrics["average_reward"],
                    metrics["max_successes"],
                ),
                flush=True,
            )
        else:
            temporary_root = Path("/tmp/isaacgym_rokae_best") / digest[:12]
            temporary_root.mkdir(parents=True, exist_ok=True)
            print(
                "[{}/{}] RUN {} {}".format(
                    index, total, digest[:12], metadata["checkpoint"]
                ),
                flush=True,
            )
            with log_path.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command(metadata["checkpoint"]),
                    cwd=str(temporary_root),
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            returncode = completed.returncode
            metrics = parse_log(log_path)
            if returncode == 0 and metrics and not metrics["traceback"] and not metrics["cuda_error"]:
                status = "passed"
                print(
                    "[{}/{}] PASS {} reward={:.6f} max_success={:.0f}".format(
                        index,
                        total,
                        digest[:12],
                        metrics["average_reward"],
                        metrics["max_successes"],
                    ),
                    flush=True,
                )
            else:
                status = "failed"
                print(
                    "[{}/{}] FAIL {} returncode={}".format(
                        index, total, digest[:12], returncode
                    ),
                    flush=True,
                )

        row = dict(metadata)
        row.update(metrics or {})
        row["status"] = status
        row["returncode"] = returncode
        row["log"] = str(log_path.resolve())
        for internal in ("traceback", "cuda_error", "fbx_warning"):
            row.pop(internal, None)
        rows.append(row)
        write_summary(summary_path, rows)

    failed = sum(row["status"] == "failed" for row in rows)
    print(
        "COMPLETE unique={} failed={} summary={}".format(
            len(rows), failed, summary_path.resolve()
        ),
        flush=True,
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
