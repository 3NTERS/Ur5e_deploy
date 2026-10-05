#!/usr/bin/env python3
"""Export the UR5e hover actor from a checkpoint and rebuild its ONNX metadata."""

import argparse
import copy
import hashlib
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import yaml


ROOT = Path(__file__).resolve().parents[1]
ISAAC_ROOT = Path("/home/liang/Workspace/Isaacgym_cjlu/IsaacGymEnvs")
if str(ISAAC_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ISAAC_ROOT / "scripts"))

from export_rokae_policy_onnx import (  # noqa: E402
    build_rl_games_model,
    load_checkpoint,
    load_hydra_yaml,
    make_export_wrapper,
    model_io,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved(path):
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def reference_observations(paths):
    batches = []
    descriptions = []
    for item in paths:
        path = resolved(item)
        with np.load(str(path), allow_pickle=False) as payload:
            batch = np.asarray(payload["observation"], dtype=np.float32)
        if batch.ndim != 2 or batch.shape[1] != 69 or not np.isfinite(batch).all():
            raise ValueError("Expected finite [N, 69] observations in {}".format(path))
        batches.append(batch)
        descriptions.append({"path": str(path), "count": len(batch)})
    if not batches:
        raise ValueError("At least one real observation NPZ is required for export validation")
    return np.concatenate(batches), descriptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(ISAAC_ROOT / "runs/Ur5eRobotiqHoverGripperPPO_actual_q_05-13-06-58/nn/Ur5eRobotiqHoverGripperPPO_actual_q.pth"),
    )
    parser.add_argument(
        "--output", default="resources/models/ur5e_hover_gripper/policy_actual_q_reexport.onnx"
    )
    parser.add_argument(
        "--metadata", default="resources/models/ur5e_hover_gripper/policy_actual_q_reexport.meta.yaml"
    )
    parser.add_argument(
        "--contract-template", default="resources/models/ur5e_hover_gripper/policy.meta.yaml"
    )
    parser.add_argument("--reference", action="append", required=True,
                        help="NPZ containing real 69-D observations; may be repeated")
    args = parser.parse_args()

    checkpoint_path = resolved(args.checkpoint)
    output_path = resolved(args.output)
    metadata_path = resolved(args.metadata)
    template_path = resolved(args.contract_template)
    if output_path == metadata_path or output_path.exists() or metadata_path.exists():
        raise FileExistsError("Choose unused ONNX and metadata output paths")

    task_config_path = ISAAC_ROOT / "isaacgymenvs/cfg/task/Ur5eRobotiqHoverGripper.yaml"
    train_config_path = ISAAC_ROOT / "isaacgymenvs/cfg/train/Ur5eRobotiqHoverGripperPPO.yaml"
    task_config = load_hydra_yaml(task_config_path)
    train_config = load_hydra_yaml(train_config_path)
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    env = task_config["env"]
    if env["subtask"] != "hover_gripper" or template["task"] != "Ur5eRobotiqHoverGripper":
        raise ValueError("Expected the hover task and a hover metadata contract")
    if tuple(tuple(part) for part in template["task_contract"]["observation_zero_slices"]) != (
        (56, 59), (62, 66), (67, 69)
    ):
        raise ValueError("Unexpected hover observation zero slices in contract template")

    torch, model = build_rl_games_model(train_config, 69, 7)
    checkpoint, state, state_key = load_checkpoint(checkpoint_path, torch)
    model.load_state_dict(state, strict=True)
    model.eval()
    rnn_type, example, input_names, output_names, dynamic_axes = model_io(
        None, 69, 7, 8, torch
    )
    if rnn_type != "none":
        raise RuntimeError("Hover export expects a non-recurrent actor")
    wrapper = make_export_wrapper(torch, model, rnn_type)
    wrapper.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, example, str(output_path), export_params=True, opset_version=12,
        do_constant_folding=True, input_names=input_names,
        output_names=output_names, dynamic_axes=dynamic_axes,
    )

    import onnx
    onnx.checker.check_model(onnx.load(str(output_path)))
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    if session.get_providers()[0] != "CPUExecutionProvider":
        raise RuntimeError("ONNX Runtime did not activate CPUExecutionProvider")
    observations, references = reference_observations(args.reference)
    # Include controlled XY changes while holding every other feature fixed.
    base = observations[:1].copy()
    varied = np.repeat(base, 5, axis=0)
    varied[:, 53:55] += np.asarray(
        ((0, 0), (0.12, 0), (-0.12, 0), (0, 0.12), (0, -0.12)),
        dtype=np.float32,
    )
    observations = np.concatenate((observations, varied))
    with torch.no_grad():
        expected = wrapper(torch.from_numpy(observations)).cpu().numpy()
    actual = session.run(["action"], {"observation": observations})[0]
    max_abs_error = float(np.max(np.abs(expected - actual)))
    position_action_range = np.ptp(actual[-5:, :], axis=0)
    if max_abs_error > 1e-5:
        raise RuntimeError("PyTorch/ONNX parity failed: max_abs_error={}".format(max_abs_error))
    if float(np.max(position_action_range)) <= 1e-5:
        raise RuntimeError("Exported ONNX shows no response to object XY changes")

    contract = copy.deepcopy(template["task_contract"])
    contract["episode_steps"] = int(env["episodeLength"])
    contract["policy_period_s"] = float(task_config["sim"]["dt"]) * int(
        env.get("controlFrequencyInv", 1)
    )
    contract["initial_arm_rad"] = list(env["ur5eDefaultDofPos"])
    contract["arm_action_scale"] = float(env["dofSpeedScale"])
    contract["hover_height_m"] = float(env["hoverHeight"])
    network = copy.deepcopy(template["network"])
    network.update({
        "mlp_units": list(train_config["params"]["network"]["mlp"]["units"]),
        "activation": str(train_config["params"]["network"]["mlp"]["activation"]),
        "recurrent": False,
    })
    metadata = {
        "format_version": 2,
        "task": "Ur5eRobotiqHoverGripper",
        "source": {
            "task_file": str(ISAAC_ROOT / "isaacgymenvs/tasks/ur5e_robotiq/ur5e_robotiq_hover_gripper.py"),
            "task_config": str(task_config_path),
            "task_config_sha256_at_export": sha256(task_config_path),
            "train_config": str(train_config_path),
            "train_config_sha256_at_export": sha256(train_config_path),
            "io_contract_template": str(template_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "state_key": state_key,
            "epoch": checkpoint.get("epoch"),
            "frame": checkpoint.get("frame"),
        },
        "onnx": {
            "path": str(output_path),
            "sha256": sha256(output_path),
            "deterministic": True,
            "output_semantics": "clamp(actor_mean, -1, 1)",
            "validation": {
                "provider": "CPUExecutionProvider",
                "max_abs_error": max_abs_error,
                "tolerance": 1e-5,
                "passed": True,
                "real_observation_sources": references,
                "position_perturbation_action_range": position_action_range.tolist(),
            },
        },
        "io": copy.deepcopy(template["io"]),
        "task_contract": contract,
        "network": network,
        "export_task_config": {
            "object_position_center_base_m": list(env["objectStartOffset"]),
            "object_position_noise_xy_m": [
                float(env["resetPositionNoiseX"]), float(env["resetPositionNoiseY"])
            ],
            "palm_center_offset_base_link_m": list(env["graspCenterOffset"]),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print("checkpoint_epoch={} frame={} sha256={}".format(
        metadata["checkpoint"]["epoch"], metadata["checkpoint"]["frame"],
        metadata["checkpoint"]["sha256"]
    ))
    print("onnx={} sha256={}".format(output_path, metadata["onnx"]["sha256"]))
    print("metadata={}".format(metadata_path))
    print("parity_max_abs_error={} position_action_range={}".format(
        max_abs_error, position_action_range.tolist()
    ))


if __name__ == "__main__":
    main()
