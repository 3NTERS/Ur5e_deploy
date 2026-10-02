#!/usr/bin/env python3
"""Export a trained rl-games Ur5eRobotiq LSTM checkpoint to deterministic ONNX."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/home/liang/Workspace/Isaacgym_cjlu/IsaacGymEnvs")


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_hydra_yaml(path, active=None):
    path = Path(path).resolve()
    active = set() if active is None else active
    if path in active:
        raise ValueError("Cyclic YAML defaults at {}".format(path))
    active.add(path)
    current = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = current.pop("defaults", []) or []
    merged = {}
    for entry in defaults:
        if entry == "_self_":
            continue
        if isinstance(entry, str):
            relative = Path(entry)
        elif isinstance(entry, dict) and len(entry) == 1:
            group, name = next(iter(entry.items()))
            relative = Path(group) / str(name)
        else:
            raise ValueError("Unsupported defaults entry {!r}".format(entry))
        if relative.suffix not in (".yaml", ".yml"):
            relative = relative.with_suffix(".yaml")
        merged = deep_merge(merged, load_hydra_yaml(path.parent / relative, active))
    active.remove(path)
    return deep_merge(merged, current)


def load_checkpoint(path, torch):
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected an rl-games checkpoint dictionary")
    for key in ("model", "state_dict"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint, checkpoint[key], key
    raise KeyError("Checkpoint has neither model nor state_dict")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(train_config, torch):
    from rl_games.algos_torch import model_builder

    params = copy.deepcopy(train_config["params"])
    factory = model_builder.ModelBuilder().load(params)
    algorithm = params.get("config", {})
    model = factory.build(
        {
            "actions_num": 7,
            "input_shape": (69,),
            "num_seqs": 1,
            "value_size": 1,
            "normalize_input": bool(algorithm.get("normalize_input", False)),
            "normalize_value": bool(algorithm.get("normalize_value", False)),
        }
    )
    model.to("cpu").eval()
    return model


def make_wrapper(torch, model):
    class DeterministicPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, observation, hidden_state, cell_state):
            normalized = self.model.norm_obs(observation)
            mu, _sigma, _value, states = self.model.a2c_network(
                {
                    "obs": normalized,
                    "seq_length": 1,
                    "rnn_states": (hidden_state, cell_state),
                }
            )
            return torch.clamp(mu, -1.0, 1.0), states[0], states[1]

    return DeterministicPolicy().eval()


def validate_onnx(path, wrapper, inputs, torch, provider):
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(path)))
    provider_name = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}[provider]
    if provider_name not in ort.get_available_providers():
        raise RuntimeError("{} unavailable: {}".format(provider_name, ort.get_available_providers()))
    session = ort.InferenceSession(str(path), providers=[provider_name])
    generator = torch.Generator().manual_seed(0)
    observations = torch.randn(5, 69, generator=generator)
    torch_hidden, torch_cell = inputs[1], inputs[2]
    ort_hidden = torch_hidden.detach().numpy()
    ort_cell = torch_cell.detach().numpy()
    errors = {name: 0.0 for name in ("action", "next_hidden_state", "next_cell_state")}
    action_sequence = []
    for observation in observations:
        observation = observation.unsqueeze(0)
        with torch.no_grad():
            expected = wrapper(observation, torch_hidden, torch_cell)
        actual = session.run(
            None,
            {
                "observation": observation.numpy(),
                "hidden_state": ort_hidden,
                "cell_state": ort_cell,
            },
        )
        for name, reference, value in zip(errors, expected, actual):
            errors[name] = max(errors[name], float(np.max(np.abs(reference.numpy() - value))))
        action_sequence.append(actual[0][0].copy())
        torch_hidden, torch_cell = expected[1], expected[2]
        ort_hidden, ort_cell = actual[1], actual[2]
    if max(errors.values()) > 1e-4:
        raise RuntimeError("ONNX numerical validation failed: {}".format(errors))
    return {
        "provider": provider_name,
        "sequence_steps": 5,
        "max_abs_error_by_output": errors,
        "action_span": float(np.ptp(np.asarray(action_sequence), axis=0).max()),
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--train-config",
        default=str(DEFAULT_SOURCE / "isaacgymenvs/cfg/train/Ur5eRobotiqGraspLSTMPPO.yaml"),
    )
    parser.add_argument("--output", default=str(ROOT / "resources/models/ur5e_robotiq/policy.onnx"))
    parser.add_argument("--metadata", default=str(ROOT / "resources/models/ur5e_robotiq/policy.meta.yaml"))
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument(
        "--inertia-report",
        default=str(ROOT / "resources/reports/ur5e_inertia_audit.json"),
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Run this script in the Isaac Gym/rl-games environment") from error
    train_config = load_hydra_yaml(args.train_config)
    model = build_model(train_config, torch)
    checkpoint, state, state_key = load_checkpoint(args.checkpoint, torch)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint restore failed: {}".format(incompatible))
    wrapper = make_wrapper(torch, model)
    rnn = train_config["params"]["network"].get("rnn")
    if not rnn or rnn.get("name") != "lstm":
        raise ValueError("This deployment exporter requires the Ur5eRobotiq LSTM training config")
    shape = (int(rnn["layers"]), 1, int(rnn["units"]))
    inputs = (torch.zeros(1, 69), torch.zeros(*shape), torch.zeros(*shape))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        str(output),
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["observation", "hidden_state", "cell_state"],
        output_names=["action", "next_hidden_state", "next_cell_state"],
        dynamic_axes={
            "observation": {0: "batch"},
            "action": {0: "batch"},
            "hidden_state": {1: "batch"},
            "cell_state": {1: "batch"},
            "next_hidden_state": {1: "batch"},
            "next_cell_state": {1: "batch"},
        },
    )
    validation = validate_onnx(output, wrapper, inputs, torch, args.provider)
    metadata_path = Path(args.metadata)
    template_path = (
        metadata_path
        if metadata_path.is_file()
        else ROOT / "resources/models/ur5e_robotiq/policy.meta.yaml"
    )
    metadata = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    inertia_report = Path(args.inertia_report)
    if not inertia_report.is_file():
        raise FileNotFoundError("Missing inertia report: {}".format(inertia_report))
    metadata["format_version"] = 2
    metadata["task"] = "Ur5eRobotiqGrasp"
    metadata["deployment_class"] = "smoke_only"
    metadata["source"] = {
        "task_file": str(DEFAULT_SOURCE / "isaacgymenvs/tasks/ur5e_robotiq/ur5e_robotiq_grasp.py"),
        "task_config": str(DEFAULT_SOURCE / "isaacgymenvs/cfg/task/Ur5eRobotiqGrasp.yaml"),
        "train_config": str(Path(args.train_config).resolve()),
    }
    metadata["training"] = {
        "seed": 0,
        "num_environments": 256,
        "horizon": 16,
        "minibatch_size": 4096,
        "lstm_units": int(rnn["units"]),
        "epochs": 30,
    }
    metadata["task_contract"] = {
        "subtask": "grasp",
        "policy_period_s": 0.01667,
        "episode_steps": 600,
        "object_size_m": [0.04, 0.04, 0.04],
        "object_density_kg_m3": 567.0,
        "goal_mode": "initial_object_offset",
        "goal_offset_m": [0.0, 0.0, 0.12],
        "success_lift_m": 0.10,
        "randomization": False,
    }
    metadata["inertia_audit"] = {
        "path": str(inertia_report.resolve()),
        "sha256": sha256(inertia_report),
    }
    metadata["checkpoint"] = {
        "path": str(Path(args.checkpoint).resolve()),
        "sha256": sha256(args.checkpoint),
        "state_key": state_key,
        "epoch": checkpoint.get("epoch"),
        "frame": checkpoint.get("frame"),
    }
    metadata["network"] = {
        "model": train_config["params"]["model"]["name"],
        "builder": train_config["params"]["network"]["name"],
        "recurrent": copy.deepcopy(rnn),
    }
    metadata["onnx"] = {
        "path": str(output.resolve()),
        "sha256": sha256(output),
        "deterministic": True,
        "output_semantics": "clamp(actor_mean, -1, 1)",
        "validation": validation,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    print("Exported {} and validated {}".format(output.resolve(), validation))


if __name__ == "__main__":
    main()
