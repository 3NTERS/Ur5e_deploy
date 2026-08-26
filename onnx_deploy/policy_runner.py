from pathlib import Path

import numpy as np
import onnxruntime as ort


class PolicyRunner:
    """Stateful batch-one ONNX policy runner."""

    def __init__(self, model_path, provider="cuda"):
        requested = {
            "cuda": "CUDAExecutionProvider",
            "cpu": "CPUExecutionProvider",
        }.get(provider)
        if requested is None:
            raise ValueError("provider must be 'cuda' or 'cpu'")
        available = ort.get_available_providers()
        if requested not in available:
            raise RuntimeError(f"{requested} is unavailable; available providers: {available}")
        try:
            self.session = ort.InferenceSession(str(Path(model_path)), providers=[requested])
        except Exception as error:
            raise RuntimeError(
                f"Failed to create an ONNX Runtime session with {requested}; "
                "verify nvidia-smi, the CUDA driver, and CUDA runtime libraries. "
                f"Original error: {error}"
            ) from error
        if self.session.get_providers()[0] != requested:
            raise RuntimeError(f"Failed to activate {requested}: {self.session.get_providers()}")
        self.provider = requested
        self.inputs = {item.name: item for item in self.session.get_inputs()}
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.observation_dim = self._fixed_dim(self.inputs["observation"].shape[-1])
        self.state = {}
        self.reset()

    @staticmethod
    def _fixed_dim(value):
        if not isinstance(value, int):
            raise ValueError(f"Expected a fixed ONNX dimension, got {value!r}")
        return value

    def reset(self):
        self.state.clear()
        for name in ("hidden_state", "cell_state"):
            if name not in self.inputs:
                continue
            shape = self.inputs[name].shape
            fixed = tuple(1 if not isinstance(dim, int) else dim for dim in shape)
            self.state[name] = np.zeros(fixed, dtype=np.float32)

    def infer(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape == (self.observation_dim,):
            observation = observation[None, :]
        if observation.shape != (1, self.observation_dim):
            raise ValueError(
                f"Expected observation shape (1, {self.observation_dim}), got {observation.shape}"
            )
        if not np.isfinite(observation).all():
            raise ValueError("Observation contains NaN or Inf")
        outputs = self.session.run(None, {"observation": observation, **self.state})
        values = dict(zip(self.output_names, outputs))
        for current, next_name in (
            ("hidden_state", "next_hidden_state"),
            ("cell_state", "next_cell_state"),
        ):
            if next_name in values:
                self.state[current] = np.asarray(values[next_name], dtype=np.float32)
        action = np.asarray(values["action"], dtype=np.float32)
        if not np.isfinite(action).all():
            raise RuntimeError("Policy action contains NaN or Inf")
        return action
