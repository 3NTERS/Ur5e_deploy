import csv
from pathlib import Path
from time import perf_counter

import numpy as np


class EpisodeRunner:
    def __init__(self, adapter, policy):
        self.adapter = adapter
        self.policy = policy
        self.stopped = False

    def stop(self):
        self.stopped = True

    def run(self, steps=600, viewer=None, npz_path=None, csv_path=None):
        self.stopped = False
        self.policy.reset()
        observation = self.adapter.reset()
        observations, actions, targets, times = [], [], [], []
        started = perf_counter()
        for step in range(steps):
            if self.stopped or self.adapter.done:
                break
            action = self.policy.infer(observation)[0]
            target = self.adapter.apply_action(action)
            target_time = (step + 1) * self.adapter.policy_period
            self.adapter.step_physics_to(target_time)
            observation = self.adapter.observe()
            observations.append(observation.copy())
            actions.append(action.copy())
            targets.append(target)
            times.append(self.adapter.data.time)
            if viewer is not None:
                viewer.sync()
        elapsed = perf_counter() - started
        result = {
            "observation": np.asarray(observations, dtype=np.float32),
            "action": np.asarray(actions, dtype=np.float32),
            "target": np.asarray(targets, dtype=np.float32),
            "sim_time": np.asarray(times, dtype=np.float64),
        }
        if npz_path:
            path = Path(npz_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **result)
        if csv_path:
            path = Path(csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["step", "sim_time", *[f"action_{i}" for i in range(23)], *[f"target_{i}" for i in range(23)]])
                for index, (time, action, target) in enumerate(zip(result["sim_time"], result["action"], result["target"])):
                    writer.writerow([index, time, *action, *target])
        return {"steps": len(times), "elapsed_seconds": elapsed, "trajectory": result}
