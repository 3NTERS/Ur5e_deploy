"""Check the runtime prerequisites used by Allegro-Rokae sim2sim."""

import sys

import mujoco
import onnxruntime as ort
import torch


def main():
    providers = ort.get_available_providers()
    print("python={}.{}.{}".format(*sys.version_info[:3]))
    print("mujoco={}".format(mujoco.__version__))
    print("onnxruntime={}".format(ort.__version__))
    print("providers={}".format(providers))
    print("torch_cuda={}".format(torch.cuda.is_available()))
    if "CUDAExecutionProvider" not in providers or not torch.cuda.is_available():
        raise SystemExit("CUDA preflight failed: check nvidia-smi and the NVIDIA driver")


if __name__ == "__main__":
    main()
