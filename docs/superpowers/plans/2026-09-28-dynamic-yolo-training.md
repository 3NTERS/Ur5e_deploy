# Dynamic YOLO Training Implementation Plan

> **For agentic workers:** Use available tools to implement this plan task-by-task within the user-authorized scope. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the YOLO training workflow accept any contiguous class mapping from `data.yaml`, then provide an isolated Conda environment that can train with the installed NVIDIA GPU.

**Architecture:** Dataset validation will normalize and retain class names in `DatasetSpec`; every downstream manifest, checkpoint check, and metadata record will consume that normalized list rather than a hard-coded `object` value. The environment will use a current Windows-compatible Python, PyTorch CUDA build, Ultralytics, and PyYAML, verified against the real dataset before training.

**Tech Stack:** Python 3.10, Ultralytics YOLO, PyTorch CUDA, PyYAML, unittest, Conda

**Spec:** User request in the active Codex task; no separate design document.

## Global Constraints

- Class IDs must be contiguous integers beginning at 0.
- Each YOLO label class ID must be an integer within the configured class range.
- Existing single-class datasets must remain supported.
- Do not modify or overwrite the user's untracked `resources/config/data.yaml`.
- Use a separate Conda environment named `yolo-train`.

---

### Task 1: Dynamic dataset class contract

**Files:**
- Modify: `ur5e_comm/yolo_training.py`
- Test: `tests/test_yolo_training.py`

**Interfaces:**
- Consumes: `data.yaml` `names` as a list or integer-keyed mapping.
- Produces: `DatasetSpec.class_names: tuple[str, ...]` and label validation against `class_count`.

- [ ] **Step 1: Add failing multiclass validation tests**

Add a six-class dataset fixture and assertions that class ID 5 is accepted, while class ID 6 and non-integral IDs are rejected.

- [ ] **Step 2: Run the focused tests and confirm the hard-coded single-class failure**

Run: `python -m unittest tests.test_yolo_training.TestDatasetValidation -v`

Expected: the multiclass acceptance test fails with the current `exactly one class` error.

- [ ] **Step 3: Implement normalized dynamic class names**

Return non-empty class names from `_normalise_names`, reject blank or duplicate names, store the tuple on `DatasetSpec`, and pass its length into `_validate_label` and `_validate_split`.

- [ ] **Step 4: Run the focused tests**

Run: `python -m unittest tests.test_yolo_training.TestDatasetValidation -v`

Expected: all dataset validation tests pass.

### Task 2: Dynamic training outputs and modern Ultralytics compatibility

**Files:**
- Modify: `ur5e_comm/yolo_training.py`
- Test: `tests/test_yolo_training.py`

**Interfaces:**
- Consumes: `DatasetSpec.class_names`.
- Produces: resolved manifest, checkpoint class verification, and metadata that preserve the configured class order.

- [ ] **Step 1: Add failing workflow assertions**

Extend the fake model to expose configurable class names and assert the resolved manifest and metadata contain all six names.

- [ ] **Step 2: Replace every hard-coded `object` class payload**

Generate `names` with `enumerate(spec.class_names)`, compare the trained model to `list(spec.class_names)`, and write that same list to metadata.

- [ ] **Step 3: Remove the legacy `ultralytics.yolo.utils` font dependency**

Resolve `USER_CONFIG_DIR` from the supported `ultralytics.utils` namespace when available, retain a guarded fallback for Ultralytics 8.0.20, and include Windows Arial/DejaVu font candidates.

- [ ] **Step 4: Run the complete training unit test module**

Run: `python -m unittest tests.test_yolo_training -v`

Expected: all tests pass.

### Task 3: CLI/documentation and real-dataset verification

**Files:**
- Modify: `README.md`
- Verify: `scripts/train_yolo_object.py`
- Verify: `D:/Workspace/RobotArm/Code/object_yolo_dataset/data.yaml`

**Interfaces:**
- Consumes: the existing CLI and real six-class dataset.
- Produces: an accurate training command and a successful pre-training validation.

- [ ] **Step 1: Update documentation language and examples**

Replace the fixed `object` requirement with the dynamic contiguous-class contract and document the actual entry point `scripts/train_yolo_object.py`.

- [ ] **Step 2: Validate the real dataset**

Run a short Python command importing `validate_dataset` and assert 37 training images, 31 validation images, and six class names.

- [ ] **Step 3: Run all repository tests relevant to the changed module**

Run: `python -m unittest tests.test_yolo_training -v`

Expected: all tests pass without modifying deployment weights.

### Task 4: Isolated GPU training environment

**Files:**
- Create externally: Conda environment `yolo-train`

**Interfaces:**
- Consumes: NVIDIA driver, Conda, project requirements, and the modified training package.
- Produces: a working Python environment where Ultralytics imports, CUDA is available, and the real dataset validates.

- [ ] **Step 1: Create the environment**

Run: `conda create -n yolo-train python=3.10 pip -y`

- [ ] **Step 2: Install the focused training dependencies**

Install PyTorch/Torchvision with a CUDA build compatible with the machine, then install Ultralytics and PyYAML without pulling the unrelated robot-deployment stack.

- [ ] **Step 3: Verify imports and GPU access**

Run a Python probe that prints package versions, asserts `torch.cuda.is_available()`, and reports the RTX 5060 device.

- [ ] **Step 4: Verify code and dataset in the new environment**

Run the YOLO training unit tests and `validate_dataset` against the real six-class `data.yaml`.

- [ ] **Step 5: Perform a non-destructive one-epoch smoke training only if pretrained weights are locally available**

Use an explicit local model path to avoid an implicit network download; otherwise report the exact command for the user to start full training.

## Self-Review

- Spec coverage: dynamic classes, tests, documentation, isolated environment, CUDA validation, and training entry command are covered.
- Placeholder scan: no deferred implementation placeholders remain.
- Type consistency: `DatasetSpec.class_names` is the shared tuple used by validation, manifests, model checks, and metadata.
