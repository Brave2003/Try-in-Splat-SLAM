---
name: 3dgs-slam-expert
description: "Expert workflow for 3D Gaussian Splatting SLAM debugging and refactoring. Use when working on 3DGS, SLAM tracking or mapping, PyTorch autograd breakage, NaN or Inf instability, tensor shape or device mismatches, CUDA rasterizer backward issues, pose transform bugs, or maintainable code refactors."
argument-hint: "Describe the bug, stack trace, file path, and whether you want diagnosis, patch, or refactor"
---

# 3DGS SLAM Expert

## Outcome
Produce a high-confidence diagnosis and practical fix plan for 3DGS/SLAM development tasks, especially:
- Tracking or mapping optimization not converging
- Missing gradients or broken autograd graphs
- NaN or Inf explosions in rendering or optimization
- Tensor shape or device mismatch failures
- Coordinate transform logic mistakes (W2C and C2W)
- Safe refactors that preserve mathematical behavior

## When To Use
Use this skill when the user asks to debug, inspect gradients, fix runtime errors, or refactor code in a 3DGS/SLAM project.

Common trigger phrases:
- "loss does not decrease"
- "params are not updating"
- "grad is None"
- "found NaN or Inf"
- "shape mismatch" or "device mismatch"
- "refactor tracking or mapping code"

## Required Inputs
Gather these first:
1. Target files and function entry points
2. Error messages, stack trace, and reproducible command
3. Training or optimization phase (tracking, mapping, or both)
4. Model and renderer modules involved (PyTorch and CUDA ops)

## Workflow

### Step 1: Frame The Failure
1. Identify whether the failure is primarily:
- Optimization failure (loss stagnant, no parameter update)
- Numerical instability (NaN or Inf)
- Data contract issue (shape, dtype, or device)
- Pose or transform logic issue
- Architecture and maintainability issue
2. Define success criteria before patching:
- Loss trend improves over a short controlled run
- Target parameters show non-zero finite gradients
- No NaN or Inf in critical tensors
- Refactor preserves functional behavior and outputs

### Step 2: Autograd Integrity Checks
1. Verify target tensors are leaf parameters where required and included in optimizer param groups.
2. Check for graph-breaking operations:
- `.detach()` on active optimization path
- `.item()` used before loss composition
- `.numpy()` conversion before backward path
3. If custom CUDA or third-party rasterizer is used, verify backward path receives valid upstream gradients.
4. Add temporary gradient diagnostics:
- Print `requires_grad`, `is_leaf`, and `.grad` status
- Register hooks to detect zero, NaN, or Inf gradients

Suggested diagnostics snippet:

```python
import torch

def print_tensor_state(name: str, t: torch.Tensor) -> None:
    print(
        f"[{name}] shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
        f"requires_grad={t.requires_grad} is_leaf={t.is_leaf}"
    )


def grad_probe(name: str):
    def _hook(g: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(g).all().item()
        mean_abs = g.abs().mean().item()
        print(f"[grad:{name}] finite={finite} mean_abs={mean_abs:.6e}")
        return g
    return _hook

# Example usage:
# print_tensor_state("camera_pose", camera_pose)
# if camera_pose.requires_grad:
#     camera_pose.register_hook(grad_probe("camera_pose"))
```

### Step 3: 3DGS Numerical Stability Checks
1. Quaternion normalization:
- Confirm quaternion is normalized before rotation matrix conversion.
- Assert finite values after normalization.
2. Scale and covariance validity:
- Guard against over-compressed scales that can produce ill-conditioned covariance.
- Clamp or regularize where mathematically justified.
3. Alpha blending and opacity:
- Verify bounds and finite values through renderer outputs.
4. Propagation guards:
- Add finite checks at key boundaries and fail fast with clear messages.

### Step 4: Tensor Contract Verification
1. Ensure all participating tensors are on consistent device and compatible dtype.
2. Validate shape contracts at module boundaries, especially between tracking and mapping threads.
3. Check copy semantics:
- Avoid unintended shallow references across asynchronous components.

### Step 5: Pose Transform Validation
1. Verify W2C and C2W conversion logic and inversion assumptions.
2. Check transpose versus inverse usage for rotation blocks.
3. Add small deterministic unit checks for transform round-trip consistency.

### Step 6: Refactor With Minimal Behavior Change
1. Separate responsibilities cleanly:
- Tracking frontend
- Mapping backend
- Renderer
- Loss construction
2. Preserve original math unless there is confirmed bug evidence.
3. Add concise comments only where intent is non-obvious.
4. Add type hints and explicit tensor shape intent in docstrings or comments.

## Decision Branches
- If gradients are missing or zero: prioritize Step 2 before any architecture changes.
- If NaN or Inf appears early: prioritize Step 3 and add finite guards first.
- If runtime fails with mismatch errors: prioritize Step 4 contract assertions.
- If trajectory or pose quality is wrong without hard errors: prioritize Step 5.
- If behavior works but code is brittle: execute Step 6 after correctness baseline is locked.

## Completion Checks
The task is complete when all relevant checks pass:
1. Repro path executes without new regressions.
2. Target parameters receive finite non-trivial gradients.
3. No NaN or Inf in monitored critical tensors.
4. Loss or quality metric trends in the expected direction for a short run.
5. Refactor patches keep behavior equivalent unless bug fix intentionally changes it.

## Response Format
Return results in this order:
1. Problem analysis: most likely root cause in one focused section
2. Diagnosis or fix code: concrete assert or hook or patch snippets
3. Architecture note: short maintainability recommendation when relevant

## Example Prompts
- "Use 3dgs-slam-expert: mapping loss is flat and gaussian params do not update in src/mapper.py"
- "Use 3dgs-slam-expert: track down NaN in quaternion to rotation conversion"
- "Use 3dgs-slam-expert: refactor tracker and mapper boundaries without changing math"
- "Use 3dgs-slam-expert: add gradient probes for camera pose and opacity"
