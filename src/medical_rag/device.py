"""Hardware acceleration resolution — the one place that decides what runs where.

This machine's GPU is an **AMD Radeon RX590** (Polaris, gfx803), not NVIDIA.
ROCm does not support this card on Windows, and only patchily on Linux, so the
only real GPU path here is **ONNX Runtime's DirectML execution provider**
(`DmlExecutionProvider`), used through sentence-transformers' `backend="onnx"`
(see `requirements-dml.txt` for the package layout and why install order
matters). torch itself stays on the CPU wheel; DirectML acceleration happens
inside ONNX Runtime, not through a torch device string.

Non-negotiable: **no silent CPU fallback**. If neither CUDA nor a *working*
DirectML provider resolves, `get_device()` raises `RuntimeError` rather than
quietly returning CPU. Set `ALLOW_CPU=1` to override for a machine that really
has no GPU (e.g. CI, or a laptop) — that override is explicit and logged at
WARNING, never silent.

Availability is not proof: ONNX Runtime lists `DmlExecutionProvider` in
`get_available_providers()` whenever the DLL is importable, even on a machine
where it cannot actually initialize. If a session then fails to init the
requested provider, ORT drops it and silently serves the next one (typically
CPU) with no error. So the only real check is creating a session and reading
back `session.get_providers()` *after* construction, not before.

PRODUCTION IS CPU-ONLY as of 2026-09-14 -- CORRECTED same day: the first pass
of this investigation saw two reboots on 2026-09-14 with a display-driver TDR
event next to one of them and wrongly concluded sustained DirectML/GPU load
was the cause. A fuller look at the 30-day power-event history showed this
machine has been losing power unexpectedly **1-7 times a day for at least a
month**, including days with no GPU load at all (e.g. three reboots on
2026-09-09, the day the vector index was built on CPU). That is a general
power-delivery/hardware-reliability problem -- most likely a PSU degrading
under load, possibly failing RAM, thermal shutdown, or a loose power
connector -- not something DirectML specifically triggers.

`config.Settings.device` still defaults to `"cpu"`, which resolves through
`device_info_for("cpu")` below and never reaches `_resolve()`/`_directml()`
at all -- kept as a precaution, since the RX590 draws ~225W a workstation PSU
may not be specced for and this machine's power delivery is already
demonstrably marginal, but this is risk-reduction against a *plausible
contributing factor*, not a fix applied to a *confirmed root cause*. The
CUDA/DirectML probes in this module are kept working for deliberate,
supervised use (e.g. `scripts/bench_device.py`, or `DEVICE=dml` set by hand
for a one-off benchmark or a supervised full-corpus rebuild) -- the GPU path
itself is not known to be broken or dangerous, it is just not the default
while the underlying hardware fault is still undiagnosed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    backend: str  # "cuda" | "dml" | "cpu"
    torch_device: str  # what to hand torch itself: "cuda" | "cpu"
    st_backend: str  # sentence-transformers `backend=`: "torch" | "onnx"
    provider: str | None  # ONNX Runtime execution provider, when st_backend == "onnx"
    driver: str | None  # CUDA version / ORT version / None, for the log line


def _cuda() -> DeviceInfo | None:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    return DeviceInfo("cuda", "cuda", "torch", None, torch.version.cuda)


def _register_onnxruntime_directml_alias() -> None:
    """Teach `optimum` that `onnxruntime-directml` counts as `onnxruntime`.

    `optimum.utils.import_utils._is_package_available("onnxruntime", ...)`
    checks a fixed list of known GPU-variant distribution names
    (onnxruntime-gpu, onnxruntime-rocm, ...) via `importlib.metadata.version`.
    `onnxruntime-directml` is not on that list, even though it installs the
    exact same `onnxruntime` import module (see requirements-dml.txt). Left
    unpatched, optimum's ONNX export step raises "onnx and onnxruntime
    packages are necessary" even though onnxruntime demonstrably works -- this
    module's own DirectML probe in `_directml()` already proved that.

    Must run before optimum is imported anywhere in the process, since the
    flag this feeds is computed once at `optimum.utils.import_utils` import
    time. Safe to call unconditionally: it is a no-op whenever a plain
    `onnxruntime` distribution is actually installed.
    """
    import importlib.metadata as im

    try:
        im.version("onnxruntime")
        return  # a plain "onnxruntime" distribution is genuinely installed
    except im.PackageNotFoundError:
        pass

    try:
        im.version("onnxruntime-directml")
    except im.PackageNotFoundError:
        return  # neither present; let the real "not installed" error surface

    orig_version = im.version

    def _version(name, *args, **kwargs):
        if name == "onnxruntime":
            return orig_version("onnxruntime-directml", *args, **kwargs)
        return orig_version(name, *args, **kwargs)

    im.version = _version


_register_onnxruntime_directml_alias()


def _make_probe_model_bytes() -> bytes:
    """A one-node ONNX graph, just to force real session construction."""
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "device_probe", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


def _directml() -> DeviceInfo | None:
    try:
        import onnxruntime as ort
    except Exception:
        return None

    available = ort.get_available_providers()
    if "DmlExecutionProvider" not in available:
        return None

    # Listed != working (see module docstring): only a real session proves it.
    try:
        session = ort.InferenceSession(
            _make_probe_model_bytes(),
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
    except Exception as exc:
        logger.warning("DirectML provider listed but session creation failed: %s", exc)
        return None

    active = session.get_providers()
    if "DmlExecutionProvider" not in active:
        logger.warning(
            "DirectML provider listed but ORT silently fell back to %s after session init",
            active,
        )
        return None

    return DeviceInfo("dml", "cpu", "onnx", "DmlExecutionProvider", ort.__version__)


def _resolve() -> DeviceInfo:
    for probe in (_cuda, _directml):
        info = probe()
        if info is not None:
            return info

    if os.environ.get("ALLOW_CPU") == "1":
        logger.warning(
            "No GPU backend resolved (no CUDA, no working DirectML provider); "
            "running on CPU because ALLOW_CPU=1"
        )
        return DeviceInfo("cpu", "cpu", "torch", None, None)

    raise RuntimeError(
        "No GPU backend available: no CUDA device, and DirectML did not "
        "initialize (or onnxruntime-directml / optimum are not installed — "
        "see requirements-dml.txt). This project refuses to run medical "
        "inference on a silent CPU fallback. Set ALLOW_CPU=1 to override "
        "explicitly (e.g. for CI or a machine with no GPU)."
    )


_resolved: DeviceInfo | None = None


def get_device() -> DeviceInfo:
    """Resolve once, at first call, and cache for the life of the process."""
    global _resolved
    if _resolved is None:
        _resolved = _resolve()
        logger.info(
            "Resolved device: backend=%s torch_device=%s st_backend=%s provider=%s driver=%s",
            _resolved.backend,
            _resolved.torch_device,
            _resolved.st_backend,
            _resolved.provider,
            _resolved.driver,
        )
    return _resolved


def reset_device_cache() -> None:
    """Test-only escape hatch — production code never needs to re-resolve."""
    global _resolved
    _resolved = None


def device_info_for(explicit: str) -> DeviceInfo:
    """Build a `DeviceInfo` for an explicit `DEVICE=` override (not "auto").

    An explicit choice is a deliberate override, not a fallback, so it is
    honoured without the raise-unless-ALLOW_CPU check in `_resolve()` -- the
    caller already said what they want.
    """
    if explicit == "cuda":
        return DeviceInfo("cuda", "cuda", "torch", None, None)
    if explicit == "dml":
        return DeviceInfo("dml", "cpu", "onnx", "DmlExecutionProvider", None)
    if explicit == "cpu":
        return DeviceInfo("cpu", "cpu", "torch", None, None)
    raise ValueError(f"Unknown explicit device override: {explicit!r}")
