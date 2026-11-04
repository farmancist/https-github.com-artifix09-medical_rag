"""Phase 0 proof: embed 512 synthetic sentences and report device + throughput.

    python scripts/bench_device.py

Required by the master prompt's Phase 0 gate: "benchmark shows real GPU
throughput, and get_device() provably raises on CPU." A CPU-speed number on a
"GPU" backend means the device check is lying, not that DirectML is slow --
that's the whole point of reading the numbers rather than trusting the label.

Runs through the real `DenseEncoder`, not a one-off script, so this exercises
exactly the code path ingestion and querying use.

CPU-only production note (2026-09-14, see src/medical_rag/device.py
docstring for the full story -- this machine has an unrelated month-long
power-reliability problem, not a GPU-load-specific one): this reads `DEVICE`
the same way `DenseEncoder` does, so with the current default (`DEVICE=cpu`)
it never touches CUDA/DirectML and never spins up the GPU. To actually
benchmark the GPU, set `DEVICE=dml` (or `auto`/`cuda`) in `.env`
*deliberately* for that one run, and put it back to `cpu` afterwards.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_rag.config import configure_logging, get_settings  # noqa: E402
from medical_rag.device import device_info_for, get_device  # noqa: E402
from medical_rag.embeddings.encoder import DenseEncoder  # noqa: E402

N_SENTENCES = 512

_TOPICS = [
    "hypertension", "type 2 diabetes", "sepsis", "asthma exacerbation",
    "atrial fibrillation", "community-acquired pneumonia", "acute kidney injury",
    "iron-deficiency anaemia", "peptic ulcer disease", "deep vein thrombosis",
    "chronic obstructive pulmonary disease", "migraine", "epilepsy",
    "rheumatoid arthritis", "hypothyroidism", "urinary tract infection",
]


def synthetic_sentences(n: int) -> list[str]:
    return [
        f"Sentence {i}: first-line management of {_TOPICS[i % len(_TOPICS)]} "
        f"in an adult patient, including monitoring and follow-up."
        for i in range(n)
    ]


def main() -> int:
    configure_logging()

    settings = get_settings()
    # Mirror DenseEncoder's own resolution: only probe CUDA/DirectML (and
    # thus only wake the GPU) when DEVICE is explicitly "auto" -- an explicit
    # "cpu" (the production default) or "dml"/"cuda" is honoured as-is.
    info = get_device() if settings.device == "auto" else device_info_for(settings.device)
    print("=== device ===")
    print(f"{'backend':>15} : {info.backend}")
    print(f"{'torch_device':>15} : {info.torch_device}")
    print(f"{'st_backend':>15} : {info.st_backend}")
    print(f"{'provider':>15} : {info.provider}")
    print(f"{'driver':>15} : {info.driver}")

    encoder = DenseEncoder(settings)
    sentences = synthetic_sentences(N_SENTENCES)

    # Warm-up: loads the model and pays first-call session/graph-build cost
    # once, outside the timed region.
    encoder.encode(sentences[:8])

    start = time.perf_counter()
    vectors = encoder.encode(sentences, show_progress=False)
    elapsed = time.perf_counter() - start
    rate = len(sentences) / elapsed

    print("\n=== throughput ===")
    print(f"{'sentences':>15} : {len(sentences)}")
    print(f"{'wall time':>15} : {elapsed:.2f}s")
    print(f"{'sentences/sec':>15} : {rate:.1f}")
    print(f"{'output shape':>15} : {vectors.shape}")

    if info.backend == "cpu":
        print(
            "\nNOTE: backend is cpu -- this is a CPU baseline, not a GPU proof. "
            "Set DEVICE=dml (or auto/cuda) in .env for a deliberate one-off GPU run."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
