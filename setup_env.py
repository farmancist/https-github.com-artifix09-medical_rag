"""One-shot environment bootstrap. Run this FIRST on any new machine.

    python setup_env.py

It is deliberately dependency-free (standard library only) so it can run with a
bare Python install, before anything else exists.

What it does:
  1. checks the Python version
  2. creates .venv/
  3. installs torch — CUDA build if it detects an NVIDIA GPU, CPU build otherwise
  4. installs requirements.txt
  5. installs this project in editable mode, so `import medical_rag` just works
  6. creates .env from .env.example if missing, and reports which keys are unset
  7. creates the data directories
  8. prints the exact next commands to run

Useful flags:
    --cpu             force the CPU torch build even if a GPU is present
    --cuda cu124      pin a specific CUDA wheel index (default: auto-detected)
    --recreate        delete an existing .venv and start clean
    --skip-torch      leave torch alone (already installed / managing it yourself)
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
MIN_PYTHON = (3, 10)

# Driver version -> the newest CUDA wheel index that driver supports.
# PyTorch wheels bundle their own CUDA runtime, so only the driver matters.
CUDA_BY_DRIVER = [(550, "cu124"), (525, "cu121"), (450, "cu118")]
DEFAULT_CUDA = "cu121"

GREEN, YELLOW, RED, DIM, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"
if platform.system() == "Windows" and not os.environ.get("WT_SESSION"):
    # Older consoles do not render ANSI; degrade rather than print escape codes.
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7
        )
    except Exception:
        GREEN = YELLOW = RED = DIM = RESET = ""


def say(msg: str, colour: str = "") -> None:
    print(f"{colour}{msg}{RESET}", flush=True)


def step(n: int, total: int, msg: str) -> None:
    say(f"\n[{n}/{total}] {msg}", GREEN)


def run(cmd: list[str], *, check: bool = True) -> int:
    say(f"  $ {' '.join(str(c) for c in cmd)}", DIM)
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        say(f"  command failed with exit code {result.returncode}", RED)
        sys.exit(result.returncode)
    return result.returncode


# --------------------------------------------------------------------------


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        say(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required; "
            f"this is {platform.python_version()}",
            RED,
        )
        sys.exit(1)
    say(f"  Python {platform.python_version()} on {platform.system()}")


def detect_cuda() -> str | None:
    """Return a CUDA wheel tag (e.g. 'cu124') if an NVIDIA GPU is usable."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        first = out.stdout.strip().splitlines()[0]
        say(f"  GPU detected: {first}")
        match = re.search(r"(\d+)\.\d+", first.split(",")[1] if "," in first else "")
        if not match:
            return DEFAULT_CUDA
        driver = int(match.group(1))
        for min_driver, tag in CUDA_BY_DRIVER:
            if driver >= min_driver:
                return tag
        say("  NVIDIA driver is older than 450; falling back to the CPU build.", YELLOW)
        return None
    except Exception as exc:
        say(f"  could not query nvidia-smi ({exc}); assuming CPU", YELLOW)
        return None


def create_venv(recreate: bool) -> None:
    if VENV_DIR.exists():
        if not recreate:
            say(f"  reusing existing venv at {VENV_DIR}")
            return
        say(f"  removing {VENV_DIR}", YELLOW)
        shutil.rmtree(VENV_DIR)
    say(f"  creating {VENV_DIR}")
    venv.EnvBuilder(with_pip=True, clear=False, upgrade_deps=False).create(VENV_DIR)


def pip(*args: str, check: bool = True) -> int:
    return run([str(venv_python()), "-m", "pip", *args], check=check)


def install_torch(cuda_tag: str | None) -> None:
    if cuda_tag:
        index = f"https://download.pytorch.org/whl/{cuda_tag}"
        say(f"  installing CUDA torch ({cuda_tag}) — this is a large download")
    else:
        index = "https://download.pytorch.org/whl/cpu"
        say("  installing CPU torch")
    pip("install", "torch", "--index-url", index)


def ensure_env_file() -> list[str]:
    """Create .env from the template if absent; report keys still unset."""
    env_path = ROOT / ".env"
    example = ROOT / ".env.example"
    if not env_path.exists():
        if not example.exists():
            say("  .env.example is missing — cannot create .env", RED)
            return ["GROQ_API_KEY", "QDRANT_URL"]
        shutil.copyfile(example, env_path)
        say("  created .env from .env.example", YELLOW)

    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if match:
            values[match.group(1).upper()] = match.group(2).strip()

    missing: list[str] = []
    # `GROK` / `QDRANT` are accepted aliases, matching the existing .env.
    if not (values.get("GROQ_API_KEY") or values.get("GROK") or values.get("GROQ")):
        missing.append("GROQ_API_KEY (or GROK)")
    # QDRANT_URL is optional: empty means embedded on-disk mode, which needs
    # no server and no key. Only a placeholder value is a real problem.
    if "YOUR-CLUSTER" in values.get("QDRANT_URL", ""):
        missing.append("QDRANT_URL (still the placeholder; clear it for embedded mode)")
    return missing


def make_dirs() -> None:
    for path in (ROOT / "RAW_DATA", ROOT / "data" / "processed", ROOT / "data" / "cache"):
        path.mkdir(parents=True, exist_ok=True)
    say("  RAW_DATA/, data/processed/, data/cache/ ready")


def count_pdfs() -> int:
    raw = ROOT / "RAW_DATA"
    return len(list(raw.rglob("*.pdf"))) if raw.exists() else 0


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the Medical RAG environment")
    parser.add_argument("--cpu", action="store_true", help="Force the CPU torch build")
    parser.add_argument("--cuda", default=None, help="Pin a CUDA wheel tag, e.g. cu124")
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild .venv")
    parser.add_argument("--skip-torch", action="store_true", help="Do not touch torch")
    args = parser.parse_args()

    total = 6
    say("=" * 70)
    say("  Medical RAG — environment setup")
    say("=" * 70)

    step(1, total, "Checking Python")
    check_python()

    step(2, total, "Detecting hardware")
    if args.cpu:
        cuda_tag = None
        say("  --cpu given; using the CPU build")
    elif args.cuda:
        cuda_tag = args.cuda
        say(f"  --cuda given; pinning {cuda_tag}")
    else:
        cuda_tag = detect_cuda()
        if cuda_tag is None:
            say("  no NVIDIA GPU found; using the CPU build")

    step(3, total, "Creating the virtual environment")
    create_venv(args.recreate)
    pip("install", "--upgrade", "pip", "setuptools", "wheel")

    step(4, total, "Installing dependencies")
    if args.skip_torch:
        say("  --skip-torch given")
    else:
        install_torch(cuda_tag)
    pip("install", "-r", str(ROOT / "requirements.txt"))
    # Editable install so `import medical_rag` resolves without PYTHONPATH games.
    pip("install", "-e", str(ROOT))

    step(5, total, "Preparing configuration and directories")
    missing = ensure_env_file()
    make_dirs()
    n_pdfs = count_pdfs()
    say(f"  {n_pdfs} PDF(s) in RAW_DATA/")

    step(6, total, "Verifying the installation")
    rc = run(
        [str(venv_python()), "-c",
         "import torch, transformers, qdrant_client, groq, fitz;"
         "print('  torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"],
        check=False,
    )
    if rc != 0:
        say("  verification failed — see the error above", RED)

    # ---- summary ----
    activate = (
        r".venv\Scripts\activate" if platform.system() == "Windows" else "source .venv/bin/activate"
    )
    say("\n" + "=" * 70)
    say("  Setup complete", GREEN)
    say("=" * 70)

    if missing:
        say("\n  Fill these into .env before continuing:", YELLOW)
        for key in missing:
            say(f"    - {key}", YELLOW)
    if n_pdfs == 0:
        say("\n  RAW_DATA/ has no PDFs — put your source documents there.", YELLOW)

    say("\n  Next:")
    say(f"    1. {activate}")
    say("    2. python scripts/check_setup.py --models")
    say("    3. python scripts/ingest.py --recreate")
    say("    4. python scripts/evaluate.py --retrieval-only")
    say("    5. python scripts/query.py")
    say("")
    say("  Qdrant needs no server: leave QDRANT_URL empty for the embedded")
    say("  on-disk store, or set it to a cluster URL to use that instead.", DIM)
    say("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
