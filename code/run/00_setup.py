"""00_setup.py -- clone the pinned starter kit, create/verify the local env, sanity-check data.

Run once per machine (and once per Colab runtime, where the Drive copy of the repo is
already synced so only the clone check is needed):

    python code/run/00_setup.py [--bump-swap]

Exit codes: 0 ok, 1 something the next steps need is missing.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fec_common.config import (  # noqa: E402
    CODE_DIR,
    DATA_ROOT,
    PINNED_COMMIT,
    REPO_ID,
    RUN_DIR,
    STARTER_DIR,
    TRAIN_CONFIG,
    load_env,
    log,
)


def sh(cmd: list[str], cwd: Path | None = None) -> None:
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_starter_clone() -> None:
    """Clone the official starter kit at the pinned commit (idempotent)."""
    if (STARTER_DIR / ".git").exists():
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=STARTER_DIR, capture_output=True, text=True
        ).stdout.strip()
        if head != PINNED_COMMIT:
            log(f"starter at {head[:10]}, checking out {PINNED_COMMIT[:10]}")
            sh(["git", "fetch", "--depth", "1", "origin", PINNED_COMMIT], cwd=STARTER_DIR)
            sh(["git", "checkout", "--force", PINNED_COMMIT], cwd=STARTER_DIR)
        else:
            log(f"starter already at pinned commit {PINNED_COMMIT[:10]}")
        return

    STARTER_DIR.parent.mkdir(parents=True, exist_ok=True)
    sh(["git", "clone", "https://github.com/Sophelio/fusion-equilibrium-challenge-starter.git",
        str(STARTER_DIR)])
    sh(["git", "checkout", PINNED_COMMIT], cwd=STARTER_DIR)


def ensure_env_file() -> None:
    """Make sure code/.env exists with an HF_TOKEN slot (values are never printed)."""
    env_path = CODE_DIR / ".env"
    if not env_path.exists():
        env_path.write_text("# Hugging Face token (dataset download + prediction push)\n"
                            "HF_TOKEN=\n")
        log(f"created {env_path} -- fill in HF_TOKEN for 06/07")
    token = load_env()
    log(f"HF_TOKEN: {'set' if token else 'MISSING (07 --push will need it)'}")


def check_python_deps() -> list[str]:
    missing = []
    for mod in ["numpy", "pandas", "scipy", "sklearn", "joblib", "pyarrow",
                "datasets", "huggingface_hub", "dotenv"]:
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    return missing


def check_data() -> dict:
    counts = {}
    for name in [TRAIN_CONFIG, "diii_d_public_test", "mast_public_test"]:
        d = DATA_ROOT / name
        counts[name] = len(list(d.glob("*.parquet"))) if d.exists() else 0
    return counts


def check_memory() -> None:
    with open("/proc/meminfo") as fh:
        info = dict(
            line.split(":", 1) for line in fh.read().splitlines() if ":" in line
        )
    def gb(key):
        return int(info.get(key, "0 kB").split()[0]) / 1024 ** 2
    log(f"RAM: {gb('MemTotal'):.1f} GiB total, {gb('MemAvailable'):.1f} GiB available; "
        f"swap: {gb('SwapTotal'):.1f} GiB")


def bump_swap(size_gb: int = 32) -> None:
    """Optional: raise the hibernation-disabled swap ceiling for the full-scale run.

    The stock 16 GiB swap is enough for a careful pipeline but not for a retry of the
    22 GB peak the first Colab run hit.  Needs sudo -- we only run it when the caller
    asks (and only non-interactively, so it fails loudly rather than hanging).
    """
    swapfile = Path("/swapfile_fec")
    cmds = [
        ["sudo", "-n", "fallocate", "-l", f"{size_gb}G", str(swapfile)],
        ["sudo", "-n", "chmod", "600", str(swapfile)],
        ["sudo", "-n", "mkswap", str(swapfile)],
        ["sudo", "-n", "swapon", str(swapfile)],
    ]
    for cmd in cmds:
        try:
            sh(cmd)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            log(f"bump-swap step failed ({exc}); run manually:\n  " + "\n  ".join(" ".join(c) for c in cmds))
            return
    log(f"swap extended by {size_gb} GiB at {swapfile}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bump-swap", action="store_true",
                    help="add a 32 GiB swapfile (needs passwordless sudo; off by default)")
    ap.add_argument("--swap-gb", type=int, default=32)
    args = ap.parse_args()

    log("=== 00_setup ===")
    check_memory()
    ensure_starter_clone()
    ensure_env_file()

    missing = check_python_deps()
    if missing:
        log(f"MISSING python deps: {', '.join(missing)}")
        log(f"  install with:  uv pip install --python {CODE_DIR / '.venv'} -r {RUN_DIR.parent / 'requirements-local.txt'}")
        return 1

    counts = check_data()
    for name, n in counts.items():
        expected = {"diii_d_train": 7041, "diii_d_public_test": 874, "mast_public_test": 1206}.get(name)
        ok = n > 0 and (expected is None or n == expected)
        log(f"  {name}: {n} parquet files {'OK' if ok else 'PROBLEM'}"
            + (f" (expected {expected})" if expected and n != expected else ""))
    if counts[TRAIN_CONFIG] == 0:
        log("no training data -- run code/download_dataset.py or the notebook download cell")
        return 1

    if args.bump_swap:
        bump_swap(args.swap_gb)
    else:
        log("swap left as-is (use --bump-swap to extend it)")

    log("setup OK -- next: 01_extract_features.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
