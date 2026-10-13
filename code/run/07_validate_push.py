"""07_validate_push.py -- run the official validator and (optionally) the official pusher.

    python code/run/07_validate_push.py --validate-only --max-shots 5
    python code/run/07_validate_push.py --validate-only            # full npz check
    python code/run/07_validate_push.py --push --repo <user>/fusion-eq-predictions

Both subcommands shell out to the **unmodified** starter-kit CLIs
(``validate_submission.py`` / ``push_predictions.py``), so what we check locally is
what Codabench runs.  Pushing needs two different tokens (see CLAUDE.md): a write
token already logged in via ``huggingface-cli login`` for the upload, and a
fine-grained read-only token passed with ``--read-token`` and shipped inside the zip.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fec_common.config import CODE_DIR, STARTER_DIR, SUBMISSION_DIR, ensure_dirs, ensure_starter, log  # noqa: E402

CFG_FOR = {
    "diii_d_public_test.npz": "diii_d_public_test",
    "mast_public_test.npz": "mast_public_test",
}


_SECRET_FLAGS = {"--read-token", "--token"}


def _redact(cmd: list[str]) -> list[str]:
    """Replace the value right after any flag in _SECRET_FLAGS before this ever hits a log
    line -- a prior version logged the raw --read-token value in plaintext (see git history /
    CLAUDE.md's Credentials section on why that token must not leak)."""
    out = list(cmd)
    for i, tok in enumerate(out):
        if tok in _SECRET_FLAGS and i + 1 < len(out):
            out[i + 1] = "<redacted>"
    return out


def run(cmd: list[str], cwd: Path) -> None:
    import threading
    import time

    log("$ " + " ".join(str(c) for c in _redact(cmd)))
    proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd))

    # Used by push_predictions.py (a single upload, no per-shot loop to show real progress
    # for) -- validate() below gets a real tqdm bar instead, see its own docstring.
    t0 = time.time()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            log(f"  ... still running ({time.time()-t0:.0f}s elapsed, no output expected "
                f"until it finishes -- this is normal for the vendored scripts)")

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    proc.wait()
    stop.set()

    if proc.returncode:
        raise SystemExit(f"command failed with exit code {proc.returncode}")


# Expected shot counts per the README -- used only to size the tqdm bar below, not to validate
# anything (the vendored validate() function still does all the real checking).
_EXPECTED_SHOTS = {"diii_d_public_test": 874, "mast_public_test": 1206}


def validate(npz: Path, config: str, max_shots: int) -> None:
    """Runs the vendored validate_submission.validate() IN-PROCESS (not subprocess) so we can
    wrap its internal ``datasets.load_dataset(...)`` return value in a real per-shot tqdm bar
    -- the vendored function itself prints nothing for the whole streaming loop. The vendored
    function's CODE is untouched; only datasets.load_dataset is monkeypatched for the duration
    of this call. IMPORTANT: unlike an earlier (reverted) version of this function, this does
    NOT force streaming=False/num_proc -- that downloaded+cached a second, duplicate copy of
    data already sitting locally under code/fusion-equilibrium-challenge/data/. The kwargs
    passed through here are exactly what validate_submission.py's own code already uses
    (streaming=True); we only wrap the returned iterable, never change how it fetches data."""
    import datasets
    from tqdm.auto import tqdm

    sys.path.insert(0, str(STARTER_DIR))
    import validate_submission as vs  # noqa: PLC0415 -- vendored, imported lazily on purpose

    total = max_shots if max_shots else _EXPECTED_SHOTS.get(config)
    real_load_dataset = datasets.load_dataset

    def patched_load_dataset(*args, **kwargs):
        ds = real_load_dataset(*args, **kwargs)  # kwargs unchanged -- still whatever vs.validate() passed
        return tqdm(ds, total=total, desc=f"validating {config}", unit="shot")

    datasets.load_dataset = patched_load_dataset
    try:
        rc = vs.validate(npz, config, max_shots)
    finally:
        datasets.load_dataset = real_load_dataset
    if rc:
        raise SystemExit(f"validate_submission.py reported {rc} problem(s) for {npz.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate-only", action="store_true", default=True)
    ap.add_argument("--push", action="store_true",
                    help="push without prompting (e.g. for non-interactive/automation use)")
    ap.add_argument("--no-push", action="store_true",
                    help="validate only, never prompt to push")
    ap.add_argument("--max-shots", type=int, default=0, help="0 = validate every shot")
    ap.add_argument("--npz", nargs="*", default=None,
                    help="npz files (default: everything in artifacts/submission)")
    ap.add_argument("--repo", default=None,
                    help="<user>/fusion-eq-predictions for --push (or set FEC_HF_REPO)")
    ap.add_argument("--read-token", default=None,
                    help="fine-grained read-only token (or set HF_READ_TOKEN)")
    ap.add_argument("--dry-run", action="store_true", help="push_predictions --dry-run")
    args = ap.parse_args()

    import os

    from dotenv import load_dotenv

    load_dotenv(CODE_DIR / ".env", override=True)  # .env is the source of truth for these --
                                    # override=True because a stale HF_TOKEN can be sitting in
                                    # the shell env from an old session (e.g. a leftover export
                                    # from before it was removed from ~/.bashrc), and
                                    # load_dotenv's default override=False would silently keep
                                    # using that stale value instead of the current .env one

    hf_token = (os.getenv("HF_TOKEN") or "").strip()
    if hf_token:
        # Anonymous Hub requests get a much stricter rate limit, which is what was causing the
        # "read operation timed out" retries while streaming the (public, read-only) reference
        # dataset during validate() -- log in with the write token so huggingface_hub/datasets
        # pick it up for every request in this process, not just the ones we control directly.
        # .strip() matters: a trailing newline/CR from .env's line ending becomes part of the
        # token string otherwise, and a token with a stray trailing byte reads as "invalid" to
        # the API even though the token itself (e.g. checked via `hf auth login`) is fine.
        # Non-fatal: a still-bad token must not crash validation, which works fine (just
        # slower, subject to anonymous rate limits) without it.
        import huggingface_hub

        os.environ["HF_TOKEN"] = hf_token
        try:
            huggingface_hub.login(token=hf_token, add_to_git_credential=False)
            log("  authenticated to HF Hub (fewer rate-limit retries while streaming reference data)")
        except Exception as exc:
            log(f"  WARNING: HF_TOKEN in .env is invalid, continuing unauthenticated ({exc})")

    ensure_dirs()
    ensure_starter()
    npzs = [Path(p) for p in args.npz] if args.npz else sorted(SUBMISSION_DIR.glob("*.npz"))
    if not npzs:
        raise SystemExit(f"no .npz under {SUBMISSION_DIR} -- run 06_build_submission.py")

    log("=== 07_validate ===")
    for npz in npzs:
        config = CFG_FOR.get(npz.name)
        if config is None:
            log(f"  skipping {npz.name} (unknown config)")
            continue
        validate(npz, config, args.max_shots)
        log(f"  {npz.name}: VALID")

    if args.no_push:
        log("validation only (--no-push given)")
        return 0

    do_push = args.push
    if not do_push:
        # Validation above (esp. the ~1200-shot MAST config) can take close to an hour --
        # ask here instead of making the user re-run with --push, which would re-validate
        # everything from scratch for a second time just to get to the push.
        answer = input("All files validated. Push to the HF repo now? [y/N]: ").strip().lower()
        do_push = answer in ("y", "yes")
    if not do_push:
        log("validation only (not pushing)")
        return 0

    repo = args.repo or os.getenv("FEC_HF_REPO")
    if not repo:
        raise SystemExit("--push needs --repo <user>/fusion-eq-predictions (or set FEC_HF_REPO)")

    read_token = args.read_token or os.getenv("HF_READ_TOKEN")
    if not read_token:
        raise SystemExit(
            "--push needs a fine-grained READ-ONLY token for the predictions repo "
            "(pass --read-token or set HF_READ_TOKEN). The upload itself uses the "
            "write token from `huggingface-cli login` -- do not reuse it here."
        )

    log("=== 07_push ===")
    out_zip = SUBMISSION_DIR / "submission_pointer.zip"
    cmd = [sys.executable, "push_predictions.py",
           "--repo", repo, "--read-token", read_token,
           "--dir", SUBMISSION_DIR, "--out", out_zip]
    if args.dry_run:
        cmd.append("--dry-run")
    run(cmd, cwd=STARTER_DIR)
    log(f"upload zip ready: {out_zip} -- submit it on Codabench (competition 17456)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
