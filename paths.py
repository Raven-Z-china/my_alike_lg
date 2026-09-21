#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Every path the tooling resolves, in one place.

Why this module exists
----------------------
Paths used to be literals scattered across fifteen files, absolute and naming one
particular machine.  That made the repository look like it shipped models only its
author could score, and it leaked that author's filesystem layout into committed
reports.  Nothing here is a portability nicety: a path that resolves to the wrong
file, or to no file, does not raise where the mistake is made - it raises inside a
conversion or, worse, produces a directory listing where a tensor was expected.

Resolution rules
----------------
Everything inside the repository resolves from the checkout's own location, so a
clone works anywhere and no absolute path is ever stored.

The three inputs that genuinely live outside it resolve in a fixed order:

1. the environment variable, if set (and a set-but-wrong value is an error, not a
   reason to fall through - see `_find`);
2. `data/<name>/`, `../<name>/`, or `<name>/` **relative to the checkout**, which
   is what makes a self-contained bundle work;
3. otherwise a clear failure naming the variable to set.

There is deliberately **no** absolute fallback.  Guessing a location that happens
to exist on one machine is how a run silently scores a different dataset than the
one intended, and it is the same mistake as hard-coding the path in the first
place.  Point the variable at it instead.

    HPATCHES_ROOT      HPatches sequences (open benchmark, used by every
                       documented number)
    GLUEFACTORY_ROOT   the training checkout (`pruning/finetune.py` and
                       `scripts/verify_vendored_matcher.py` only)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent

# --- inside the repository: no configuration needed -------------------------
WEIGHTS = REPO / "weights"
CHECKPOINTS = WEIGHTS / "checkpoints"
CKPT_9L = CHECKPOINTS / "alike_native_gl_s1_9L.tar"
CKPT_D7 = CHECKPOINTS / "alike_native_gl_s1_d7.tar"
TRAINED_CKPT = CKPT_9L          # the 9-layer model: ALIKE + the original matcher
SHIPPED_CKPT = CKPT_D7          # the 9->7 recovered matcher: what `optimized/` came from
REFS = REPO / "refs"


def outputs() -> Path:
    """Where tool output goes.  **Outside** the repository.

    No code in this repository reads anything back from here, so the tools never
    depend on a previous run's report and the directory can be deleted, moved or
    archived at any time without touching the code.  Reports are an artefact of
    measuring, not an input to anything.

    Override with `OUTPUTS_ROOT` to point the tools somewhere else.
    """
    root = os.environ.get("OUTPUTS_ROOT")
    p = Path(root) if root else REPO.parent / "alike_lightglue_outputs/reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _find(env_var, env_value, *candidates, what):
    """First existing candidate, or raise naming the environment variable.

    A set-but-wrong `env_var` is a hard error rather than a fall-through.  The
    alternative - ignore the variable and quietly use a different directory - is
    how a run ends up reporting numbers for a dataset nobody chose, which is the
    failure mode this whole file exists to make impossible.
    """
    if env_value:
        p = Path(env_value)
        if not p.exists():
            raise FileNotFoundError(
                f"{env_var} is set to {p} but that path does not exist")
        return p
    tried = []
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(
        f"cannot find {what}; set {env_var}=<path> "
        f"(tried: {', '.join(tried)})")


def hpatches():
    """HPatches sequences root, e.g. `<root>/i_ajuntament/1.ppm`.

    Call this WHERE THE DATA IS USED, not at module import: a clone with no
    datasets nearby must still be able to import every tool and read `--help`,
    and resolving at import time turns "this machine has no HPatches" into a
    failure to start rather than a failure to load images.
    """
    return _find("HPATCHES_ROOT", os.environ.get("HPATCHES_ROOT"),
                 REPO / "data/hpatches",
                 REPO.parent / "hpatches-sequences-release",
                 REPO.parent / "hpatches",
                 what="the HPatches sequences")


def gluefactory():
    """The training checkout, or None. Callers decide whether that is fatal."""
    root = os.environ.get("GLUEFACTORY_ROOT")
    for c in (root, REPO.parent / "glue-factory", REPO / "glue-factory"):
        if c and (Path(c) / "gluefactory").is_dir():
            return Path(c)
    return None


def scratch(name):
    """A file under the git-ignored scratch directory, created on demand."""
    REFS.mkdir(exist_ok=True)
    return REFS / name
