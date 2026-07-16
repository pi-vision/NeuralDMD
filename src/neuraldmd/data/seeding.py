"""Seeding helpers for synthetic data generation.

Deliberately free of heavy imports (no ehtim), so the reproducibility guarantees below
can be unit-tested in the fast CI lane rather than only where ehtim is installed.

Reproducibility here needed two separate fixes, and both failure modes were silent:

1. ehtim's ``observe`` seeds with ``if seed: np.random.seed(seed)``. ``seed=0`` is
   FALSY, so the most natural default disabled seeding entirely and every dataset was
   an unrepeatable draw. (ehtim's own docstring says "DO NOT set to 0!".)
   :func:`ehtim_seed` maps user seeds into a guaranteed-nonzero range.
2. ``PYTHONHASHSEED`` is randomised per process, which changes dict/set iteration order
   inside ehtim and hence the ORDER random numbers are consumed -- so even a correctly
   seeded run differs process to process (measured: same seed, two processes,
   ``max |dV| = 0.14`` Jy on 52 of 63 points). That one cannot be fixed from inside
   Python; ``PYTHONHASHSEED=0`` is exported by the SLURM template.

Either alone leaves the data irreproducible, which silently confounds any comparison
between runs that regenerate their data.
"""

from __future__ import annotations

#: Offset applied to user seeds so the result is never falsy. Prime, so distinct user
#: seeds stay distinct.
EHTIM_SEED_OFFSET = 1_000_003


def ehtim_seed(seed: int) -> int:
    """Map a user seed to a nonzero seed ehtim will actually honour.

    Parameters
    ----------
    seed : int
        User-facing seed. ``0`` is allowed and is the common default.

    Returns
    -------
    int
        A strictly positive seed. ehtim skips seeding for falsy values, so this must
        never return 0.
    """
    return int(seed) + EHTIM_SEED_OFFSET
