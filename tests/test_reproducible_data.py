"""Synthetic data must be reproducible, or no comparison between runs means anything.

Two independent bugs made it irreproducible, and both were silent:

1. ehtim's ``observe`` seeds with ``if seed: np.random.seed(seed)``. ``seed=0`` is
   FALSY, so passing the most natural default disabled seeding altogether. (ehtim's
   docstring does say "DO NOT set to 0!".)
2. Python randomises ``PYTHONHASHSEED`` per process, which changes dict/set iteration
   order inside ehtim and hence the ORDER random numbers are consumed -- so even a
   correctly seeded RNG produced different noise per process.

Consequence: every run that regenerated its own data drew a fresh noise realization,
and cross-run comparisons silently mixed the effect under test with realization noise.
"""

import numpy as np
import pytest

# from .seeding, NOT .generation: generation imports ehtim at module scope, which is
# absent in the fast CI lanes ("not ehtim"). Importing it here would fail COLLECTION
# and take the whole lane down, not just skip these tests.
from neuraldmd.data.seeding import EHTIM_SEED_OFFSET, ehtim_seed


def test_ehtim_seed_is_never_falsy():
    """The whole bug: a falsy seed silently disables seeding in ehtim."""
    for s in (0, 1, 7, 42):
        assert ehtim_seed(s), f"seed {s} maps to a falsy value -- ehtim would not seed"
        assert ehtim_seed(s) > 0


def test_ehtim_seed_keeps_distinct_seeds_distinct():
    """Different user seeds must still give different data; the offset must not collapse
    them onto one value."""
    mapped = [ehtim_seed(s) for s in range(16)]
    assert len(set(mapped)) == 16


def test_ehtim_seed_is_a_pure_offset():
    """Documents the mapping so a future reader can reproduce an old dataset."""
    assert ehtim_seed(0) == EHTIM_SEED_OFFSET
    assert ehtim_seed(5) == EHTIM_SEED_OFFSET + 5


def test_zero_is_the_dangerous_case_and_is_handled():
    """seed=0 is the default in our SLURM template and the natural default anywhere.
    It MUST survive the mapping as a usable, nonzero, reproducible seed."""
    assert bool(0) is False  # this is why the original code silently did nothing
    assert bool(ehtim_seed(0)) is True


@pytest.mark.slow
@pytest.mark.ehtim
def test_generation_is_reproducible_in_process():
    """Same seed, same data. Cross-PROCESS reproducibility additionally needs
    PYTHONHASHSEED pinned (see the module docstring); that is set in the SLURM
    template and cannot be asserted from inside one interpreter."""
    import hashlib
    import tempfile

    pytest.importorskip("ehtim")
    from neuraldmd.data.generation import generate_polarized_dataset

    kw = dict(
        stokes=("RR", "LL", "RL", "LR"),
        basis="circular",
        npix=16,
        num_frames=4,
        fov_uas=200,
        frac_pol=0.2,
        truth_model="mring_hs",
        direction="CW",
    )
    h = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as d:
            op = generate_polarized_dataset(d, seed=0, **kw)
            h.append(hashlib.md5(np.ascontiguousarray(op.targets["RR"])).hexdigest())
    assert h[0] == h[1], "same seed produced different data"
