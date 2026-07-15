"""Typed, fail-fast configuration dataclasses for NeuralDMD.

Kept import-light (no jax) so configs can be constructed and validated anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_STOKES: tuple[str, ...] = ("I", "Q", "U", "V")


@dataclass(frozen=True)
class StokesConfig:
    """Which Stokes parameters a polarized model represents.

    Attributes
    ----------
    stokes : tuple of str
        Ordered, unique Stokes parameters drawn from I/Q/U/V. Stokes I must be
        present -- it anchors the total intensity that the polarized components
        are defined relative to. Defaults to ``("I", "Q", "U")`` (linear
        polarization, no circular V).
    """

    stokes: tuple[str, ...] = ("I", "Q", "U")

    def __post_init__(self) -> None:
        """Coerce ``stokes`` to a tuple and validate it.

        Raises
        ------
        ValueError
            If ``stokes`` is empty, contains a value outside I/Q/U/V, has
            duplicates, or omits Stokes I.
        """
        object.__setattr__(self, "stokes", tuple(self.stokes))
        if not self.stokes:
            raise ValueError("stokes must be non-empty")
        unknown = [s for s in self.stokes if s not in _VALID_STOKES]
        if unknown:
            raise ValueError(f"unknown Stokes {unknown}; valid are {_VALID_STOKES}")
        if len(set(self.stokes)) != len(self.stokes):
            raise ValueError(f"duplicate Stokes in {self.stokes}")
        if "I" not in self.stokes:
            raise ValueError(f"Stokes I must be present, got {self.stokes}")
