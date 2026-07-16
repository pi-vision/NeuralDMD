"""Polarized NeuralDMD: polarization parameterized as a *fraction of Stokes I*.

Stokes I is a NeuralDMD field; the linear polarization is derived from it as

    Q = m_l * I * cos(2 xi),    U = m_l * I * sin(2 xi),    V = m_c * I,

with the fractional magnitude and EVPA direction produced by additional NeuralDMD
raw fields passed through bounded activations. This ties polarization to the total
intensity (pol only where I is), enforces the physical bound ``P <= I`` by
construction, and -- via an output bias -- starts the source *unpolarized* and
grows polarization as the cross-hand data demand it. Crucially the model
*multiplies* by I (never divides), so the likelihood stays on absolute Stokes (or
correlation products).

Why this and not free (Q, U) fields: a physical Stokes vector obeys
``P = sqrt(Q^2+U^2+V^2) <= I`` and ``Q=U=V=0 where I=0``. Free (Q, U) fields
violate both -- they can fit sparse visibilities to chi^2 ~ 1 while producing
unphysical polarization images. ``(I, m_l, xi)`` is the same three degrees of
freedom in the coordinate chart where those constraints are a simple box on
``m_l``. See ``docs/polarization_parameterization.tex``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .config import StokesConfig
from .model import NeuralDMD, physical_intensities


class PolarizedNeuralDMD(eqx.Module):
    """Polarized model with polarization parameterized as a fraction of I.

    Sub-models (each a :class:`NeuralDMD` raw field, mapped through a bounded
    activation)::

        m_l              = sigmoid(raw - outshift) * scaling_ml   in [0, scaling_ml]
        (cos2xi, sin2xi) = (c_raw, s_raw) / ||(c_raw, s_raw)||     unit EVPA direction
        m_c              = (sigmoid(raw - outshift) - 0.5) * 2     in [-1, 1]   (V only)

    Because the EVPA direction is unit-normalized, ``P = sqrt(Q^2+U^2) = m_l*|I|``
    exactly, so ``P <= scaling_ml*|I| <= I`` holds strictly. ``outshift`` biases
    ``m_l`` small at init (a modest initial polarization). NB: too large an
    ``outshift`` saturates the sigmoid and starves its gradient, so polarization
    never grows -- with the gauge-fixed O(1) raw scale, ``~2`` works well (``10``
    leaves the fit unpolarized, RL/LR chi^2 huge).

    Attributes
    ----------
    intensity : NeuralDMD
        Stokes-I field (physical scaling via frame_max/frame_min; disk-pretrainable).
    frac, cos2xi, sin2xi : NeuralDMD
        Raw fields for ``m_l`` and the EVPA direction.
    circ : NeuralDMD or None
        Raw field for ``m_c`` (present iff V is modeled).
    stokes : tuple of str
        Stokes produced (static). ``outshift``, ``scaling_ml`` are static controls.
    """

    intensity: NeuralDMD
    frac: NeuralDMD
    cos2xi: NeuralDMD
    sin2xi: NeuralDMD
    circ: NeuralDMD | None
    stokes: tuple[str, ...] = eqx.field(static=True)
    outshift: float = eqx.field(static=True)
    scaling_ml: float = eqx.field(static=True)
    pol_param: str = eqx.field(static=True)

    def __init__(
        self,
        stokes: tuple[str, ...] | StokesConfig,
        r: int,
        *,
        key: jax.Array,
        outshift: float = 2.0,
        scaling_ml: float = 1.0,
        r_pol: int | None = None,
        pol_param: str = "fractional",
        pol_model_kwargs: dict | None = None,
        **model_kwargs,
    ):
        """Build the I field and the fractional-pol fields from independent split keys.

        Parameters
        ----------
        stokes : tuple of str or StokesConfig
            Must contain I; Q/U enable linear pol, V enables circular pol.
        r : int
            Complex DMD modes for the Stokes-I field.
        key : jax.Array
            PRNG key (split per field).
        outshift : float
            Sigmoid bias for ``m_l``/``m_c`` -> unpolarized initialization.
        scaling_ml : float
            Cap on the linear polarization fraction (``<= 1``).
        r_pol : int or None
            Complex DMD modes for the polarization fields (``m_l``, EVPA, ``m_c``).
            Defaults to ``r``. Set ``r_pol < r`` to deliberately starve the
            polarization's *temporal* capacity relative to I.
        pol_param : {"fractional", "direct", "iscaled"}
            Polarization parameterization.
            ``"fractional"`` (default) derives ``Q,U`` from ``(m_l, EVPA)`` fields
            (``P <= I`` by construction, but the *unit-normalized* EVPA direction
            must wind ``2 * (azimuthal mode)`` times around the source -- a
            topological constraint that makes an m>=2 EVPA spiral very hard to
            optimize, collapsing to m=1).
            ``"direct"`` makes ``Q,U`` independent signed :class:`NeuralDMD` fields
            in Stokes-I units (plan D8): no winding, but -- being untied from I --
            they leak an off-source polarized haze into the cross-hand null space
            that a soft ``p_weight``/support penalty must fight (an optimization
            barrier the penalty does not reliably clear).
            ``"iscaled"`` makes ``Q = I*tanh(q)``, ``U = I*tanh(u)`` with ``q,u``
            free signed fields: ties pol to I so NO off-source haze can form
            (support by construction) AND keeps ``q,u`` free of the winding
            obstruction (m>=2 representable). ``P <= sqrt(2) I``; residual
            ``P <= I`` supplied softly by ``p_weight``.
            ``"expm"`` is the matrix-exponential fractional form of Arras et al.
            (2025) applied on top of our I field: ``Q,U,V = I*tanh(p)*(q,u,v)/p``
            with ``p = sqrt(q^2+u^2+v^2)``. Same support + no-winding as ``iscaled``
            but with EXACT ``P <= I`` (``m = tanh(p) <= 1``, no ``p_weight``) and
            native Stokes V. (P<=I only where the I field is positive.)
            ``"expm_full"`` (recommended; exactly what resolve does) is the FULL
            matrix exponential ``X = exp([[s+v,q+iu],[q-iu,s-v]])`` with ``s`` the
            LOG-intensity field: ``I = e^s cosh(p)``, ``(Q,U,V) = e^s sinh(p)/p
            (q,u,v)``. X is positive semi-definite by construction, so ``I>0`` AND
            ``P<=I`` hold EVERYWHERE -- a physically complete coherency matrix for
            RIME. Requires ``--no-pretrain`` (log-I is incompatible with the
            physical-I disk pretrain). See ``docs/polarization_parameterization.tex``.
        pol_model_kwargs : dict or None
            Overrides applied on top of ``model_kwargs`` for the polarization
            fields only (e.g. ``{"hidden_size": 128, "num_layers": 2,
            "num_frequencies": 1}``). A smaller/lower-frequency spatial network
            band-limits the polarization *spatially* -- the pol structure is
            smoother than the total intensity, and the sparse cross-hand
            coverage cannot pin down a full-size field (it gets over-fit below
            the noise floor).
        **model_kwargs
            Forwarded to every :class:`NeuralDMD`.
        """
        cfg = stokes if isinstance(stokes, StokesConfig) else StokesConfig(tuple(stokes))
        self.stokes = cfg.stokes
        self.outshift = float(outshift)
        self.scaling_ml = float(scaling_ml)
        if pol_param not in ("fractional", "direct", "iscaled", "expm", "expm_full"):
            raise ValueError(
                "pol_param must be 'fractional', 'direct', 'iscaled', 'expm', or "
                f"'expm_full', got {pol_param!r}"
            )
        self.pol_param = str(pol_param)
        r_pol = r if r_pol is None else int(r_pol)
        pol_kwargs = {**model_kwargs, **(pol_model_kwargs or {})}
        want_v = "V" in self.stokes
        keys = jax.random.split(key, 5 if want_v else 4)
        self.intensity = NeuralDMD(r=r, key=keys[0], **model_kwargs)
        self.frac = NeuralDMD(r=r_pol, key=keys[1], **pol_kwargs)
        self.cos2xi = NeuralDMD(r=r_pol, key=keys[2], **pol_kwargs)
        self.sin2xi = NeuralDMD(r=r_pol, key=keys[3], **pol_kwargs)
        self.circ = NeuralDMD(r=r_pol, key=keys[4], **pol_kwargs) if want_v else None

    def stokes_fields(self, xy, times, frame_max: dict, frame_min: dict):
        """Physical per-Stokes image cubes and per-network modes (loss/eval interface).

        Parameters
        ----------
        xy : jax.Array
            ``(P, 2)`` pixel coordinates.
        times : jax.Array
            ``(T,)`` normalized frame times.
        frame_max, frame_min : dict of str -> float
            Output scaling for Stokes I (only the ``"I"`` entry is used; the pol
            fields are bounded by their own activations).

        Returns
        -------
        images : dict of str -> jax.Array
            ``{stokes: (P, T)}`` with ``Q=m_l*I*cos2xi``, ``U=m_l*I*sin2xi``,
            ``V=m_c*I``.
        modes : list of tuple
            ``(W0, W, b0, b)`` per sub-network, for the sparsity penalty.
        """
        if self.pol_param == "expm_full":
            # FULL matrix-exponential brightness matrix (Arras et al. 2025; the form
            # resolve uses). X = exp([[s+v, q+iu],[q-iu, s-v]]) with s the LOG
            # intensity field, giving
            #   I = e^s cosh(p),  (Q,U,V) = e^s (sinh(p)/p) (q,u,v),  p=|(q,u,v)|.
            # Because X = exp(Hermitian) it is POSITIVE SEMI-DEFINITE by construction:
            # I>0 EXACTLY (I is now log-parameterized -- no negativity penalty needed)
            # AND det X = I^2 - P^2 >= 0 (P<=I) EVERYWHERE, not just where a separate
            # I field happens to stay positive. This is the physically complete
            # coherency matrix a correct polarized RIME (gains, D-terms, feed rotation)
            # acts on. ``self.intensity`` supplies s (raw, O(1)); e^s is scaled by
            # frame_max["I"] so the brightness lands in physical units, and s is
            # clipped to avoid exp overflow (the flux anchor keeps it near 0). NB
            # incompatible with the physical-I disk pretrain -- run with --no-pretrain.
            s_raw, s_modes = physical_intensities(self.intensity, xy, times, 1.0, 0.0)
            q_raw, q_modes = physical_intensities(self.frac, xy, times, 1.0, 0.0)
            u_raw, u_modes = physical_intensities(self.cos2xi, xy, times, 1.0, 0.0)
            _, sp_modes = physical_intensities(self.sin2xi, xy, times, 1.0, 0.0)  # spare
            want_v = self.circ is not None
            if want_v:
                v_raw, v_modes = physical_intensities(self.circ, xy, times, 1.0, 0.0)
                p_sq = q_raw**2 + u_raw**2 + v_raw**2
            else:
                p_sq = q_raw**2 + u_raw**2
            p = jnp.sqrt(p_sq + 1e-12)
            base = frame_max["I"] * jnp.exp(jnp.clip(s_raw, -10.0, 10.0))
            sinh_over_p = jnp.sinh(p) / p  # smooth, -> 1 as p -> 0
            images = {"I": base * jnp.cosh(p)}
            modes = [s_modes, q_modes, u_modes, sp_modes]
            if "Q" in self.stokes:
                images["Q"] = base * sinh_over_p * q_raw
            if "U" in self.stokes:
                images["U"] = base * sinh_over_p * u_raw
            if want_v:
                images["V"] = base * sinh_over_p * v_raw
                modes.append(v_modes)
            return images, modes

        i_img, i_modes = physical_intensities(
            self.intensity, xy, times, frame_max["I"], frame_min["I"]
        )

        if self.pol_param == "direct":
            # Direct signed Q, U (plan D8): independent NeuralDMD fields scaled to
            # Stokes-I units. No m_l/EVPA rotation -> no topological winding
            # constraint on the EVPA direction, so an m=2 (spiral) pattern fits as
            # readily as the ring itself. P<=I and off-source pol suppression come
            # from the loss's soft ``p_weight`` penalty, not by construction.
            q_img, q_modes = physical_intensities(self.frac, xy, times, frame_max["I"], 0.0)
            u_img, u_modes = physical_intensities(self.cos2xi, xy, times, frame_max["I"], 0.0)
            # sin2xi kept live (spare) so the pol freeze/warmup update partition and
            # the modes/sparsity list stay identical across both parameterizations.
            _, s_modes = physical_intensities(self.sin2xi, xy, times, 1.0, 0.0)
            images = {"I": i_img}
            modes = [i_modes, q_modes, u_modes, s_modes]
            if "Q" in self.stokes:
                images["Q"] = q_img
            if "U" in self.stokes:
                images["U"] = u_img
            if self.circ is not None:
                v_img, v_modes = physical_intensities(self.circ, xy, times, frame_max["I"], 0.0)
                images["V"] = v_img
                modes.append(v_modes)
            return images, modes

        if self.pol_param == "iscaled":
            # I-scaled direct: Q = I*q, U = I*u with q,u free tanh-bounded signed
            # fields. Ties pol to I so Q,U -> 0 where I -> 0 -- NO off-source haze
            # can form (the support constraint holds by construction, not via a
            # penalty that must fight an optimization barrier) -- while keeping
            # q,u free of the unit-EVPA winding obstruction, so an m>=2 (spiral)
            # EVPA is representable. P = I*sqrt(q^2+u^2); tanh caps |q|,|u|<=1 so
            # P <= sqrt(2)*I, and the residual P<=I is the soft p_le_i penalty.
            q_raw, q_modes = physical_intensities(self.frac, xy, times, 1.0, 0.0)
            u_raw, u_modes = physical_intensities(self.cos2xi, xy, times, 1.0, 0.0)
            _, s_modes = physical_intensities(self.sin2xi, xy, times, 1.0, 0.0)  # spare
            q = jnp.tanh(q_raw)
            u = jnp.tanh(u_raw)
            images = {"I": i_img}
            modes = [i_modes, q_modes, u_modes, s_modes]
            if "Q" in self.stokes:
                images["Q"] = i_img * q
            if "U" in self.stokes:
                images["U"] = i_img * u
            if self.circ is not None:
                v_raw, v_modes = physical_intensities(self.circ, xy, times, 1.0, 0.0)
                images["V"] = i_img * jnp.tanh(v_raw)
                modes.append(v_modes)
            return images, modes

        if self.pol_param == "expm":
            # Matrix-exponential polarization (Arras et al. 2025, as used in
            # resolve): X = exp([[s+v, q+iu],[q-iu, s-v]]) yields
            # Q,U,V = I * tanh(p) * (q,u,v)/p with p = sqrt(q^2+u^2+v^2). Applied on
            # top of our (separately regularized) I field, this gives the exact
            # matrix-exp fractional/EVPA structure: EXACT P<=I (m = tanh(p) <= 1),
            # support (Q,U,V prop I -> 0 where I=0), and no winding (q,u,v are free
            # fields; tanh(p)/p is smooth through p=0). Cleaner P<=I than iscaled
            # (exact vs sqrt(2) bound) and V-capable; no p_le_i penalty required.
            q_raw, q_modes = physical_intensities(self.frac, xy, times, 1.0, 0.0)
            u_raw, u_modes = physical_intensities(self.cos2xi, xy, times, 1.0, 0.0)
            _, s_modes = physical_intensities(self.sin2xi, xy, times, 1.0, 0.0)  # spare
            want_v = self.circ is not None
            if want_v:
                v_raw, v_modes = physical_intensities(self.circ, xy, times, 1.0, 0.0)
                p_sq = q_raw**2 + u_raw**2 + v_raw**2
            else:
                p_sq = q_raw**2 + u_raw**2
            p = jnp.sqrt(p_sq + 1e-12)
            m_over_p = jnp.tanh(p) / p  # -> m/p; smooth, -> 1 as p -> 0
            images = {"I": i_img}
            modes = [i_modes, q_modes, u_modes, s_modes]
            if "Q" in self.stokes:
                images["Q"] = i_img * m_over_p * q_raw
            if "U" in self.stokes:
                images["U"] = i_img * m_over_p * u_raw
            if want_v:
                images["V"] = i_img * m_over_p * v_raw
                modes.append(v_modes)
            return images, modes

        ml_raw, ml_modes = physical_intensities(self.frac, xy, times, 1.0, 0.0)
        c_raw, c_modes = physical_intensities(self.cos2xi, xy, times, 1.0, 0.0)
        s_raw, s_modes = physical_intensities(self.sin2xi, xy, times, 1.0, 0.0)
        # m_l in [0, scaling_ml]; outshift -> m_l ~ 0 at init (unpolarized)
        m_l = jax.nn.sigmoid(ml_raw - self.outshift) * self.scaling_ml
        # EVPA direction as a UNIT vector (cos2xi^2 + sin2xi^2 = 1), so that
        # P = sqrt(Q^2+U^2) = m_l*|I| exactly -> P <= scaling_ml*|I| <= I (strict).
        norm = jnp.sqrt(c_raw**2 + s_raw**2 + 1e-8)
        cos2xi = c_raw / norm
        sin2xi = s_raw / norm

        images = {"I": i_img}
        modes = [i_modes, ml_modes, c_modes, s_modes]
        if "Q" in self.stokes:
            images["Q"] = m_l * i_img * cos2xi
        if "U" in self.stokes:
            images["U"] = m_l * i_img * sin2xi
        if self.circ is not None:
            v_raw, v_modes = physical_intensities(self.circ, xy, times, 1.0, 0.0)
            m_c = (jax.nn.sigmoid(v_raw - self.outshift) - 0.5) * 2.0
            images["V"] = m_c * i_img
            modes.append(v_modes)
        return images, modes

    @property
    def i_submodel(self) -> NeuralDMD:
        """The Stokes-I :class:`NeuralDMD` (target of disk-template pretraining)."""
        return self.intensity

    def replace_i_submodel(self, new_i: NeuralDMD) -> PolarizedNeuralDMD:
        """Return a copy with the Stokes-I field replaced (e.g. by a pretrained one)."""
        return eqx.tree_at(lambda m: m.intensity, self, new_i)
