"""Synthesize an m-ring + orbiting hot spot movie (Sgr A*-like).

The scene is a thick ring with a mild azimuthal brightness asymmetry (an
"m-ring") plus a compact Gaussian hot spot orbiting it — the standard test
case for dynamical black-hole imaging. Frame synthesis is pure NumPy; ehtim
is only needed to wrap the frames into an ehtim Movie and save it in the
hdf5 format that eht2017/data_generation.py consumes.

Default parameters reproduce the movie used in the NeuralDMD experiments:
ring radius 23 uas (FWHM 25 uas, |beta_1| = 0.12), hot spot of FWHM 28 uas
and ~0.28 Jy orbiting at 25.6 uas with an 80-minute period, over a 6-hour
observation (4.5 orbits).
"""

import numpy as np

# Sgr A* at 1.3 mm during the April 2017 EHT campaign
SGRA = dict(ra=17.761121055814954, dec=-29.0078430557251, rf=227.07e9, mjd=57854)
RADPERUAS = np.pi / 180.0 / 3600.0 / 1e6


# hoisted from a function default (ruff B008): m=1 azimuthal asymmetry phasor
_DEFAULT_BETA1 = 0.12 * np.exp(1j * np.deg2rad(35.0))


def _grid(npix, fov_uas):
    x = (np.arange(npix) - npix / 2.0) * (fov_uas / npix)
    return np.meshgrid(x, x)  # X, Y in uas


def mring_image(
    npix=200,
    fov_uas=200.0,
    r_ring=23.0,  # uas
    width_fwhm=25.0,  # uas
    beta1=_DEFAULT_BETA1,  # m=1 azimuthal asymmetry
    flux=2.47,  # Jy
):
    """Static m-ring: radial Gaussian ring profile x azimuthal modulation."""
    X, Y = _grid(npix, fov_uas)
    R = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)

    radial = np.exp(-4 * np.log(2) * (R - r_ring) ** 2 / width_fwhm**2)
    azimuthal = 1.0 + 2.0 * np.real(beta1 * np.exp(-1j * phi))
    ring = radial * np.clip(azimuthal, 0.0, None)
    return ring * (flux / ring.sum())


def hotspot_frame(
    t_hr,
    npix=200,
    fov_uas=200.0,
    r_orbit=25.6,
    period_min=80.0,
    phase0_deg=178.0,
    direction=-1,
    spot_fwhm=28.0,
    spot_flux=0.28,
):
    """Gaussian hot spot at its orbital position at time t_hr (hours).

    direction is the sense of rotation in array coordinates; -1 reproduces
    the counterclockwise-on-sky spot (east-west flips in the sky display).
    """
    X, Y = _grid(npix, fov_uas)
    phase = np.deg2rad(phase0_deg) + direction * 2 * np.pi * (t_hr * 60.0) / period_min
    x0 = r_orbit * np.cos(phase)
    y0 = r_orbit * np.sin(phase)

    sigma = spot_fwhm / 2.355
    spot = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))
    return spot * (spot_flux / spot.sum())


def make_frames(
    num_frames=411,
    tstart_hr=9.0,
    tstop_hr=15.0,
    npix=200,
    fov_uas=200.0,
    **kwargs,
):
    """(T, npix, npix) movie of the m-ring + orbiting hot spot, plus times.

    kwargs are split between mring_image and hotspot_frame by name.
    """
    import inspect

    ring_keys = set(inspect.signature(mring_image).parameters)
    spot_keys = set(inspect.signature(hotspot_frame).parameters)
    ring_kwargs = {k: v for k, v in kwargs.items() if k in ring_keys}
    spot_kwargs = {k: v for k, v in kwargs.items() if k in spot_keys}
    unknown = set(kwargs) - ring_keys - spot_keys
    if unknown:
        raise TypeError(f"Unknown parameters: {unknown}")

    times = np.linspace(tstart_hr, tstop_hr, num_frames)
    ring = mring_image(npix=npix, fov_uas=fov_uas, **ring_kwargs)

    frames = np.empty((num_frames, npix, npix), dtype=np.float64)
    for i, t in enumerate(times):
        frames[i] = ring + hotspot_frame(t, npix=npix, fov_uas=fov_uas, **spot_kwargs)
    return frames, times


def polarization_maps(
    npix=200,
    fov_uas=200.0,
    frac_pol=0.3,
    evpa_winding=1,
    evpa_offset_deg=0.0,
):
    """Static fractional-polarization and EVPA maps for the polarized test movie.

    The linear-polarization pattern is fixed in the sky plane; it multiplies the
    time-varying Stokes I of each frame. The electric-vector position angle winds
    azimuthally, ``chi = evpa_winding * phi + offset`` (a smooth spiral), and the
    fractional polarization ``m`` is uniform -- a simple, exactly recoverable
    truth for which ``sqrt(Q^2 + U^2) / I = m`` and ``0.5 * atan2(U, Q) = chi``.

    Parameters
    ----------
    npix : int
        Grid side length.
    fov_uas : float
        Field of view [uas].
    frac_pol : float
        Uniform linear fractional polarization ``m``, in ``[0, 1]``.
    evpa_winding : int
        Number of EVPA turns around the ring (1 = one full rotation).
    evpa_offset_deg : float
        Constant EVPA offset [degrees].

    Returns
    -------
    m, chi : numpy.ndarray
        Two ``(npix, npix)`` arrays: fractional polarization and EVPA [radians].
    """
    X, Y = _grid(npix, fov_uas)
    phi = np.arctan2(Y, X)
    m = np.full((npix, npix), float(frac_pol))
    chi = evpa_winding * phi + np.deg2rad(evpa_offset_deg)
    return m, chi


def make_polarized_frames(
    num_frames=411,
    tstart_hr=9.0,
    tstop_hr=15.0,
    npix=200,
    fov_uas=200.0,
    frac_pol=0.3,
    evpa_winding=1,
    evpa_offset_deg=0.0,
    **kwargs,
):
    """Polarized m-ring + hot-spot movie in Stokes (I, Q, U); V = 0.

    Stokes I comes from :func:`make_frames`; a static polarization field
    (:func:`polarization_maps`) then gives, per frame, ``Q = m * I * cos(2*chi)``
    and ``U = m * I * sin(2*chi)`` (EVPA ``chi = 0.5*atan2(U, Q)``, matching
    :func:`neuraldmd.physics.stokes.evpa`). V is identically zero and not returned.

    Parameters
    ----------
    num_frames, tstart_hr, tstop_hr, npix, fov_uas
        Movie sampling and grid, as in :func:`make_frames`.
    frac_pol, evpa_winding, evpa_offset_deg
        Polarization-field parameters (see :func:`polarization_maps`).
    **kwargs
        Forwarded to :func:`make_frames` (m-ring / hot-spot geometry).

    Returns
    -------
    I, Q, U, times : numpy.ndarray
        ``I``, ``Q``, ``U`` each ``(T, npix, npix)``; ``times`` ``(T,)`` [hours].
    """
    intensity, times = make_frames(
        num_frames=num_frames,
        tstart_hr=tstart_hr,
        tstop_hr=tstop_hr,
        npix=npix,
        fov_uas=fov_uas,
        **kwargs,
    )
    m, chi = polarization_maps(
        npix=npix,
        fov_uas=fov_uas,
        frac_pol=frac_pol,
        evpa_winding=evpa_winding,
        evpa_offset_deg=evpa_offset_deg,
    )
    q = m * intensity * np.cos(2 * chi)
    u = m * intensity * np.sin(2 * chi)
    return intensity, q, u, times


def _add_radial_pol(im, linpol_frac, circpol_frac, qu_shift=np.pi / 2):
    """Add a spiral-EVPA linear polarization + uniform circular pol to an ehtim
    Image in place (the ``mring+hs`` polarization pattern).

    Uses the ehtim ``qimage``/``uimage`` convention with a global EVPA twist
    ``qu_shift`` split symmetrically between Q and U:
    ``Q = m·I·cos(2(φ + qu_shift/2))``, ``U = m·I·sin(2(φ − qu_shift/2))`` where
    ``φ = atan2(y, x)`` and ``m = linpol_frac``. With ``qu_shift = π/2`` (default)
    the EVPA is a constant 45° offset from radial -- a spiral ("twisty") pattern
    that winds around the ring; ``qu_shift`` rotates it globally (``0`` gives a
    radial B-field / tangential EVPA). ``V = circpol_frac · I``.

    Parameters
    ----------
    im : ehtim.image.Image
        Image whose ``qvec``/``uvec``/``vvec`` are set in place.
    linpol_frac : float
        Fractional linear polarization ``|m|``.
    circpol_frac : float
        Fractional circular polarization ``|v|`` (uniform, ``V = v·I``).
    qu_shift : float, optional
        Global EVPA twist [rad]. Default ``π/2`` (spiral, 45° from radial).

    Returns
    -------
    None
        ``im`` is modified in place.
    """
    npix = im.xdim
    grid = np.linspace(-1.0, 1.0, npix)
    xg, yg = np.meshgrid(grid, grid)
    phi = np.angle(xg + 1j * yg)
    intensity = np.asarray(im.imvec).reshape(im.ydim, im.xdim)
    im.qvec = (linpol_frac * intensity * np.cos(2 * (phi + qu_shift / 2))).flatten()
    im.uvec = (linpol_frac * intensity * np.sin(2 * (phi - qu_shift / 2))).flatten()
    im.vvec = circpol_frac * np.asarray(im.imvec)


def make_mring_hs_pol_movie(
    npix=50,
    fov_uas=200.0,
    num_frames=64,
    tstart_hr=9.0,
    tstop_hr=15.0,
    period_min=80.0,
    direction="CW",
    total_flux=2.7,
    hs_flux=0.3,
    pa_deg=120.0,
    diameter_uas=52.0,
    alpha_uas=15.0,
    beta1_abs=0.23,
    ring_radius_uas=26.0,
    hs_fwhm_uas=20.0,
    linpol_frac=0.2,
    circpol_frac=0.002,
    qu_shift_rad=np.pi / 2,
    phase0_deg=0.0,
    source="SgrA",
    **sky,
):
    """Canonical polarized ``mring+hsCW`` movie via an ehtim thick m-ring.

    Reproduces the standard Sgr A* dynamics ``mring+hsCW`` test model: an ehtim
    thick m-ring (``add_thick_mring``) with diameter, width (``alpha``) and an
    m=1 azimuthal asymmetry, carrying a **spiral-EVPA** linear polarization (the
    EVPA is a constant 45° offset from radial, winding around the ring; see
    :func:`_add_radial_pol`) and a small uniform circular polarization, plus an
    **unpolarized** Gaussian hot spot orbiting the ring. Defaults are the
    standard model's parameter values. Requires ehtim.

    Parameters
    ----------
    npix, fov_uas, num_frames, tstart_hr, tstop_hr
        Image grid, field of view [uas], and observation sampling.
    period_min : float
        Hot-spot orbital period [minutes].
    direction : {"CW", "CCW"}
        Orbital sense.
    total_flux, hs_flux : float
        Total and hot-spot flux [Jy] (ring flux = total - hs).
    pa_deg, diameter_uas, alpha_uas, beta1_abs : float
        m-ring position angle [deg], diameter [uas], thickness/width [uas], and
        m=1 asymmetry amplitude.
    ring_radius_uas, hs_fwhm_uas : float
        Hot-spot orbital radius and Gaussian FWHM [uas].
    linpol_frac, circpol_frac : float
        Fractional linear (|m|) and circular (|v|) polarization of the ring.
    qu_shift_rad : float
        Global EVPA twist [rad]; ``π/2`` gives the spiral (45°-from-radial)
        pattern of the standard model.
    phase0_deg : float
        Initial hot-spot orbital phase [deg].
    source : str
        Source name. **sky overrides ra/dec/rf/mjd.

    Returns
    -------
    ehtim.movie.Movie
        The polarized movie (each frame carries I, Q, U, V).
    """
    import ehtim as eh

    sky = {**SGRA, **sky}
    fov = fov_uas * eh.RADPERUAS
    times = np.linspace(tstart_hr, tstop_hr, num_frames)
    mring_flux = total_flux - hs_flux
    # linear orbital-angle ramp over the window (n_loops full orbits; +1 for CW)
    n_loops = (tstop_hr - tstart_hr) / (period_min / 60.0)
    sign = 1.0 if str(direction).upper() == "CW" else -1.0
    angles = np.deg2rad(phase0_deg) + np.linspace(0.0, sign, num_frames) * 2 * np.pi * n_loops
    r = ring_radius_uas * eh.RADPERUAS
    fwhm = hs_fwhm_uas * eh.RADPERUAS

    frames = []
    for angle, t in zip(angles, times, strict=False):
        model = eh.model.Model(ra=sky["ra"], dec=sky["dec"], rf=sky["rf"], source=source)
        im = model.add_thick_mring(
            F0=mring_flux,
            d=diameter_uas * eh.RADPERUAS,
            alpha=alpha_uas * eh.RADPERUAS,
            x0=0.0,
            y0=0.0,
            beta_list=[beta1_abs * np.exp(-1j * np.deg2rad(-pa_deg))],
        ).make_image(fov, npix)
        im.mjd = sky["mjd"]
        _add_radial_pol(im, linpol_frac, circpol_frac, qu_shift_rad)  # spiral ring pol
        # unpolarized orbiting hot spot
        im = im.add_gauss(
            hs_flux, [fwhm, fwhm, 0.0, r * np.cos(angle), r * np.sin(angle)], pol=None
        )
        im.time = float(t)
        frames.append(im)

    return eh.movie.merge_im_list(frames, interp="linear", bounds_error=True)


def make_mring_hs_polarized_movie(
    npix=50,
    fov_uas=200.0,
    num_frames=64,
    tstart_hr=9.0,
    tstop_hr=15.0,
    period_min=80.0,
    direction="CW",
    total_flux=2.7,
    hs_flux=0.3,
    pa_deg=120.0,
    diameter_uas=52.0,
    alpha_uas=15.0,
    beta1_abs=0.23,
    ring_radius_uas=26.0,
    hs_fwhm_uas=20.0,
    crescent_linpol=0.05,
    crescent_circpol=0.001,
    hs_linpol=0.2,
    hs_circpol=0.002,
    phase0_deg=0.0,
    source="SgrA",
    **sky,
):
    """m-ring + a **polarized** orbiting hot spot: the ``mring+hs-pol`` test model.

    Ported from the ehteval model of the same name. Unlike ``mring+hsCW`` (a static
    spiral EVPA with an *unpolarized* spot), here the ring carries a weak radial EVPA
    (5%) and the hot spot carries its own stronger polarization (20%) whose EVPA
    rotates with the orbital angle. The polarized structure therefore **moves with the
    spot**: this is the only model in the suite with both Stokes-I dynamics AND
    polarization dynamics, which makes it the strictest of the four.

    Score with :func:`neuraldmd.evaluation.beta2_dynamics_error` -- the polarization
    is time-varying, so the time-averaged ``beta2`` is not the right measure.

    Parameters
    ----------
    npix, fov_uas, num_frames, tstart_hr, tstop_hr
        Image grid, field of view [uas], and observation sampling.
    period_min : float
        Hot-spot orbital period [minutes].
    direction : {"CW", "CCW"}
        Orbital sense.
    total_flux, hs_flux : float
        Total and hot-spot flux [Jy]; the ring carries the remainder.
    pa_deg, diameter_uas, alpha_uas, beta1_abs
        Thick m-ring geometry.
    ring_radius_uas, hs_fwhm_uas : float
        Hot-spot orbital radius and FWHM [uas].
    crescent_linpol, crescent_circpol : float
        Ring linear / circular polarization fractions (weak).
    hs_linpol, hs_circpol : float
        Hot-spot linear / circular polarization fractions (strong, EVPA follows the orbit).
    phase0_deg : float
        Initial orbital phase [deg].
    source : str
        Source name for the ehtim model.
    **sky
        Overrides for the default sky coordinates (see ``SGRA``).

    Returns
    -------
    ehtim.movie.Movie
        The polarized movie (each frame carries I, Q, U, V).
    """
    import ehtim as eh

    sky = {**SGRA, **sky}
    fov = fov_uas * eh.RADPERUAS
    times = np.linspace(tstart_hr, tstop_hr, num_frames)
    mring_flux = total_flux - hs_flux
    n_loops = (tstop_hr - tstart_hr) / (period_min / 60.0)
    sign = 1.0 if str(direction).upper() == "CW" else -1.0
    angles = np.deg2rad(phase0_deg) + np.linspace(0.0, sign, num_frames) * 2 * np.pi * n_loops
    r = ring_radius_uas * eh.RADPERUAS
    fwhm = hs_fwhm_uas * eh.RADPERUAS

    # Q = I*m*cos(2 chi), U = I*m*sin(2 chi) (ehtim's qimage/uimage, which were
    # removed in ehtim 1.4; inlined here so the model does not depend on that API)
    lin = np.linspace(-1.0, 1.0, npix)
    gx, gy = np.meshgrid(lin, lin)
    grid_angle = np.angle(gx + 1j * gy).flatten()

    frames = []
    for angle, t in zip(angles, times, strict=False):
        model = eh.model.Model(ra=sky["ra"], dec=sky["dec"], rf=sky["rf"], source=source)
        im = model.add_thick_mring(
            F0=mring_flux,
            d=diameter_uas * eh.RADPERUAS,
            alpha=alpha_uas * eh.RADPERUAS,
            x0=0.0,
            y0=0.0,
            beta_list=[beta1_abs * np.exp(-1j * np.deg2rad(-pa_deg))],
        ).make_image(fov, npix)
        im.mjd = sky["mjd"]

        # weak radial EVPA on the ring
        ivec = np.asarray(im.imvec)
        im.qvec = ivec * crescent_linpol * np.cos(2 * (grid_angle + np.pi / 2))
        im.uvec = ivec * crescent_linpol * np.sin(2 * grid_angle)
        im.vvec = crescent_circpol * ivec

        # the hot spot, built separately so its EVPA can follow the orbit
        gauss = eh.image.make_empty(
            npix=npix, fov=fov, ra=sky["ra"], dec=sky["dec"], rf=sky["rf"], source=source
        )
        gauss = gauss.add_gauss(hs_flux, [fwhm, fwhm, 0.0, 0.0, 0.0], pol=None)
        gvec = np.asarray(gauss.imvec)
        gauss.qvec = gvec * hs_linpol * np.cos(2 * (-angle))
        gauss.uvec = gvec * hs_linpol * np.sin(2 * (-angle))
        gauss.vvec = hs_circpol * gvec
        gauss = gauss.shift_fft([r * np.cos(angle), r * np.sin(angle)])

        im.imvec = np.asarray(im.imvec) + np.asarray(gauss.imvec)
        im.qvec = np.asarray(im.qvec) + np.asarray(gauss.qvec)
        im.uvec = np.asarray(im.uvec) + np.asarray(gauss.uvec)
        im.vvec = np.asarray(im.vvec) + np.asarray(gauss.vvec)
        im.time = float(t)
        frames.append(im)

    return eh.movie.merge_im_list(frames, interp="linear", bounds_error=True)


def make_varbeta2_movie(
    npix=50,
    fov_uas=200.0,
    num_frames=64,
    tstart_hr=9.0,
    tstop_hr=15.0,
    total_flux=2.7,
    pa_deg=120.0,
    diameter_uas=52.0,
    alpha_uas=15.0,
    beta1_abs=0.23,
    varbeta_period_hr=1.3333,
    linpol_frac=0.2,
    circpol_frac=0.002,
    source="SgrA",
    **sky,
):
    """Thick m-ring whose EVPA pattern ROTATES: the ``mring-varbeta2`` test model.

    Ported from the ehteval model of the same name. Stokes I is a static thick
    m-ring; the polarization is a radial EVPA field rotated by
    ``theta_rot = -2 pi t / (2 T)``, so the ``beta2`` phase turns through a full
    2 pi every ``2 * varbeta_period_hr`` (default 2.67 hr). This is the model that
    tests **polarization dynamics** -- ``mring+hsCW`` carries a static spiral, so it
    exercises pol *recovery* but never pol *variability*.

    NB score this with :func:`neuraldmd.evaluation.beta2_series` /
    ``beta2_dynamics_error``: the time-averaged ``beta2_coefficient`` cancels a
    rotating swirl and reports ~0 even for a perfect reconstruction.

    Parameters
    ----------
    npix, fov_uas, num_frames, tstart_hr, tstop_hr
        Image grid, field of view [uas], and observation sampling.
    total_flux : float
        Ring flux [Jy].
    pa_deg, diameter_uas, alpha_uas, beta1_abs
        Thick m-ring geometry (position angle, diameter, width, m=1 asymmetry).
    varbeta_period_hr : float
        Half the EVPA rotation period [hr]; the pattern turns 2 pi in ``2 * T``.
    linpol_frac, circpol_frac : float
        Linear / circular polarization fractions.
    source : str
        Source name for the ehtim model.
    **sky
        Overrides for the default sky coordinates (see ``SGRA``).

    Returns
    -------
    ehtim.movie.Movie
        The polarized movie (each frame carries I, Q, U, V).
    """
    import ehtim as eh

    sky = {**SGRA, **sky}
    fov = fov_uas * eh.RADPERUAS
    times = np.linspace(tstart_hr, tstop_hr, num_frames)
    t_elapsed = times - times[0]

    # radial unit field on the image grid, rotated by theta_rot each frame
    lin = np.linspace(-1.0, 1.0, npix)
    xx, yy = np.meshgrid(lin, lin)
    rr = np.sqrt(xx**2 + yy**2)
    radial_x, radial_y = xx / (rr + 1e-6), yy / (rr + 1e-6)

    frames = []
    for te, t in zip(t_elapsed, times, strict=False):
        model = eh.model.Model(ra=sky["ra"], dec=sky["dec"], rf=sky["rf"], source=source)
        im = model.add_thick_mring(
            F0=total_flux,
            d=diameter_uas * eh.RADPERUAS,
            alpha=alpha_uas * eh.RADPERUAS,
            x0=0.0,
            y0=0.0,
            beta_list=[beta1_abs * np.exp(-1j * np.deg2rad(-pa_deg))],
        ).make_image(fov, npix)
        im.mjd = sky["mjd"]

        theta_rot = -2.0 * np.pi * te / (2.0 * varbeta_period_hr)
        rot_x = radial_x * np.cos(theta_rot) - radial_y * np.sin(theta_rot)
        rot_y = radial_x * np.sin(theta_rot) + radial_y * np.cos(theta_rot)
        norm = np.sqrt(rot_x**2 + rot_y**2)
        chi = np.arctan2(rot_y / norm, rot_x / norm)

        stokes_i = im.imarr(pol="I")
        im.ivec = stokes_i.flatten()
        im.qvec = (stokes_i * linpol_frac * np.cos(2 * chi)).flatten()
        im.uvec = -(stokes_i * linpol_frac * np.sin(2 * chi)).flatten()  # ehtim sign
        im.vvec = circpol_frac * np.asarray(im.ivec)
        im.time = float(t)
        frames.append(im)

    return eh.movie.merge_im_list(frames, interp="linear", bounds_error=True)


def to_ehtim_movie(frames, times, fov_uas=200.0, source="SgrA", qframes=None, uframes=None, **sky):
    """Wrap ``(T, H, W)`` frames into an ehtim Movie (requires ehtim).

    Parameters
    ----------
    frames : numpy.ndarray
        Stokes-I frames of shape ``(T, H, W)``.
    times : numpy.ndarray
        Frame times [hours], length ``T``.
    fov_uas : float
        Field of view [uas].
    source : str
        Source name stored on each ehtim Image.
    qframes, uframes : numpy.ndarray or None
        Optional Stokes Q and U frames ``(T, H, W)``. When both are given, each
        Image carries its linear polarization via ``ehtim.image.Image.add_qu``.
    **sky
        Overrides for the sky metadata (``ra``/``dec``/``rf``/``mjd``).

    Returns
    -------
    ehtim.movie.Movie
        Movie assembled from the per-frame ehtim Images.
    """
    import ehtim as eh

    sky = {**SGRA, **sky}
    npix = frames.shape[-1]
    psize = fov_uas * RADPERUAS / npix

    imlist = []
    for i, t in enumerate(times):
        im = eh.image.Image(
            frames[i],
            psize=psize,
            ra=sky["ra"],
            dec=sky["dec"],
            rf=sky["rf"],
            mjd=sky["mjd"],
            source=source,
        )
        im.time = float(t)
        if qframes is not None and uframes is not None:
            im.add_qu(qframes[i], uframes[i])
            # ehtim's polarized chi-squared requires all four Stokes; a movie
            # without V cannot be scored by polchisq at all
            im.add_v(np.zeros_like(frames[i]))
        imlist.append(im)

    return eh.movie.merge_im_list(imlist)


def save_movie_hdf5(movie, path):
    """Save in ehtim's own hdf5 format (readable by eh.movie.load_hdf5)."""
    movie.save_hdf5(str(path))
    print(f"Saved movie to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./data/mring+hs.hdf5")
    parser.add_argument("--num-frames", type=int, default=411)
    args = parser.parse_args()

    frames, times = make_frames(num_frames=args.num_frames)
    movie = to_ehtim_movie(frames, times)
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_movie_hdf5(movie, args.out)
