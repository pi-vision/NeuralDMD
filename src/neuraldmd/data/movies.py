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


def to_ehtim_movie(frames, times, fov_uas=200.0, source="SgrA", **sky):
    """Wrap (T, H, W) frames into an ehtim Movie (requires ehtim)."""
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
