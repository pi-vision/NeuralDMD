"""Re-export shim: the m-ring + hot-spot movie synthesizer now lives in
neuraldmd.data.movies."""

from neuraldmd.data.movies import (
    RADPERUAS,
    SGRA,
    hotspot_frame,
    make_frames,
    mring_image,
    save_movie_hdf5,
    to_ehtim_movie,
)

__all__ = [
    "RADPERUAS",
    "SGRA",
    "hotspot_frame",
    "make_frames",
    "mring_image",
    "save_movie_hdf5",
    "to_ehtim_movie",
]
