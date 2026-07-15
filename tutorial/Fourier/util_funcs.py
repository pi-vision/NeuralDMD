"""Re-export shim: evaluation/plotting helpers now live in neuraldmd.evaluation."""

from neuraldmd.evaluation import (
    calc_psnr,
    evaluate_chi2,
    load_hdf5,
    make_comparison_gif,
    make_gif,
    pixel_grid_coords,
    plot_frames,
    plot_modes,
    plot_unit_circle,
    sort_modes_by_lambda,
    write_mp4,
)

__all__ = [
    "calc_psnr",
    "evaluate_chi2",
    "load_hdf5",
    "make_comparison_gif",
    "make_gif",
    "pixel_grid_coords",
    "plot_frames",
    "plot_modes",
    "plot_unit_circle",
    "sort_modes_by_lambda",
    "write_mp4",
]
