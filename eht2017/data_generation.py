"""Re-export shim: dataset generation now lives in neuraldmd.data.generation.

The array files (e.g. EHT2017.txt) are packaged under neuraldmd/data/arrays/;
Config.array_dir defaults there. Requires the [data] extra (ehtim).
"""

from neuraldmd.data.generation import Config, generate

__all__ = ["Config", "generate"]
