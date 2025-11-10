import os
import glob
import numpy as np
import math
import segyio

DEFAULT_RAW_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'Orange Basin_cropped 1')
)


def _pick_file_in_dir(dir_path: str) -> str:
    """Pick a plausible seismic file from a directory.
    Preference order: .npy, .npz, then SEG-Y-like extensions.
    """
    patterns = ['*.npy', '*.npz', '*.segy', '*.sgy', '*.seg', '*.segd']
    for pat in patterns:
        files = sorted(glob.glob(os.path.join(dir_path, pat)))
        if files:
            return files[0]
    raise FileNotFoundError(f"No supported files found in directory: {dir_path}")


def load_volume(input_path: str) -> np.ndarray:
    """Load 3D seismic volume as a numpy array [D, H, W] (z,y,x order)."""

    # Generic SEG-Y loader. This may need adaptation for specific Orange Basin layout.
    with segyio.open(input_path, 'r', ignore_geometry=True) as f:
        # f.trace.raw[:] -> (num_traces, num_samples)
        traces = segyio.collect(f.trace)  # shape: (num_traces, num_samples)
        # Infer grid (inline x crossline). Without geometry we assume a rectangular grid.
        num_traces, num_samples = traces.shape
        # Heuristic: try to make grid as square as possible in spatial dimensions
        # User can later replace this with known dims if available.
        # We'll search for factors close to sqrt(num_traces)
        
        s = int(math.sqrt(num_traces))
        best_hw = None
        best_diff = None
        for h in range(1, s + 1):
            if num_traces % h == 0:
                w = num_traces // h
                diff = abs(h - w)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_hw = (h, w)
        if best_hw is None:
            # fallback to (num_traces, 1)
            best_hw = (num_traces, 1)
        H, W = best_hw
        # Volume as [D, H, W] where D = time/depth samples
        vol = traces.reshape(H, W, num_samples).transpose(2, 0, 1)
        return vol


from typing import Optional

def def_volume(input_path: Optional[str] = None) -> np.ndarray:
    """Load the default Orange Basin cropped volume.

    If input_path is None, uses src/data/Orange Basin_cropped 1.
    This may be either a file (e.g., .npy/.npz/.segy) or a directory containing one.
    """
    path = input_path or DEFAULT_RAW_DIR
    if os.path.isdir(path):
        picked = _pick_file_in_dir(path)
        return load_volume(picked)
    return load_volume(path)
