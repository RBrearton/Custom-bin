import numpy as np

import custom_bin


def weighted_bin_3d(arr: np.ndarray, start: np.ndarray, stop: np.ndarray,
                    step: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    This is an alias to the native C function _weighted_bin_3d, which adds a
    useful protective layer. A lot of explicit type conversions are carried out,
    which prevents segfaults on the C side.
    """
    # pylint: disable=c-extension-no-member
    start = start.astype(np.float32)
    stop = stop.astype(np.float32)
    step = step.astype(np.float32)
    weights = weights.astype(np.float32)

    # Work out the shape array on the python end, as opposed to on the C end.
    # Life's easier in python, so do what we can here.
    shape = ((stop-start)/step).astype(np.int32)

    # Allocate a new numpy array on the python end. Originally, I malloced a
    # big array on the C end, but the numpy C api documentation wasn't super
    # clear on 1) how to cast this to a np.ndarray, or 2) how to prevent memory
    # leaks on the manually malloced array.
    # This array will be initialized to zeros on the C-end, so don't bother
    # initializing it here.
    out = np.ndarray(shape, np.float32)

    custom_bin.weighted_bin_3d(arr, start, stop, step, shape,
                               weights, out)
    return out


arr = np.array([[0.999, 0.22, 0.33], [0.2, 0.21, 0.1],
                [0.61, 0.71, 0.8], [0.92, 0.91, 0.91]]).astype(np.float32)

intensities = np.array([1, 1, 2, 1])

start = np.array((0, 0.1, 0.2), np.float32)
stop = np.array((1, 1.1, 1.2), np.float32)
step = np.array((0.02, 0.02, 0.02), np.float32)

print("About to call custom bin")
binned = weighted_bin_3d(arr, start, stop, step, intensities)

print((binned[10, 5, 0]))
