""" Tests for the functions in pixhist.api.
"""
import numpy as np
from numba import njit
import pixhist
from pixhist.histogram import hist_from_arrays_np


def test_hist_from_gen():
    """ Check that equivalent to NumPy histogram 2D,
    when make_xy_proportional=False.
    """

    @njit
    def gen():
        x = y = count = 0
        while True:
            yield x, y
            count += 1
            x = np.sin(count)
            y = np.cos(count)

    N_ITER = 100
    W, H = 5, 6

    # put the vals from the generator into arrays, so can use np.histogram2d
    x_vals = np.zeros(N_ITER)
    y_vals = np.zeros(N_ITER)
    iterator = gen()
    for i in range(N_ITER):
        x, y = next(iterator)
        x_vals[i] = x
        y_vals[i] = y

    RANGE = [[x_vals.min(), x_vals.max()], [y_vals.min(), y_vals.max()]]

    # Test without log:
    hist_g = pixhist.from_gen(
        gen, N_ITER, W, H, RANGE, make_xy_proportional=False, log=False
    )
    hist_np = hist_from_arrays_np(x_vals, y_vals, W, H)

    assert np.all(hist_g == hist_np), "hist_g != hist_np"

    # Test with log:
    hist_g = pixhist.from_gen(
        gen, N_ITER, W, H, RANGE, make_xy_proportional=False, log=True
    )

    assert np.all(hist_g == np.log(hist_np + 1)), "Logs not equal."



def test_hist_from_arrays():
    """ Check that pixhist.from_vals equivalent to NumPy histogram2d, 
    when make_xy_proportional=False.

    Test for various widths and heights.
    """

    x_vals = np.random.random(200)
    y_vals = np.random.random(200)

    for w in range(1, 4):
        for h in range(1, 4):
            grid = pixhist.from_arrays(
                x_vals, y_vals, w, h, make_xy_proportional=False, log=False
            )
            grid_np = hist_from_arrays_np(x_vals, y_vals, w, h)
            assert np.all(grid == grid_np), f"values not equal, w={w}, w={h}"

            grid = pixhist.from_arrays(
                x_vals, y_vals, w, h, make_xy_proportional=False, log=True
            )
            assert np.all(
                grid == np.log(grid_np + 1)
            ), f"Log values not equal, w={w}, h={h}"


def test_estimate_range():
    
    @njit
    def gen():
        x = y = -499
        while True:
            yield x, y
            x += 1
            y += 1

    r = pixhist.estimate_range(gen, n_iter=1000)

    assert np.all(r == [[-499., 500.], [-499., 500.]])
