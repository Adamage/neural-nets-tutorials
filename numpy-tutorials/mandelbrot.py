import numpy as np
import matplotlib.pyplot as plt


def mandelbrot(h, w, maxit=20):
    """
    Returns an image of the Mandelbrot fractal of size (h,w).
    https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
    """

    # multi-dimensional, sparse "meshgrid"  -> np.ogrid
    y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
    c = x+y*1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2            # who is diverging
        div_now = diverge & (divtime==maxit)  # who is diverging now
        divtime[div_now] = i                  # note when
        z[diverge] = 2                        # avoid diverging too much

    return divtime

plt.imshow(mandelbrot(1400, 1400, maxit=100))
plt.show()