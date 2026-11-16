import numpy as np
import matplotlib.pyplot as plt

def plot_levels(func, xrange=None, yrange=None, levels=None):
    xrange = xrange or [-6, 6]
    yrange = yrange or [-6, 6]
    x = np.linspace(*xrange, 160)
    y = np.linspace(*yrange, 160)
    X, Y = np.meshgrid(x, y)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    if levels is None:
        levels = 18
    cs = plt.contour(X, Y, Z, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, alpha=0.3)

def plot_trajectory(func, history, fit_axis=False, label=None):
    points = np.asarray(history)
    plt.plot(points[:, 0], points[:, 1], "-o", ms=3, lw=1.5, label=label)
    if fit_axis:
        plt.xlim(points[:, 0].min()-1, points[:, 0].max()+1)
        plt.ylim(points[:, 1].min()-1, points[:, 1].max()+1)
