#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def michaelis_menten(s, kcat, km):
    return kcat * s / (km + s)


def fit_curve(func, x, y):
    return curve_fit(func, x, y)


def simulate_kinetics(x, popt):
    xx = np.linspace(min(x), max(x), 10000)
    yy = michaelis_menten(xx, *popt)
    return xx, yy


def r_sq_adj(observed, fitted, n_params):
    obs_mean = observed.mean()
    total_ss = ((observed - obs_mean) ** 2).sum()
    resid_ss = ((observed - fitted) ** 2).sum()
    total_ms = total_ss / (n_params - 1)
    resid_ms = resid_ss / (len(observed) - n_params)
    return (total_ms - resid_ms) / (total_ms)


def plot(df, x_fitted, y_fitted, r_sq):
    fig, ax = plt.subplots()
    grouped = df.groupby("concentration")
    v_mean = grouped["response"].mean()
    v_std = grouped["response"].std()
    ax.scatter(v_mean.index, v_mean, label="observed")
    ax.errorbar(
        v_mean.index,
        v_mean,
        v_std,
        color="black",
        zorder=-1,
        ls="none",
    )
    ax.plot(x_fitted, y_fitted, color="C1", label="fitted")
    ax.legend(loc="lower right")
    for position in ("top", "right", "bottom", "left"):
        ax.spines[position].set_visible(False)
    ax.grid(ls=":")
    ax.set_xlabel("Substrate concentration ($\mu$M)", fontsize=14)
    ax.set_ylabel("Velocity (A.U.)", fontsize=14)
    text_y = max(v_mean) / 2
    text_x = 0.8 * max(v_mean.index)
    plt.text(text_x, text_y, f"$R^{2}$ {r_sq:.4f}")
    plt.show()


def main():
    if len(sys.argv) > 2:
        print("Usage: model.py filename")
        return
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        filename = "data.csv"
    df = pd.read_csv(filename)
    v_mean = df.groupby("concentration")["response"].mean()
    popt, pcov = fit_curve(michaelis_menten, v_mean.index, v_mean)
    x_sim, y_sim = simulate_kinetics(v_mean.index, popt)
    y_fitted = michaelis_menten(v_mean.index, *popt)
    r_sq = r_sq_adj(v_mean, y_fitted, len(popt))
    std_err = np.sqrt(np.diag(pcov))
    print(f"kcat: {popt[0]:.6f} +/- {std_err[0]:.6f}")
    print(f"Km: {popt[1]:.6f} +/- {std_err[1]:.6f}")
    plot(df, x_sim, y_sim, r_sq)


if __name__ == "__main__":
    main()
