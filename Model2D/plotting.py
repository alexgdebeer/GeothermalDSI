import cmocean
import h5py
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde


plt.rc("text", usetex=True)
plt.rc("font", family="serif")

TITLE_SIZE = 16
LABEL_SIZE = 12
LEGEND_SIZE = 10
TICK_SIZE = 10

LIMS_XS = (0, 1000)
LIMS_TS = (0, 160)

MIN_PRES = 15
MAX_PRES = 20

TICKS_XS = [0, 500, 1000]
TICKS_TS = [0, 80, 160]

CMAP_PERM = cmocean.cm.thermal
CMAP_PRES = "viridis"

C_PRI = "gray"
C_PCN = "tomato"
C_LMAP = "limegreen"
C_DSI = "lightskyblue"
C_PRES = "lightskyblue"

LABEL_X1 = "$x_{1}$ [m]"
LABEL_X2 = "$x_{2}$ [m]"
LABEL_PERM = "ln(Permeability) [ln(m$^{2}$)]"
LABEL_TIME = "Time [Days]"
LABEL_PRES = "Pressure [MPa]"
LABEL_PROB = "Probability Density"

OBS_END = 80

FULL_WIDTH = 10


well_nums = [7, 4, 1, 8, 5, 2, 9, 6, 3]
well_nums_r = np.argsort(well_nums)
def get_well_name(n):
    return r"\texttt{WELL " + f"{well_nums[n]}" + r"}"


def tufte_axis(ax, bnds_x, bnds_y, gap=0.1, xticks=None, yticks=None):

    if xticks is None:
        xticks = bnds_x
    if yticks is None:
        yticks = bnds_y
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_bounds(*bnds_x)
    ax.spines["left"].set_bounds(*bnds_y)

    dx = bnds_x[1] - bnds_x[0]
    dy = bnds_y[1] - bnds_y[0]

    ax.set_xlim(bnds_x[0] - gap*dx, bnds_x[1] + gap*dx)
    ax.set_ylim(bnds_y[0] - gap*dy, bnds_y[1] + gap*dy)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)


def read_setup_data():

    data = {}

    with h5py.File("data/setup.h5", "r") as f:

        data["lnperm_t"] = f["lnperm_t"][:, :]

        data["p_wells"] = f["well_pressures"][:, :].T / 1e6
        data["xs"] = f["xs"][:]
        data["ts"] = f["ts"][:]

        data["d_obs"] = f["d_obs"][:, :].T / 1e6
        data["t_obs"] = f["t_obs"][:]

        data["well_centres"] = f["well_centres"][:]
        data["us_pri"] = f["us_pri"][:, :].T
        data["states"] = f["states"][:, :].T / 1e6

    return data


def read_results_data():

    data = {}

    with h5py.File("data/results.h5", "r") as f:

        data["pri_preds"] = f["pri_preds"][:, :] / 1e6
        data["pcn_preds"] = f["pcn_preds"][:, :] / 1e6
        data["lmap_preds"] = f["lmap_preds"][:, :] / 1e6
        data["dsi_preds"] = f["dsi_preds"][:, :] / 1e6
        data["ts_preds"] = f["ts"][:]

        for i in [10, 100, 500, 1000, 2000, 5000, 10_000]:
            data[f"dsi_m_{i}"] = f[f"dsi_m_{i}"][:, :].T / 1e6
            data[f"dsi_s_{i}"] = f[f"dsi_s_{i}"][:, :].T / 1e6

    return data


def plot_setup(data_setup, fname, well_to_plot=3):

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 0.26*FULL_WIDTH))

    xs = data_setup["xs"]
    lnperm_t = data_setup["lnperm_t"].T

    ts = data_setup["ts"]
    pres_t = data_setup["p_wells"][well_to_plot]

    t_obs = data_setup["t_obs"]
    pres_obs = data_setup["d_obs"][well_to_plot]

    well_centres = data_setup["well_centres"]

    for ax in axes.flat:
        ax.set_box_aspect(1)

    m = axes[0].pcolormesh(xs, xs, lnperm_t, cmap=CMAP_PERM, rasterized=True)

    cbar = fig.colorbar(m, ax=axes[0], label=LABEL_PERM)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    axes[0].set_xlim(LIMS_XS)
    axes[0].set_ylim(LIMS_XS)
    axes[0].set_xlabel(LABEL_X1, fontsize=LABEL_SIZE)
    axes[0].set_ylabel(LABEL_X2, fontsize=LABEL_SIZE)
    axes[0].set_xticks(TICKS_XS)
    axes[0].set_yticks(TICKS_XS)

    for i, c in enumerate(well_centres):

        cx, cy = c
        name = get_well_name(i)
        
        axes[1].scatter(cx, cy, c="k", s=15)
        axes[1].text(cx, cy+40, name, ha="center", va="bottom", fontsize=TICK_SIZE)

    axes[1].set_facecolor("lightskyblue")
    axes[1].set_xlim(LIMS_XS)
    axes[1].set_ylim(LIMS_XS)
    axes[1].set_xlabel(LABEL_X1, fontsize=LABEL_SIZE)
    axes[1].set_ylabel(LABEL_X2, fontsize=LABEL_SIZE)
    axes[1].set_xticks(TICKS_XS)
    axes[1].set_yticks(TICKS_XS)

    axes[2].plot(ts, pres_t, c="k", zorder=2)
    axes[2].scatter(t_obs, pres_obs, c="k", s=10, zorder=2)
    axes[2].axvline(OBS_END, ls="--", c="gray", ymin=1/12, ymax=11/12, zorder=1)

    axes[2].set_xlabel(LABEL_TIME, fontsize=LABEL_SIZE)
    axes[2].set_ylabel(LABEL_PRES, fontsize=LABEL_SIZE)

    tufte_axis(axes[2], bnds_x=LIMS_TS, bnds_y=(13, 21), 
               xticks=TICKS_TS, yticks=(13, 17, 21))

    plt.tight_layout()
    plt.savefig(fname)
    plt.clf()


def plot_states(data_setup, fname):

    states = data_setup["states"]
    xs = data_setup["xs"]

    selected_states = [
        states[:, :, 9],
        states[:, :, 19],
        states[:, :, 29],
        states[:, :, 39]
    ]

    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 0.23*FULL_WIDTH), 
                             layout="constrained")

    for ax, state in zip(axes, selected_states):
        ax.set_box_aspect(1)
        ax.set_xticks(TICKS_XS)
        ax.set_yticks(TICKS_XS)
        ax.set_xlabel(LABEL_X1, fontsize=LABEL_SIZE)
        m = ax.pcolormesh(xs, xs, state.T, cmap=CMAP_PRES, 
                          vmin=MIN_PRES, vmax=MAX_PRES, rasterized=True)

    axes[0].set_ylabel(LABEL_X2, fontsize=LABEL_SIZE)

    axes[0].set_title("$t$ = 40 Days", fontsize=LABEL_SIZE)
    axes[1].set_title("$t$ = 80 Days", fontsize=LABEL_SIZE)
    axes[2].set_title("$t$ = 120 Days", fontsize=LABEL_SIZE)
    axes[3].set_title("$t$ = 160 Days", fontsize=LABEL_SIZE)

    cbar = fig.colorbar(m, ax=axes[-1], label=LABEL_PRES)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    plt.savefig(fname)


def plot_prior_draws(data_setup, fname):
    
    prior_draws = data_setup["us_pri"]
    xs = np.linspace(10, 990, 50)

    vmin = np.min(prior_draws)
    vmax = np.max(prior_draws)
    
    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 0.22*FULL_WIDTH), 
                             layout="constrained")
    
    for i, ax in enumerate(axes):
        ax.set_box_aspect(1)
        ax.set_xticks([0, 500, 1000])
        ax.set_yticks([0, 500, 1000])
        ax.set_xlabel(LABEL_X1, fontsize=LABEL_SIZE)
        m = ax.pcolormesh(xs, xs, prior_draws[:, :, i].T, cmap=CMAP_PERM, 
                          vmin=vmin, vmax=vmax, rasterized=True)
    
    axes[0].set_ylabel(LABEL_X2, fontsize=LABEL_SIZE)

    cbar = fig.colorbar(m, ax=axes[-1], label=LABEL_PERM)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    plt.savefig(fname)


def plot_results(data_setup, data_results, fname):

    wells_to_plot = [2, 7, 3]

    _, axes = plt.subplots(3, 4, figsize=(10, 7.5))

    ts = data_setup["ts"]
    t_obs = data_setup["t_obs"]

    ts_preds = data_results["ts_preds"]

    for i, well_num in enumerate(wells_to_plot):

        pri_preds = data_results["pri_preds"][well_num, :, :].T
        pcn_preds = data_results["pcn_preds"][well_num, :, :].T
        lmap_preds = data_results["lmap_preds"][well_num, :, :].T
        dsi_preds = data_results["dsi_preds"][well_num, :, :].T

        pres_t = data_setup["p_wells"][well_num]
        pres_obs = data_setup["d_obs"][well_num]

        axes[i][0].plot(ts_preds, pri_preds, c=C_PRES, alpha=0.05)
        axes[i][1].plot(ts_preds, pcn_preds, c=C_PRES, alpha=0.05)
        axes[i][2].plot(ts_preds, lmap_preds, c=C_PRES, alpha=0.05)
        axes[i][3].plot(ts_preds, dsi_preds, c=C_PRES, alpha=0.05)

        for ax in axes[i]:

            ax.plot(ts, pres_t, c="k", zorder=2)
            ax.scatter(t_obs, pres_obs, c="k", s=10, zorder=2)

            tufte_axis(ax, bnds_x=(0, 160), bnds_y=(14, 21), 
                       xticks=(0, 80, 160), yticks=(14, 17.5, 21))

            ax.set_box_aspect(1)
            ax.axvline(OBS_END, ls="--", c="gray", ymin=1/12, ymax=11/12, zorder=1)

        n = well_nums[well_num]
        axes[i][0].set_ylabel(f"Well {n} Pressure [MPa]", fontsize=LABEL_SIZE)

    for ax in axes[-1]:
        ax.set_xlabel(LABEL_TIME, fontsize=LABEL_SIZE)

    axes[0][0].set_title("Prior", fontsize=LABEL_SIZE)
    axes[0][1].set_title("MCMC", fontsize=LABEL_SIZE)
    axes[0][2].set_title("LMAP", fontsize=LABEL_SIZE)
    axes[0][3].set_title("DSI", fontsize=LABEL_SIZE)

    plt.tight_layout()
    plt.savefig(fname)


def plot_final_pressures(data_setup, data_results, fname):

    _, axes = plt.subplots(3, 3, figsize=(0.75*FULL_WIDTH, 0.8*FULL_WIDTH))

    for i in range(9):

        well_num = well_nums_r[i]

        pri_preds = data_results["pri_preds"][well_num, :, -1]
        pcn_preds = data_results["pcn_preds"][well_num, :, -1]
        lmap_preds = data_results["lmap_preds"][well_num, :, -1]
        truth = data_setup["p_wells"][well_num][-1]

        dsi_m = data_results["dsi_m_1000"][well_num][-1]
        dsi_s = data_results["dsi_s_1000"][well_num][-1]
        
        xmin = np.min([
            np.min(pcn_preds),
            np.min(lmap_preds),
            dsi_m - 4 * dsi_s
        ])

        xmax = np.max([
            np.max(pcn_preds),
            np.max(lmap_preds),
            dsi_m + 4 * dsi_s
        ])

        xmin = np.min([np.floor(xmin * 2) / 2, 14.5])
        xmax = np.max([np.ceil(xmax * 2) / 2, 17.5])

        xs = np.linspace(xmin, xmax, 1000)

        pri_density = gaussian_kde(pri_preds)(xs)
        pcn_density = gaussian_kde(pcn_preds)(xs)
        lmap_density = gaussian_kde(lmap_preds)(xs)
        dsi_density = stats.norm(dsi_m, dsi_s).pdf(xs)

        ymin = 0
        ymax = np.max([
            pcn_density,
            lmap_density,
            dsi_density
        ])

        ymax = np.ceil(ymax * 2) / 2

        p_truth = axes.flat[i].axvline(truth, c="k", ymin=1/12, ymax=11/12, zorder=0, label="Truth")
        p_pri,  = axes.flat[i].plot(xs, pri_density, c=C_PRI, label="Prior", zorder=1, lw=1.2)
        p_pcn,  = axes.flat[i].plot(xs, pcn_density, c=C_PCN, ls="dashed", label="MCMC", zorder=4, lw=1.2)
        p_lmap, = axes.flat[i].plot(xs, lmap_density, c=C_LMAP, label="LMAP", zorder=3, lw=1.2)
        p_dsi,  = axes.flat[i].plot(xs, dsi_density, c=C_DSI, label="DSI", zorder=5, lw=1.2)

        tufte_axis(axes.flat[i], bnds_x=(xmin, xmax), bnds_y=(ymin, ymax))
        axes.flat[i].set_title(f"Well {i+1}", fontsize=LABEL_SIZE)
        axes.flat[i].set_box_aspect(1)

    for ax in axes[-1]:
        ax.set_xlabel(LABEL_PRES, fontsize=LABEL_SIZE)

    for ax in axes[:, 0]:
        ax.set_ylabel(LABEL_PROB, fontsize=LABEL_SIZE)

    handles = [p_truth, p_pri, p_pcn, p_lmap, p_dsi]
    plt.figlegend(handles=handles, loc="lower center", ncols=5, 
                  frameon=False, fontsize=LEGEND_SIZE)

    plt.subplots_adjust(left=0.1,bottom=0.1, right=0.95, top=0.95)
    plt.savefig(fname)


def plot_sample_comparison(data_setup, data_results, fname):

    _, axes = plt.subplots(3, 3, figsize=(0.75*FULL_WIDTH, 0.8*FULL_WIDTH))

    dsi_samples = [10, 100, 500, 1000, 2000, 5000, 10_000]

    cmap = mpl.colormaps["Blues"].resampled(1000)
    cmap = cmap(np.linspace(0.25, 0.9, 7))

    handles = []

    for i in range(9):

        well_num = well_nums_r[i]

        means = [data_results[f"dsi_m_{n}"][well_num][-1] for n in dsi_samples]
        stds = [data_results[f"dsi_s_{n}"][well_num][-1] for n in dsi_samples]
        truth = data_setup["p_wells"][well_num][-1]

        xmin = np.min([m-5*s for m, s in zip(means, stds)])
        xmax = np.max([m+5*s for m, s in zip(means, stds)])
        xmin = np.round(xmin * 2) / 2
        xmax = np.round(xmax * 2) / 2
        xmin = min(xmin, 14.5)
        xmax = max(xmax, 17.5)
        xs = np.linspace(xmin, xmax, 1000)
        
        pri_preds = data_results["pri_preds"][well_num, :, -1]
        pcn_preds = data_results["pcn_preds"][well_num, :, -1]
        pri_density = gaussian_kde(pri_preds)(xs)
        pcn_density = gaussian_kde(pcn_preds)(xs)

        ymin = 0
        ymax = np.max(pcn_density)

        p_pri, = axes.flat[i].plot(xs, pri_density, c=C_PRI, lw=1.2, label="Prior")
        p_pcn, = axes.flat[i].plot(xs, pcn_density, c=C_PCN, lw=1.2, label="MCMC")
        p_truth = axes.flat[i].axvline(truth, c="k", ymin=1/12, ymax=11/12, lw=1.2, zorder=0, label="Truth")
        
        if i == 0:
            handles.append(p_truth)
            handles.append(p_pri)
            handles.append(p_pcn)

        for n, (m, s) in enumerate(zip(means, stds)):
            
            dsi_density = stats.norm(m, s).pdf(xs)
            label = f"DSI ($l={dsi_samples[n]}$)"
            p, = axes.flat[i].plot(xs, dsi_density, c=cmap[n], label=label, lw=1.2)
            
            if np.max(dsi_density) > ymax:
                ymax = np.max(dsi_density)
            
            if i == 0:
                handles.append(p)

        axes.flat[i].set_box_aspect(1)
        tufte_axis(axes.flat[i], bnds_x=(xmin, xmax), bnds_y=(ymin, np.ceil(ymax)))
        axes.flat[i].set_title(f"Well {i+1}", fontsize=LABEL_SIZE)

    for ax in axes[:, 0]:
        ax.set_ylabel(LABEL_PROB, fontsize=LABEL_SIZE)

    for ax in axes[-1]:
        ax.set_xlabel(LABEL_PRES, fontsize=LABEL_SIZE)

    plt.figlegend(handles=handles, loc="lower center", ncols=5, 
                  frameon=False, fontsize=TICK_SIZE)

    plt.subplots_adjust(left=0.1, bottom=0.13, right=0.95, top=0.95)
    plt.savefig(fname)


if __name__ == "__main__":

    data_setup = read_setup_data()
    data_results = read_results_data()

    plot_setup(data_setup, fname="figures/fig1.pdf")
    plot_states(data_setup, fname="figures/fig2.pdf")
    plot_prior_draws(data_setup, fname="figures/fig3.pdf")
    plot_results(data_setup, data_results, fname="figures/fig4.pdf")
    plot_final_pressures(data_setup, data_results, fname="figures/fig5.pdf")
    plot_sample_comparison(data_setup, data_results, fname="figures/fig6.pdf")