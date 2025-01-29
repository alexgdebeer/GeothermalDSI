import cmocean
import h5py
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import patches
import numpy as np
import pyvista as pv
from scipy import stats

from setup import *


plt.style.use("../paperstyle.mplstyle")

LIMS_XS = (0, 6000)
LIMS_TS = (0, 2)
LIMS_UPFLOW = (0, 2.75e-4)

MIN_PERM = -17.0
MAX_PERM = -12.5

P_MIN, P_MAX = 0.5, 10.5
E_MIN, E_MAX = 100, 1500
T_MIN, T_MAX = 20, 320

TICKS_XS = [0, 3000, 6000]
TICKS_TS = [0, 1, 2]

LABEL_X1 = "$x_{1}$ [m]"
LABEL_X2 = "$x_{2}$ [m]"
LABEL_TEMP = r"Temperature [$^\circ$C]"
LABEL_TIME = "Time [Years]"
LABEL_ELEV = "Elevation [m]"
LABEL_PERM = "log$_{10}$(Perm) [log$_{10}$(m$^2$)]"
LABEL_PRES = "Pressure [MPa]"
LABEL_ENTH = "Enthalpy [kJ kg$^{-1}$]"
LABEL_UPFLOW = "Upflow [kg s$^{-1}$ m$^{-2}$]"
LABEL_PROB = "Probability Density"

COL_CUR_WELLS = "royalblue"
COL_NEW_WELLS = "firebrick"

COL_TEMP = "tab:orange"
COL_PRES = "cornflowerblue"
COL_ENTH = "tab:green"
COL_CAP = "peru"

COL_GRID = "darkgrey"
COL_DEEP = "whitesmoke"

CMAP_PERM = cmocean.cm.turbid.reversed()
CMAP_UPFLOW = cmocean.cm.thermal
CMAP_TEMP = cmocean.cm.balance

PLOT_GRID = False
PLOT_TRUTH = False
PLOT_UPFLOWS = False
PLOT_CAPS = False
PLOT_PERMS = False
PLOT_DATA = True
PLOT_PRIOR_PREDICTIONS = False
PLOT_DSI_PREDICTIONS_A = False
PLOT_DSI_PREDICTIONS_B = False
PLOT_SAMPLE_COMP = False

FULL_WIDTH = 10

CAMERA_POSITION = (13_000, 15_000, 6_000)


ps = np.load("data/ps.npy")
F_t = np.load("data/truth/F_true.npy")


def get_ns_temps(fname):

    with h5py.File(f"{fname}.h5", "r") as f:
        cell_inds = f["cell_index"][:, 0]
        temp = f["cell_fields"]["fluid_temperature"][0][cell_inds]

    return temp


def get_well_name(i):
    return r"\texttt{WELL" + f" {i+1}" + r"}"


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


def get_well_tubes(wells, feedzone_depths):
        
    lines = pv.MultiBlock()
    for well in wells:
        line = pv.Line(*well.coords)
        lines.append(line)
    bodies = lines.combine().extract_geometry().clean().split_bodies()

    tubes = pv.MultiBlock()
    for body in bodies:
        tubes.append(body.extract_geometry().tube(radius=35))

    for i, well in enumerate(wells):
        feedzone = (well.x, well.y, feedzone_depths[i])
        tubes.append(pv.SolidSphere(outer_radius=70, center=feedzone))

    return tubes


def get_layer_polys(mesh: lm.mesh, cmap):

    verts = [[n.pos for n in c.column.node] 
             for c in mesh.layer[-1].cell]

    polys = PolyCollection(verts, cmap=cmap)
    return polys


def convert_inds(mesh, fem_mesh, inds):
    return [1 if mesh.find(c.center, indices=True) in inds else 0 
            for c in fem_mesh.cell]


def plot_colourbar(cmap, vmin, vmax, label, fname, power=False):
    
    _, ax = plt.subplots(figsize=(2.5, 2.5))
    m = ax.pcolormesh(np.zeros((10, 10)), cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(m, ax=ax, label=label)

    if power:
        cbar.formatter.set_powerlimits((0, 0))

    plt.tight_layout()
    plt.savefig(fname)


def plot_mesh(fem_mesh: pv.UnstructuredGrid, wells, feedzone_depths, fname):

    cur_tubes = get_well_tubes(wells[:n_cur_wells], feedzone_depths)
    new_tubes = get_well_tubes(wells[n_cur_wells:], feedzone_depths)
    edges = fem_mesh.extract_geometry().extract_all_edges()

    p = pv.Plotter(off_screen=True, window_size=(2400, 2400))
    p.add_mesh(fem_mesh, color=COL_DEEP, opacity=0.2)
    p.add_mesh(cur_tubes, color=COL_CUR_WELLS)
    p.add_mesh(new_tubes, color=COL_NEW_WELLS)
    p.add_mesh(edges, line_width=2, color=COL_GRID)

    p.camera.position = CAMERA_POSITION
    p.add_light(pv.Light(position=CAMERA_POSITION, intensity=0.2))
    p.screenshot(fname)


def plot_grid_layer(mesh, cur_well_centres, new_well_centres, fname):

    _, ax = plt.subplots(figsize=(0.5*FULL_WIDTH, 0.5*FULL_WIDTH))

    mesh.m.layer_plot(axes=ax, linewidth=0.75, linecolour=COL_GRID)

    for i, (x, y) in enumerate(cur_well_centres):
        ax.scatter(x, y, s=20, c=COL_CUR_WELLS)
        plt.text(x-360, y+160, s=get_well_name(i), 
                 color=COL_CUR_WELLS, fontsize=12)
    
    for i, (x, y) in enumerate(new_well_centres):
        ax.scatter(x, y, s=20, c=COL_NEW_WELLS)
        plt.text(x-360, y+160, s=get_well_name(len(cur_well_centres)+i), 
                 color=COL_NEW_WELLS, fontsize=12)
    
    ax.set_xlabel(LABEL_X1)
    ax.set_ylabel(LABEL_X2)
    
    for s in ax.spines.values():
        s.set_edgecolor(COL_GRID)
    
    ax.set_xlim(LIMS_XS)
    ax.set_ylim(LIMS_XS)
    ax.set_xticks(TICKS_XS)
    ax.set_yticks(TICKS_XS)
    ax.tick_params(length=0)
    ax.set_facecolor(COL_DEEP)
    
    plt.tight_layout()
    plt.savefig(fname)


def plot_truth(mesh, fem_mesh, temps, logks, fname):
    
    p = pv.Plotter(shape=(1, 3), window_size=(900, 300), border=False, off_screen=True)

    outline = fem_mesh.outline()
    p.add_light(pv.Light(position=(4000, 3500, 1000), intensity=0.5))

    fem_mesh["vals"] = temps[mesh.mesh_mapping]
    slice_x = fem_mesh.clip(normal="x")
    slice_y = fem_mesh.clip(normal="y")

    p.subplot(0, 0)
    p.add_mesh(outline, line_width=6, color=COL_GRID)
    p.add_mesh(slice_y, cmap=cmocean.cm.balance, clim=(0, 250), show_scalar_bar=False)
    p.camera.position = CAMERA_POSITION

    p.subplot(0, 1)
    p.add_mesh(outline, line_width=6, color=COL_GRID)
    p.add_mesh(slice_x, cmap=cmocean.cm.balance, clim=(0, 250), show_scalar_bar=False)
    p.camera.position = CAMERA_POSITION

    fem_mesh["vals"] = logks[mesh.mesh_mapping]
    slice_logks = fem_mesh.clip(normal="x")

    p.subplot(0, 2)
    p.add_mesh(outline, line_width=6, color=COL_GRID)
    p.add_mesh(slice_logks, cmap=CMAP_PERM, clim=(MIN_PERM, MAX_PERM), show_scalar_bar=False)
    p.camera.position = CAMERA_POSITION
            
    p.screenshot(fname, scale=3)


def plot_true_upflows(mesh: lm.mesh, upflows, fname):

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.2))
    polys = get_layer_polys(mesh, cmap=CMAP_UPFLOW)
    polys.set_clim(0, 2.75e-4)

    p = ax.add_collection(polys)
    polys.set_array(upflows)

    ax.set_xlim(LIMS_XS)
    ax.set_ylim(LIMS_XS)
    ax.set_box_aspect(1)
    ax.set_xticks(LIMS_XS)
    ax.set_yticks(LIMS_XS)
    ax.tick_params(which="both", bottom=False, left=False)
    ax.set_xlabel(LABEL_X1)
    ax.set_ylabel(LABEL_X2)

    cbar = fig.colorbar(p, ax=ax, label=LABEL_UPFLOW)
    cbar.ax.ticklabel_format(style="sci", scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(fname)


def plot_caps(mesh, fem_mesh, cap_cell_inds, fname):

    camera_position_top = (6000, 7000, 3000)
    camera_position_bottom = (6000, 7000, -4000)

    caps = [convert_inds(mesh, fem_mesh, cap_cell_inds[i]) for i in range(4)]

    p = pv.Plotter(shape=(2, 4), window_size=(4800, 2000), border=False, off_screen=True)

    light = pv.Light(position=(0, 0, -2000), intensity=0.35)
    p.add_light(light)

    for j in range(4):

        fem_mesh["vals"] = caps[j]

        cap = fem_mesh.threshold(0.5)
        cap_r = cap.rotate_x(0)

        outline = cap.extract_surface().extract_feature_edges()
        outline_r = cap_r.extract_surface().extract_feature_edges()

        p.subplot(0, j)
        p.add_mesh(cap, color=COL_CAP, lighting=True)
        p.add_mesh(outline, line_width=3, color="k")
        p.camera.position = camera_position_top

        p.subplot(1, j)
        p.add_mesh(cap_r, color=COL_CAP, lighting=True)
        p.add_mesh(outline_r, line_width=3, color="k")
        p.camera.position = camera_position_bottom

    p.screenshot(fname)


def plot_slices(mesh, fem_mesh, vals, fname, n_slices=4, 
                cmap=CMAP_PERM, vmin=MIN_PERM, vmax=MAX_PERM):

    p = pv.Plotter(shape=(1, n_slices), window_size=(300*n_slices, 300), 
                   border=False, off_screen=True)

    outline = fem_mesh.outline()
    p.add_light(pv.Light(position=(4000, 3500, 1000), intensity=0.5))

    for i in range(n_slices):

        fem_mesh["vals"] = vals[i][mesh.mesh_mapping]
        slice = fem_mesh.clip(normal="x")

        p.subplot(0, i)
        p.add_mesh(outline, line_width=6, color=COL_GRID)
        p.add_mesh(slice, cmap=cmap, clim=(vmin, vmax), show_scalar_bar=False)
        p.camera.position = CAMERA_POSITION
            
    p.screenshot(fname, scale=3)


def plot_data(elev, time, temp, pres, enth, 
              elev_obs, time_obs, temp_obs, pres_obs, enth_obs,
              fname):

    _, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))

    axes[0].plot(temp, elev, c="k", lw=1.5, zorder=2)
    axes[1].plot(time, pres, c="k", lw=1.5, zorder=2)
    axes[2].plot(time, enth, c="k", lw=1.5, zorder=2)

    axes[0].scatter(temp_obs, elev_obs, c="k", s=6, zorder=2)
    axes[1].scatter(time_obs, pres_obs, c="k", s=6, zorder=2)
    axes[2].scatter(time_obs, enth_obs, c="k", s=6, zorder=2)

    # axes[1].axvline(1, ls="--", c="gray", lw=1.5, ymin=1/12, ymax=11/12, zorder=1)
    # axes[2].axvline(1, ls="--", c="gray", lw=1.5, ymin=1/12, ymax=11/12, zorder=1)

    forecast_p = patches.Rectangle(
        xy=(1, 5), 
        width=1, 
        height=3,
        facecolor="gainsboro",
        zorder=0
    )
    forecast_e = patches.Rectangle(
        xy=(1, 300), 
        width=1, 
        height=300, 
        facecolor="gainsboro", 
        zorder=0
    )

    axes[1].add_patch(deepcopy(forecast_p))
    axes[2].add_patch(deepcopy(forecast_e))

    axes[0].set_xlabel(LABEL_TEMP)
    axes[1].set_xlabel(LABEL_TIME)
    axes[2].set_xlabel(LABEL_TIME)

    axes[0].set_ylabel(LABEL_ELEV)
    axes[1].set_ylabel(LABEL_PRES)
    axes[2].set_ylabel(LABEL_ENTH)

    for ax in axes:
        ax.set_box_aspect(1)

    ymin = min(elev)
    ymax = max(elev)

    tufte_axis(axes[0], bnds_x=(0, 200), bnds_y=(ymin, ymax))
    tufte_axis(axes[1], bnds_x=LIMS_TS, bnds_y=(5, 8), xticks=TICKS_TS)
    tufte_axis(axes[2], bnds_x=LIMS_TS, bnds_y=(300, 600), xticks=TICKS_TS)

    plt.tight_layout()
    plt.savefig(fname)


def plot_upflows(mesh: lm.mesh, upflows, fname):

    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 0.255*FULL_WIDTH), 
                             layout="constrained", sharey=True)
    
    polys = get_layer_polys(mesh, cmap=CMAP_UPFLOW)
    polys.set_clim(LIMS_UPFLOW)

    for i, ax in enumerate(axes.flat):

        polys_i = deepcopy(polys)
        polys_i.set_array(upflows[i])
        p = ax.add_collection(polys_i)

        ax.set_xlim(LIMS_XS)
        ax.set_ylim(LIMS_XS)
        ax.set_box_aspect(1)
        ax.set_xticks(LIMS_XS)
        ax.set_yticks(LIMS_XS)
        ax.tick_params(which="both", bottom=False, left=False)
        ax.set_xlabel(LABEL_X1)

    axes[0].set_ylabel(LABEL_X2)
    cbar = fig.colorbar(p, ax=axes[-1], label=LABEL_UPFLOW)
    cbar.ax.ticklabel_format(style="sci", scilimits=(0, 0))

    plt.savefig(fname)


def plot_prior_vs_dsi(elevs_crse, 
                      temp_pri, pres_pri, enth_pri, 
                      temp_dsi, pres_dsi, enth_dsi,
                      well_nums, fname):

    fig, axes = plt.subplots(3, 4, figsize=(10, 9), sharey="row")

    ts_crse = data_handler_crse.ts / (SECS_PER_WEEK * 52)

    forecast_p = patches.Rectangle(
        xy=(1, P_MIN), 
        width=1, 
        height=P_MAX-P_MIN,
        facecolor="gainsboro",
        zorder=0
    )
    forecast_e = patches.Rectangle(
        xy=(1, E_MIN), 
        width=1, 
        height=E_MAX-E_MIN, 
        facecolor="gainsboro", 
        zorder=0
    )

    zmin = min([min(elevs) for elevs in elevs_crse])
    zmax = max([max(elevs) for elevs in elevs_crse])
    
    for i in range(2):

        j0 = 2*i
        j1 = 2*i+1

        axes[0][j0].plot(temp_pri[i].T, elevs_crse[i], c=COL_TEMP, lw=1, alpha=0.5, zorder=1)
        axes[0][j1].plot(temp_dsi[i].T, elevs_crse[i], c=COL_TEMP, lw=1, alpha=0.5, zorder=1)
            
        axes[1][j0].plot(ts_crse, pres_pri[i].T, c=COL_PRES, lw=1, alpha=0.5, zorder=1)
        axes[1][j1].plot(ts_crse, pres_dsi[i].T, c=COL_PRES, lw=1, alpha=0.5, zorder=1)

        axes[2][j0].plot(ts_crse, enth_pri[i].T, c=COL_ENTH, lw=1, alpha=0.5, zorder=1)
        axes[2][j1].plot(ts_crse, enth_dsi[i].T, c=COL_ENTH, lw=1, alpha=0.5, zorder=1)

        axes[1][j0].add_patch(deepcopy(forecast_p))
        axes[1][j1].add_patch(deepcopy(forecast_p))

        axes[2][j0].add_patch(deepcopy(forecast_e))
        axes[2][j1].add_patch(deepcopy(forecast_e))

        tufte_axis(axes[0][j0], bnds_x=(T_MIN, T_MAX), bnds_y=(zmin, zmax), xticks=(T_MIN, 0.5*(T_MIN+T_MAX), T_MAX), yticks=(zmin, 0.5*(zmin+zmax), zmax), gap=0.05)
        tufte_axis(axes[0][j1], bnds_x=(T_MIN, T_MAX), bnds_y=(zmin, zmax), xticks=(T_MIN, 0.5*(T_MIN+T_MAX), T_MAX), yticks=(zmin, 0.5*(zmin+zmax), zmax), gap=0.05)

        tufte_axis(axes[1][j0], bnds_x=(0, 2), bnds_y=(P_MIN, P_MAX), xticks=(0, 1, 2), yticks=(P_MIN, 0.5*(P_MIN+P_MAX), P_MAX), gap=0.05)
        tufte_axis(axes[1][j1], bnds_x=(0, 2), bnds_y=(P_MIN, P_MAX), xticks=(0, 1, 2), yticks=(P_MIN, 0.5*(P_MIN+P_MAX), P_MAX), gap=0.05)

        tufte_axis(axes[2][j0], bnds_x=(0, 2), bnds_y=(E_MIN, E_MAX), xticks=(0, 1, 2), yticks=(E_MIN, 0.5*(E_MIN+E_MAX), E_MAX), gap=0.05)
        tufte_axis(axes[2][j1], bnds_x=(0, 2), bnds_y=(E_MIN, E_MAX), xticks=(0, 1, 2), yticks=(E_MIN, 0.5*(E_MIN+E_MAX), E_MAX), gap=0.05)

        axes[0][j0].set_title("Prior")
        axes[0][j1].set_title("DSI (Unconditional)")

        axes[0][j0].set_xlabel(LABEL_TEMP)
        axes[0][j1].set_xlabel(LABEL_TEMP)

        axes[1][j0].set_xlabel(LABEL_TIME)
        axes[1][j1].set_xlabel(LABEL_TIME)

        axes[2][j0].set_xlabel(LABEL_TIME)
        axes[2][j1].set_xlabel(LABEL_TIME)

    axes[0][0].set_ylabel(LABEL_ELEV)
    axes[1][0].set_ylabel(LABEL_PRES)
    axes[2][0].set_ylabel(LABEL_ENTH)

    fig.text(0.31, 0.975, get_well_name(well_nums[0]), ha="center", va="top", fontsize=16)
    fig.text(0.76, 0.975, get_well_name(well_nums[1]), ha="center", va="top", fontsize=16)

    for ax in axes.flat:
        ax.set_box_aspect(1)

    fig.align_labels()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(fname)


def plot_dsi_predictions(elevs_fine, temp_t, pres_t, enth_t, 
                         elevs_crse, temp_pri, pres_pri, enth_pri, 
                         temp_dsi, pres_dsi, enth_dsi,
                         pres_obs, enth_obs, well_nums, fname,
                         forecast_only=False):

    fig, axes = plt.subplots(3, 4, figsize=(10, 9), sharey="row")

    ts_crse = data_handler_crse.ts / (SECS_PER_WEEK * 52)
    ts_fine = data_handler_fine.ts / (SECS_PER_WEEK * 52)

    if forecast_only:
        
        n_forecast_crse = len(ts_crse) // 2
        ts_crse = ts_crse[n_forecast_crse:]
        n_forecast_fine = len(ts_fine) // 2
        ts_fine = ts_fine[n_forecast_fine:]

        # temp_pri = [t[n_forecast_crse:, :] for t in temp_pri]
        pres_pri = [p[:, n_forecast_crse:] for p in pres_pri]
        enth_pri = [e[:, n_forecast_crse:] for e in enth_pri]
        pres_dsi = [p[:, n_forecast_crse:] for p in pres_dsi]
        enth_dsi = [e[:, n_forecast_crse:] for e in enth_dsi]

        pres_t = pres_t[:, n_forecast_fine:]
        enth_t = enth_t[:, n_forecast_fine:]
    
    time_min, time_max = min(ts_crse), max(ts_crse)

    temp_qs_pri = [np.quantile(t, q=(0.025, 0.975), axis=0) for t in temp_pri]
    pres_qs_pri = [np.quantile(p, q=(0.025, 0.975), axis=0) for p in pres_pri]
    enth_qs_pri = [np.quantile(e, q=(0.025, 0.975), axis=0) for e in enth_pri]

    temp_qs_dsi = [np.quantile(t, q=(0.025, 0.975), axis=0) for t in temp_dsi]
    pres_qs_dsi = [np.quantile(p, q=(0.025, 0.975), axis=0) for p in pres_dsi]
    enth_qs_dsi = [np.quantile(e, q=(0.025, 0.975), axis=0) for e in enth_dsi]

    forecast_p = patches.Rectangle(
        xy=(1, P_MIN), 
        width=1, 
        height=P_MAX-P_MIN,
        facecolor="gainsboro",
        zorder=0
    )
    forecast_e = patches.Rectangle(
        xy=(1, E_MIN), 
        width=1, 
        height=E_MAX-E_MIN, 
        facecolor="gainsboro", 
        zorder=0
    )
    
    for i in range(2):

        j0 = 2*i
        j1 = 2*i+1

        axes[0][j0].plot(temp_pri[i].T, elevs_crse[i], c=COL_TEMP, lw=1.0, alpha=0.1, zorder=1)
        axes[0][j1].plot(temp_dsi[i].T, elevs_crse[i], c=COL_TEMP, lw=1.0, alpha=0.1, zorder=1)

        axes[0][j0].plot(temp_qs_pri[i].T, elevs_crse[i], c="dimgrey", lw=1.0, zorder=2)
        axes[0][j1].plot(temp_qs_dsi[i].T, elevs_crse[i], c="dimgrey", lw=1.0, zorder=2)
            
        axes[1][j0].plot(ts_crse, pres_pri[i].T, c=COL_PRES, lw=1.0, alpha=0.1, zorder=1)
        axes[1][j1].plot(ts_crse, pres_dsi[i].T, c=COL_PRES, lw=1.0, alpha=0.1, zorder=1)

        axes[1][j0].plot(ts_crse, pres_qs_pri[i].T, c="dimgrey", lw=1.0, zorder=2)
        axes[1][j1].plot(ts_crse, pres_qs_dsi[i].T, c="dimgrey", lw=1.0, zorder=2)

        axes[2][j0].plot(ts_crse, enth_pri[i].T, c=COL_ENTH, lw=1.0, alpha=0.1, zorder=1)
        axes[2][j1].plot(ts_crse, enth_dsi[i].T, c=COL_ENTH, lw=1.0, alpha=0.1, zorder=1)

        axes[2][j0].plot(ts_crse, enth_qs_pri[i].T, c="dimgrey", lw=1.0, zorder=2)
        axes[2][j1].plot(ts_crse, enth_qs_dsi[i].T, c="dimgrey", lw=1.0, zorder=2)

        # Plot truth
        axes[0][j0].plot(temp_t[i], elevs_fine[i], c="k", lw=1.2, zorder=3)
        axes[0][j1].plot(temp_t[i], elevs_fine[i], c="k", lw=1.2, zorder=3)
        
        axes[1][j0].plot(ts_fine, pres_t[i], c="k", lw=1.2, zorder=3)
        axes[1][j1].plot(ts_fine, pres_t[i], c="k", lw=1.2, zorder=3)

        axes[2][j0].plot(ts_fine, enth_t[i], c="k", lw=1.2, zorder=3)
        axes[2][j1].plot(ts_fine, enth_t[i], c="k", lw=1.2, zorder=3)

        if pres_obs is not None:
            axes[1][j0].scatter(prod_obs_ts / (SECS_PER_WEEK * 52), pres_obs[i], s=6, c="k", zorder=3)
            axes[1][j1].scatter(prod_obs_ts / (SECS_PER_WEEK * 52), pres_obs[i], s=6, c="k", zorder=3)
        
        if enth_obs is not None:
            axes[2][j0].scatter(prod_obs_ts / (SECS_PER_WEEK * 52), enth_obs[i], s=6, c="k", zorder=3)
            axes[2][j1].scatter(prod_obs_ts / (SECS_PER_WEEK * 52), enth_obs[i], s=6, c="k", zorder=3)

        axes[1][j0].add_patch(deepcopy(forecast_p))
        axes[1][j1].add_patch(deepcopy(forecast_p))

        axes[2][j0].add_patch(deepcopy(forecast_e))
        axes[2][j1].add_patch(deepcopy(forecast_e))

        zmin = min(elevs_fine[i])
        zmax = max(elevs_fine[i])

        tufte_axis(axes[0][j0], bnds_x=(T_MIN, T_MAX), bnds_y=(zmin, zmax), xticks=(T_MIN, 0.5*(T_MIN+T_MAX), T_MAX), yticks=(zmin, 0.5*(zmin+zmax), zmax), gap=0.05)
        tufte_axis(axes[0][j1], bnds_x=(T_MIN, T_MAX), bnds_y=(zmin, zmax), xticks=(T_MIN, 0.5*(T_MIN+T_MAX), T_MAX), yticks=(zmin, 0.5*(zmin+zmax), zmax), gap=0.05)

        tufte_axis(axes[1][j0], bnds_x=(time_min, time_max), bnds_y=(P_MIN, P_MAX), xticks=(time_min, time_max), yticks=(P_MIN, 0.5*(P_MIN+P_MAX), P_MAX), gap=0.05)
        tufte_axis(axes[1][j1], bnds_x=(time_min, time_max), bnds_y=(P_MIN, P_MAX), xticks=(time_min, time_max), yticks=(P_MIN, 0.5*(P_MIN+P_MAX), P_MAX), gap=0.05)

        tufte_axis(axes[2][j0], bnds_x=(time_min, time_max), bnds_y=(E_MIN, E_MAX), xticks=(time_min, time_max), yticks=(E_MIN, 0.5*(E_MIN+E_MAX), E_MAX), gap=0.05)
        tufte_axis(axes[2][j1], bnds_x=(time_min, time_max), bnds_y=(E_MIN, E_MAX), xticks=(time_min, time_max), yticks=(E_MIN, 0.5*(E_MIN+E_MAX), E_MAX), gap=0.05)

        axes[0][j0].set_title("Prior")
        axes[0][j1].set_title("DSI")

        axes[0][j0].set_xlabel(LABEL_TEMP)
        axes[0][j1].set_xlabel(LABEL_TEMP)

        axes[1][j0].set_xlabel(LABEL_TIME)
        axes[1][j1].set_xlabel(LABEL_TIME)

        axes[2][j0].set_xlabel(LABEL_TIME)
        axes[2][j1].set_xlabel(LABEL_TIME)

    axes[0][0].set_ylabel(LABEL_ELEV)
    axes[1][0].set_ylabel(LABEL_PRES)
    axes[2][0].set_ylabel(LABEL_ENTH)

    fig.text(0.31, 0.975, get_well_name(well_nums[0]), ha="center", va="top", fontsize=16)
    fig.text(0.76, 0.975, get_well_name(well_nums[1]), ha="center", va="top", fontsize=16)

    for ax in axes.flat:
        ax.set_box_aspect(1)

    fig.align_labels()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(fname)


def plot_sample_comparison(temp_t_n, pres_t_n, enth_t_n, 
                           temp_pri, pres_pri, enth_pri,
                           temp_post, pres_post, enth_post,
                           well_nums, n_samples, fname):
    
    cmap = mpl.colormaps["Blues"].resampled(1000)
    cmap = cmap(np.linspace(0.25, 0.9, len(n_samples)))

    handles = []

    t_min = [20, 220, 50, 0]
    t_max = [80, 300, 280, 100]

    p_min = [3, 5, 2, 3]
    p_max = [8, 10, 8, 8]

    e_min = [400, 1000, 300, 200]
    e_max = [700, 1200, 800, 800]

    xs_temp = [np.linspace(t0, t1, 10000) for (t0, t1) in zip(t_min, t_max)]
    xs_pres = [np.linspace(p0, p1, 10000) for (p0, p1) in zip(p_min, p_max)]
    xs_enth = [np.linspace(e0, e1, 10000) for (e0, e1) in zip(e_min, e_max)]

    fig, axes = plt.subplots(3, 4, figsize=(10, 9.5))

    t_den_max = [0.30, 0.10, 0.05, 0.10]
    p_den_max = [2, 5, 3, 1]
    e_den_max = [0.05, 0.1, 0.01, 0.02]

    for j in range(4):

        kde_temp = stats.gaussian_kde(temp_pri[j], bw_method=2.0)
        p_pri, = axes[0][j].plot(xs_temp[j], kde_temp(xs_temp[j]), c="gray", lw=1.2, zorder=1, label="Prior")

        kde_pres = stats.gaussian_kde(pres_pri[j])
        axes[1][j].plot(xs_pres[j], kde_pres(xs_pres[j]), c="gray", lw=1.2, zorder=1)

        kde_enth = stats.gaussian_kde(enth_pri[j])
        axes[2][j].plot(xs_enth[j], kde_enth(xs_enth[j]), c="gray", lw=1.2, zorder=1)

        p_truth = axes[0][j].axvline(temp_t_n[j], c="k", ymin=1/12, ymax=11/12, lw=1.2, zorder=0, label="Truth")
        axes[1][j].axvline(pres_t_n[j], c="k", ymin=1/12, ymax=11/12, lw=1.2, zorder=0)
        axes[2][j].axvline(enth_t_n[j], c="k", ymin=1/12, ymax=11/12, lw=1.2, zorder=0)

        if j == 0:
            handles.extend([p_pri, p_truth])

        axes[0][j].set_xlim([t_min[j], t_max[j]])
        axes[1][j].set_xlim([p_min[j], p_max[j]])
        axes[2][j].set_xlim([e_min[j], e_max[j]]) 

        for k, temps in enumerate(temp_post):
            kde_temp = stats.gaussian_kde(temps[j])
            p, = axes[0][j].plot(xs_temp[j], kde_temp(xs_temp[j]), c=cmap[k], lw=1.2, zorder=2, label=f"DSI ($J={n_samples[k]}$)")
            if j == 0:
                handles.append(p)

        for k, press in enumerate(pres_post):
            kde_pres = stats.gaussian_kde(press[j])
            axes[1][j].plot(xs_pres[j], kde_pres(xs_pres[j]), c=cmap[k], lw=1.2, zorder=2)

        for k, enths in enumerate(enth_post):
            kde_enth = stats.gaussian_kde(enths[j])
            axes[2][j].plot(xs_enth[j], kde_enth(xs_enth[j]), c=cmap[k], lw=1.2, zorder=2)

    for ax in axes.flat:
        ax.set_box_aspect(1)

    for i in range(4):

        tufte_axis(axes[0][i], bnds_x=(t_min[i], t_max[i]), bnds_y=(0, t_den_max[i]))
        tufte_axis(axes[1][i], bnds_x=(p_min[i], p_max[i]), bnds_y=(0, np.ceil(p_den_max[i])))
        tufte_axis(axes[2][i], bnds_x=(e_min[i], e_max[i]), bnds_y=(0, e_den_max[i]))

        axes[0][i].set_title(get_well_name(well_nums[i]))
        axes[0][i].set_xlabel(LABEL_TEMP)
        axes[1][i].set_xlabel(LABEL_PRES)
        axes[2][i].set_xlabel(LABEL_ENTH)
    
    for i in range(3):
        axes[i][0].set_ylabel(LABEL_PROB)

    fig.align_labels()
    plt.figlegend(handles=handles, loc="lower center", ncols=4, frameon=False, fontsize=14)
    plt.subplots_adjust(left=0.1, bottom=0.13, right=0.95, top=0.95)
    plt.savefig(fname)


if PLOT_GRID:

    fname = f"figures/fig9a.png"
    plot_mesh(mesh_crse.fem_mesh, wells_crse, feedzone_depths, fname)

    fname = "figures/fig9b.pdf"
    plot_grid_layer(mesh_crse, cur_well_centres, new_well_centres, fname)


if PLOT_TRUTH:

    fname = "figures/fig10a.png"
    temps = get_ns_temps("models/FL13383_NS")
    logks = p_t[:mesh_fine.m.num_cells]
    plot_truth(mesh_fine, mesh_fine.fem_mesh, temps, logks, fname)

    fname = "figures/fig10b.pdf"
    plot_colourbar(CMAP_TEMP, 0, 250, LABEL_TEMP, fname)

    upflows = p_t[-mesh_fine.m.num_columns:]
    fname = "figures/fig10c.pdf"
    plot_true_upflows(mesh_fine.m, upflows, fname)


if PLOT_UPFLOWS:

    upflows = ps[-mesh_crse.m.num_columns:, 4:8].T
    fname = "figures/fig11.pdf"
    plot_upflows(mesh_crse.m, upflows, fname)


if PLOT_CAPS:

    logks = ps[:mesh_crse.m.num_cells, 1:5].T
    cap_cell_inds = [np.nonzero(k < -15.75)[0] for k in logks]

    fname = "figures/fig12.png"
    plot_caps(mesh_crse.m, mesh_crse.fem_mesh, cap_cell_inds, fname)


if PLOT_PERMS:

    logks = ps[:mesh_crse.m.num_cells, 1:5].T
    fname = "figures/fig13a.png"
    plot_slices(mesh_crse, mesh_crse.fem_mesh, logks, fname)

    fname = "figures/fig13b.pdf"
    plot_colourbar(CMAP_PERM, MIN_PERM, MAX_PERM, LABEL_PERM, fname)


if PLOT_DATA:

    well_num = 1

    elev_t = data_handler_fine.downhole_elevs[well_num]
    time_t = data_handler_fine.ts / (SECS_PER_WEEK * 52)

    temp_t = data_handler_fine.get_pr_temperatures(F_t)[well_num]
    pres_t = data_handler_fine.get_pr_pressures(F_t).T[well_num]
    enth_t = data_handler_fine.get_pr_enthalpies(F_t).T[well_num]

    elev_obs = temp_obs_zs
    time_obs = data_handler_crse.ts_prod_obs / (SECS_PER_WEEK * 52)

    temp_obs, pres_obs, enth_obs = data_handler_fine.split_obs(y)

    temp_obs = temp_obs.T[well_num]
    pres_obs = pres_obs.T[well_num]
    enth_obs = enth_obs.T[well_num]

    fname = "figures/fig14.pdf"

    plot_data(elev_t, time_t, temp_t, pres_t, enth_t, 
              elev_obs, time_obs, temp_obs, pres_obs, enth_obs, fname)


if PLOT_PRIOR_PREDICTIONS:

    well_nums = [2, 5]

    temp_t = data_handler_fine.get_pr_temperatures(F_t)
    pres_t = data_handler_fine.get_pr_pressures(F_t)
    enth_t = data_handler_fine.get_pr_enthalpies(F_t)

    temp_t = [temp_t[n] for n in well_nums]
    pres_t = pres_t.T[well_nums]
    enth_t = enth_t.T[well_nums]

    Fs_pri = np.load("data/Fs.npy")[:, :100]
    Fs_dsi = np.load("data/Fs_pri.npy")[:, :100]

    temp_pri = [np.array([data_handler_crse.get_pr_temperatures(F_i)[n] for F_i in Fs_pri.T]) for n in well_nums]
    temp_dsi = [np.array([data_handler_crse.get_pr_temperatures(F_i)[n] for F_i in Fs_dsi.T]) for n in well_nums]

    pres_pri = [np.array([data_handler_crse.get_pr_pressures(F_i).T[n] for F_i in Fs_pri.T]) for n in well_nums]
    pres_dsi = [np.array([data_handler_crse.get_pr_pressures(F_i).T[n] for F_i in Fs_dsi.T]) for n in well_nums]

    enth_pri = [np.array([data_handler_crse.get_pr_enthalpies(F_i).T[n] for F_i in Fs_pri.T]) for n in well_nums]
    enth_dsi = [np.array([data_handler_crse.get_pr_enthalpies(F_i).T[n] for F_i in Fs_dsi.T]) for n in well_nums]

    elevs_crse = [data_handler_crse.downhole_elevs[n] for n in well_nums]

    fname = "figures/fig15.pdf"

    plot_prior_vs_dsi(elevs_crse, 
                      temp_pri, pres_pri, enth_pri, 
                      temp_dsi, pres_dsi, enth_dsi, 
                      well_nums, fname)


if PLOT_DSI_PREDICTIONS_A:

    well_nums = [2, 3]

    temp_t = data_handler_fine.get_pr_temperatures(F_t)
    pres_t = data_handler_fine.get_pr_pressures(F_t)
    enth_t = data_handler_fine.get_pr_enthalpies(F_t)

    _, pres_obs, enth_obs = data_handler_fine.split_obs(y)

    temp_t = [temp_t[n] for n in well_nums]
    pres_t = pres_t.T[well_nums]
    enth_t = enth_t.T[well_nums]
    
    pres_obs = pres_obs.T[well_nums] 
    enth_obs = enth_obs.T[well_nums]

    Fs_pri = np.load("data/Fs.npy")
    Fs_dsi = np.load("data/Fs_post.npy")

    temp_pri = [np.array([data_handler_crse.get_pr_temperatures(F_i)[n] for F_i in Fs_pri.T]) for n in well_nums]
    temp_dsi = [np.array([data_handler_crse.get_pr_temperatures(F_i)[n] for F_i in Fs_dsi.T]) for n in well_nums]

    pres_pri = [np.array([data_handler_crse.get_pr_pressures(F_i).T[n] for F_i in Fs_pri.T]) for n in well_nums]
    pres_dsi = [np.array([data_handler_crse.get_pr_pressures(F_i).T[n] for F_i in Fs_dsi.T]) for n in well_nums]

    enth_pri = [np.array([data_handler_crse.get_pr_enthalpies(F_i).T[n] for F_i in Fs_pri.T]) for n in well_nums]
    enth_dsi = [np.array([data_handler_crse.get_pr_enthalpies(F_i).T[n] for F_i in Fs_dsi.T]) for n in well_nums]

    elevs_fine = [data_handler_fine.downhole_elevs[n] for n in well_nums]
    elevs_crse = [data_handler_crse.downhole_elevs[n] for n in well_nums]

    fname = "figures/fig16.pdf"

    plot_dsi_predictions(elevs_fine, temp_t, pres_t, enth_t, 
                         elevs_crse, temp_pri, pres_pri, enth_pri, 
                         temp_dsi, pres_dsi, enth_dsi,
                         pres_obs, enth_obs, well_nums, fname)


if PLOT_DSI_PREDICTIONS_B:

    well_nums = [7, 8]

    temp_t = data_handler_fine.get_pr_temperatures(F_t)
    pres_t = data_handler_fine.get_pr_pressures(F_t)
    enth_t = data_handler_fine.get_pr_enthalpies(F_t)

    temp_t = [temp_t[n] for n in well_nums]
    pres_t = pres_t.T[well_nums]
    enth_t = enth_t.T[well_nums]
    
    pres_obs = None 
    enth_obs = None

    Fs_pri = np.load("data/Fs.npy")
    Fs_dsi = np.load("data/Fs_post.npy")

    temp_pri = [np.array([data_handler_crse.get_pr_temperatures(F_i)[n] for F_i in Fs_pri.T]) for n in well_nums]
    temp_dsi = [np.array([data_handler_crse.get_pr_temperatures(F_i)[n] for F_i in Fs_dsi.T]) for n in well_nums]

    pres_pri = [np.array([data_handler_crse.get_pr_pressures(F_i).T[n] for F_i in Fs_pri.T]) for n in well_nums]
    pres_dsi = [np.array([data_handler_crse.get_pr_pressures(F_i).T[n] for F_i in Fs_dsi.T]) for n in well_nums]

    enth_pri = [np.array([data_handler_crse.get_pr_enthalpies(F_i).T[n] for F_i in Fs_pri.T]) for n in well_nums]
    enth_dsi = [np.array([data_handler_crse.get_pr_enthalpies(F_i).T[n] for F_i in Fs_dsi.T]) for n in well_nums]

    elevs_fine = [data_handler_fine.downhole_elevs[n] for n in well_nums]
    elevs_crse = [data_handler_crse.downhole_elevs[n] for n in well_nums]

    fname = "figures/fig17.pdf"

    plot_dsi_predictions(elevs_fine, temp_t, pres_t, enth_t, 
                         elevs_crse, temp_pri, pres_pri, enth_pri, 
                         temp_dsi, pres_dsi, enth_dsi,
                         pres_obs, enth_obs, well_nums, fname,
                         forecast_only=True)


if PLOT_SAMPLE_COMP: 

    well_nums = [2, 3, 7, 8]

    n_samples = [100, 250, 500, 1000, 1300]

    Fs_pri = np.load("data/Fs.npy")

    Fs_post = [np.load(f"data/Fs_post_{n}.npy") for n in n_samples]

    temp_t = data_handler_fine.get_pr_temperatures(F_t)
    pres_t = data_handler_fine.get_pr_pressures(F_t)
    enth_t = data_handler_fine.get_pr_enthalpies(F_t)

    temp_t_n = [temp_t[n][-1] for n in well_nums]
    pres_t_n = pres_t[-1][well_nums]
    enth_t_n = enth_t[-1][well_nums]

    temp_post = [[data_handler_crse.get_pr_temperatures(F_i) for F_i in Fs_post_i.T] for Fs_post_i in Fs_post]
    temp_post = [np.array([[t[-1] for t in temp] for temp in temp_post_i]).T[well_nums] for temp_post_i in temp_post]

    pres_post = [np.array([data_handler_crse.get_pr_pressures(F_i)[-1] for F_i in Fs_post_i.T]).T[well_nums] for Fs_post_i in Fs_post]
    enth_post = [np.array([data_handler_crse.get_pr_enthalpies(F_i)[-1] for F_i in Fs_post_i.T]).T[well_nums] for Fs_post_i in Fs_post]

    temp_pri = [data_handler_crse.get_pr_temperatures(F_i) for F_i in Fs_pri.T]
    temp_pri = np.array([[t[-1] for t in temp] for temp in temp_pri]).T
    pres_pri = np.array([data_handler_crse.get_pr_pressures(F_i)[-1] for F_i in Fs_pri.T]).T
    enth_pri = np.array([data_handler_crse.get_pr_enthalpies(F_i)[-1] for F_i in Fs_pri.T]).T

    temp_pri = temp_pri[well_nums]
    pres_pri = pres_pri[well_nums]
    enth_pri = enth_pri[well_nums]

    fname = "figures/fig18.pdf"

    plot_sample_comparison(temp_t_n, pres_t_n, enth_t_n, 
                           temp_pri, pres_pri, enth_pri,
                           temp_post, pres_post, enth_post,
                           well_nums, n_samples, fname)