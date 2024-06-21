import cmocean
import h5py
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pyvista as pv
from scipy import stats

from setup import *


plt.rc("text", usetex=True)
plt.rc("font", family="serif")

TITLE_SIZE = 16
LABEL_SIZE = 12
LEGEND_SIZE = 10
TICK_SIZE = 10

LIMS_XS = (0, 6000)
LIMS_TS = (0, 2)
LIMS_UPFLOW = (0, 2.75e-4)

MIN_PERM = -17.0
MAX_PERM = -12.5

P_MIN, P_MAX = 3, 11
E_MIN, E_MAX = 100, 1500
T_MIN, T_MAX = 20, 300

TICKS_XS = [0, 3000, 6000]
TICKS_TS = [0, 1, 2]

LABEL_X1 = "$x_{1}$ [m]"
LABEL_X2 = "$x_{2}$ [m]"
LABEL_TEMP = "Temperature [$^\circ$C]"
LABEL_TIME = "Time [Years]"
LABEL_ELEV = "Elevation [m]"
LABEL_PERM = "log$_{10}$(Permeability) [log$_{10}$(m$^2$)]"
LABEL_PRES = "Pressure [MPa]"
LABEL_ENTH = "Enthalpy [kJ kg$^{-1}$]"
LABEL_UPFLOW = "Upflow [kg s$^{-1}$ m$^{-3}$]"
LABEL_PROB = "Probability Density"

COL_CUR_WELLS = "royalblue"
COL_NEW_WELLS = "firebrick"

COL_TEMP = "coral"
COL_PRES = "lightskyblue"
COL_ENTH = "lightgreen"
COL_CAP = "peru"

COL_GRID = "darkgrey"
COL_DEEP = "whitesmoke"

CMAP_PERM = cmocean.cm.turbid.reversed()
CMAP_UPFLOW = cmocean.cm.thermal
CMAP_TEMP = cmocean.cm.balance

PLOT_GRID = True
PLOT_TRUTH = True
PLOT_UPFLOWS = True
PLOT_CAPS = True
PLOT_PERMS = True
PLOT_DATA = False
PLOT_DSI_PREDICTIONS = False
PLOT_TRANSFORMATION = False
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
    cbar = plt.colorbar(m, ax=ax)
    cbar.set_label(label, fontsize=LABEL_SIZE)

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
                 color=COL_CUR_WELLS, fontsize=TICK_SIZE)
    
    for i, (x, y) in enumerate(new_well_centres):
        ax.scatter(x, y, s=20, c=COL_NEW_WELLS)
        plt.text(x-360, y+160, s=get_well_name(len(cur_well_centres)+i), 
                 color=COL_NEW_WELLS, fontsize=TICK_SIZE)
    
    ax.set_xlabel(LABEL_X1, fontsize=LABEL_SIZE)
    ax.set_ylabel(LABEL_X2, fontsize=LABEL_SIZE)
    
    for s in ax.spines.values():
        s.set_edgecolor(COL_GRID)
    
    ax.set_xlim(LIMS_XS)
    ax.set_ylim(LIMS_XS)
    ax.set_xticks(TICKS_XS)
    ax.set_yticks(TICKS_XS)
    ax.tick_params(labelsize=TICK_SIZE, length=0)
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

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
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
    ax.set_xlabel(LABEL_X1, fontsize=LABEL_SIZE)
    ax.set_ylabel(LABEL_X2, fontsize=LABEL_SIZE)

    cbar = fig.colorbar(p, ax=ax, label=LABEL_UPFLOW)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
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

    axes[0].plot(temp, elev, c="k", zorder=2)
    axes[1].plot(time, pres, c="k", zorder=2)
    axes[2].plot(time, enth, c="k", zorder=2)

    axes[0].scatter(temp_obs, elev_obs, c="k", s=10, zorder=2)
    axes[1].scatter(time_obs, pres_obs, c="k", s=10, zorder=2)
    axes[2].scatter(time_obs, enth_obs, c="k", s=10, zorder=2)

    axes[1].axvline(1, ls="--", c="gray", ymin=1/12, ymax=11/12, zorder=1)
    axes[2].axvline(1, ls="--", c="gray", ymin=1/12, ymax=11/12, zorder=1)

    axes[0].set_xlabel(LABEL_TEMP, fontsize=LABEL_SIZE)
    axes[1].set_xlabel(LABEL_TIME, fontsize=LABEL_SIZE)
    axes[2].set_xlabel(LABEL_TIME, fontsize=LABEL_SIZE)

    axes[0].set_ylabel(LABEL_ELEV, fontsize=LABEL_SIZE)
    axes[1].set_ylabel(LABEL_PRES, fontsize=LABEL_SIZE)
    axes[2].set_ylabel(LABEL_ENTH, fontsize=LABEL_SIZE)

    for ax in axes:
        ax.set_box_aspect(1)

    ymin = min(elev)
    ymax = max(elev)

    tufte_axis(axes[0], bnds_x=(0, 250), bnds_y=(ymin, ymax))
    tufte_axis(axes[1], bnds_x=LIMS_TS, bnds_y=(4, 8), xticks=TICKS_TS)
    tufte_axis(axes[2], bnds_x=LIMS_TS, bnds_y=(600, 750), xticks=TICKS_TS)

    plt.tight_layout()
    plt.savefig(fname)


def plot_upflows(mesh: lm.mesh, upflows, fname):

    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 0.25*FULL_WIDTH), 
                             layout="constrained", sharey=True)
    
    polys = get_layer_polys(mesh, cmap=CMAP_UPFLOW)
    polys.set_clim(LIMS_UPFLOW)

    for i, ax in enumerate(axes.flat):

        polys_i = deepcopy(polys)
        p = ax.add_collection(polys_i)
        polys_i.set_array(upflows[i])

        ax.set_xlim(LIMS_XS)
        ax.set_ylim(LIMS_XS)
        ax.set_box_aspect(1)
        ax.set_xticks(LIMS_XS)
        ax.set_yticks(LIMS_XS)
        ax.tick_params(which="both", bottom=False, left=False)
        ax.set_xlabel(LABEL_X1, fontsize=LABEL_SIZE)

    axes[0].set_ylabel(LABEL_X2, fontsize=LABEL_SIZE)
    cbar = fig.colorbar(p, ax=axes[-1], label=LABEL_UPFLOW)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.ax.ticklabel_format(style="sci", scilimits=(0, 0))

    plt.savefig(fname)


def plot_dsi_predictions(elevs_fine, temp_t, pres_t, enth_t, 
                         elevs_crse, temp_pri, pres_pri, enth_pri, 
                         temp_dsi, pres_dsi, enth_dsi,
                         pres_obs, enth_obs, well_nums, fname):

    fig, axes = plt.subplots(3, 4, figsize=(10, 7.8))

    ts_crse = data_handler_crse.ts / (SECS_PER_WEEK * 52)
    ts_fine = data_handler_fine.ts / (SECS_PER_WEEK * 52)

    temp_qs_pri = [np.quantile(t, q=(0.025, 0.975), axis=0) for t in temp_pri]
    pres_qs_pri = [np.quantile(p, q=(0.025, 0.975), axis=0) for p in pres_pri]
    enth_qs_pri = [np.quantile(e, q=(0.025, 0.975), axis=0) for e in enth_pri]

    temp_qs_dsi = [np.quantile(t, q=(0.025, 0.975), axis=0) for t in temp_dsi]
    pres_qs_dsi = [np.quantile(p, q=(0.025, 0.975), axis=0) for p in pres_dsi]
    enth_qs_dsi = [np.quantile(e, q=(0.025, 0.975), axis=0) for e in enth_dsi]
    
    for i in range(2):

        j0 = 2*i
        j1 = 2*i+1

        axes[0][j0].plot(temp_pri[i].T, elevs_crse[i], c=COL_TEMP, lw=0.5, alpha=0.4, zorder=1)
        axes[0][j1].plot(temp_dsi[i].T, elevs_crse[i], c=COL_TEMP, lw=0.5, alpha=0.4, zorder=1)

        axes[0][j0].plot(temp_qs_pri[i].T, elevs_crse[i], c="dimgrey", lw=1.0, zorder=2)
        axes[0][j1].plot(temp_qs_dsi[i].T, elevs_crse[i], c="dimgrey", lw=1.0, zorder=2)
            
        axes[1][j0].plot(ts_crse, pres_pri[i].T, c=COL_PRES, lw=0.5, alpha=0.4, zorder=1)
        axes[1][j1].plot(ts_crse, pres_dsi[i].T, c=COL_PRES, lw=0.5, alpha=0.4, zorder=1)

        axes[1][j0].plot(ts_crse, pres_qs_pri[i].T, c="dimgrey", lw=1.0, zorder=2)
        axes[1][j1].plot(ts_crse, pres_qs_dsi[i].T, c="dimgrey", lw=1.0, zorder=2)

        axes[2][j0].plot(ts_crse, enth_pri[i].T, c=COL_ENTH, lw=0.5, alpha=0.4, zorder=1)
        axes[2][j1].plot(ts_crse, enth_dsi[i].T, c=COL_ENTH, lw=0.5, alpha=0.4, zorder=1)

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

        axes[1][j0].axvline(1, ls="--", c="gray", lw=1, ymin=1/12, ymax=11/12, zorder=0)
        axes[1][j1].axvline(1, ls="--", c="gray", lw=1, ymin=1/12, ymax=11/12, zorder=0)

        axes[2][j0].axvline(1, ls="--", c="gray", lw=1, ymin=1/12, ymax=11/12, zorder=0)
        axes[2][j1].axvline(1, ls="--", c="gray", lw=1, ymin=1/12, ymax=11/12, zorder=0)

        zmin = min(elevs_fine[i])
        zmax = max(elevs_fine[i])

        tufte_axis(axes[0][j0], bnds_x=(T_MIN, T_MAX), bnds_y=(zmin, zmax), xticks=(T_MIN, 0.5*(T_MIN+T_MAX), T_MAX), yticks=(zmin, 0.5*(zmin+zmax), zmax))
        tufte_axis(axes[0][j1], bnds_x=(T_MIN, T_MAX), bnds_y=(zmin, zmax), xticks=(T_MIN, 0.5*(T_MIN+T_MAX), T_MAX), yticks=(zmin, 0.5*(zmin+zmax), zmax))

        tufte_axis(axes[1][j0], bnds_x=(0, 2), bnds_y=(P_MIN, P_MAX), xticks=(0, 1, 2), yticks=(P_MIN, 0.5*(P_MIN+P_MAX), P_MAX))
        tufte_axis(axes[1][j1], bnds_x=(0, 2), bnds_y=(P_MIN, P_MAX), xticks=(0, 1, 2), yticks=(P_MIN, 0.5*(P_MIN+P_MAX), P_MAX))

        tufte_axis(axes[2][j0], bnds_x=(0, 2), bnds_y=(E_MIN, E_MAX), xticks=(0, 1, 2), yticks=(E_MIN, 0.5*(E_MIN+E_MAX), E_MAX))
        tufte_axis(axes[2][j1], bnds_x=(0, 2), bnds_y=(E_MIN, E_MAX), xticks=(0, 1, 2), yticks=(E_MIN, 0.5*(E_MIN+E_MAX), E_MAX))

        axes[0][j0].set_title("Prior", fontsize=LABEL_SIZE)
        axes[0][j1].set_title("DSI", fontsize=LABEL_SIZE)

        axes[0][j0].set_xlabel(LABEL_TEMP, fontsize=LABEL_SIZE)
        axes[0][j1].set_xlabel(LABEL_TEMP, fontsize=LABEL_SIZE)

        axes[1][j0].set_xlabel(LABEL_TIME, fontsize=LABEL_SIZE)
        axes[1][j1].set_xlabel(LABEL_TIME, fontsize=LABEL_SIZE)

        axes[2][j0].set_xlabel(LABEL_TIME, fontsize=LABEL_SIZE)
        axes[2][j1].set_xlabel(LABEL_TIME, fontsize=LABEL_SIZE)

    axes[0][0].set_ylabel(LABEL_ELEV, fontsize=LABEL_SIZE)
    axes[1][0].set_ylabel(LABEL_PRES, fontsize=LABEL_SIZE)
    axes[2][0].set_ylabel(LABEL_ENTH, fontsize=LABEL_SIZE)

    fig.text(0.27, 0.975, f"Well {well_nums[0]+1}", ha="center", va="top", fontsize=LABEL_SIZE)
    fig.text(0.76, 0.975, f"Well {well_nums[1]+1}", ha="center", va="top", fontsize=LABEL_SIZE)

    for ax in axes.flat:
        ax.set_box_aspect(1)

    fig.align_labels()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(fname)


def plot_transformation(pres_t, pres_obs, pres_dsi, pres_trn, well_nums, fname):
    
    ts_crse = data_handler_crse.ts / (SECS_PER_WEEK * 52)
    ts_fine = data_handler_fine.ts / (SECS_PER_WEEK * 52)
    ts_obs = data_handler_fine.ts_prod_obs / (SECS_PER_WEEK * 52)

    pres_qs_dsi = [np.quantile(p, q=(0.025, 0.975), axis=0) for p in pres_dsi]
    pres_qs_trn = [np.quantile(p, q=(0.025, 0.975), axis=0) for p in pres_trn]

    _, axes = plt.subplots(2, 4, figsize=(10, 5))

    for i in range(4):

        axes[0][i].plot(ts_fine, pres_t[i], c="k", lw=1.5, zorder=3)
        axes[1][i].plot(ts_fine, pres_t[i], c="k", lw=1.5, zorder=3)

        axes[0][i].plot(ts_crse, pres_dsi[i].T, c=COL_PRES, lw=0.5, alpha=0.4, zorder=1)
        axes[1][i].plot(ts_crse, pres_trn[i].T, c=COL_PRES, lw=0.5, alpha=0.4, zorder=1)

        axes[0][i].plot(ts_crse, pres_qs_dsi[i].T, c="dimgrey", lw=1.0)
        axes[1][i].plot(ts_crse, pres_qs_trn[i].T, c="dimgrey", lw=1.0)

        if pres_obs[i] is not None:
            axes[0][i].scatter(ts_obs, pres_obs[i], c="k", s=6, zorder=4)
            axes[1][i].scatter(ts_obs, pres_obs[i], c="k", s=6, zorder=4)

    for ax in axes.flat:
        ax.axvline(1, ls="--", c="gray", lw=1, ymin=1/12, ymax=11/12, zorder=0)
        ax.set_box_aspect(1)
        tufte_axis(ax, bnds_x=(0, 2), bnds_y=(1, 9), xticks=(0, 1, 2), yticks=(1, 5, 9))
        
    for ax in axes[-1]:
        ax.set_xlabel(LABEL_TIME, fontsize=LABEL_SIZE)

    axes[0][0].set_ylabel(LABEL_PRES, fontsize=LABEL_SIZE)
    axes[1][0].set_ylabel(LABEL_PRES, fontsize=LABEL_SIZE)

    for i, n in enumerate(well_nums):
        axes[0][i].set_title(f"Well {n+1}", fontsize=LABEL_SIZE)

    plt.tight_layout()
    plt.savefig(fname)


def plot_sample_comparison(temp_t_n, pres_t_n, enth_t_n, 
                           temp_pri, pres_pri, enth_pri,
                           temp_ms, temp_ss,
                           pres_ms, pres_ss, 
                           enth_ms, enth_ss, 
                           well_nums, n_samples, fname):
    
    cmap = mpl.colormaps["Blues"].resampled(1000)
    cmap = cmap(np.linspace(0.25, 0.9, len(temp_ms)))

    handles = []

    t_min = [20, 20, 20, 20]
    t_max = [250, 250, 300, 250]

    p_min = [4, 2, 5, 5]
    p_max = [9, 9, 10, 7]

    e_min = [800, 400, 700, 100]
    e_max = [1100, 800, 1400, 1100]

    xs_temp = [np.linspace(t0, t1, 1000) for (t0, t1) in zip(t_min, t_max)]
    xs_pres = [np.linspace(p0, p1, 1000) for (p0, p1) in zip(p_min, p_max)]
    xs_enth = [np.linspace(e0, e1, 1000) for (e0, e1) in zip(e_min, e_max)]

    _, axes = plt.subplots(3, 4, figsize=(10, 8))

    t_den_max = [0.2, 0.25, 0.40, 0.25]
    p_den_max = [9, 6, 10, 24]
    e_den_max = [0.1, 0.1, 0.06, 0.07]

    for j in range(4):

        kde_temp = stats.gaussian_kde(temp_pri[j])
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

        for k, (m, s) in enumerate(zip(temp_ms, temp_ss)):
            density = stats.norm(m[j], s[j]).pdf(xs_temp[j])
            p, = axes[0][j].plot(xs_temp[j], density, c=cmap[k], lw=1.2, zorder=2, label=f"DSI ($l={n_samples[k]}$)")
            if j == 0:
                handles.append(p)

        for k, (m, s) in enumerate(zip(pres_ms, pres_ss)):
            density = stats.norm(m[j], s[j]).pdf(xs_pres[j])
            axes[1][j].plot(xs_pres[j], density, c=cmap[k], lw=1.2, zorder=2)

        for k, (m, s) in enumerate(zip(enth_ms, enth_ss)):
            density = stats.norm(m[j], s[j]).pdf(xs_enth[j])
            axes[2][j].plot(xs_enth[j], density, c=cmap[k], lw=1.2, zorder=2)

    for ax in axes.flat:
        ax.set_box_aspect(1)

    for i in range(4):

        tufte_axis(axes[0][i], bnds_x=(t_min[i], t_max[i]), bnds_y=(0, t_den_max[i]))
        tufte_axis(axes[1][i], bnds_x=(p_min[i], p_max[i]), bnds_y=(0, np.ceil(p_den_max[i])))
        tufte_axis(axes[2][i], bnds_x=(e_min[i], e_max[i]), bnds_y=(0, e_den_max[i]))

        axes[0][i].set_title(f"Well {well_nums[i]+1}", fontsize=LABEL_SIZE)
        axes[0][i].set_xlabel(LABEL_TEMP, fontsize=LABEL_SIZE)
        axes[1][i].set_xlabel(LABEL_PRES, fontsize=LABEL_SIZE)
        axes[2][i].set_xlabel(LABEL_ENTH, fontsize=LABEL_SIZE)
    
    for i in range(3):
        axes[i][0].set_ylabel(LABEL_PROB, fontsize=LABEL_SIZE)

    plt.figlegend(handles=handles, loc="lower center", ncols=4, frameon=False, fontsize=TICK_SIZE)
    plt.subplots_adjust(left=0.1, bottom=0.13, right=0.95, top=0.95)
    plt.savefig(fname)


if PLOT_GRID:

    fname = f"figures/fig7a.png"
    plot_mesh(mesh_crse.fem_mesh, wells_crse, feedzone_depths, fname)

    fname = "figures/fig7b.pdf"
    plot_grid_layer(mesh_crse, cur_well_centres, new_well_centres, fname)


if PLOT_TRUTH:

    fname = "figures/fig8a.png"
    temps = get_ns_temps("models/FL13383_NS")
    logks = p_t[:mesh_fine.m.num_cells]
    plot_truth(mesh_fine, mesh_fine.fem_mesh, temps, logks, fname)

    fname = "figures/fig8b.pdf"
    plot_colourbar(CMAP_TEMP, 0, 250, LABEL_TEMP, fname)

    upflows = p_t[mesh_fine.m.num_cells:]
    fname = "figures/fig8c.pdf"
    plot_true_upflows(mesh_fine.m, upflows, fname)


if PLOT_UPFLOWS:

    upflows = ps[mesh_crse.m.num_cells:, 4:8].T
    fname = "figures/fig9.pdf"
    plot_upflows(mesh_crse.m, upflows, fname)


if PLOT_CAPS:

    logks = ps[:mesh_crse.m.num_cells, 1:5].T
    cap_cell_inds = [np.nonzero(k < -15.75)[0] for k in logks]

    fname = "figures/fig10.png"
    plot_caps(mesh_crse.m, mesh_crse.fem_mesh, cap_cell_inds, fname)


if PLOT_PERMS:

    logks = ps[:mesh_crse.m.num_cells, 1:5].T
    fname = "figures/fig11a.png"
    plot_slices(mesh_crse, mesh_crse.fem_mesh, logks, fname)

    fname = "figures/fig11b.pdf"
    plot_colourbar(CMAP_PERM, MIN_PERM, MAX_PERM, LABEL_PERM, fname)


if PLOT_DATA:

    well_num = 0

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

    fname = "figures/fig12.pdf"

    plot_data(elev_t, time_t, temp_t, pres_t, enth_t, 
              elev_obs, time_obs, temp_obs, pres_obs, enth_obs, fname)


if PLOT_DSI_PREDICTIONS:

    well_nums = [7, 8]

    temp_t = data_handler_fine.get_pr_temperatures(F_t)
    pres_t = data_handler_fine.get_pr_pressures(F_t)
    enth_t = data_handler_fine.get_pr_enthalpies(F_t)

    temp_obs, pres_obs, enth_obs = data_handler_fine.split_obs(y)

    temp_t = [temp_t[n] for n in well_nums]
    pres_t = pres_t.T[well_nums]
    enth_t = enth_t.T[well_nums]
    
    pres_obs_n = None 
    enth_obs_n = None

    Fs_pri = np.load("data/Fs.npy")
    Fs_dsi = np.load("data/Fs_post.npy").T

    temp_pri = [np.array([data_handler_crse.get_pr_temperatures(F_i)[n] for F_i in Fs_pri.T]) for n in well_nums]
    temp_dsi = [np.array([data_handler_crse.get_pr_temperatures(F_i)[n] for F_i in Fs_dsi.T]) for n in well_nums]

    pres_pri = [np.array([data_handler_crse.get_pr_pressures(F_i).T[n] for F_i in Fs_pri.T]) for n in well_nums]
    pres_dsi = [np.array([data_handler_crse.get_pr_pressures(F_i).T[n] for F_i in Fs_dsi.T]) for n in well_nums]

    enth_pri = [np.array([data_handler_crse.get_pr_enthalpies(F_i).T[n] for F_i in Fs_pri.T]) for n in well_nums]
    enth_dsi = [np.array([data_handler_crse.get_pr_enthalpies(F_i).T[n] for F_i in Fs_dsi.T]) for n in well_nums]

    elevs_fine = [data_handler_fine.downhole_elevs[n] for n in well_nums]
    elevs_crse = [data_handler_crse.downhole_elevs[n] for n in well_nums]

    fname = "figures/fig14.pdf"

    plot_dsi_predictions(elevs_fine, temp_t, pres_t, enth_t, 
                         elevs_crse, temp_pri, pres_pri, enth_pri, 
                         temp_dsi, pres_dsi, enth_dsi,
                         pres_obs_n, enth_obs_n, well_nums, fname)


if PLOT_TRANSFORMATION:

    Fs_dsi = np.load("data/Fs_post.npy").T
    Fs_trn = np.load("data/Fs_post_trans.npy")

    well_nums = [2, 3, 7, 8]

    pres_t = data_handler_fine.get_pr_pressures(F_t).T[well_nums]

    pres_obs = data_handler_fine.split_obs(y)[1]
    pres_obs = [*pres_obs.T[[2, 3]], None, None]

    pres_dsi = [np.array([data_handler_crse.get_pr_pressures(F_i).T[n] for F_i in Fs_dsi.T]) for n in well_nums]
    pres_trn = [np.array([data_handler_crse.get_pr_pressures(F_i).T[n] for F_i in Fs_trn.T]) for n in well_nums]

    fname = "figures/fig15.pdf"

    plot_transformation(pres_t, pres_obs, pres_dsi, pres_trn, well_nums, fname)


if PLOT_SAMPLE_COMP: 

    well_nums = [2, 3, 7, 8]

    n_samples = [10, 100, 250, 500, 676]

    Fs_pri = np.load("data/Fs.npy")

    ms = [np.load(f"data/m_post_{n}.npy") for n in n_samples]
    Cs = [np.load(f"data/C_post_{n}.npy") for n in n_samples]
    ss = [np.sqrt(np.diag(C)) for C in Cs]

    temp_t = data_handler_fine.get_pr_temperatures(F_t)
    pres_t = data_handler_fine.get_pr_pressures(F_t)
    enth_t = data_handler_fine.get_pr_enthalpies(F_t)

    temp_t_n = [temp_t[n][-1] for n in well_nums]
    pres_t_n = pres_t[-1][well_nums]
    enth_t_n = enth_t[-1][well_nums]

    temp_ms = [data_handler_crse.get_pr_temperatures(m) for m in ms]
    temp_ss = [data_handler_crse.get_pr_temperatures(s) for s in ss]

    temp_ms = np.array([[m[n][-1] for n in well_nums] for m in temp_ms])
    temp_ss = np.array([[s[n][-1] for n in well_nums] for s in temp_ss])

    pres_ms = np.array([data_handler_crse.get_pr_pressures(m)[-1][well_nums] for m in ms])
    pres_ss = np.array([data_handler_crse.get_pr_pressures(s)[-1][well_nums] for s in ss])

    enth_ms = np.array([data_handler_crse.get_pr_enthalpies(m)[-1][well_nums] for m in ms])
    enth_ss = np.array([data_handler_crse.get_pr_enthalpies(s)[-1][well_nums] for s in ss])

    temp_pri = [data_handler_crse.get_pr_temperatures(F_i) for F_i in Fs_pri.T]
    temp_pri = np.array([[t[-1] for t in temp] for temp in temp_pri]).T
    pres_pri = np.array([data_handler_crse.get_pr_pressures(F_i)[-1] for F_i in Fs_pri.T]).T
    enth_pri = np.array([data_handler_crse.get_pr_enthalpies(F_i)[-1] for F_i in Fs_pri.T]).T

    temp_pri = temp_pri[well_nums]
    pres_pri = pres_pri[well_nums]
    enth_pri = enth_pri[well_nums]

    fname = "figures/fig16.pdf"

    plot_sample_comparison(temp_t_n, pres_t_n, enth_t_n, 
                           temp_pri, pres_pri, enth_pri,
                           temp_ms, temp_ss,
                           pres_ms, pres_ss, 
                           enth_ms, enth_ss,
                           well_nums, n_samples, fname)