from matplotlib import pyplot as plt

plt.style.use("paperstyle.mplstyle")


def tufte_axis(ax, bnds_x, bnds_y, gap=0.05, xticks=None, yticks=None):

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


fig, ax = plt.subplots(figsize=(6, 4))

zs = [-1.0, 0.0, 3.2, 4.4, 6.0, 7.2, 9.5, 11.0, 12.0]
Fs = [0.0, 0.0, 0.02, 0.25, 0.50, 0.75, 0.98, 1.0, 1.0]

ax.plot(zs, Fs, c="k", zorder=2)
ax.plot([-3.0, -1.0], [0.0, 0.0], c="k", zorder=2, ls="--")
ax.plot([12.0, 14.0], [1.0, 1.0], c="k", zorder=2, ls="--")
ax.scatter(zs[1:-1], Fs[1:-1], s=15, c="k", zorder=2)
ax.axhline(y=0.0, xmin=3/22, xmax=21/22, c="grey", ls="dotted", zorder=1)
ax.axhline(y=1.0, xmin=1/22, xmax=19/22, c="grey", ls="dotted", zorder=1)

tufte_axis(
    ax, 
    bnds_x=(-3.0, 14.0), 
    bnds_y=(0.0, 1.0), 
    xticks=zs[1:-1],
    yticks=(0.0, 0.25, 0.5, 0.75, 1.0)
)

xticklabels = [
    r"$y^{\mathrm{min}}_{i}$", 
    r"$y^{(2)}_{i}$", 
    r"$y^{(5)}_{i}$", 
    r"$y^{(4)}_{i}$", 
    r"$y^{(3)}_{i}$", 
    r"$y^{(1)}_{i}$", 
    r"$y^{\mathrm{max}}_{i}$", 
]

ax.set_xticklabels(labels=xticklabels)
ax.set_xlabel(r"$y_{i}$")
ax.set_ylabel(r"$\mathcal{F}^{\bm{y}}_{i}(y_{i})$")

plt.tight_layout()
plt.savefig("fig1.pdf")
