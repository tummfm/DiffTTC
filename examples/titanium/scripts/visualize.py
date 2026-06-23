"""Functions to evaluate and visualize results from multiple computations."""

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "figure.figsize": (7, 3),
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "legend.fontsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "lines.markersize": 4,
    "lines.linewidth": 1.5,
    "figure.dpi": 400,
    'figure.constrained_layout.use': True,
    "text.latex.preamble": "\n".join([
        r"\usepackage{amsmath}",
        r"\usepackage{helvet}",
        r"\usepackage{sfmath}",
        r"\renewcommand{\familydefault}{\sfdefault}",
        r"\usepackage{siunitx}",
        r"\DeclareSIUnit\atom{atom}",
    ])
})

import numpy as onp

def plot_fe_diff(press, temps, fa_hcp, fa_bcc, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(7, 3))
    else:
        fig = ax.get_figure()

    ax.scatter(press, fa_hcp, marker="x", label="HCP")
    ax.scatter(press, fa_bcc, marker="+", label="BCC")
    ax.plot([press, press], [fa_hcp, fa_hcp], c="k")

    for ti, pi, hi, bi in zip(temps, press, fa_hcp, fa_bcc):
        ax.annotate(
            f"$T = {ti} K$\n$\Delta G \\approx {hi - bi:.4f} \\frac{{kJ}}{{mol\\cdot atom}}$",
            (pi, (hi + bi) / 2), xytext=(0.0, -0.1), textcoords="offset points",
            ha="center", va="bottom", fontsize=6)

    ax.set_xlabel("Pressure [GPa]")
    ax.set_ylabel("Free energy per atom [kJ/mol]")
    ax.set_xlim([-1, 6])
    ax.set_ylim([-100, 0])
    ax.set_xticks(onp.arange(6), onp.arange(6))
    ax.legend()

    return fig

def plot_transition_temperature(temps, f_hcp, f_bcc, t_melt=None, labels=None):
    fig, axes = plt.subplots(int(onp.ceil(len(temps) / 3)), 3, figsize=(7, 3), layout="constrained")
    for idx, ax in enumerate(axes.flat):
        ax.annotate(f"\\textbf{{({chr(ord('`') + idx + 1)})}}", (0, 1),
                    (-35, 2), xycoords="axes fraction",
                    textcoords="offset points", ha="center", va="bottom")

    axes = axes.ravel()

    if labels is None:
        labels=["HCP", "BCC"]

    TT = []
    dG = []
    for idx, (T, F_hcp, F_bcc) in enumerate(zip(temps, f_hcp, f_bcc)):
        if t_melt is None:
            eq = onp.argmin(onp.abs(F_hcp - F_bcc))
            label=f"$T \\approx {T[eq] :.1f} K$"
        else:
            eq = onp.argmin(onp.abs(T - t_melt[idx]))
            F_hcp -= F_hcp[eq]
            F_bcc -= F_bcc[eq]
            dG.append(F_hcp[0] - F_bcc[0])

            label=f"$\\Delta G \\approx {F_hcp[0] - F_bcc[0] :.3f}~\\frac{{\\mathrm{{kJ}}}}{{\\mathrm{{mol}}\\cdot\\mathrm{{atom}}}}$"

        axes[idx].plot(T, F_hcp, label=labels[0])
        axes[idx].plot(T, F_bcc, label=labels[1])

        if t_melt is None:
            axes[idx].axvline(T[eq], color="red", linestyle="--", label=label)
        else:
            axes[idx].axvline(T[0], color="red", linestyle="--", label=label)

        axes[idx].legend()
        axes[idx].set_xlabel("Temperature [K]")
        axes[idx].set_ylabel("G [kJ/mol/atom]")
        axes[idx].set_title(f"P = {idx} GPa")

        TT.append(T[eq])

    if t_melt is not None:
        return fig, onp.asarray(dG)
    else:
        return fig, onp.asarray(TT)


def plot_rs_cst(temps, press, forward, backward, col=1):
    label = ["$f_\\text{fcc}$", "$f_\\text{hcp}$", "$f_\\text{bcc}$", "$f_\\text{ico}$", "$f_\\text{amp}$"][col]
    fig, axes = plt.subplots(int(onp.ceil(len(temps) / 3)), 3, figsize=(7, 3), layout="constrained")
    axes = axes.ravel()

    for idx, (T, P, fcst, bcst) in enumerate(zip(temps, press, forward, backward)):
        axes[idx].plot(onp.linspace(T.min(), T.max(), fcst[col, :].size), fcst[col, :], label=f"Forward {label}")
        axes[idx].plot(onp.linspace(T.min(), T.max(), fcst[col, :].size), bcst[col, ::-1], label=f"Backward {label}")
        axes[idx].legend()
        axes[idx].set_xlabel("Temperature")
        axes[idx].set_ylabel("Fraction of Crystal Structure")
        axes[idx].set_title(f"P = {idx} GPa, T = {idx} K")

    return fig


def plot_statepoints(paths, col=0, label="Quantity", unit="", structure="hcp", delimiter=";"):
    time_step = 0.004  # in ps, used for x-axis scaling
    segments = 10

    fig, axes = plt.subplots(len(paths) //  3, 3, figsize=(7, 3), layout="constrained", sharex=True, sharey=False)

    for idx, ax in enumerate(axes.flat):
        ax.annotate(f"\\textbf{{({chr(ord('`') + idx + 1)})}}", (0, 1),
                    (-35, 2), xycoords="axes fraction",
                    textcoords="offset points", ha="center", va="bottom")

    axes = axes.ravel()

    # Sort the paths to increasing pressure
    press = []
    for idx, path in enumerate(paths):
        # Read out estimated temp and simulated pressure from filename
        _, _, name = path.name.partition(f"{structure}_equilibrated_")
        name, _, _ = name.partition(".csv")
        print(name)
        press.append(name.split("_")[1])

    sort_idx = onp.argsort(onp.array(press).astype(float))

    means = []
    temps = []
    press = []

    for idx, path in enumerate(onp.asarray(paths)[sort_idx]):
        # Read out estimated temp and simulated pressure from filename
        _, _, name = path.name.partition(f"{structure}_equilibrated_")
        name, _, _ = name.partition(".csv")
        _, pressure, _, temp, *_ = name.split("_")

        # Load coexistence temperature and crystal data
        data = onp.loadtxt(path, skiprows=1, delimiter=delimiter)
        values = data[:,col]
        time = 0.004 * onp.arange(values.size)

        axes[idx].plot(time / 1000., values)
        axes[idx].axhline(values.mean(), label=f"{label} = {values.mean():.4f} {unit}")
        axes[idx].legend(loc="upper right")

        axes[idx].set_title(f"$P = {pressure}$ bar, $T = {temp}$ K", fontsize=10)

        if idx % 3 == 0:
            axes[idx].get_yaxis().set_label_text(f"{label} [{unit}]")

        means.append(values.mean())
        temps.append(float(temp))
        press.append(float(pressure))

    axes[4].set_xlabel("Time [ps]")

    return fig, onp.asarray(means), onp.asarray(temps), onp.asarray(press)


def plot_coexistence(paths, temps, press, delta=None, title="Coexistence", err_est=None):
    fig, axes = plt.subplots(int(onp.ceil(len(paths) / 3)), 3, figsize=(7, int(onp.ceil(len(paths)  /  3)) * 1.75), layout="constrained", sharex=True, sharey=False)

    for idx, ax in enumerate(axes.flat):
        ax.annotate(f"\\textbf{{({chr(ord('`') + idx + 1)})}}", (0, 1),
                    (-35, 2), xycoords="axes fraction",
                    textcoords="offset points", ha="center", va="bottom")

    axes = axes.ravel()

    print(f"Create {int(onp.ceil(len(paths) / 3))} rows for {len(paths)} plots.")

    time_step = 0.004  # in ps, used for x-axis scaling

    segments = 10

    if delta is None:
        delta = onp.zeros_like(temps)

    sort_idx = onp.lexsort((temps, press))
    # sort_idx = onp.arange(len(paths))[sort_idx]

    est_temps = onp.zeros(len(paths))
    avg_cst = onp.zeros(len(paths))

    for idx, (path, temp, pressure, delta) in enumerate(zip(paths, temps, press, delta)):
        idx = onp.argmax(sort_idx == idx)

        # Load coexistence temperature and crystal data
        coexistence_data = onp.loadtxt(path / "coexistence.txt", skiprows=1, delimiter=" ")
        crystal_data = onp.zeros((coexistence_data.shape[0], 5))
        bcc_idx=3
        try:
            data = onp.loadtxt(next(path.glob("*cst.csv")), skiprows=1, delimiter=" ")
            crystal_data = data
        except StopIteration:
            pass

        try:
            data = onp.loadtxt(next(path.glob("coexistence_cst.txt")), skiprows=1, delimiter=" ")
            crystal_data = data
            bcc_idx=4
        except StopIteration:
            pass

        coex_time = coexistence_data[:, 0] * time_step
        coex_temp = coexistence_data[:, 2]

        coex_temp_uc = coex_temp[:(coex_temp.size // segments) * segments].reshape((segments, -1))
        steps_per_segment = coex_temp_uc.shape[1]
        coex_temp_uc = onp.mean(coex_temp_uc, axis=1)
        axes[idx].scatter(coex_time[steps_per_segment // 2::steps_per_segment], coex_temp_uc, label=f"_Coexistence {temp}K", alpha=0.5)

        cst_time = crystal_data[:, 0] * time_step
        cst_f_bcc = crystal_data[:, bcc_idx]

        axes[idx].plot(coex_time, coex_temp, label=f"_Coexistence {temp}K", alpha=0.5)
        twinx = axes[idx].twinx()
        twinx.plot(cst_time, cst_f_bcc * 100, label=f"_CST {temp}K", color="orange", alpha=0.5)
        twinx.set_ylim([0, 100])

        axes[idx].spines['left'].set_color("tab:blue")
        axes[idx].tick_params(axis='y', colors="tab:blue")
        twinx.spines['right'].set_color("tab:orange")
        twinx.tick_params(axis='y', colors="tab:orange")

        mean_temp = coex_temp.mean()
        if err_est is None:
            std_temp = coex_temp.std()
        else:
            std_temp = err_est[idx]

        axes[idx].axhline(mean_temp, color="tab:blue", linestyle="--", label=f"$T \\approx {mean_temp : .0f} \pm {onp.round(std_temp, 0) :.0f}$")
        twinx.axhline(onp.mean(cst_f_bcc) * 100. , color="tab:orange", linestyle="--", label=f"$f_{{bcc}} \\approx {onp.mean(cst_f_bcc) * 100 :.1f} \pm {onp.std(cst_f_bcc) * 100 :.1f}$ %")
        axes[idx].axhline(mean_temp + std_temp, color="tab:blue", linestyle=":", label=f"_$T = {mean_temp:.2f} \pm {std_temp:.2f}$ K")
        axes[idx].axhline(mean_temp - std_temp, color="tab:blue", linestyle=":", label=f"_$T = {mean_temp:.2f} \pm {std_temp:.2f}$ K")

        axes[idx].set_title(f"$P = {pressure}$ bar, $\\tilde T = {temp}$ K", fontsize=10)

        hdls, lbls = axes[idx].get_legend_handles_labels()
        hdlst, lblst = twinx.get_legend_handles_labels()

        axes[idx].legend(hdls + hdlst, lbls + lblst, fontsize=7, loc="upper right")

        if idx % 3 == 2:
            twinx.get_yaxis().set_label_text("Fraction BCC [%]", color="tab:orange")
        if idx % 3 == 0:
            axes[idx].get_yaxis().set_label_text("Temperature [K]", color="tab:blue")

        est_temps[idx] = mean_temp
        avg_cst[idx] = onp.mean(cst_f_bcc) * 100

    axes[4].set_xlabel("Time [ps]")
    fig.suptitle(title)

    return fig, onp.asarray(est_temps), onp.asarray(avg_cst)


def plot_coexistence_convergence(paths, temps, press, delta=None, title="Coexistence"):
    fig, axes = plt.subplots(int(onp.ceil(len(paths) / 3)), 3, figsize=(7, int(onp.ceil(len(paths)  /  3)) * 1.75), layout="constrained", sharex=True, sharey=False)

    for idx, ax in enumerate(axes.flat):
        ax.annotate(f"\\textbf{{({chr(ord('`') + idx + 1)})}}", (0, 1),
                    (-15, 2), xycoords="axes fraction",
                    textcoords="offset points", ha="center", va="bottom")

    axes = axes.ravel()

    print(f"Create {int(onp.ceil(len(paths) / 3))} rows for {len(paths)} plots.")

    time_step = 0.004  # in ps, used for x-axis scaling

    segments = 10

    if delta is None:
        delta = onp.zeros_like(temps)

    sort_idx = onp.lexsort((temps, press))
    # sort_idx = onp.arange(len(paths))[sort_idx]

    est_uc = onp.zeros(len(paths))

    for idx, (path, temp, pressure, delta) in enumerate(zip(paths, temps, press, delta)):
        idx = onp.argmax(sort_idx == idx)

        # Load coexistence temperature and crystal data
        coexistence_data = onp.loadtxt(path / "coexistence.txt", skiprows=1, delimiter=" ")
        crystal_data = onp.zeros((coexistence_data.shape[0], 5))
        bcc_idx=3
        try:
            data = onp.loadtxt(next(path.glob("*cst.csv")), skiprows=1, delimiter=" ")
            crystal_data = data
        except StopIteration:
            pass

        try:
            data = onp.loadtxt(next(path.glob("coexistence_cst.txt")), skiprows=1, delimiter=" ")
            crystal_data = data
            bcc_idx=4
        except StopIteration:
            pass

        coex_time = coexistence_data[:, 0] * time_step
        coex_temp = coexistence_data[:, 2]

        min_bs = 4
        max_bs = 2 ** int(onp.floor(onp.log2(len(coex_time))) - 2)

        coex_time = coex_time[-4*max_bs:]
        coex_temp = coex_temp[-4*max_bs:]

        xvals = []
        yvals = []
        zvals = []

        twinx = axes[idx].twinx()
        for exp in range(int(onp.log2(min_bs)), int(onp.log2(max_bs))):
            blocked = onp.reshape(coex_temp, (-1, 2 ** exp)).mean(axis=1)
            # print(f"{blocked.shape} ({2 ** exp}): {blocked.std()}")
            xvals.append(1000 * time_step * 2 ** exp)
            yvals.append(blocked.std())
            zvals.append(blocked.std() / onp.sqrt(blocked.size))

        axes[idx].plot(xvals, zvals, marker="o", label=f"_Coexistence {temp}K")
        twinx.plot(xvals, yvals, marker="o", label=f"_Coexistence {temp}K", color="tab:orange")
        # twinx.set_ylim([-1, 1])

        xvals = onp.asarray(xvals)
        yvals = onp.asarray(yvals)

        # def f(x, a, b):
        #     return a * onp.sqrt(x)
        #
        # popt, pcov = scpo.curve_fit(
        #     f, onp.asarray(xvals), onp.asarray(zvals),
        #     p0=(10, 0.01),
        # )
        # a, b = popt
        #
        # scale, shift = onp.polyfit(
        #     onp.log10(onp.asarray(xvals)),
        #     onp.log10(onp.asarray(yvals)),
        #     1, w=xvals)
        #
        # axes[idx].plot(xvals, a * onp.sqrt(xvals), linestyle="--", label=f"Slope = {scale :.2f}")

        twinx.spines['right'].set_color("tab:orange")
        axes[idx].spines['left'].set_color("tab:blue")

        twinx.set_ylim([0, 15])
        axes[idx].set_ylim([0, 5])


        if idx % 3 == 2:
            twinx.get_yaxis().set_label_text("Block Std. Dev. [K]", color="tab:orange")
        if idx % 3 == 0:
            axes[idx].get_yaxis().set_label_text("Standard Error [K]", color="tab:blue")

        est_uc[idx] = (zvals[-1])

    axes[4].set_xlabel("Block Size [fs]")
    fig.suptitle(title)

    return fig, 2 * onp.asarray(est_uc)


def plot_coexistence_relax(paths, temps, press, delta=None, title="Coexistence"):
    fig, axes = plt.subplots(int(onp.ceil(len(paths) / 3)), 3, figsize=(7, int(onp.ceil(len(paths) / 3)) * 1.75), layout="constrained", sharex=True, sharey=False)
    axes = axes.ravel()

    time_step = 0.004  # in ps, used for x-axis scaling

    segments = 10

    if delta is None:
        delta = onp.zeros_like(temps)

    sort_idx = onp.lexsort((temps, press))
    # sort_idx = onp.arange(len(paths))[sort_idx]


    for idx, (path, temp, pressure, delta) in enumerate(zip(paths, temps, press, delta)):
        idx = onp.argmax(sort_idx == idx)
        print(f"{idx}: {pressure}" )

        # Load coexistence temperature and crystal data
        coexistence_data = onp.loadtxt(path / "relax.txt", skiprows=1, delimiter=" ")
        crystal_data = onp.zeros((coexistence_data.shape[0], 5))
        bcc_idx=3
        try:
            data = onp.loadtxt(next(path.glob("*cst.csv")), skiprows=1, delimiter=" ")
            crystal_data = data
        except StopIteration:
            pass

        try:
            data = onp.loadtxt(next(path.glob("relax_cst.txt")), skiprows=1, delimiter=" ")
            crystal_data = data
            bcc_idx=4
        except StopIteration:
            pass

        coex_time = coexistence_data[:, 0] * time_step
        coex_temp_liquid, coex_temp_solid = coexistence_data[:, 2:4].T

        cst_time = crystal_data[:, 0] * time_step
        cst_f_bcc = crystal_data[:, bcc_idx]

        axes[idx].plot(coex_time, coex_temp_liquid, ":", label=f"Liquid", color="tab:blue")
        axes[idx].plot(coex_time, coex_temp_solid, "--", label=f"Solid", color="tab:blue")
        twinx = axes[idx].twinx()
        twinx.plot(cst_time, cst_f_bcc * 100, label=f"_CST {temp}K", color="orange", alpha=0.5)
        twinx.set_ylim([0, 100])

        axes[idx].spines['left'].set_color("tab:blue")
        axes[idx].tick_params(axis='y', colors="tab:blue")
        twinx.spines['right'].set_color("tab:orange")
        twinx.tick_params(axis='y', colors="tab:orange")

        axes[idx].set_title(f"$P = {pressure}$ bar, $\\tilde T = {temp}$ K", fontsize=10)

        hdls, lbls = axes[idx].get_legend_handles_labels()
        hdlst, lblst = twinx.get_legend_handles_labels()

        axes[idx].legend(hdls + hdlst, lbls + lblst, fontsize=7, loc="upper right")

        if idx % 3 == 2:
            twinx.get_yaxis().set_label_text("Fraction BCC [%]", color="tab:orange")
        if idx % 3 == 0:
            axes[idx].get_yaxis().set_label_text("Temperature [K]", color="tab:blue")


    axes[4].set_xlabel("Time [ps]")
    fig.suptitle(title)

    return fig


def plot_phase_diagram(melting_data, solid_data):
    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(7, 3))

    exp_linestyles = ["--", ":", "-.", "-", "--"]
    min_melt, max_melt = 2200.0, 0.0

    cmap = plt.get_cmap("tab10").colors
    for idx, args in enumerate(melting_data):
        if len(args) == 3:
            press, temp, label = args
            delta_temp = None
        else:
            press, temp, delta_temp, label = args

        label: str = label
        if label.lower().startswith("exp"):
            style = exp_linestyles.pop()
            ax.plot(temp, press, linestyle=style, color="k", label=label)

            if len(temp) == 0: continue

            if (temp[2] + temp[3]) / 2 < min_melt:
                min_melt = (temp[2] + temp[3]) / 2
            if (temp[2] + temp[3]) / 2 > max_melt:
                max_melt = (temp[2] + temp[3]) / 2

            if delta_temp is not None:
                ax.plot(temp + delta_temp, press, linestyle=style, color="k", label="_", linewidth=0.8)
                ax.plot(temp - delta_temp, press, linestyle=style, color="k", label="_", linewidth=0.8)
            continue

        if len(args) == 4:
            ax.errorbar(x=temp, y=press, xerr=delta_temp, linestyle="none", marker="d", label=label, color=cmap[idx - 1], capsize=3)
        else:
            ax.scatter(temp, press, marker="d", label=label, color=cmap[idx - 1])

    ax.annotate("BCC", (min_melt, 2.5),
                textcoords="offset points", xytext=(-10, 0), ha="right",
                va="center")
    ax.annotate("Liquid", (max_melt, 2.5),
                textcoords="offset points", xytext=(10, 0), ha="left",
                va="center")

    exp_linestyles = ["--", ":", "-.", "-", "--"]
    for idx, args in enumerate(solid_data):
        if len(args) == 3:
            press, temp, label = args
            delta_temp = None
        else:
            press, temp, delta_temp, label = args

        if label.lower().startswith("exp"):
            style = exp_linestyles.pop()
            ax.plot(temp, press, linestyle=style, color="k", label="_Exp")

            if len(temp) == 0: continue

            ax.annotate("HCP", ((temp[2] + temp[3]) / 2, 2.5),
                        textcoords="offset points", xytext=(-10, 0), ha="right", va="center")
            ax.annotate("BCC", ((temp[2] + temp[3]) / 2, 2.5),
                        textcoords="offset points", xytext=(10, 0), ha="left", va="center")
            continue

        if len(args) == 4:
            ax.errorbar(x=temp, y=press, xerr=delta_temp, linestyle="none", marker="d", label="_" + label, color=cmap[idx - 1], capsize=3)
        else:
            ax.scatter(temp, press, marker="d", label="_" + label, color=cmap[idx - 1])

        # ax.plot(temp, press, "o-", label="_" + label, color=cmap[idx - 1])

    ax.set_ylabel("Pressure [GPa]")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylim([-0.5, 5.5])
    ax.set_xlim([800, 2200])
    ax.set_yticks(onp.arange(6), onp.arange(6))
    fig.legend(loc="outside lower center", bbox_to_anchor=(0.5, 1.0), ncol=5)
    # ax.grid()

    return fig
