"""Build vector figures exclusively from archived experimental outputs."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import RESULTS

COLORS = ("#145d88", "#b75d11", "#317859", "#76518d")
PLANTS = ("A", "B", "C", "D")
METHODS = ("Nyquist", "Routh-Hurwitz", "Lugar das Raízes", "Bode")
METHOD_MARKERS = ("o", "s", "^", "D")


def _rows(path: str) -> list[dict]:
    with (RESULTS / path).open() as stream:
        return list(csv.DictReader(stream))


def _json(path: str):
    return json.loads((RESULTS / path).read_text())


def _select(rows: list[dict], **filters) -> list[dict]:
    return [row for row in rows if all(str(row[key]) == str(val) for key, val in filters.items())]


def _values(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows])


def _save(fig, name: str) -> str:
    filename = f"{name}.pdf"
    output = RESULTS / "figures" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.025,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return filename


def _complex_axes(axis) -> None:
    axis.axhline(0, color="#829098", linewidth=0.55, zorder=0)
    axis.axvline(0, color="#829098", linewidth=0.55, zorder=0)
    axis.set_aspect("equal", adjustable="datalim")


def _nyquist() -> str:
    rows = _rows("raw/nyquist-contours.csv")
    nominal = _json("processed/plant-summary.json")
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.0), layout="constrained")
    for axis, label, color, result in zip(axes.flat, PLANTS, COLORS, nominal, strict=True):
        plant = _select(rows, system=label)
        path = _values(plant, "mapped_real") + 1j * _values(plant, "mapped_imag")
        s_real = _values(plant, "s_real")
        _complex_axes(axis)
        axis.plot(path.real, path.imag, lw=1.25, color=color)
        axis.scatter(-1, 0, marker="x", linewidth=1.35, s=28, color="#a92432", zorder=3)
        branch = np.flatnonzero(s_real == 0)
        if branch.size:
            candidates = branch[(abs(path[branch]) > 0.15) & (path[branch].imag > 0)]
            if len(candidates) > 3:
                k = candidates[min(len(candidates) - 3, len(candidates) * 3 // 4)]
                end = min(k + 4, len(path) - 1)
                axis.annotate("", (path[end].real, path[end].imag), (path[k].real, path[k].imag),
                              arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.9})
        axis.set_title(f"{label}   $P={result['p']},\\ N={result['n']},\\ Z={result['z']}$",
                       loc="left", fontweight="bold")
    fig.supxlabel(r"$\operatorname{Re} L(s)$")
    fig.supylabel(r"$\operatorname{Im} L(s)$")
    return _save(fig, "nyquist-abcd")


def _bode() -> str:
    frequency = _rows("raw/frequency.csv")
    nominal = _json("processed/plant-summary.json")
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.5), sharex=True, layout="constrained")
    for column, label, color, result in zip(range(4), PLANTS, COLORS, nominal, strict=True):
        plant = _select(frequency, system=label)
        w = _values(plant, "omega_rad_s")
        selection = (w >= 0.01) & (w <= 15)
        mag, phase = axes[:, column]
        mag.semilogx(w[selection], _values(plant, "magnitude_db")[selection], color=color)
        phase.semilogx(w[selection], _values(plant, "phase_deg")[selection], color=color)
        mag.axhline(0, linestyle=":", color="#63717b", linewidth=0.7)
        phase.axhline(-180, linestyle=":", color="#63717b", linewidth=0.7)
        for wc, _ in result["gain_crossings_rad_s_pm_deg"]:
            if 0.01 < wc < 15:
                mag.axvline(wc, color="#2c638a", linewidth=0.6, linestyle="--")
                phase.axvline(wc, color="#2c638a", linewidth=0.6, linestyle="--")
        for wc, _ in result["phase_crossings_rad_s_gm"]:
            if 0.01 < wc < 15:
                mag.axvline(wc, color="#a92432", linewidth=0.6, linestyle="-.")
                phase.axvline(wc, color="#a92432", linewidth=0.6, linestyle="-.")
        mag.set(ylim=(-80, 13), title=label)
        phase.set(xlim=(0.01, 15), xticks=(0.01, 0.1, 1, 10))
        if label in "ABC":
            phase.set_ylim(-285, 20)
    axes[0, 0].set_ylabel("Magnitude (dB)")
    axes[1, 0].set_ylabel("Fase (°)")
    fig.supxlabel("Frequência angular (rad/s)")
    return _save(fig, "bode-abcd")


def _locus() -> str:
    rows = _rows("raw/root-locus.csv")
    nominal = _json("processed/plant-summary.json")
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.15), layout="constrained")
    for axis, label, result in zip(axes.flat, PLANTS, nominal, strict=True):
        plant = _select(rows, system=label)
        _complex_axes(axis)
        for branch in sorted({int(row["branch"]) for row in plant}):
            data = _select(plant, branch=branch)
            re, im = _values(data, "pole_real"), _values(data, "pole_imag")
            color = COLORS[branch % len(COLORS)]
            axis.plot(re, im, linewidth=1.12, color=color)
            axis.plot(re[0], im[0], marker="x", markersize=4,
                      color="#293640", linestyle="none")
        crossings = result["root_locus_crossings"]
        if crossings:
            _, re, im = np.asarray(crossings).T
            axis.scatter(re, im, color="#a92432", marker="D", s=13, zorder=3)
        axis.set_title(label if label != "D" else "D   Padé [3/3]", loc="left", fontweight="bold")
    fig.supxlabel(r"$\operatorname{Re}s$ (s$^{-1}$)")
    fig.supylabel(r"$\operatorname{Im}s$ (rad/s)")
    return _save(fig, "root-locus-abcd")


def _margins_and_stability() -> str:
    nominal = _json("processed/plant-summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25), layout="constrained")
    width = 0.24
    for offset, attr, color in zip((-width, 0, width), ("p", "n", "z"), COLORS, strict=False):
        axes[0].bar(np.arange(4) + offset, [row[attr] for row in nominal], width,
                    label=attr.upper(), color=color)
    axes[0].set(xticks=range(4), xticklabels=PLANTS, ylabel="Contagem",
                yticks=[-1, 0, 1], ylim=(-1.35, 1.32))
    axes[0].axhline(0, linewidth=0.55, color="#7a8990")
    axes[0].legend(ncol=3, loc="upper left", fontsize=6, frameon=False, handlelength=1)
    for idx, (row, color) in enumerate(zip(nominal, COLORS, strict=True)):
        gm = row["phase_crossings_rad_s_gm"]
        pm = [pair for pair in row["gain_crossings_rad_s_pm_deg"] if pair[0] > 0]
        for axis, margin, conversion in ((axes[1], gm, lambda x: 20 * np.log10(x)),
                                         (axes[2], pm, float)):
            if margin:
                axis.bar(idx, conversion(margin[0][1]), width=0.56, color=color)
            else:
                axis.text(idx, 0, "—", ha="center", va="bottom", color="#444c58")
    axes[1].set(xticks=range(4), xticklabels=PLANTS, ylabel="Primeira MG (dB)")
    axes[1].axhline(0, linewidth=0.55, color="#7a8990")
    axes[2].set(xticks=range(4), xticklabels=PLANTS, ylabel="MF para ω > 0 (°)")
    return _save(fig, "stability-margins")


def _delay() -> str:
    frequency = _rows("raw/pade-frequency-error.csv")
    sweep = _rows("raw/pade-delay-sweep.csv")
    summary = _json("processed/pade-summary.json")
    orders = [row["order"] for row in summary["orders"]]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.9, 2.6), layout="constrained")
    for order, color in zip(orders, COLORS, strict=True):
        sub = _select(frequency, order=order)
        omega, error = _values(sub, "omega_rad_s"), _values(sub, "absolute_complex_error")
        left.loglog(omega, np.maximum(error, 1e-17), color=color, label=f"[{order}/{order}]")
    left.set(xlabel="Frequência angular (rad/s)", ylabel="Erro complexo absoluto",
             ylim=(1e-17, 3))
    left.legend(frameon=False, fontsize=6, ncol=2)
    first = _select(sweep, order=orders[0])
    right.step(_values(first, "delay_s"), _values(first, "rhp_exact"),
               where="post", linewidth=1.15, color="#293640", label="Exato (malha)")
    for row, color in zip(summary["orders"], COLORS, strict=True):
        right.axvline(row["critical_delay_s"], color=color, linestyle=":", linewidth=1)
    right.axvline(summary["critical_exact_analytic_s"], color="#293640", linestyle="--",
                  linewidth=0.9, label="Exato (limite)")
    right.set(xlabel="Atraso (s)", ylabel="Polos à direita", yticks=[0, 2], ylim=(-0.15, 2.75))
    right.legend(frameon=False, fontsize=6, loc="upper left")
    return _save(fig, "delay-pade")


def _performance() -> str:
    summary = _json("processed/benchmark-summary.json")["timings"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65), layout="constrained")
    for axis, metric, title in zip(axes, ("wall", "cpu"), ("Parede", "CPU"), strict=True):
        for method, color, marker in zip(METHODS, COLORS, METHOD_MARKERS, strict=True):
            rows = _select(summary, method=method)
            orders = _values(rows, "order")
            med = _values(rows, f"median_{metric}_ms")
            min_ms = _values(rows, f"min_{metric}_ms")
            max_ms = _values(rows, f"max_{metric}_ms")
            axis.errorbar(orders, med, yerr=np.vstack((med - min_ms, max_ms - med)),
                          color=color, marker=marker, markersize=3.5, capsize=2.5,
                          linewidth=1.1, label=method)
        axis.set(ylabel="Tempo por execução (ms)", xlabel="Ordem", title=title, yscale="log",
                 xticks=(5, 10, 20))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=6)
    return _save(fig, "scalability")


def _precision() -> str:
    summary = _json("processed/benchmark-summary.json")["precision"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.8, 2.65), layout="constrained")
    for family, color, marker in zip(("dispersa", "agrupada"), COLORS, METHOD_MARKERS,
                                     strict=False):
        rows = _select(summary, family=family)
        order = _values(rows, "order")
        error = _values(rows, "root_absolute_error_to_design")
        shift = _values(rows, "max_pole_shift_after_perturbation")
        condition = _values(rows, "max_relative_condition")
        left.semilogy(order, error, color=color, marker=marker, label=f"Erro · {family}")
        left.semilogy(order, shift, color=color, marker=marker, linestyle="--",
                      label=f"Perturbação · {family}")
        right.semilogy(order, condition, color=color, marker=marker, label=family)
    for axis, ylabel in ((left, "Distância máxima (s⁻¹)"), (right, "Cond. relativo máx.")):
        axis.set(xlabel="Ordem", ylabel=ylabel, xticks=(5, 10, 20))
        axis.legend(frameon=False, fontsize=6)
    return _save(fig, "numerical-precision")


def _step() -> str:
    rows = _rows("raw/step.csv")
    fig, axis = plt.subplots(figsize=(3.35, 2.25), layout="constrained")
    for label, color in zip(PLANTS, COLORS, strict=True):
        plant = _select(rows, system=label)
        axis.plot(_values(plant, "time_s"), _values(plant, "response"), color=color,
                  label=label if label != "D" else "D · Padé [3/3]")
    axis.axhline(0, color="#829098", linewidth=0.55)
    axis.set(xlabel="Tempo (s)", ylabel="Saída (u.a.)", xlim=(0, 15))
    axis.legend(frameon=False, fontsize=6)
    return _save(fig, "step-response")


def run() -> list[str]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "axes.axisbelow": True, "grid.color": "#d8dfe3", "grid.linewidth": 0.45,
        "axes.edgecolor": "#536169", "axes.linewidth": 0.65, "lines.linewidth": 1.0,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
    })
    return [builder() for builder in (
        _nyquist, _bode, _locus, _margins_and_stability, _delay, _performance, _precision, _step
    )]


if __name__ == "__main__":
    print(run())
