"""Interface da ferramenta acadêmica de estabilidade de sistemas contínuos."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from nyquist_research.bode import bode
from nyquist_research.nyquist import nyquist
from nyquist_research.pade import approximate_delay
from nyquist_research.root_locus import root_locus
from nyquist_research.routh_hurwitz import RouthResult, routh_hurwitz
from nyquist_research.step_response import step_response
from nyquist_research.time_delay import critical_delay
from nyquist_research.transfer_function import TransferFunction

_PRESETS = Path(__file__).resolve().parents[2] / "experiments/configurations/systems.json"
_STATUS = {
    "stable": "Estável",
    "unstable": "Instável",
    "marginal": "No limite",
    "indeterminate": "Indeterminada",
}
_LINE = "#175f75"
_ACCENT = "#ac482e"
_AXES = "#8395a0"
_LIMIT = 20


def _tokens(text):
    return re.split(r"[\s,;]+", text.strip()) if text.strip() else []


def _coefficients(text, name):
    tokens = _tokens(text)
    if not tokens:
        raise ValueError(f"{name} precisa de pelo menos um coeficiente.")
    if len(tokens) > _LIMIT + 1:
        raise ValueError(f"{name} deve ter ordem máxima {_LIMIT}.")
    try:
        coefficients = [float(token) for token in tokens]
    except ValueError as error:
        raise ValueError(f"Confira {name.lower()} e use ponto decimal.") from error
    return coefficients


def _root_polynomial(text, name):
    tokens = _tokens(text)
    if len(tokens) > _LIMIT:
        raise ValueError(f"Use no máximo {_LIMIT} {name.lower()}.")
    try:
        roots = np.asarray([complex(token.lower().replace("i", "j")) for token in tokens])
    except ValueError as error:
        raise ValueError(f"Use raízes como -1 ou -1+2j no campo {name.lower()}.") from error
    if np.any(~np.isfinite(roots)):
        raise ValueError(f"Use valores finitos no campo {name.lower()}.")
    with np.errstate(over="ignore", invalid="ignore"):
        polynomial = np.atleast_1d(np.poly(roots))
    if np.any(~np.isfinite(polynomial)):
        raise ValueError(f"Os {name.lower()} produzem coeficientes fora da escala suportada.")
    if np.any(np.abs(polynomial.imag) > 1e-10 * np.maximum(1, np.abs(polynomial.real))):
        raise ValueError(f"Informe os pares conjugados no campo {name.lower()}.")
    return polynomial.real


def _format_list(values):
    return ", ".join(format(value, ".16g") for value in values)


def _inputs(presets):
    with st.sidebar:
        st.header("Modelo")
        case = st.selectbox(
            "Sistema",
            (*presets, "Personalizado"),
            format_func=lambda key: f"{key} · {presets[key]['label']}" if key in presets else key,
        )
        preset = presets.get(case, presets["A"])
        edited_models = st.session_state.setdefault("edited_models", {})
        default_system = edited_models.get(case) or TransferFunction(
            preset["numerator"], preset["denominator"], preset["gain"], preset["delay"]
        )
        mode = st.radio("Representação", ("Coeficientes", "Polos e zeros"))
        if mode == "Coeficientes":
            numerator_raw = st.text_input(
                "Numerador",
                _format_list(default_system.numerator),
                help="Coeficientes em ordem decrescente. Separe por vírgula e use ponto decimal.",
                key=f"numerator-{case}",
            )
            denominator_raw = st.text_input(
                "Denominador",
                _format_list(default_system.denominator),
                help="Inclua coeficientes nulos dos termos intermediários.",
                key=f"denominator-{case}",
            )
        else:
            zeros_raw = st.text_input(
                "Zeros", _format_list(default_system.zeros), key=f"zeros-{case}"
            )
            poles_raw = st.text_input(
                "Polos",
                _format_list(default_system.poles),
                key=f"poles-{case}",
                help="Separe os valores por vírgula e inclua os complexos conjugados.",
            )
            numerator_scale = st.number_input(
                "Coeficiente líder do numerador",
                value=default_system.numerator[0],
                key=f"scale-{case}",
            )
        gain = st.number_input(
            "Ganho K", min_value=0.0, value=default_system.gain, step=0.1, key=f"gain-{case}"
        )
        delay = st.number_input(
            "Atraso τ (s)",
            min_value=0.0,
            value=default_system.delay,
            step=0.05,
            key=f"delay-{case}",
        )
        with st.expander("Intervalos e aproximação"):
            pade_order = st.slider(
                "Ordem de Padé",
                1,
                8,
                3,
                help="Usada em Routh-Hurwitz, Lugar das Raízes e resposta ao degrau se τ > 0.",
            )
            gain_max = st.number_input(
                "Ganho máximo no Lugar das Raízes",
                min_value=0.01,
                value=max(15.0, 2 * float(preset["gain"])),
                step=1.0,
                key=f"gain-end-{case}",
            )
            duration = st.number_input(
                "Duração do degrau (s)", min_value=0.01, value=12.0, step=1.0
            )
            zoom = st.checkbox("Destacar região de −1 no Nyquist")
    if mode == "Coeficientes":
        numerator = _coefficients(numerator_raw, "Numerador")
        denominator = _coefficients(denominator_raw, "Denominador")
    else:
        numerator = numerator_scale * _root_polynomial(zeros_raw, "Zeros")
        denominator = _root_polynomial(poles_raw, "Polos")
    system = TransferFunction(numerator, denominator, gain, delay)
    edited_models[case] = system
    return (
        case,
        system,
        pade_order,
        gain_max,
        duration,
        zoom,
    )


def _attempt(function, *args):
    try:
        return function(*args), None
    except (ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as error:
        return None, str(error)


def _degrau(model, duration):
    time = np.linspace(0, duration, 401)
    return {"time": time, "output": step_response(model, time)}


@dataclass(frozen=True)
class _RouthIndeterminate:
    reason: str
    status: str = "indeterminate"


def _routh(model):
    characteristic = model.characteristic()
    if characteristic[0] == 0:
        return _RouthIndeterminate(
            "Realimentação algébrica singular. A característica perde grau neste ganho."
        )
    return routh_hurwitz(characteristic)


@st.cache_data(show_spinner=False, max_entries=24)
def _compute(system, pade_order, gain_max, duration):
    results = {
        "nyquist": _attempt(nyquist, system),
        "bode": _attempt(bode, system),
    }
    if system.delay:
        results["critical_delay"] = _attempt(critical_delay, system)
        rational, approximation_error = _attempt(approximate_delay, system, pade_order)
    else:
        rational, approximation_error = system, None
    if rational is None:
        results.update({key: (None, approximation_error) for key in ("routh", "locus", "step")})
    else:
        gains = np.unique(np.r_[np.linspace(0, max(gain_max, system.gain), 141), system.gain])
        results["routh"] = _attempt(_routh, rational)
        results["locus"] = _attempt(root_locus, rational, gains)
        results["step"] = _attempt(_degrau, rational, duration)
    return rational, results


def _scatter(x, y, name, **kwargs):
    return go.Scatter(x=x, y=y, name=name, **kwargs)


def _layout(figure, x_label, y_label, height=340):
    figure.update_layout(
        height=height,
        margin={"l": 12, "r": 16, "t": 31, "b": 12},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
        hovermode="closest",
    )
    figure.update_xaxes(title_text=x_label)
    figure.update_yaxes(title_text=y_label)
    return figure


def _nyquist_figure(result, zoom):
    chart = go.Figure()
    chart.add_trace(
        _scatter(
            result.mapped.real,
            result.mapped.imag,
            "Contorno mapeado",
            mode="lines",
            line={"color": _LINE, "width": 2},
            customdata=np.c_[result.contour.real, result.contour.imag],
            hovertemplate=(
                "L(s) = %{x:.4g} + j(%{y:.4g})<br>"
                "s = %{customdata[0]:.4g} + j(%{customdata[1]:.4g})<extra></extra>"
            ),
        )
    )
    chart.add_trace(
        _scatter(
            [-1],
            [0],
            "Ponto crítico",
            mode="markers",
            marker={"symbol": "x", "color": _ACCENT, "size": 10},
        )
    )
    chart.update_yaxes(scaleanchor="x", scaleratio=1)
    if zoom:
        chart.update_xaxes(range=[-4.0, 2.0])
        chart.update_yaxes(range=[-2.0, 2.0])
    return _layout(chart, "Re L(s)", "Im L(s)")


def _locus_figure(result, model):
    chart = go.Figure()
    gaps = np.searchsorted(result.gains, result.degree_drops)
    gains = np.insert(result.gains, gaps, np.nan)
    for branch in result.roots.T:
        chart.add_trace(
            _scatter(
                np.insert(branch.real, gaps, np.nan),
                np.insert(branch.imag, gaps, np.nan),
                "Ramo",
                mode="lines",
                line={"color": _LINE, "width": 2},
                showlegend=False,
                customdata=gains,
                hovertemplate="Re %{x:.4g}<br>Im %{y:.4g}<br>K %{customdata:.4g}<extra></extra>",
            )
        )
    if model.gain not in result.degree_drops:
        index = int(np.searchsorted(result.gains, model.gain))
        current = result.roots[index]
        chart.add_trace(
            _scatter(
                current.real,
                current.imag,
                "Polos no ganho K",
                mode="markers",
                marker={"color": _ACCENT, "size": 10, "symbol": "x"},
            )
        )
    zeros = model.zeros
    if zeros.size:
        chart.add_trace(
            _scatter(
                zeros.real,
                zeros.imag,
                "Zeros",
                mode="markers",
                marker={"color": _LINE, "size": 9, "symbol": "circle-open"},
            )
        )
    chart.add_vline(x=0, line_dash="dot", line_color=_AXES)
    return _layout(chart, "Re s (s⁻¹)", "Im s (rad/s)")


def _bode_figure(result):
    chart = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    chart.add_trace(
        _scatter(
            result.omega, result.magnitude_db, "Magnitude", mode="lines", line={"color": _LINE}
        ),
        row=1,
        col=1,
    )
    chart.add_trace(
        _scatter(result.omega, result.phase_deg, "Fase", mode="lines", line={"color": _LINE}),
        row=2,
        col=1,
    )
    chart.add_hline(y=0, line_dash="dot", line_color=_AXES, row=1, col=1)
    chart.add_hline(y=-180, line_dash="dot", line_color=_AXES, row=2, col=1)
    for crossing in result.gain_crossings:
        if crossing.omega > 0:
            chart.add_vline(x=crossing.omega, line_dash="dot", line_color=_ACCENT)
    chart.update_xaxes(type="log")
    chart.update_xaxes(title_text="Frequência ω (rad/s)", row=2, col=1)
    chart.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
    chart.update_yaxes(title_text="Fase (°)", row=2, col=1)
    chart.update_layout(height=430, margin={"l": 12, "r": 16, "t": 12, "b": 12}, showlegend=False)
    return chart


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, complex):
        return {"real": _json_value(value.real), "imag": _json_value(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_asdict"):
        return _json_value(value._asdict())
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _export(case, system, pade_order, results):
    return json.dumps(
        _json_value(
            {
                "system": case,
                "model": system,
                "pade_order": pade_order if system.delay else None,
                "units": {"delay": "s", "frequency": "rad/s", "phase": "deg"},
                "results": {
                    key: {"data": result, "error": error}
                    for key, (result, error) in results.items()
                },
            }
        ),
        ensure_ascii=False,
        allow_nan=False,
    )


def _status(result):
    return _STATUS[result.status] if result is not None else "Indisponível"


def _show_margins(result):
    if result.gain_crossings:
        st.dataframe(
            [
                {"ω (rad/s)": m.omega, "Margem de fase (°)": m.phase_margin_deg}
                for m in result.gain_crossings
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("Sem cruzamento de ganho isolado encontrado.")
    if result.phase_crossings:
        st.dataframe(
            [
                {
                    "ω (rad/s)": m.omega,
                    "Margem de ganho (fator)": m.gain_margin,
                    "Margem de ganho (dB)": m.gain_margin_db,
                }
                for m in result.phase_crossings
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("Sem cruzamento de fase isolado encontrado na faixa avaliada.")
    if result.phase_band:
        lower, upper = result.phase_band
        st.caption(f"Busca de cruzamentos de fase em [{lower:.3g}, {upper:.3g}] rad/s.")
    if result.phase_band_limited:
        st.caption("A busca de cruzamentos de fase atingiu o limite numérico da malha.")
    if result.gain_continuum or result.phase_continuum:
        st.caption("Há um intervalo contínuo de cruzamentos. Margens isoladas são insuficientes.")


def _show_comparison(system, results, pade_order):
    ny, _ = results["nyquist"]
    rh, _ = results["routh"]
    locus, _ = results["locus"]
    frequency, _ = results["bode"]
    rational_model = f"Padé [{pade_order}/{pade_order}]" if system.delay else "Racional"
    frequency_model = "Atraso exponencial" if system.delay else "Racional"
    locus_result = (
        f"{len(locus.crossings)} polos de fronteira localizados na faixa de ganho"
        if locus is not None
        else "Indisponível"
    )
    bode_result = (
        f"{len(frequency.gain_crossings)} cruzamentos de ganho, "
        f"{len(frequency.phase_crossings)} cruzamentos de fase"
        if frequency is not None
        else "Indisponível"
    )
    rows = (
        ("Nyquist", frequency_model, _status(ny)),
        ("Routh-Hurwitz", rational_model, _status(rh)),
        ("Lugar das Raízes", rational_model, locus_result),
        ("Bode", frequency_model, bode_result),
    )
    with st.expander("Comparar métodos"):
        st.dataframe(
            [
                {"Método": method, "Modelo": model, "Resultado": value}
                for method, model, value in rows
            ],
            hide_index=True,
            width="stretch",
        )


def main():
    st.set_page_config(page_title="Laboratório de estabilidade", layout="wide")
    st.title("Laboratório de estabilidade")
    st.caption("Sistemas contínuos SISO com realimentação unitária negativa.")
    presets = json.loads(_PRESETS.read_text(encoding="utf-8"))
    try:
        case, system, pade_order, gain_max, duration, zoom = _inputs(presets)
    except (ValueError, FloatingPointError, OverflowError) as error:
        st.error(str(error))
        return

    rational, results = _compute(system, pade_order, gain_max, duration)
    ny, ny_error = results["nyquist"]
    rh, rh_error = results["routh"]
    locus, locus_error = results["locus"]
    frequency, frequency_error = results["bode"]
    step, step_error = results["step"]
    columns = st.columns(4)
    columns[0].metric("Estabilidade por Nyquist", _status(ny), border=True)
    columns[1].metric("Polos abertos à direita (P)", ny.p if ny else "—", border=True)
    columns[2].metric(
        "Envoltórios horários N", ny.n if ny and ny.n is not None else "—", border=True
    )
    columns[3].metric(
        "Polos fechados à direita (Z)", ny.z if ny and ny.z is not None else "—", border=True
    )
    if system.delay:
        st.info(
            f"Nyquist e Bode usam τ = {system.delay:.4g} s no termo exponencial. "
            f"Routh-Hurwitz, Lugar das Raízes e degrau usam Padé [{pade_order}/{pade_order}]."
        )
        limit, limit_error = results["critical_delay"]
        if limit is not None:
            first = (
                f"Atraso no primeiro limite: {limit.delay:.5g} s. "
                if limit.delay is not None
                else ""
            )
            st.caption(first + limit.reason)
        elif limit_error:
            st.caption(f"Limite de atraso indisponível. {limit_error}")

    first, second = st.columns(2)
    with first:
        st.subheader("Nyquist")
        if ny is not None:
            if ny.mapped.size:
                st.plotly_chart(_nyquist_figure(ny, zoom), width="stretch", key="nyquist-plot")
            st.caption(ny.reason)
        if ny_error:
            st.warning(ny_error)
    with second:
        st.subheader("Lugar das Raízes")
        if locus is not None:
            st.plotly_chart(_locus_figure(locus, rational), width="stretch", key="locus-plot")
            if locus.crossings:
                with st.expander("Cruzamentos do eixo imaginário"):
                    st.dataframe(
                        [
                            {"Ganho K": gain, "Im s (rad/s)": pole.imag}
                            for gain, pole in locus.crossings
                        ],
                        hide_index=True,
                        width="stretch",
                    )
            for gain in locus.degree_drops:
                st.caption(f"A equação característica perde grau em K = {gain:.6g}.")
            for event in locus.events:
                st.caption(event)
        if locus_error:
            st.warning(locus_error)

    first, second = st.columns(2)
    with first:
        st.subheader("Bode")
        if frequency is not None:
            st.plotly_chart(_bode_figure(frequency), width="stretch", key="bode-plot")
            with st.expander("Margens de ganho e fase"):
                _show_margins(frequency)
                st.caption("As margens locais devem ser interpretadas com a estabilidade global.")
        if frequency_error:
            st.warning(frequency_error)
    with second:
        st.subheader("Resposta ao degrau")
        if step is not None:
            time, output = step["time"], step["output"]
            chart = go.Figure(_scatter(time, output, "Saída", mode="lines", line={"color": _LINE}))
            st.plotly_chart(_layout(chart, "Tempo (s)", "Saída"), width="stretch", key="step-plot")
            if system.delay:
                st.caption("Padé pode produzir transiente antes do tempo morto do modelo original.")
        if step_error:
            st.warning(step_error)

    st.subheader("Routh-Hurwitz")
    if isinstance(rh, RouthResult):
        count = str(rh.rhp) if rh.rhp is not None else "indeterminada"
        st.caption(f"{_status(rh)} · contagem à direita {count} · polos no eixo {rh.axis}")
        table = {"Potência de s": np.arange(rh.table.shape[0] - 1, -1, -1)}
        table.update({f"Coluna {i + 1}": values for i, values in enumerate(rh.table.T)})
        st.dataframe(table, width="stretch", hide_index=True)
        for event in rh.events:
            st.caption(event)
        if ny is not None and ny.status != rh.status:
            st.caption("Os estados de Nyquist e Routh-Hurwitz diferem para os modelos avaliados.")
    elif isinstance(rh, _RouthIndeterminate):
        st.warning(f"{_status(rh)} · {rh.reason}")
    if rh_error:
        st.warning(rh_error)
    _show_comparison(system, results, pade_order)
    st.download_button(
        "Baixar análise (JSON)",
        _export(case, system, pade_order, results),
        file_name="estabilidade.json",
        mime="application/json",
        on_click="ignore",
    )


if __name__ == "__main__":
    main()
