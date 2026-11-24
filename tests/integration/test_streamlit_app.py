from pathlib import Path

import numpy as np
import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
pytest.importorskip("plotly")

_APP = Path(__file__).resolve().parents[2] / "application/streamlit/app.py"


def _app():
    return AppTest.from_file(str(_APP), default_timeout=20).run()


def _metrics(app):
    assert not app.exception
    return tuple(metric.value for metric in app.metric)


@pytest.mark.parametrize(
    ("system", "expected"),
    [
        ("A", ("Estável", "0", "0", "0")),
        ("B", ("Estável", "1", "-1", "0")),
        ("C", ("Estável", "0", "0", "0")),
        ("D", ("Estável", "0", "0", "0")),
    ],
)
def test_presets_render_the_four_diagrams_and_download(system, expected):
    app = _app()
    app.selectbox[0].set_value(system).run()
    assert _metrics(app) == expected
    assert len(app.get("plotly_chart")) == 4
    assert len(app.get("download_button")) == 1
    assert any(header.value == "Routh-Hurwitz" for header in app.subheader)


def test_delay_edit_recomputes_stability_and_pade():
    app = _app()
    app.selectbox[0].set_value("D").run()
    app.number_input(key="delay-D").set_value(1.5).run()
    app.slider[0].set_value(2).run()
    assert _metrics(app) == ("Instável", "0", "2", "2")
    assert any("Padé [2/2]" in info.value for info in app.info)
    assert any("1.2092 s" in caption.value for caption in app.caption)
    assert len(app.get("plotly_chart")) == 4


def test_custom_coefficients_and_pole_edit_are_consistent():
    app = _app()
    app.selectbox[0].set_value("Personalizado").run()
    app.text_input(key="numerator-Personalizado").set_value("3")
    app.text_input(key="denominator-Personalizado").set_value("1, 5, 6").run()
    assert _metrics(app)[0] == "Estável"
    app.radio[0].set_value("Polos e zeros").run()
    model = app.session_state["edited_models"]["Personalizado"]
    np.testing.assert_allclose(model.denominator, [1, 5, 6])
    np.testing.assert_allclose(model.numerator, [3])
    app.text_input(key="poles-Personalizado").set_value("-1+2j").run()
    assert not app.exception
    assert any("conjugados" in error.value for error in app.error)
    assert not app.get("plotly_chart")
    app.text_input(key="poles-Personalizado").set_value("-1+2j, -1-2j").run()
    assert _metrics(app)[0] == "Estável"
    assert len(app.get("plotly_chart")) == 4
    app.radio[0].set_value("Coeficientes").run()
    updated = app.session_state["edited_models"]["Personalizado"]
    np.testing.assert_allclose(updated.denominator, [1, 2, 5])
    assert not app.exception


def test_singular_feedback_does_not_report_routh_stability():
    app = _app()
    app.selectbox[0].set_value("Personalizado").run()
    app.text_input(key="numerator-Personalizado").set_value("-1, 0")
    app.text_input(key="denominator-Personalizado").set_value("1, 1").run()
    assert _metrics(app)[0] == "Indeterminada"
    assert any(
        "Indeterminada · Realimentação algébrica singular" in warning.value
        for warning in app.warning
    )
    assert any("Laço algébrico singular" in warning.value for warning in app.warning)
    assert any("perde grau em K = 1" in caption.value for caption in app.caption)
    assert not any("Estável · contagem" in caption.value for caption in app.caption)
    charts = app.get("plotly_chart")
    assert len(charts) == 2
    assert "Polos no ganho K" not in charts[0].proto.spec
