import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


def test_app_starts_without_errors():
    at = st_testing.AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception
