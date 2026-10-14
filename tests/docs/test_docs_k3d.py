"""Focused tests for documentation execution and embedding."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.exceptions import PluginError
from mkdocs.structure.files import File, Files

spec = importlib.util.spec_from_file_location(
    "docs_k3d", Path(__file__).resolve().parents[2] / "scripts/docs_k3d.py"
)
assert spec is not None and spec.loader is not None
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


class Plot:
    def __init__(self, value):
        self.value = value

    def get_snapshot(self):
        return str(self.value)

    def close(self):
        pass


def test_sequential_blocks_and_immediate_snapshots():
    calls = []

    def volumetric(value, **kwargs):
        calls.append(kwargs)
        return Plot(value)

    viz = SimpleNamespace(volumetric=volumetric)
    body = """```python
from builtins import _test_viz
value = [1]
_test_viz.volumetric(value)
value.append(2)
```
<!-- k3d-output: demo/first -->
```python
_test_viz.volumetric(value)
```
<!-- k3d-output: demo/second -->"""
    import builtins
    from unittest.mock import patch

    with patch.object(builtins, "_test_viz", viz, create=True):
        result = hook.execute_example("demo", 1, body, "example.md", viz)
    assert result == {"demo/first": "[1]", "demo/second": "[1, 2]"}
    assert calls == [dict(show=False, save=False, snapshot_type="full")] * 2
    assert viz.volumetric is volumetric


def test_missing_plot_fails_and_restores_wrapper():
    original = lambda *args, **kwargs: Plot(1)
    viz = SimpleNamespace(volumetric=original)
    with pytest.raises(ValueError, match="requires one plot"):
        hook.execute_example(
            "demo", 1, "<!-- k3d-output: demo/missing -->", "x.md", viz
        )
    assert viz.volumetric is original


def test_boundaries_and_unclosed_example():
    source = (
        "ignore\n    <!-- k3d-example: demo -->\n    content\n    <!-- /k3d-example -->"
    )
    assert list(hook.examples(source, "x.md")) == [("demo", 2, "content")]
    with pytest.raises(PluginError, match="missing K3D end"):
        list(hook.examples("<!-- k3d-example: demo -->", "x.md"))


@pytest.mark.parametrize(
    "url, expected",
    [
        ("", "assets/k3d/demo/first.html"),
        ("api_reference/morphology/", "../../assets/k3d/demo/first.html"),
    ],
)
def test_embed_relative_to_page(url, expected, monkeypatch):
    monkeypatch.setattr(hook, "_assets", {"demo/first": "assets/k3d/demo/first.html"})
    page = SimpleNamespace(url=url, file=SimpleNamespace(src_uri="test.md"))
    result = hook.on_page_content(
        "before<!-- k3d-output: demo/first -->after", page, None, None
    )
    assert f'src="{expected}"' in result
    assert result.startswith("before<iframe") and result.endswith("</iframe>after")


def test_orphan_output_is_rejected(monkeypatch):
    monkeypatch.setattr(hook, "_assets", {})
    page = SimpleNamespace(url="", file=SimpleNamespace(src_uri="index.md"))
    with pytest.raises(PluginError, match="missing K3D output"):
        hook.on_page_content("<!-- k3d-output: missing/plot -->", page, None, None)


def test_on_files_registers_markdown_and_docstring_snapshots(tmp_path, monkeypatch):
    import qim3d.viz

    monkeypatch.setattr(hook, "_assets", {})
    monkeypatch.setattr(qim3d.viz, "volumetric", lambda value, **kwargs: Plot(value))

    def example(name, value):
        return f"""<!-- k3d-example: {name} -->
```python
import qim3d.viz
qim3d.viz.volumetric({value})
```
<!-- k3d-output: {name}/plot -->
<!-- /k3d-example -->
"""

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "index.md").write_text(example("markdown", 1), encoding="utf-8")
    package = tmp_path / "qim3d"
    package.mkdir()
    (package / "example.py").write_text(
        'def documented():\n    """\n'
        + "".join("    " + line + "\n" for line in example("docstring", 2).splitlines())
        + '    """\n',
        encoding="utf-8",
    )
    config = MkDocsConfig(config_file_path=str(tmp_path / "mkdocs.yml"))
    config.load_dict(
        {
            "site_name": "Test",
            "docs_dir": str(docs),
            "site_dir": str(tmp_path / "site"),
            "plugins": [],
        }
    )
    errors, _ = config.validate()
    assert not errors
    files = Files([File("index.md", str(docs), config.site_dir, True)])

    config.plugins["docs_k3d"] = hook
    result = config.plugins.on_files(files, config=config)

    assert (
        result.get_file_from_path("assets/k3d/markdown/plot.html").content_string == "1"
    )
    assert (
        result.get_file_from_path("assets/k3d/docstring/plot.html").content_string
        == "2"
    )
    assert hook._assets == {
        "markdown/plot": "assets/k3d/markdown/plot.html",
        "docstring/plot": "assets/k3d/docstring/plot.html",
    }
