"""Execute marked documentation examples and embed K3D snapshots.

See CONTRIBUTING.md for marker syntax and usage.
"""

import ast
import html
import logging
import re
import textwrap
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.exceptions import PluginError
from mkdocs.structure.files import File, Files
from mkdocs.structure.pages import Page
from mkdocs.utils import get_relative_url

START = re.compile(r"<!-- k3d-example: ([a-z0-9_-]+) -->")
OUTPUT = re.compile(r"<!-- k3d-output: ([a-z0-9_-]+/[a-z0-9_-]+) -->")
END = "<!-- /k3d-example -->"
_assets: dict[str, str] = {}
_logger = logging.getLogger("mkdocs.plugins.k3d")


def examples(source: str, filename: str | Path) -> Iterator[tuple[str, int, str]]:
    """Yield marked regions; retain source line numbers for execution errors."""
    active = None
    body = []
    for number, line in enumerate(source.splitlines(), 1):
        start = START.fullmatch(line.strip())
        if start:
            if active:
                raise PluginError(f"{filename}:{number}: nested K3D example")
            active = (start[1], number)
            body = []
        elif line.strip() == END:
            if not active:
                raise PluginError(f"{filename}:{number}: unmatched K3D end marker")
            yield active[0], active[1], textwrap.dedent("\n".join(body))
            active = None
        elif active:
            body.append(line)
    if active:
        raise PluginError(f"{filename}:{active[1]}: missing K3D end marker")


def execute_example(
    name: str, first_line: int, body: str, filename: str | Path, viz
) -> dict[str, str]:
    """Run code blocks in a shared namespace and capture each volumetric plot."""
    import matplotlib.pyplot as plt

    namespace = {"__name__": "__docs_example__"}
    pending = []
    results = {}
    original = viz.volumetric

    def capture(*args, **kwargs):
        kwargs.update(show=False, save=False, snapshot_type="full")
        plot = original(*args, **kwargs)
        try:
            pending.append(plot.get_snapshot())
        finally:
            plot.close()
        return plot

    try:
        with (
            patch.object(plt, "show"),
            patch.object(viz, "volumetric", capture),
        ):
            code = None
            code_line = None
            for offset, line in enumerate(body.splitlines(), first_line + 1):
                if code is not None:
                    if line.strip() == "```":
                        source = "\n" * (code_line - 1) + textwrap.dedent(
                            "\n".join(code)
                        )
                        exec(compile(source, str(filename), "exec"), namespace)
                        code = None
                    else:
                        code.append(line)
                elif line.strip() == "```python":
                    code = []
                    code_line = offset + 1
                elif match := OUTPUT.fullmatch(line.strip()):
                    key = match[1]
                    if key.split("/")[0] != name:
                        raise ValueError(f"Output {key} must belong to example {name}")
                    if len(pending) != 1:
                        raise ValueError(
                            f"Output {key} requires one plot, got {len(pending)}"
                        )
                    if key in results:
                        raise ValueError(f"Duplicate output {key}")
                    results[key] = pending.pop()
            if code is not None:
                raise ValueError("Unclosed Python code block")
            if pending:
                raise ValueError("Plot has no following K3D output marker")
            if not results:
                raise ValueError("Example has no K3D output")
            return results
    finally:
        plt.close("all")


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Generate static files before MkDocs copies assets and renders pages."""
    import qim3d.viz as viz

    _assets.clear()
    root = Path(config.config_file_path).parent
    sources = [(Path(f.abs_src_path), None) for f in files.documentation_pages()]
    sources += [(p, "python") for p in sorted((root / "qim3d").rglob("*.py"))]
    names = set()
    for path, kind in sources:
        source = path.read_text(encoding="utf-8")
        if "<!-- k3d-example:" not in source:
            continue
        # Read docstrings without importing every module or executing source code.
        regions = [source]
        if kind == "python":
            regions = []
            for node in ast.walk(ast.parse(source)):
                if isinstance(
                    node,
                    (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    doc = ast.get_docstring(node, clean=False)
                    if doc and "<!-- k3d-example:" in doc:
                        regions.append("\n" * (node.body[0].lineno - 1) + doc)
        for region in regions:
            for name, line, body in examples(region, path):
                if name in names:
                    raise PluginError(f"{path}:{line}: duplicate K3D example {name}")
                names.add(name)
                try:
                    _logger.info("Generating K3D example: %s", name)
                    outputs = execute_example(name, line, body, path, viz)
                except Exception as exc:
                    raise PluginError(
                        f"{path}:{line}: K3D example {name} failed: {exc}"
                    ) from exc
                for key, snapshot in outputs.items():
                    asset = f"assets/k3d/{key}.html"
                    files.append(File.generated(config, asset, content=snapshot))
                    _assets[key] = asset
    return files


def on_page_content(
    html_content: str, page: Page, config: MkDocsConfig, files: Files
) -> str:
    """Replace markers after mkdocstrings expands Python docstrings."""
    if "<!-- k3d-" not in html_content:
        return html_content

    def embed(match: re.Match[str]) -> str:
        key = match[1]
        if key not in _assets:
            raise PluginError(f"{page.file.src_uri}: missing K3D output {key}")
        url = html.escape(get_relative_url(_assets[key], page.url), quote=True)
        return (
            f'<iframe src="{url}" title="{html.escape(key)}" '
            'width="100%" height="500" style="border:0" loading="lazy"></iframe>'
        )

    rendered = OUTPUT.sub(embed, html_content)
    return START.sub("", rendered).replace(END, "")
