set minimum-version := '1.55.0'
set default-list := true

# Installs the project dependencies
[group: 'dev']
install:
    uv sync --all-extras

# Formats the code using ruff
[group: 'format']
format:
    uv run ruff check . --fix
    uv run ruff format .

# Checks the code for linting errors
[group: 'format']
lint:
    uv run ruff check .

# Runs the test suite
[group: 'test']
test *args:
    uv run pytest {{args}}

# Serves documentation locally
[group: 'docs']
docs-serve:
    uv run mkdocs serve

[group: 'docs']
docs-build:
    uv run mkdocs build
