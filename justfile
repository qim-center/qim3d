set minimum-version := '1.55.0'
set default-list := true

### --- SETUP ---
# Installs the project dependencies
[group: 'setup']
install:
    uv sync --all-extras

# Installs the pre-commit hook (prek)
[group: 'setup']
install-pre-commit:
    uv run --group misc prek install

### --- DEV ---
# Runs the pre-commit hook on changes
[group: 'dev']
pre-commit:
    uv run --group misc prek

alias pc := pre-commit

# Runs the pre-commit hook on all files
[group: 'dev']
pre-commit-all:
    uv run --group misc prek --all-files

alias pc-all := pre-commit-all

### --- FORMAT ---
# Formats the code using ruff
[group: 'format']
format:
    uv run ruff check . --fix
    uv run ruff format .

# Checks the code for linting errors
[group: 'format']
lint:
    uv run ruff check .

### --- TEST ---
# Runs the test suite
[group: 'test']
test *args:
    uv run pytest {{args}}

### --- DOCS ---
# Serves documentation locally
[group: 'docs']
docs-serve:
    uv run mkdocs serve

[group: 'docs']
docs-build:
    uv run mkdocs build
