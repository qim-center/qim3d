set minimum-version := '1.55.0'
set default-list := true
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

### --- SETUP ---
# Installs the project dependencies
[group: 'setup']
install:
    uv sync

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
    uv run --group lint ruff format .

# Formats the code using ruff and applies lint fixes
[group: 'format']
fix:
    uv run --group lint ruff check . --fix
    uv run --group lint ruff format .

# Checks the code for linting errors
[group: 'format']
lint:
    uv run --group lint ruff check .

### --- TEST ---
# Runs the test suite
[group: 'test']
test *args:
    uv run --group test pytest {{args}}

### --- DOCS ---
# Serves documentation locally
[group: 'docs']
docs-serve:
    uv run --group docs mkdocs serve

[group: 'docs']
docs-build:
    uv run --group docs mkdocs build

# Bumps the version (major, minor, patch, stable, alpha, beta, rc, post or dev; several combine) and commits it
[group: 'release']
bump +bumps:
    git diff --exit-code HEAD -- pyproject.toml uv.lock
    uv version {{ prepend("--bump=", bumps) }}
    git commit --message "chore(version): release $(uv version --short)" -- pyproject.toml uv.lock

# Tags HEAD on master as v<version in pyproject.toml> and pushes the tag, which publishes it to PyPI
[group: 'release']
[confirm("Tag HEAD as a release and push the tag? This publishes it to PyPI. (N/y): ")]
tag-release:
    {{ if trim(`git branch --show-current`) != "main" { error("Releases are tagged on main.") } else { "" } }}
    uv lock --check
    git diff --exit-code HEAD -- uv.lock
    just pc-all
    git tag --annotate v{{ trim(`uv version --short`) }} --message "v{{ trim(`uv version --short`) }}"
    git push origin v{{ trim(`uv version --short`) }}
