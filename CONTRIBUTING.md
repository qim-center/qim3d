# Contributor Guide

This file provides guidance for contributors to `qim3d` and other [QIM Centre](https://qim.dk/) projects.

## Contributor Setup

Please follow these steps to get a development environment set up.

### Project manager: `uv`

The recommended tooling for local development is to install [`uv`](https://docs.astral.sh/uv/):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Pre-Commit

To ensure consistent styling and reduce git diffs, please install the pre-commit git hook by running the following:

```sh
uvx pre-commit install
```

You may wish to install pre-commit as a tool to be able to run it outside the git hook:

```sh
uv tool install pre-commit
```

## Review Policy

For submitting pull-requests, please make _atomic_ pull requests (and ideally also atomic commits)
and _select a QIM team member_ to do a technical review. The reviewer should in most cases be a _peer_ as opposed to a lead (Felipe, Jakob).

After implementing any recommended changes and obtaining approval, request a brief review from a lead (Felipe, Jakob).

After approval has been obtained, then the request may be merged (preferably with a `rebase` to keep the git history linear).
It should generally be the code author who performs the merge action.
