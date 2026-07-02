# Contributor Guide

This file provides guidance for contributors to `qim3d` and other [QIM Centre](https://qim.dk/) projects.

## Contributor Setup

Please follow these steps to get a development environment set up.

### Project manager: `uv`

The recommended tooling for local development is to install [`uv`](https://docs.astral.sh/uv/):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You may find it helpful to read their [getting started guide](https://docs.astral.sh/uv/getting-started/) if you are not familiar with `uv`.

### Pre-Commit

To ensure consistent styling and reduce git diffs, please install the pre-commit git hook by running the following:

```sh
uvx pre-commit install
```

You may wish to install pre-commit as a tool to be able to run it outside the git hook:

```sh
uv tool install pre-commit
```

## Git

The sections below describe some of the best practices of `git` that we strive to abide by.

In summary, they are:

1. Use branches on fork or `qim3d` for your PRs; don't push to other people's work branches
2. Clean up the git history using `rebase` before merging a PR, at least if it is very bad
3. Keep PRs small, clean, and atomic
4. PRs must get:
   1. Technical review by a QIM Developer
   2. Final review by a Senior QIM Team Member (Jakob: @jakobsj or Felipe: @delestro)
5. PRs should be merged with the `rebase` strategy to avoid littering commits

### Crash Course

If you have the required permissions to create branches on the `qim3d` repository, you can put the branches for PR's there. If you don't, you can use the GitHub _fork_ feature to create a fork of the `qim3d` repository under your own user account and make pull requests and associated branches there.

Please note that feature branches on `qim3d` used for PRs are generally _"owned"_ by the developer using them and that they may be force pushed to. For this reason, please refrain from pushing changes to other developer's feature branches.

In order to have a relatively clean git history, it is appreciated if PRs have relatively few, clean commits. You can rewrite them with `git rebase`, see for example [this guide on clean git histories](https://mainmatter.com/blog/2021/05/26/keeping-a-clean-git-history/).

The merge strategy onto `main` should also ideally use the `rebase` strategy. See [this blog post](https://graphite.com/blog/why-ban-merge-commits) for justification.

## Review Policy

For submitting pull-requests, please make _atomic_[^1] pull requests (and ideally also atomic commits)
and _select a QIM team member_ to do a technical review. The reviewer should in most cases be a _peer_ as opposed to a lead (Felipe, Jakob).

After implementing any recommended changes and obtaining approval, request a brief review from a lead (Felipe, Jakob).

After approval has been obtained, then the request may be merged (preferably with a `rebase` to keep the git history linear).
It should generally be the code author who performs the merge action.

[^1]: Here _atomic_ implies that the content of the PR can be applied as a single, atomic change to the codebase and leaves it in a working state afterwards.
