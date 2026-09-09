import itertools
import json
import subprocess
import tomllib as toml
from pathlib import Path

from ruff import find_ruff_bin

PROJECT_ROOT = Path(__file__).parents[1].resolve()

# Configuration
# CONFIG_SOURCE = "pyproject.toml"
CONFIG_SOURCE = ".ruff.toml"
EXCLUDE_PREVIEW_CODES = True  # if preview enabled
PROHIBIT_PREVIEW_CODES = True  # if preview enabled

# Load configuration
pyproject_config = toml.loads(
    (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
)
config_path = PROJECT_ROOT / CONFIG_SOURCE
if CONFIG_SOURCE == "pyproject.toml":
    try:
        config = pyproject_config["tool"]["ruff"]
    except KeyError as e:
        msg = f"Could not find [tool.ruff] section in {config_path}."
        raise RuntimeError(msg) from e
else:
    config = toml.loads(config_path.read_text(encoding="utf-8"))

preview = config["lint"].get("preview", False)

ruff_bin = find_ruff_bin()

# Get ruff version
version_call = subprocess.run(
    [ruff_bin, "--version"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    check=True,
)
ruff_version = version_call.stdout.strip().split()[1]
ruff_dependency_version = None
for dep in itertools.chain(
    pyproject_config["project"]["dependencies"],
    *pyproject_config["dependency-groups"].values(),
):
    if not isinstance(dep, str):
        continue
    if dep.startswith("ruff"):
        ruff_dependency_version = dep.split("==")[1]
        break
assert ruff_dependency_version is not None, (
    "Could not find ruff dependency in pyproject.toml"
)
print(f"Using ruff version: {ruff_version}")
print(f"Configured ruff dependency version: {ruff_dependency_version}")
if not ruff_version == ruff_dependency_version:
    raise RuntimeError(
        f"Ruff version mismatch: {ruff_version} (installed) != {ruff_dependency_version} (configured)"
    )

# Load rules using ruff CLI
all_rules_call = subprocess.run(
    [ruff_bin, "rule", "--all", "--output-format", "json"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    check=True,
)
all_rules = json.loads(all_rules_call.stdout)

# Check that shape of json doc has not changed
for rule in all_rules:
    assert len(rule["status"]) == 1

# Remove removed rules
all_rules = [
    rule for rule in all_rules if next(iter(rule["status"].keys())) != "Removed"
]

rule_map = {rule["code"]: rule for rule in all_rules}

if not preview and EXCLUDE_PREVIEW_CODES:
    filtered_rules = [rule for rule in all_rules if not rule["preview"]]
else:
    filtered_rules = all_rules

all_codes = {rule["code"] for rule in all_rules}
filtered_codes = {rule["code"] for rule in filtered_rules}


def is_code_group(code: str) -> bool:
    """Check if a code is a group code.

    Group code examples are be "E" or "AIR",
    whereas specific codes are "E501".
    """
    return not code[-1].isdigit()


def expand_codes(code_prefix: str, *, expand_only_fixable: bool = False) -> set[str]:
    """Expand a code prefix to all codes that start with that prefix."""
    codes = {}
    if code_prefix == "ALL":
        codes = filtered_codes
    else:
        codes = {code for code in filtered_codes if code.startswith(code_prefix)}

    if expand_only_fixable:
        return {code for code in codes if is_fixable(code)}

    return codes


def is_fixable(code: str) -> bool:
    """Check if a code is fixable."""
    rule = rule_map[code]
    assert rule["fix_availability"] in ("None", "Sometimes", "Always")
    return rule["fix_availability"] != "None"


def is_preview(code: str) -> bool:
    """Check if a code is a preview rule."""
    rule = rule_map[code]
    return rule["preview"]


def fetch_rules(
    selector: str, *, only_explicit: bool = False, expand_only_fixable: bool = False
) -> set[str]:
    """Fetch specific set of rules from config."""
    rules = set()
    lst = config["lint"].get(selector, [])

    for code in lst:
        if is_code_group(code):
            if only_explicit:
                continue

            rules.update(expand_codes(code, expand_only_fixable=expand_only_fixable))
        else:
            rules.add(code)
    return rules


configured_select_codes = fetch_rules("select")
configured_select_codes |= fetch_rules("extend-select")

configured_ignore_codes = fetch_rules("ignore")
# Note: extend-ignore is deprecated, so we don't support it.

configured_fixable_codes = fetch_rules("fixable", expand_only_fixable=True)
configured_fixable_codes |= fetch_rules("extend-fixable", expand_only_fixable=True)

configured_unfixable_codes = fetch_rules("unfixable", expand_only_fixable=True)
# Note: extend-unfixable is not implemented in ruff (as of v0.16.4).

configured_codes = configured_select_codes | configured_ignore_codes
enabled_codes = configured_select_codes - configured_ignore_codes
unconfigured_codes = filtered_codes - configured_codes

configured_fixablelike_codes = configured_fixable_codes | configured_unfixable_codes

all_configured_codes = configured_codes | configured_fixablelike_codes

if not preview and PROHIBIT_PREVIEW_CODES:
    # We need to fetch with only_explicit to check this
    all_explicit_configured_codes = set()
    selectors = [
        "select",
        "extend-select",
        "ignore",
        "fixable",
        "extend-fixable",
        "unfixable",
    ]
    for selector in selectors:
        all_explicit_configured_codes |= fetch_rules(selector, only_explicit=True)

    for code in all_explicit_configured_codes:
        if is_preview(code):
            rule = rule_map[code]
            name = rule["name"]
            raise RuntimeError(
                f'Configured code "{code}" ({name}) is a preview rule, but preview rules are prohibited.'
            )

if unconfigured_codes:
    print("The following rules were not found in either the select or ignore lists:")
    for code in sorted(unconfigured_codes):
        rule = rule_map[code]
        name = rule["name"]

        print(f'    "{code}",  # {name}')

# Check for invalid codes
invalid_codes = all_configured_codes - all_codes
if invalid_codes:
    print("The following codes are configured but not valid Ruff codes:")
    for code in sorted(invalid_codes):
        print(f'    "{code}",')

# Check fixable rules
# We keep an explicit list of these to prevent updates to Ruff causing unexpected fixes.
# This means we don't want to use the default `--fixable = "ALL"`.
# To ensure we don't miss out on fixable rules, we keep track of all potentially fixable
# enabled rules.
# Thus, every fixable, enabled rule must be marked either "fixable" or "unfixable".
fixablelike_enabled_codes = set()

for code in enabled_codes:
    if is_fixable(code):
        fixablelike_enabled_codes.add(code)

# Check that configured fixablelike codes are actually fixable
for code in configured_fixablelike_codes:
    if not is_fixable(code):
        rule = rule_map[code]
        name = rule["name"]
        raise RuntimeError(
            f'Configured code "{code}" ({name}) is not fixable, but is listed in fixable or unfixable.'
        )

configured_enabled_fixablelike_codes = (
    configured_fixable_codes | configured_unfixable_codes
)
unconfigured_fixablelike_enabled_codes = (
    fixablelike_enabled_codes - configured_enabled_fixablelike_codes
)
if unconfigured_fixablelike_enabled_codes:
    print(
        "The following fixable rules are enabled but not listed in either the fixable or unfixable lists:"
    )
    for code in sorted(unconfigured_fixablelike_enabled_codes):
        rule = rule_map[code]
        name = rule["name"]

        print(f'    "{code}",  # {name}')

print("End of script.")
