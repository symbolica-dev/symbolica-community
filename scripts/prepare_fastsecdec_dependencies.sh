#!/usr/bin/env bash
# Prepare the exact local owner patches while retaining FastSecDec's Git identity.
set -euo pipefail
if [[ $# != 1 ]]; then
  printf 'Usage: %s NEW_OUTPUT_DIRECTORY_OUTSIDE_THIS_CHECKOUT\n' "$0" >&2
  exit 2
fi
community_root=$(cd "$(dirname "$0")/.." && pwd -P)
output_parent=$(cd "$(dirname "$1")" && pwd -P)
dependency_output="$output_parent/$(basename "$1")"
case "$dependency_output/" in
  "$community_root/"*)
    printf 'Choose an output directory outside the community checkout to avoid nested Cargo workspaces.\n' >&2
    exit 2
    ;;
esac
[[ ! -e "$dependency_output" && ! -L "$dependency_output" ]] || {
  printf 'Output already exists: %s\n' "$dependency_output" >&2
  exit 2
}
mkdir "$dependency_output"
cargo read-manifest --manifest-path "$community_root/Cargo.toml" > "$dependency_output/community-manifest.json"
revision=$(python - "$dependency_output/community-manifest.json" <<'PY'
import json
import re
import sys

with open(sys.argv[1]) as stream:
    dependencies = json.load(stream)["dependencies"]
sources = [entry["source"] for entry in dependencies if entry["name"] == "fastsecdec"]
match = re.fullmatch(r"git\+https://github.com/alphal00p/fastSecDec\?rev=([0-9a-f]{40})", sources[0] or "") if len(sources) == 1 else None
if match is None:
    sys.exit("Cargo.toml must pin FastSecDec to one exact published Git revision")
print(match.group(1))
PY
)
checkout="$dependency_output/fastsecdec"
git init --quiet "$checkout"
git -C "$checkout" remote add origin https://github.com/alphal00p/fastSecDec
git -C "$checkout" fetch --quiet --depth=1 origin "$revision"
git -C "$checkout" checkout --quiet --detach FETCH_HEAD
[[ $(git -C "$checkout" rev-parse HEAD) == "$revision" ]]
bash "$checkout/scripts/bootstrap-dependencies.sh" "$checkout" "$dependency_output/owners"
# This clone supplies patch files and bootstrap instructions only. Cargo must
# consume FastSecDec from the published Git revision declared in Cargo.toml.
awk '
  /^\[/ { skip = ($0 == "[patch.\"https://github.com/alphal00p/fastSecDec\"]") }
  !skip { print }
' "$dependency_output/owners/overlay.toml" > "$dependency_output/overlay-community-git.toml"
# Maturin 1.15 does not forward --config to its metadata subprocess. Both
# metadata and compilation honor this command-scoped Cargo home instead.
dependency_cargo_home="$dependency_output/cargo-home"
mkdir "$dependency_cargo_home"
cp "$dependency_output/overlay-community-git.toml" "$dependency_cargo_home/config.toml"
printf 'Prepared: %s\n' "$dependency_output/overlay-community-git.toml"
printf 'For maturin/Pyodide, set CARGO_HOME=%s for that command.\n' "$dependency_cargo_home"
printf 'Cargo caches and locks are isolated; existing config, credentials and caches are untouched.\n'
