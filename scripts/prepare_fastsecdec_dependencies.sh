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
sources = [entry["source"] for entry in dependencies if entry["name"] == "fastsecdec-python"]
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
bash "$checkout/bindings/python/scripts/prepare-community.sh" "$community_root" "$dependency_output"
