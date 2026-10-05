"""Check the active native-object owners and the published FastSecDec source."""
import json
import sys

with open(sys.argv[1]) as stream:
    metadata = json.load(stream)
backend = sys.argv[2]
assert backend in {"native", "portable"}
nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
packages = [package for package in metadata["packages"] if package["id"] in nodes]
owners = {}
for name in (
    "fastsecdec", "fastsecdec-sectors", "symbolica", "graphica", "numerica",
    "feynkit-graph", "feynkit-model", "feynkit-kinematics", "feynkit-tensor",
    "feynkit-py", "spynso3", "linnet", "idenso", "spenso", "pyo3",
):
    matches = [package for package in packages if package["name"] == name]
    assert len(matches) == 1, (name, [package["id"] for package in matches])
    owners[name] = matches[0]
features = nodes[owners["fastsecdec"]["id"]]["features"]
assert set(features) & {"native", "portable"} == {backend}, features
community = next(package for package in packages if package["name"] == "symbolica_community")
declared = next(dependency["source"] for dependency in community["dependencies"] if dependency["name"] == "fastsecdec")
assert declared.startswith("git+https://github.com/alphal00p/fastSecDec?rev="), declared
revision = declared.rsplit("=", 1)[1]
assert len(revision) == 40 and all(character in "0123456789abcdef" for character in revision)
for name in ("fastsecdec", "fastsecdec-sectors"):
    assert owners[name]["source"] == declared + "#" + revision, owners[name]["source"]
print(f"Unique native owners; {backend} backend; published FastSecDec {revision}")
