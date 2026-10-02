"""Run the Pyodide comparison workloads against a native Symbolica package."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package_dir.resolve()))
    start = time.perf_counter()
    import symbolica
    import symbolica.core
    from symbolica.community import hep, tensor

    if os.environ.get("SYMBOLICA_LICENSE_KEY"):
        symbolica.set_license_key(os.environ["SYMBOLICA_LICENSE_KEY"])
    import_ms = (time.perf_counter() - start) * 1000
    root = Path(__file__).resolve().parents[2]
    workload = Path(__file__).with_name("benchmark_pyodide.py")
    scope = {"hep_model_json": (root / "examples/hep/scalar_phi3.json").read_text()}
    exec(compile(workload.read_text(), str(workload), "exec"), scope)
    result = json.loads(scope["benchmark_result"])
    module = Path(symbolica.core.__file__)
    result.update({
        "platform": platform.platform(), "executable": sys.executable,
        "native_module": str(module),
        "native_module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "import_ms": import_ms, "network_time_included": False,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
    })
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
