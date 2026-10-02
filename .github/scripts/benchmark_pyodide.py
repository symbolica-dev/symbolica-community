"""Representative warm-runtime workloads for comparing identical WASM builds."""

import json
import statistics
import sys
import time
from importlib.metadata import version

from symbolica import E, S
from symbolica.community import hepkit as hep
from symbolica.community.tensor import Representation, Tensor, TensorLibrary, TensorName, TensorNetwork

x, y, z, a, b = S("x", "y", "z", "a", "b")
power = (x + y + z + a + b) ** 12
expanded = power.expand()
assert expanded.derivative(x) == (12 * (x + y + z + a + b) ** 11).expand()

common = (x + y + z) ** 4
left = (common * (x - y + 2 * z) ** 3).expand().to_polynomial()
right = (common * (2 * x + y - z) ** 3).expand().to_polynomial()
assert left.gcd(right) == common.expand().to_polynomial()

rational = E("(x^24-y^24)/(x-y)")
cancelled = rational.cancel()
expected_cancel = sum((x ** (23 - i) * y**i for i in range(24)), E("0"))
assert (cancelled - expected_cancel).expand().cancel() == E("0")

integrands = [E(s) for s in ("x^7", "1/(1+x^2)", "exp(2*x)", "sin(x)", "x*exp(x)")]
for expression in integrands:
    assert (expression.integrate(x).derivative(x) - expression).expand().cancel() == E("0")

space = Representation.euc(8)
matrix = Tensor.dense(TensorName("benchmark_matrix")(space, space), [E(str(i + 1)) for i in range(64)])
library = TensorLibrary.hep_lib()
library.register(matrix)
matrix_expression = matrix.expression()


def tensor_trace():
    network = TensorNetwork(matrix_expression(1, 2) * matrix_expression(2, 1), library=library)
    network.execute(library=library)
    return list(network.result_tensor(library=library))[0]


assert tensor_trace() == E(str(sum((8*i+j+1)*(8*j+i+1) for i in range(8) for j in range(8))))
model = hep.Model.from_json(hep_model_json)
process = model.process(["scalar_0"], ["scalar_0", "scalar_0"])


def generate():
    result = process.generate_diagrams(loops=1, max_vertices=3, allow_self_loops=False)
    assert result.report.completed
    return len(result)


assert generate() > 0
workloads = {
    "expand_degree12_5vars": lambda: power.expand(),
    "differentiate_expanded": lambda: expanded.derivative(x),
    "multivariate_gcd": lambda: left.gcd(right),
    "rational_cancel": lambda: rational.cancel(),
    "integrate_5_expressions": lambda: [expression.integrate(x) for expression in integrands],
    "tensor_matrix_trace_8x8": tensor_trace,
    "generate_one_loop_phi3": generate,
}


def measure(function):
    for _ in range(4):
        function()
    count = 1
    while True:
        start = time.perf_counter()
        for _ in range(count):
            function()
        elapsed = time.perf_counter() - start
        if elapsed >= 0.15 or count >= 2048:
            break
        count *= 2
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(count):
            function()
        samples.append((time.perf_counter() - start) * 1000 / count)
    return {"median_ms": statistics.median(samples), "samples_ms": samples, "iterations_per_sample": count}


benchmark_result = json.dumps({
    "python": sys.version,
    "symbolica": version("symbolica"),
    "correctness_checks_passed": True,
    "workloads": {name: measure(function) for name, function in workloads.items()},
})
