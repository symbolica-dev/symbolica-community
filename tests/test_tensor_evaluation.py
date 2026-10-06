"""Tensor evaluators preserve values and layout with and without NumPy."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("numpy_available", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("number_type", ["real", "complex"])
def test_tensor_evaluation_optional_numpy(
    numpy_available, compiled, number_type, tmp_path
):
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        from pathlib import Path

        numpy_available, compiled, number_type, directory = sys.argv[1:]
        if numpy_available == "False":
            class HideNumpy(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "numpy" or fullname.startswith("numpy."):
                        raise ModuleNotFoundError("NumPy unavailable", name=fullname)
            sys.meta_path.insert(0, HideNumpy())
        else:
            import numpy as np

        from symbolica import S
        from symbolica.community.tensor import Representation, Tensor, TensorName

        x = S("optional_numpy::x")
        descriptor = TensorName("optional_numpy::A")(
            7, Representation.euc(2), Representation.euc(3)
        )
        tensor = Tensor.dense(descriptor, [x, x**2, 0, 2*x, 3, x+1])
        tensor = tensor.permute_axes([1, 0]).to_sparse()
        evaluator = tensor.evaluator([x], jit_compile=False, n_cores=1)
        points = [[2.0], [-1.0]] if number_type == "real" else [[2+1j], [-1-2j]]
        if numpy_available == "True":
            points = np.asarray(points)
        scalar_method = "evaluate" if number_type == "real" else "evaluate_complex"
        expected = getattr(evaluator.scalar_evaluator, scalar_method)(points)
        if numpy_available == "True":
            expected = expected.tolist()
        else:
            assert isinstance(expected, list)
        if compiled == "True":
            prefix = Path(directory) / number_type
            evaluator = evaluator.compile(
                "evaluate_tensor", str(prefix.with_suffix(".cpp")), str(prefix),
                number_type=number_type, inline_asm="none", optimization_level=0,
                native=False,
            )
            method = "evaluate"
        else:
            method = scalar_method
        result = getattr(evaluator, method)(points)
        assert len(result) == len(expected) == 2
        assert evaluator.output_shape == (3, 2)
        for value, components in zip(result, expected, strict=True):
            assert value[:] == components
            assert value.shape == (3, 2)
            assert value.structure == tensor.structure
            assert value.expression().to_expression() == tensor.expression().to_expression()

        constant = Tensor.dense(TensorName.vector("optional_numpy::constant")(
            Representation.euc(2)
        ), [1, 0]).evaluator([], jit_compile=False, n_cores=1)
        assert constant.evaluate([[]])[0][:] == [1.0, 0.0]
        assert constant.evaluate_complex([[]])[0][:] == [1+0j, 0j]
        try:
            getattr(evaluator, method)([[1.0, 2.0]])
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input width was accepted")
        if numpy_available == "False":
            assert "numpy" not in sys.modules
    """)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(numpy_available),
            str(compiled),
            number_type,
            str(tmp_path),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
