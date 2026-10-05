"""Tensor evaluation follows the shared Symbolica evaluator interface."""

from typing_extensions import assert_type

from symbolica import Evaluator, Expression, FunctionDefinition
from symbolica.community.tensor import CompiledTensorEvaluator, Tensor, TensorEvaluator


def prepare(
    tensor: Tensor, parameters: list[Expression], functions: list[FunctionDefinition]
) -> TensorEvaluator:
    evaluator = tensor.evaluator(
        params=parameters, functions=functions, jit_compile=False,
        cpe_iterations=1, max_horner_scheme_variables=100,
    )
    assert_type(evaluator, TensorEvaluator)
    assert_type(evaluator.scalar_evaluator, Evaluator)
    assert_type(evaluator.evaluate([[1.0]]), list[Tensor])
    assert_type(evaluator.evaluate_complex([[1.0 + 2.0j]]), list[Tensor])
    evaluator.set_real_params([0], sqrt_real=True)
    return evaluator


def compile_complex(evaluator: TensorEvaluator, prefix: str) -> CompiledTensorEvaluator:
    compiled = evaluator.compile(
        "evaluate_tensor", prefix + ".cpp", prefix, number_type="complex", inline_asm="none",
    )
    assert_type(compiled, CompiledTensorEvaluator)
    assert_type(compiled.evaluate([[1.0 + 2.0j]]), list[Tensor])
    return compiled
