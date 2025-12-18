"""
# Spenso

A comprehensive Python tensor library for symbolic and numerical tensor computations, with a focus on physics applications.

## Overview

The Spenso Python API provides powerful tools for:
- **Tensor Algebra**: Dense and sparse tensors with flexible data types
- **Symbolic Computation**: Integration with Symbolica for tensors with symbolic expressions
- **Network Operations**: Tensor networks for optimized computation graphs
- **Physics Applications**: Built-in support for HEP tensors (gamma matrices, color structures, etc.)
- **Performance**: Compiled evaluators for high-speed numerical computation

# Contributors
- Lucien Huber mail@lucien.ch

"""

from ..spenso_native import *

initialize_module()
