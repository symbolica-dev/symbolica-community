"""Maintained compatibility imports retain the canonical native type identity."""

from symbolica.community import tensor
from symbolica.community.hepkit.vakint import Vakint
from symbolica.community.spenso import TensorExpression as LegacyTensorExpression
from symbolica.community.vakint import Vakint as LegacyVakint

legacy_tensor_type: type[tensor.TensorExpression] = LegacyTensorExpression
legacy_vakint_type: type[Vakint] = LegacyVakint
