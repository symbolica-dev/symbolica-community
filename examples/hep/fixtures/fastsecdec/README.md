These are native HEPKit input fixtures from `alphal00p/fastSecDec`:

- `examples/models/scalar.json`
- `examples/graphs/triangle.dot`
- `examples/graphs/box.dot`
- `examples/graphs/box_rank2_numerator.dot`
- `examples/graphs/sunset_2loop_numerator.dot`

The files are copied unchanged. HEPKit's model and DOT readers own parsing,
momentum routing and tensor metadata. `showcase/inputs.py` supplies parameter
cards and scalar products through public HEPKit objects. The bridge receives
those objects directly.

Form values are converted from their Python binary64 value to its exact integer
ratio before entering scalar bindings or kinematics. The model card receives
that same binary64 value. This preserves the numerical point while keeping
native affine/polynomial preparation in the exact coefficient ring.
This is a representation policy for symbolic preparation, not extra physical
precision: binary64 `0.1` becomes `3602879701896397/36028797018963968`, not
`1/10`. It cannot restore exact on-shell relations from independently rounded
data. Supply intended exact relations as Symbolica expressions when they matter.
The current native family and conservative domain checks need exact coefficient
arithmetic; this is not a blanket restriction on every HEPKit scalar or numerical
numerator coefficient.

All local scalar vertex/propagator numerators and overall weights are one except
the two explicit polynomial numerators. The integral measure is
`prod d^D k / (i*pi^(D/2))`, with `D=4-2*eps`, no coupling factor and no extra
Euler-gamma normalization. The sunset's requested expansion reaches epsilon
power one; the one-loop examples reach the finite part. This folder contains no
integration results or claim of browser performance.
