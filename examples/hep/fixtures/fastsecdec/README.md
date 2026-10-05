These are native HEPKit input fixtures from `alphal00p/fastSecDec`:

- `examples/models/scalar.json`
- `examples/graphs/triangle.dot`
- `examples/graphs/box.dot`
- `examples/graphs/box_rank2_numerator.dot`
- `examples/graphs/sunset_2loop_numerator.dot`

The files are copied unchanged. HEPKit's model and DOT readers own parsing,
momentum routing and tensor metadata. `fastsecdec_inputs.py` supplies parameter
cards and scalar products through public HEPKit objects. The bridge receives
those objects directly.

Form values are converted from their Python binary64 value to its exact integer
ratio before entering scalar bindings or kinematics. The model card receives
that same binary64 value. This preserves the numerical point while keeping
native affine/polynomial preparation in the exact coefficient ring.

All local scalar vertex/propagator numerators and overall weights are one except
the two explicit polynomial numerators. The integral measure is
`prod d^D k / (i*pi^(D/2))`, with `D=4-2*eps`, no coupling factor and no extra
Euler-gamma normalization. The sunset's requested expansion reaches epsilon
power one; the one-loop examples reach the finite part. This folder contains no
integration results or claim of browser performance.
