# Native gg → HH input identity

These are input files from the native FastSecDec example named in `origin.json`.
The manifest records the exact source commit, input SHA-256 hashes and Rust
generator source hashes. The original native selector uses Linnet connectivity
and cycles to select the six-top hexagon with a central gluon and one g/H pair
on each four-cycle.

The notebook generates the diagrams through HEPKit on every explicit run. It
uses the archived raw diagram only to check the stable content ID and the
entire native physical input after native JSON readback. The generated-order
display name may differ across frontends and is recorded separately. Model,
topology, half-edge order, routing, numerator and factors must match.

The model and parameter card set mt = ymt = 172.5 GeV, mH = 125 GeV and zero
widths. The helper constructs the fixed external point and numerical helicity
projection with native HEPKit APIs; it does not load prepared scalar products,
kernels or numerical results from these files.
