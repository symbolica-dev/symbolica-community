export const examples = [
  {
    id: "expressions", title: "Build an expression", label: "Symbols & functions",
    description: "S creates symbols you can combine with Python operators. A symbol can also act as a function. Symbolica keeps the result exact, including fractions.",
    experiment: "Try changing the power or the arguments of f.",
    code: `from symbolica import S

x, y, f = S("x", "y", "f")
expression = (x + 2*y)**2 + f(x, y)/5

print(expression)
print("Expanded:")
expression.expand()`,
  },
  {
    id: "expand", title: "Parse, expand, collect", label: "Polynomial algebra",
    description: "E parses mathematical text; inside that text, ^ means a power. Expand the product, then group the terms by powers of a chosen symbol.",
    experiment: "Change collect(x) to collect(y) and compare the grouping.",
    code: `from symbolica import E, S

x, y = S("x", "y")
expression = E("(x + 2*y)^3 + x*y")
expanded = expression.expand()

print("Expanded:", expanded)
print("Collected in x:", expanded.collect(x))`,
  },
  {
    id: "replace", title: "Replace values and patterns", label: "Pattern matching",
    description: "Replace a specific symbol, or describe a family of expressions with a wildcard. A symbol whose name ends in an underscore can capture part of a match.",
    experiment: "Replace each f argument with its cube instead of its square.",
    code: `from symbolica import S

x, y, f, a_ = S("x", "y", "f", "a_")
print("Substitution:", (x**2 + x*y).replace(x, y + 2).expand())

expression = f(2) + f(4) + f(6)
print("Pattern:", expression.replace(f(a_), a_**2))`,
  },
  {
    id: "differentiate", title: "Differentiate and factor", label: "Exact calculus",
    description: "Differentiate with respect to a symbol. For polynomials, factor reverses expansion and exposes repeated factors.",
    experiment: "Take a second derivative by calling derivative(x) twice.",
    code: `from symbolica import E, S

x = S("x")
expression = E("(x + 2)^4 + sin(x)")
print("Derivative:", expression.derivative(x))

polynomial = E("(x - 2)^2 * (x + 3)").expand()
print("Polynomial:", polynomial)
print("Factors:")
polynomial.factor()`,
  },
  {
    id: "integrate", title: "Integrate with an explanation", label: "Integration steps",
    description: "Find an antiderivative and inspect the integrator’s explanation. Differentiate the result to check it against the original integrand.",
    experiment: "Try 1 / (1 + x^2) for an inverse-trigonometric result.",
    code: `from symbolica import E, S

x = S("x")
integrand = E("x / (x + 2)")
result, overview, steps = integrand.integrate_with_steps(x)

print("Antiderivative:", result)
print("Check (should be 0):", (result.derivative(x) - integrand).together().cancel())
print("\\n" + overview)
print("Recorded steps:", len(steps))
result`,
  },
  {
    id: "rational", title: "Rearrange rational expressions", label: "Fractions",
    description: "Combine fractions over a common denominator, split them into partial fractions, or cancel shared factors. Choose the representation that helps your next calculation.",
    experiment: "Change one denominator and inspect the new partial fractions.",
    code: `from symbolica import E, S

x = S("x")
expression = E("1 / (x + 1) + 2 / (x + 3)")
combined = expression.together()

print("Together:", combined)
print("Apart:", combined.apart(x))
print("Cancelled:", E("(x^2 - 4) / (x - 2)").cancel())`,
  },
  {
    id: "series", title: "Expand a series, solve a system", label: "Series & equations",
    description: "Compute a local series around a point. To solve equations, write each one as an expression equal to zero and provide the unknowns.",
    experiment: "Increase the series order, or change the constants in the equations.",
    code: `from symbolica import E, S, Expression

x, y = S("x", "y")
print("Series:", E("exp(x)").series(x, 0, 4))

solutions = Expression.solve([2*x + y - 7, x - y - 2], [x, y])
for solution in solutions:
    print("x =", solution[x], "; y =", solution[y])`,
  },
  {
    id: "evaluate", title: "Evaluate at a chosen precision", label: "Numerical values",
    description: "Build an evaluator with an explicit parameter order. Use Symbolica Float inputs with a chosen precision, then evaluate in the browser. Starting from decimal strings avoids rounding inputs to a Python float first.",
    experiment: "Change the inputs or request more decimal digits.",
    note: "The desktop guide uses evaluate for NumPy arrays and native JIT compilation. This example uses the arbitrary-precision evaluator supported by the WASM build.",
    code: `from symbolica import E, S, Float

x, y = S("x", "y")
expression = E("x^2 + x*y + 1/3")
evaluator = expression.evaluator([x, y])

# Work at 40 digits, then display 30 to leave room for rounding.
inputs = [Float("1.25", decimal_digits=40), Float("2.5", decimal_digits=40)]
values = evaluator.evaluate_with_prec(inputs, 40)
print("Value:", values[0].to_decimal(30))`,
  },
  {
    id: "polynomials", title: "Work with polynomial objects", label: "Dedicated polynomial tools",
    description: "Convert an expression into a polynomial with an explicit variable order. Polynomial objects provide their own differentiation and factorization operations.",
    experiment: "Swap the variable order to [y, x] and inspect the result.",
    code: `from symbolica import E, S

x, y = S("x", "y")
polynomial = E("(x - y)^2 * (x + 2*y)").to_polynomial(vars=[x, y])

print("Polynomial:", polynomial)
print("Derivative in x:", polynomial.derivative(x))
print("Factors:", polynomial.factor())`,
  },
];
