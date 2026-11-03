# Algebra

**Code:** `ada/algebra.py`, `ada/roots.py` · **Tests:** `tests/test_algebra.py`, `tests/test_roots.py`

A language model sets up equations well enough and then slips while solving
them: it moves a term without changing its sign, or divides one side and not
the other. `ada.solve`, and `ada compute` or the `ada_compute` tool given an
equation, solve it exactly, show each step and put the answer back into the
equations it came from.

```bash
ada compute "2x + 3 = 11"                  # x = 4
ada compute "x + y = 10; x - y = 2"        # x = 6, y = 4
ada compute "solve d = r*t for t"          # t = d/r, when r ≠ 0
ada compute "x^2 - 3x + 1 = 0"             # x = (3 - sqrt(5))/2 or x = (3 + sqrt(5))/2
ada compute "3(x + 2) - x"                 # 2x + 6
```

```python
import ada

result = ada.solve("x + y = 10; x - y = 2")
result.values       # {"x": "6", "y": "4"}
result.steps        # ["(1) x + y = 10", "(2) x - y = 2", "(2) − (1): -2y = -8",
                    #  "Divide (2) by (-2): y = 4", "(1) − (2): x = 6",
                    #  "Check x + y = 10 with x = 6, y = 4: 10 = 10, which holds.", ...]

ada.simplify("(x + 1)^2 - (x - 1)^2").result   # "4x"
ada.check_solution("2x + 3 = 11", {"x": 4})     # [Substitution(left='11', right='11', holds=True)]
```

## Letters and numbers

- **A letter is a variable**: one letter, such as x, y or t. `e` stays the
  constant, and a longer name such as `cost` or `xy` is refused with the
  reason, since `xy` could be one name or x times y.
- **Coefficients are fractions.** 0.1x + 0.2 = 0.5 gives x = 3 exactly, and
  3x = 1 gives x = 1/3 (with its decimal, 0.333…, beside it).
- **A missing multiplication sign** is read only when both readings agree, as
  in [Exact arithmetic](compute.md): `2(x + 1)` is fine, but `6/2x` is 3x or
  3/x, so it is refused with both. `3 x 4` is refused too, since x may be a
  times sign or the letter.
- **Numbers inside are worked out first**, with the same rules as
  `compute`: `sqrt(16) x + 3!` is 4x + 6. pi, and roots that are not
  fractions, are refused, since every coefficient stays exact.

## Simplifying

`ada.simplify` multiplies out and collects like terms. Terms are written
highest power first, and the result reads back as the same expression:
`x^2 + 2x*y + y^2`, with `*` between letters so `xy` is never mistaken for
one name, and fractions in parentheses: `(5/6)x`.

A quotient is kept in lowest terms where ADA can tell them: common whole
numbers and powers of letters cancel, and so does any common factor when only
one letter appears, so `(x^2 - 1)/(x - 1)` is x + 1. ADA lists what each
divisor must not be: here x ≠ 1. A common factor in two or more letters, as in
`(x^2 - y^2)/(x - y)`, is left as it is.

## Solving

`ada.solve` takes one equation, several separated by `;`, new lines, "and",
or commas between equations, or a list. A chain such as `x + 1 = 2y = 6` is
two equations.

| Result | `kind` | Example |
|---|---|---|
| One value for each unknown | `one` | `2x + 3 = 11` gives x = 4 |
| No values work | `none` | `x + 1 = x + 2` reduces to 0 = 1; `reason` says so |
| Infinitely many | `many` | `2x + 3y = 12` gives x = -1.5y + 6, for any y; `free` lists y |

Each equation is moved to one side and its terms collected, then the
equations are solved by elimination (Gauss-Jordan), with each step written as
the equation it gives: "(2) − (1): -2y = -8". The first unknown, in
alphabetical order, is solved for first, so in a `many` result the later
letters are the free ones.

**Denominators.** An equation with a letter in a denominator is multiplied by
it, and the value that would make it 0 is ruled out: `120/x = 8` gives x = 15,
when x ≠ 0, and `(x^2 - 1)/(x - 1) = 2` has no solution, since its only
candidate, x = 1, makes x - 1 zero.

**Solving for one letter.** With `unknowns`, the other letters are treated as
known, so a formula can be rearranged: `solve("d = r*t", unknowns="t")` gives
t = d/r, when r ≠ 0. Dividing by an expression in the known letters adds the
condition that it is not 0.

**The check.** Every `one` and `many` result ends with each equation, the
values put in and both sides worked out: "Check 2x + 3 = 11 with x = 4: 11 =
11, which holds."

## Powers above 1

One equation in one unknown, once denominators are multiplied away and
everything is moved to one side, is a polynomial = 0. `solve` gives its real
roots, smallest first, in `roots`; `kind` is `one`, `several` or `none`.

First the polynomial is factored over the rationals with SymPy, and the
factoring is a step: "Factor over the rationals: x^2 - 5x + 6 = (x - 2) *
(x - 3)." A factor repeated k times gives a root that carries multiplicity k.
Then each factor's roots are found:

| The factor | How its roots are found | Written as |
|---|---|---|
| Degree 1 | Its root is a whole number or a fraction | `2`, `0.5`, `1/3` |
| A quadratic | The quadratic formula, with square factors taken out of the root | `(3 - sqrt(5))/2`, `-2*sqrt(2)` |
| a*x^n + c | x^n = t, so x = root(t, n), and its negative when n is even | `root(2, 3)` |
| Anything else | SymPy isolates each real root in an interval with rational ends; up to degree 30, Sturm's theorem counts the roots again and a different count is refused. Each interval is halved until the root is known to 20 significant digits | `x ≈ 1.324717957244746026` |

Every exact value is written so `compute` reads it back, and its decimal
comes from `compute`. Each exact root is checked by putting it into the
polynomial with exact arithmetic, numbers of the form a + b*sqrt(m)
included, so "it gives exactly 0, which holds" is a calculation, not a
rounding. A root with no exact form is marked `exact: false` and comes with
its `bracket`: an interval, rounded outwards, where the polynomial changes
sign, so a root lies inside it.

A quadratic with a negative discriminant has no real root; its complex roots
are named in the steps (`x^2 + 1 = 0`: ±i) and `kind` is `none`.

When the equation divides by an expression in the unknown, the values that
make it zero are divided out of the polynomial before its roots are found:
`1/x + 1/x^2 = 1` gives (1 ± sqrt(5))/2, when x ≠ 0, and
`x/(x^2 - 1) = 1/(x^2 - 1)` has no solution. A condition on a divisor in one
letter lists each value ruled out: `x ≠ -3, x ≠ 1`.

Polynomials up to degree 100 are solved (`MAX_ROOT_DEGREE`). Whole numbers up
to 10^30 are split into primes to take square factors out of a root; a root
of a larger number stays as it is.

## Systems with powers

Several equations, or several unknowns, with a power above 1 or two
unknowns multiplied together are a system of polynomial equations. When
the system has finitely many solutions and no letters other than the
unknowns, `solve` gives its real solutions in `solutions`, ordered by the
unknowns' values; `kind` is `one`, `several` or `none`.

```python
ada.solve("x*y = 6; x + y = 5").answer
# '(x = 2, y = 3) or (x = 3, y = 2)'
ada.solve("x^2 + y^2 = 1; x + y = 1/2").answer
# '(x = (1 - sqrt(7))/4 ≈ -0.41143782776614764763, y = (1 + sqrt(7))/4 ≈ ...) or (...)'
ada.solve("x^3 = 2; y = x^2 + 1").values
# {'x': 'root(2, 3)', 'y': 'root(2, 3)^2 + 1'}
```

How it is solved:

1. SymPy eliminates the unknowns with a Gröbner basis in lex order. This
   leaves one equation in one letter, g = 0, with every other unknown a
   polynomial in that letter: "Eliminate x with a Gröbner basis (SymPy, lex
   order): y^2 - 5y + 6 = 0, with x = -y + 5." For a system with up to 16
   solutions, each unknown is tried as the letter, and the one whose g has
   the fewest factors without exact roots is kept. A solution counted more
   than once (x^2 = 0, y^2 = 0) is first counted once, by adding the
   equations the unknowns satisfy with each root once.
   When no unknown tells the solutions apart (x + y + z = 6, xy + yz + zx =
   11, xyz = 6 has six), the letter is a combination such as u = x + 3y + 7z.
2. The real roots of g are found as in "Powers above 1".
3. Each unknown is worked out at each root. It is exact when the root is a
   fraction, a + b*sqrt(m) or root(t, n), and is written so `compute` reads
   it back. Otherwise it comes to 20 significant digits, with an interval
   worked out with exact fractions (`brackets`) that holds the value.

Each entry in `solutions` has the `values`, whether each is `exact`, their
`decimals` and the `brackets` of the values with no exact form. A single
solution also fills `values` and `decimals`, as for a linear system.

**The check.** ADA writes each unknown as its polynomial in the letter and
puts it into every equation. The result must leave no remainder when divided
by the factor of g the root comes from, so the equation holds at every root
of that factor. This is exact arithmetic, whatever the roots look like.
Solutions that are fractions are also put back into each equation as
numbers. A factor whose roots make a denominator zero is dropped, with a
step that says so.

"No solution" says why: the equations contradict each other (eliminating
leaves 1 = 0), or g has no real root, so every solution is complex.

Limits: up to 6 unknowns (`MAX_UNKNOWNS`), and at most 36 solutions by
Bezout's bound, the product of the equations' powers (`MAX_SOLUTIONS`): two
equations up to the 6th power, three cubics and a quartic, or six
quadratics. Larger systems can take minutes to eliminate.

## Checking a solution

`ada.check_solution(equations, values)` puts the values in and works out both
sides exactly, so a solution a model gives can be checked without solving
again. `values` is a dict, `{"x": 4}` or `{"x": "7/2"}`, or text, `"x = 6, y =
4"`.

## Refusals

| Code | When | Example |
|---|---|---|
| `unsupported_expression` | A system with powers that has infinitely many solutions, or letters besides the unknowns; a power above 1 of an unknown with another letter; or a function, power or remainder with a letter in it | `x^2 + y^2 = 1`, `x*y = a; x + y = 1` for x and y, `a*x^2 = 1` for x, `sqrt(x)`, `2^x`, `x > 3` |
| `missing_value` | `check_solution` has no value for a letter | `check_solution("x + y = 3", {"x": 1})` |
| `ambiguous_expression` | A side reads two ways | `6/2x = 3`, `x + 10%` |
| `invalid_expression` | A name ADA does not know, no = sign, no letter to solve for | `cost = 5`, `2 + 2 = 4` |
| `division_by_zero` | An expression divides by something that is always zero | `x/(x - x)` |
| `too_large` | More than 2,000 terms once multiplied out, a power above 200, more than 50 equations, a polynomial equation above degree 100, or a system with powers that has more than 6 unknowns or could have more than 36 solutions | `(x + y + z + 1)^60`, `x^101 = x + 1`, `x^7 = 2; y^6 = 3` |

"Linear in the unknowns" means each unknown appears to the first power and no
two are multiplied together, once denominators are multiplied away. Other
letters, as in `a*x + b = c` solved for x, may appear anywhere. One equation
in one unknown may have any power up to 100, and a system with powers any
number of equations, but then the unknowns must be the only letters.

## How the results are tested

Beyond the examples, these tests check the algebra against something
independent of it:

- 120 systems of one to four equations are built around a chosen solution
  (from a matrix that can always be solved), and `solve` must give that
  solution back.
- 80 random expressions are simplified, and the simplified and written forms
  must agree, by step 1's arithmetic, at three values of x.
- 60 polynomials are built from three chosen rational roots and a random
  quadratic, and `real_roots` must give exactly those roots, the quadratic's
  checked against the formula worked out with Python's decimal module.
- 80 random cubics: the number of real roots must match the sign of the
  cubic's discriminant (three when it is positive, one when negative), and
  each root must match numpy's to six decimal places.
- 40 systems x + y = s, xy = p must give the roots of t^2 - st + p, worked
  out with Python's decimal module, or no solution when s^2 < 4p; each exact
  solution is put into both equations by step 1's arithmetic and must agree
  to 30 digits.
- 40 lines and circles must meet in two points, one or none, as the line's
  distance from the centre says.
- 12 sets of three whole numbers must come back, in every order, from
  x + y + z, xy + yz + zx and xyz.
- A system with no exact solution is checked against mpmath's roots to 50
  digits: each interval must hold the value.
