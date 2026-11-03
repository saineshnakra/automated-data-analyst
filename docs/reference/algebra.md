# Algebra

**Code:** `ada/algebra.py` · **Tests:** `tests/test_algebra.py`

A language model sets up equations well enough and then slips while solving
them: it moves a term without changing its sign, or divides one side and not
the other. `ada.solve`, and `ada compute` or the `ada_compute` tool given an
equation, solve it exactly, show each step and put the answer back into the
equations it came from.

```bash
ada compute "2x + 3 = 11"                  # x = 4
ada compute "x + y = 10; x - y = 2"        # x = 6, y = 4
ada compute "solve d = r*t for t"          # t = d/r, when r ≠ 0
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

## Checking a solution

`ada.check_solution(equations, values)` puts the values in and works out both
sides exactly, so a solution a model gives can be checked without solving
again. `values` is a dict, `{"x": 4}` or `{"x": "7/2"}`, or text, `"x = 6, y =
4"`.

## Refusals

| Code | When | Example |
|---|---|---|
| `unsupported_expression` | An equation is not linear in the unknowns, or a function, power or remainder has a letter in it | `x^2 = 4`, `x*y = 6`, `sqrt(x)`, `2^x`, `x > 3` |
| `missing_value` | `check_solution` has no value for a letter | `check_solution("x + y = 3", {"x": 1})` |
| `ambiguous_expression` | A side reads two ways | `6/2x = 3`, `x + 10%` |
| `invalid_expression` | A name ADA does not know, no = sign, no letter to solve for | `cost = 5`, `2 + 2 = 4` |
| `division_by_zero` | An expression divides by something that is always zero | `x/(x - x)` |
| `too_large` | More than 2,000 terms once multiplied out, a power above 200, or more than 50 equations | `(x + y + z + 1)^60` |

"Linear in the unknowns" means each unknown appears to the first power and no
two are multiplied together, once denominators are multiplied away. Other
letters, as in `a*x + b = c` solved for x, may appear anywhere.

## How the results are tested

Beyond the examples, two tests check the algebra against something
independent of it. 120 systems of one to four equations are built around a
chosen solution (from a matrix that can always be solved), and `solve` must
give that solution back. 80 random expressions are simplified, and the
simplified and written forms must agree, by step 1's arithmetic, at three
values of x.
