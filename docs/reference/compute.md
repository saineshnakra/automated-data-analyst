# Exact arithmetic

**Code:** `ada/compute.py` · **Tests:** `tests/test_compute.py`

A language model guesses at arithmetic. It carries a digit wrong or rounds
too early, and it cannot tell that it has. `ada compute` and the `ada_compute`
tool do the arithmetic instead and show each step, so the model only has to
write the expression and read the answer.

```bash
ada compute "0.15 * (1200 - 300)"      # 135
ada compute "1/3 + 1/6"                # 0.5 = 1/2
ada compute "comb(52, 5) / 2^10" --json
```

```python
import ada

result = ada.compute("2^10 + 3/4")
result.value        # "1024.75"
result.fraction     # "4099/4"
result.exact        # True
result.steps        # ["2^10 = 1024", "3 / 4 = 0.75", "1024 + 0.75 = 1024.75"]
```

## Numbers stay exact

- **Integers** have no size limit up to 10,000 digits (`MAX_DIGITS`).
- **Decimals are read as fractions.** 0.1 is 1/10, so 0.1 + 0.2 is exactly
  0.3. Binary floating point, which most code uses, gives
  0.30000000000000004.
- **Division gives a fraction in lowest terms.** 1/3 stays 1/3 instead of
  becoming 0.3333333333333333, so (1/3)*3 is exactly 1.
- **Roots and powers are exact when the answer is rational.** sqrt(16/9) is
  4/3, 8^(2/3) is 4, root(-8, 3) is -2.
- **Logarithms are exact when the answer is a whole number.** log(8, 2) is 3,
  log10(1000) is 3.

Only a result that is not a rational number is approximate: the square root
of 2, pi, e, most logarithms, and powers with a fractional exponent. It is
computed with Python's decimal module to the significant digits asked for,
20 unless set otherwise (`--digits`, or `digits=`). `exact` is false and the
steps write `≈` instead of `=`.

## What comes back

| Field | Meaning |
|---|---|
| `reading` | The expression as ADA read it, with every operator written out: `2(3+4)` reads as `2*(3 + 4)` |
| `value` | The value. When `exact` is true: a whole number, a decimal that ends (1024.75), or else a fraction (1/3). When false: rounded to `digits` significant digits |
| `fraction` | The value as a fraction, when it is exact |
| `decimal` | The value as a decimal; `decimal_exact` says whether it is exact or rounded |
| `repeating` | A fraction's repeating decimal, such as `0.(142857)` for 1/7, when its repeating block has 60 digits or fewer |
| `steps` | Each operation with the values it combined, in the order computed; the first 100 |
| `assumptions` | Each convention the reading rests on |
| `answer` | One line: `2^10 + 3/4 = 1024.75 = 4099/4` |

## Conventions, stated

Where a convention settles how an expression reads, ADA follows it and lists
it under `assumptions`:

| Written | Read as | Value |
|---|---|---|
| `-2^2` | `-(2^2)`, the power first | -4 |
| `2^3^2` | `2^(3^2)`, from the right | 512 |
| `50%` | `50/100` | 0.5 |
| `20% of 50` | `20/100 * 50` | 10 |
| `2(3+4)` | `2*(3 + 4)` | 14 |
| `1,234,567` | `1234567` (outside a function's parentheses) | 1234567 |
| `$1,200 * 3` | `1200 * 3`, the currency sign dropped | 3600 |
| `0^0` | 1, as in counting and power series | 1 |
| `sqrt 16 + 9` | `sqrt(16) + 9` | 13 |

`×`, `÷`, `−`, `²`, `³`, `√` and `π` are read as `*`, `/`, `-`, `^2`, `^3`,
`sqrt` and `pi`. A trailing `=` or `?` is ignored.

## Refusals

When an expression can't be computed honestly, `compute` returns a `Refusal`
with a code from `ada.REFUSAL_CODES` and a message that says what to write
instead. `ada compute` exits with 3.

| Code | When | Example |
|---|---|---|
| `ambiguous_expression` | The expression reads two ways with two values; the message gives both | `6/2(1+2)` is 9 or 1; `17 % 5` is a remainder or a percent; `100 + 10%` is 100.1 or 110; `log(8)` is base 10 or base e |
| `invalid_expression` | A name ADA does not know, a missing operator, a wrong argument | `x + 1`, `2 3`, `factorial(2.5)` |
| `division_by_zero` | A division, remainder or negative power of zero | `1/0`, `5 mod 0`, `0^-1` |
| `not_a_real_number` | The value is not a real number | `sqrt(-4)`, `(-8)^(1/3)`, `ln(0)` |
| `too_large` | An exact number would pass 10,000 digits, or the expression is longer than 2,000 characters or nests more than 100 levels deep | `2^100000`, `10000!` |
| `needs_more_digits` | An approximate value is too close to a whole number (or a halfway point) to round, floor or ceil at the digits asked for | `floor(exp(pi*sqrt(163)))` at 20 digits; at 40 it is 262537412640768743 |

A missing multiplication sign is read only when every reading gives the same
value: `2(3+4)` is 14 either way, but `6/2(1+2)` is (6/2)*(1+2) = 9 or
6/(2*(1+2)) = 1, so it is refused with both.

A percent sign between two numbers is refused. Programming languages read
`17 % 5` as the remainder, 2, and a person may mean 17% of 5, 0.85. Write
`17 mod 5` or `17% of 5`. Adding or taking away a percent is refused too: to
a calculator `100 + 10%` is 100.1, to most people it is 110. Write
`100 * 1.1` for the second.

`round`, `floor` and `ceil` of an approximate value give an exact result
only when the approximation settles it. A value within the last few digits
of a whole number, such as e^(pi·sqrt(163)) = 262537412640768743.99999999999925…,
is refused at 20 digits and answered at 40.

## Operators and functions

| Written | Meaning |
|---|---|
| `+ - * /` | Add, subtract, multiply, divide |
| `^` or `**` | Power |
| `//` | Whole-number division, rounding down: `-17 // 5` is -4 |
| `mod`, `mod(a, n)` | Remainder from 0 to n - 1: `-7 mod 3` is 2; n must be positive |
| `n!`, `factorial(n)` | 1 × 2 × … × n |
| `x%` | x/100 |
| `sqrt(x)`, `cbrt(x)`, `root(x, n)` | Square, cube and n-th roots; an odd root of a negative number is negative |
| `abs`, `floor`, `ceil` | Absolute value, round down, round up |
| `round(x)`, `round(x, places)` | Halves round away from zero, on the exact value: `round(2.675, 2)` is 2.68 |
| `gcd(a, b, ...)`, `lcm(a, b, ...)` | Greatest common divisor, least common multiple, of whole numbers |
| `comb(n, k)` (also `nCr`, `choose`) | Ways to choose k of n, order not counted |
| `perm(n, k)` (also `nPr`) | Ways to arrange k of n, order counted |
| `min(...)`, `max(...)` | Smallest and largest |
| `exp(x)`, `ln(x)`, `log10(x)`, `log2(x)`, `log(x, base)` | e^x and logarithms |
| `pi`, `e` | The constants |

## Checking written arithmetic

`ada.check_arithmetic(text)` finds each equation written in a text, such as a
worked answer a language model gave, and checks it:

```python
ada.check_arithmetic("Points: 204 + 160 + 330 = 700. Water: (2/3) * 10 = 6.67 litres.")
# [Checked(expression='204 + 160 + 330', stated='700', computed='694', correct=False),
#  Checked(expression='(2/3) * 10', stated='6.67', computed='20/3', correct=True)]
```

A stated value is correct when it is the exact value, or the exact value
rounded or cut off to as many decimal places as it has: for 22/3, 7.33, 7.3
and 7 are all correct. A stated percent may mean either reading, so
`1/4 = 25%` and `0.25 * 100 = 25%` both pass.

Only an operation whose whole left side is written out is checked, so a
flag means a slip, not a misreading:

| Written | Checked |
|---|---|
| `5,000 * (1 + 0.025) = 5,000 * 1.025 = 5,125` | The last step, `5,000 * 1.025 = 5,125`. A running total such as `3 * 4 = 12 + 1 = 13` is written the same way, so earlier steps are not compared |
| `S + B + 2 + 2 = 28`, `16 friends * 3/4 = 12` | Nothing: the left side goes on to the left |
| `15 - 5 = 2x`, `7/13 = 120/x` | Nothing: the right side goes on |
| `$450 + $300 + $750 = $1,500/month` | `450 + 300 + 750 = 1,500` |
| `- 40 + 20 = 60` | `40 + 20 = 60`: a list marker is not a minus sign |
| `**3 * 4 = 12**`, `(that is 3 * 4 = 12)` | `3 * 4 = 12` |
| `4933828 / 13 = 373, 846.92` | Nothing: the comma is misplaced |
| `only possible if 240 + 3,516,571 = 0` | Nothing: after if, unless or whether it is a condition, not a claim |
| `Suppose 2 + 2 = 5`, `0 + 0 = 2 ⇒ no solution` | Nothing: after suppose or assume, or with contradiction, impossible, no solution, not possible, not true or is false later on its line, the writer knows it is false |
| `2 * 5,000,000 = 10 million` | Nothing: the value is scaled by a word |

An equation is read within one line. `x = 5` is skipped, and so is anything
`compute` refuses. `×`, `x` between numbers, `·`, `\times`, `\cdot`, `\div`,
`\%`, currency signs and thousands separators are read as written.

This lets a model reason in its own words and still get its arithmetic
checked: hand it back each equation that comes back `correct=False`, with the
right value, and ask it to fix the steps that used it.

## Safety

Nothing is run as Python code. A small grammar reads numbers, the operators
above, parentheses and the functions in `FUNCTIONS`, and refuses anything
else. Sizes are checked before a number is computed: a power, factorial or
count whose exact value would pass 10,000 digits is refused before any work
is done. A long chain such as a sum of 600 numbers is computed in a loop
rather than by recursion.
