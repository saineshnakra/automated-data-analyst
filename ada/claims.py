"""Check a number someone states about the data: verified, or unverified with the reason.

    import ada
    check = ada.check_claim("flights.csv", "The average departure delay was 12.64 minutes.")
    check.status        # "verified" or "unverified"
    check.reason        # None, or why it is unverified
    check.to_dict()     # the claim, how it was read, the result, the calculation and the rows behind it

A claim is a sentence with one number in it. ADA takes the number out, reads
the rest as a question (see ``Analysis.ask``), computes the answer and
compares. The number stands for the interval its writing implies: "12.6" is
anything from 12.55 to 12.65, "about 13" is 12 to 14, "about 100" is 90 to
110, "more than 30%" is above 30. A number with a different scale or unit is
not taken to agree, except that a unit named in the column ("body_mass_g") is
converted to the claim's ("5.08 kg").

``status`` is "verified" only when the computed answer falls in that interval
and no check on the analysis failed. Otherwise it is "unverified", and
``reason`` says why:

    missing_context        the claim leaves out something the check needs: a number, a
                           column, a group, or the unit that decides the scale
    ambiguous_intent       the words fit more than one calculation and their answers differ
    inaccessible_data      the data could not be read
    unsupported_analysis   ADA has no way to compute what the claim describes
    conflicting_evidence   the claim's unit measures something other than the unit in the
                           column's name: minutes for a column in mm
    failed_check           the computed answer is outside the claimed interval, or a check
                           on the analysis failed

Unverified does not mean false: only ``failed_check`` with ``agreement ==
"mismatch"`` says the data give a different number. A verified claim is
verified as read: ``reading``, ``assumptions`` and ``warnings`` say what was
checked, and nothing beyond it.
"""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from .decisions import Decision, checked_choices
from .file_io import SUPPORTED_SUFFIXES as READABLE
from .formatting import counted, list_names, normalized_name
from .nlq import AGE_PHRASE, DAY_PATTERN, OUTCOME_WORDS, Condition, QueryAnswer, ValueFilter
from .sources import Source

#: Version of the ``ClaimCheck.to_dict`` layout. Bumped when a key changes meaning.
CLAIM_SCHEMA_VERSION = 1

REASONS = (
    "missing_context",
    "ambiguous_intent",
    "inaccessible_data",
    "unsupported_analysis",
    "conflicting_evidence",
    "failed_check",
)

_CURRENCY = r"(?:R\$|US\$|A\$|C\$|\$|€|£|¥|₹)"
_DIGITS = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|\.\d+"
NUMBER_PATTERN = re.compile(
    # Digits right after a letter are part of a word or a code ("CO2", "bondsba01"), not a number.
    rf"(?<![A-Za-z0-9_])(?P<currency>{_CURRENCY})?\s?(?P<sign>[-−])?(?P<number>{_DIGITS})"
    r"(?:\s?(?P<scale>thousand|million|billion|trillion|bn|mn)\b|(?P<suffix>[kKM])\b)?"
    r"(?:\s?(?P<percent>%|percent\b|per\s?cent\b))?",
    re.IGNORECASE,
)
SCALES = {
    "thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9, "trillion": 1e12,
}
# Words before a number that bound or widen it, longest first so "just over" wins over "over".
QUALIFIERS = (
    ("no less than", "at least"), ("not less than", "at least"), ("at least", "at least"),
    ("no more than", "at most"), ("not more than", "at most"), ("at most", "at most"), ("up to", "at most"),
    ("in excess of", "more than"), ("more than", "more than"), ("greater than", "more than"),
    ("higher than", "more than"), ("exceeding", "more than"), ("above", "more than"),
    ("topped", "more than"), ("exceeded", "more than"), ("surpassed", "more than"),
    ("just over", "just over"), ("just under", "just under"), ("over", "more than"),
    ("fewer than", "less than"), ("less than", "less than"), ("lower than", "less than"),
    ("below", "less than"), ("under", "less than"),
    ("close to", "about"), ("approximately", "about"), ("approx.", "about"), ("approx", "about"),
    ("around", "about"), ("roughly", "about"), ("about", "about"), ("some", "about"), ("circa", "about"),
    ("~", "about"), ("≈", "about"),
    ("nearly", "nearly"), ("almost", "nearly"), ("exactly", "exactly"), ("precisely", "exactly"),
    # "Just 48 women", "only 16 episodes": the word stresses the number and leaves its interval as written.
    ("just", None), ("only", None), ("barely", None),
)
# Words right before the number that link it to what it describes: "is 12.6", "averaged 12.6".
PREDICATE_CUES = frozenset({
    "is", "was", "were", "are", "be", "been", "equals", "equalled", "equaled", "averaged", "averages",
    "averaging", "totalled", "totaled", "totals", "totalling", "totaling", "reached", "hit", "by",
    "at", "of", "stood", "came", "amounted", "numbered", "weighed", "weighs", "weigh", "measured",
    "rose", "fell", "grew", "dropped", "increased", "decreased", "paid", "made", "had", "has", "have",
    "scored", "scores", "earned", "earns", "collected", "recorded", "reaches",
})
# Adjectives after a number that name the quantity: "6,657 metres high" is a height.
ADJECTIVE_NOUNS = {
    "high": "height", "tall": "height", "long": "length", "wide": "width", "deep": "depth", "heavy": "weight",
    "old": "age", "thick": "thickness",
}
# Verbs before a number that name the quantity: "customers paid R$16M" is a payment.
VERB_NOUNS = {
    "paid": "payment", "pays": "payment", "spent": "spending", "spends": "spending", "earned": "earnings",
    "earns": "earnings", "sold": "sales", "charged": "charge", "cost": "cost", "costs": "cost",
    "lost": "losses", "loses": "losses",
}
# Nouns for the people a table is about: a claim's subject, not a column ("Customers paid ...").
PEOPLE_NOUNS = frozenset({
    "customers", "clients", "users", "players", "passengers", "students", "members", "employees",
    "visitors", "buyers", "shoppers", "people", "patients", "participants", "respondents", "athletes",
    "drivers", "riders", "workers", "voters", "readers", "viewers", "guests", "travellers", "travelers",
})
# Verbs that add things up: in "hit 73 home runs" and "scored 3,999.5 points" the words after the
# number name the column that is totalled, not a unit.
TALLY_VERBS = frozenset({"hit", "hits", "scored", "scores", "earned", "earns", "collected", "recorded"})
# Cue words that say how rows combine, put back into the question as the aggregation.
AGGREGATION_CUES = {
    "averaged": "average", "averages": "average", "averaging": "average",
    "totalled": "total", "totaled": "total", "totals": "total", "totalling": "total", "totaling": "total",
}
# Words that end a unit after the number ("12.6 minutes in 2013"): grammar, not part of the unit.
UNIT_STOPS = frozenset({
    "in", "on", "for", "of", "per", "from", "by", "at", "during", "than", "and", "or", "with", "among",
    "across", "when", "where", "while", "since", "between", "after", "before", "to", "a", "an", "the",
    "each", "every", "over", "under", "as", "if", "is", "was", "were", "are", "who", "that", "which", "whose",
})
YEAR_RANGE = (1500, 2100)
# Words before a number that name a time point, so the number picks rows: "on day 21".
POINT_WORDS = frozenset({"day", "week", "visit", "cycle", "session", "wave", "round"})
# Words before a number that make it a filter when a column's name follows: "cars with 4 cylinders".
FILTER_WORDS = frozenset({"with", "having"})
# "Reaction time rises by about 10.5 ms per day": a change for each unit of another column is a slope.
SLOPE_PATTERN = re.compile(
    r"^\s*(?P<what>.+?)\s+(?P<verb>rises|rose|risen|increases|increased|grows|grew|climbs|climbed|"
    r"goes\s+up|went\s+up|falls|fell|fallen|drops|dropped|decreases|decreased|declines|declined|"
    r"goes\s+down|went\s+down)\s+(?:by\s+)?(?!to\b)(?P<number>.+?)\s+(?:per|for\s+each|for\s+every|with\s+each|"
    r"with\s+every)\s+(?:additional\s+|extra\s+|more\s+|further\s+)?(?P<per>.+?)\s*[.!]?\s*$",
    re.IGNORECASE,
)
FALLING_VERBS = ("fall", "fell", "drop", "decreas", "declin", "goes down", "went down")
# "has grown by about 500 million pounds a year": a change for each year is the slope of the yearly trend.
TIME_SLOPE = re.compile(
    r"(?P<verb>\b(?:(?:has|have|had)\s+)?(?:(?:been\s+)?(?:rising|growing|increasing|climbing|falling|"
    r"dropping|declining|decreasing)|rises|rose|risen|grows|grew|grown|increases|increased|climbs|climbed|"
    r"falls|fell|fallen|drops|dropped|declines|declined|decreases|decreased|shrank|shrunk|adds|added|gains|"
    r"gained|loses|lost|sheds|shed))\s+(?:by\s+)?"
    r"(?:an\s+average\s+of\s+)?(?P<number>(?:[^.;]|(?<=\d)\.(?=\d))*?\d(?:[^.;]|(?<=\d)\.(?=\d))*?)\s+"
    r"(?:a|per|each|every)\s+"
    r"(?P<unit>year|month|week|day|decade)\b(?!\s+of\b)",
    re.IGNORECASE,
)
# Verbs of a time slope that say it falls besides FALLING_VERBS: "has lost about 2,000 a year".
LOSING_VERBS = ("lose", "loses", "lost", "shed", "sheds")
# Each unit of a time slope as a share of a year.
TIME_SLOPE_UNITS = {"year": 1.0, "decade": 0.1, "month": 12.0, "week": 52.1775, "day": 365.25}
# Units of a slope's column besides those in UNITS: "per year of age", "per unit of dose".
PER_UNIT_WORDS = frozenset({"year", "month", "week", "unit", "point", "level"})
# Fractions in words: "about a quarter of", "more than half of", "two thirds of".
FRACTION_PATTERN = re.compile(
    r"\b(?:(?P<qualifier>about|roughly|around|approximately|nearly|almost|just\s+over|just\s+under|over|"
    r"under|more\s+than|less\s+than|fewer\s+than|at\s+least|at\s+most)\s+)?"
    r"(?P<count>a|an|one|two|three|four|nine)?\s*(?P<part>half|halves|thirds?|quarters?|fifths?|tenths?)"
    r"(?=\s+of\b|\s+(?:had|have|has|were|was|are|is|went|did|said|got|came|made|lived|died|voted|used|took|"
    r"chose|saw|reported|finished|passed|failed|[a-z]{3,}ed)\b)",
    re.IGNORECASE,
)
# A change by a fraction: "rose by about half", "fell by a third".
BY_FRACTION = re.compile(
    r"(?<=\bby\s)(?:(?P<qualifier>about|roughly|around|approximately|nearly|almost|more\s+than|less\s+than)"
    r"\s+)?(?:(?P<count>a|one)\s+)?(?P<part>half|third|quarter|fifth|tenth)\b(?!\s+of\b)",
    re.IGNORECASE,
)
# A fraction of one unit: "three quarters of a percentage point", "half a point", "a quarter of a mile".
FRACTION_OF_UNIT = re.compile(
    r"\b(?:(?P<qualifier>about|roughly|around|approximately|nearly|almost|more\s+than|less\s+than)\s+)?"
    r"(?:(?P<count>a|an|one|two|three|four|nine)\s+)?(?P<part>half|thirds?|quarters?|fifths?|tenths?)\s+"
    r"(?:of\s+)?an?\s+(?P<unit>percentage\s+point|point|degree|mile|kilometre|kilometer|hour|day|year)\b",
    re.IGNORECASE,
)
# A change by a multiple: "doubled", "more than sixfold", "halved", "three-fold".
MULTIPLE = re.compile(
    r"(?P<verb>\b(?:rose|grew|increased|jumped|climbed|fell|dropped|declined|decreased)\s+)?"
    r"(?P<qualifier>\b(?:more\s+than|nearly|almost|about|roughly|over|less\s+than)\s+)?"
    r"\b(?P<word>doubled|tripled|trebled|quadrupled|halved|(?P<n>two|three|four|five|six|seven|eight|nine|ten|"
    r"\d{1,3})[- ]?fold|"
    # "two hundred times as many": the number of times, in digits or words.
    r"(?P<times>\d[\d,]*(?:\.\d+)?|a\s+hundred|(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[\s-]+hundred)?)\s+times\s+as\s+"
    r"(?:many|much|large|high|big|great))\b",
    re.IGNORECASE,
)
MULTIPLE_WORDS = {"doubled": 2, "tripled": 3, "trebled": 3, "quadrupled": 4, "halved": 0.5}
# A reporting year or a period written as two years: "2021-22", "2019/20", "1985-1999".
YEAR_PAIR = re.compile(r"\b(?:1[5-9]|20)\d{2}\s*[-–/]\s*(?:\d{2}|(?:1[5-9]|20)\d{2})\b")
# Numbers that name rather than count: a rank ("No. 1"), a chart or index ("the Hot 100"), a day with no
# year ("September 11") and a time of day ("10 p.m.", "15:30").
NAMING_NUMBER = re.compile(
    r"\bNo\.\s*\d+\b|\b[Tt]he\s+(?:[A-Z][\w&]*\s+)+\d{2,4}\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"\d{1,2}(?:st|nd|rd|th)?\b(?!,?\s*\d)|"
    r"\b\d{1,2}(?:[.:]\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)(?!\w)|\b\d{1,2}:\d{2}\b"
)
# Words before a year that make it the time of the claim: "rose in 2020", "since 2009".
YEAR_PREPOSITION = re.compile(r"\b(?:in|since|until|till|before|after|during)\s*$", re.IGNORECASE)
# Two years that bound a range: "from 2023 to 2024", "between 1982 and 1988".
YEAR_SPAN = re.compile(r"\b(?:from|between)\s+\d{4}\s*(?:to|and|until|through|-|–)\s*\d{4}\b", re.IGNORECASE)
FRACTION_PARTS = {"half": 2, "halves": 2, "third": 3, "thirds": 3, "quarter": 4, "quarters": 4, "fifth": 5,
                  "fifths": 5, "tenth": 10, "tenths": 10}
COUNT_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "nine": 9}
# Whole numbers in words, read only in a claim with no number in digits: "Seventeen cases".
NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
LEFT_OUT_NOTE = "Read the claim without"
# A lead naming the group a share is of, before the share: "Among victims aged 50 and over, 9.5% died".
AMONG_LEAD = re.compile(
    r"^\s*among\s+(?P<base>[^,]+?),\s*(?P<share>(?:(?:about|nearly|almost|roughly|around|only|just|over|"
    r"under|more\s+than|less\s+than)\s+)?\d[\d,]*(?:\.\d+)?\s*%)\s+(?P<rest>.+)$",
    re.IGNORECASE,
)
# A share stated after what it counts: "Men were 79% of the victims who died", "Women made up 21% of
# the riders". The share is of what follows "of".
SUBJECT_SHARE = re.compile(
    r"^\s*(?P<part>[A-Za-z'-]+(?:\s+[A-Za-z'-]+){0,3}?)\s+(?P<verb>were|was|are|is|made\s+up|make\s+up|"
    r"makes\s+up|accounted\s+for|account\s+for|accounts\s+for|represented|represent|represents|"
    r"comprised|comprise|comprises|formed|form|forms|won|received|got|took|earned|captured)\s+"
    r"(?P<share>(?:(?:about|nearly|almost|roughly|around|only|just|over|under|more\s+than|less\s+than)\s+)?"
    r"\d[\d,]*(?:\.\d+)?\s*%)\s+of\s+(?P<rest>.+?)[.!?]?\s*$",
    re.IGNORECASE,
)
# Words in a subject that make it a measure rather than a group: "The tax rate was 45% of ...".
MEASURE_SUBJECTS = frozenset({
    "rate", "average", "mean", "median", "share", "percentage", "proportion", "growth", "increase",
    "decrease", "change", "total", "sum", "income", "tax", "cost", "price", "value", "level", "number",
})
# A clause that says who reports the claim: "Owners estimated that", "The survey found that".
REPORTING_CLAUSE = re.compile(
    r"^\s*(?:[A-Za-z']+\s+){1,4}?(?:estimated|estimates|estimate|reported|reports|report|found|finds|find|"
    r"said|says|say|showed|shows|show|stated|states|noted|notes|concluded|concludes|wrote|writes|"
    r"claimed|claims)\s+that\s+",
    re.IGNORECASE,
)
# "55 of the cats": a count of things the table's rows may be.
OF_THE = re.compile(r"\s*of\s+(?:the|these|those|all(?:\s+the)?)\s+([A-Za-z]+)\b", re.IGNORECASE)
# A lead that names a claim's base: "Of the 4,526 applicants,", "Of the 5,144 Danes in the study,".
BASE_LEAD = re.compile(
    r"^\s*(?:of|out\s+of)\s+(?:the\s+|all\s+)?(?:[A-Za-z]+['’]s\s+)?(?P<count>\d{1,3}(?:,\d{3})+|\d+)\s+"
    r"(?P<noun>[A-Za-z]+(?:\s[\w']+){0,10}),\s*",
    re.IGNORECASE,
)
# A count of a base: "34 of the 93 cars had no airbags", read as "Of the 93 cars, 34 cars had ...".
PART_OF_BASE = re.compile(
    r"^(?P<lead>\s*(?:(?:about|nearly|almost|only|just|over|more\s+than|fewer\s+than|less\s+than)\s+)?)"
    r"(?P<part>\d{1,3}(?:,\d{3})+|\d+)\s+(?:of|out\s+of)\s+(?:the\s+|all\s+)?"
    r"(?P<count>\d{1,3}(?:,\d{3})+|\d+)\s+(?P<noun>[A-Za-z]+)\b",
    re.IGNORECASE,
)
# A base a share is of: "64% of the 53,182 suicides", "about a third of the 86 departments".
SHARE_BASE = re.compile(
    r"(?:%|\b(?:half|third|thirds|quarter|quarters|fifth|fifths|tenth|tenths)"
    r"|\bin\s+(?:every\s+)?(?:two|three|four|five|six|seven|eight|nine|ten|\d{1,3}))\s+of\s+(?:the\s+|all\s+)?"
    r"(?P<count>\d{1,3}(?:,\d{3})+|\d+)\s+(?P<noun>[A-Za-z]+)\b",
    re.IGNORECASE,
)
# A base the claim spans: "across the 48 states", "in all 50 states", "among the 20 firms".
SCOPE_BASE = re.compile(
    r"\b(?:across|in|among|for|over)\s+(?:all\s+)?(?:the\s+|all\s+)?"
    r"(?P<count>\d{1,3}(?:,\d{3})+|\d+)\s+"
    # "for 511 hours" is how long, not a base.
    r"(?!(?:seconds|minutes|hours|days|weeks|months|years|decades)\b)(?P<noun>[A-Za-z]+s)\b",
    re.IGNORECASE,
)
# A base opening the claim: "The 93 models came from 28 manufacturers".
OPENING_BASE = re.compile(
    r"^\s*(?:the|all|these|those)\s+(?P<count>\d{1,3}(?:,\d{3})+|\d+)\s+(?P<noun>[A-Za-z]+)\b",
    re.IGNORECASE,
)
# "by 2024": a year a share was reached by, which may be the year itself or every year up to it.
BY_YEAR = re.compile(r"\bby\s+(?:the\s+end\s+of\s+)?(?P<year>(?:19|20)\d{2})\b", re.IGNORECASE)
# Name words of a column that says what a long table's value column holds on each row, and the names of
# that value column.
INDICATOR_WORDS = frozenset({"indicator", "measure", "statistic", "series", "variable", "metric"})
VALUE_NAMES = frozenset({"value", "values", "obs value", "observation value", "data value", "estimate"})
# A description set off by commas after a name: ", a veterinary sedative,".
APPOSITIVE = re.compile(r",\s+(?:a|an)\s+[A-Za-z][A-Za-z' -]{0,40}(?=,)(?<!\d)", re.IGNORECASE)
# "seven in ten", "one in five", "1 in 20", "about 12 of every 100": a share written as a count in a total.
# "Of every 100 birds taken in, about 25 were put down": a share said as so many of every hundred.
EVERY_BASE = re.compile(
    r"\s*(?:of|out\s+of)\s+every\s+(?P<bottom>[1-9]\d{0,2}(?:,\d{3})*)\s+(?P<base>[^,]+),\s*"
    r"(?:(?P<qualifier>about|roughly|around|nearly|almost|just\s+over|just\s+under|more\s+than|less\s+than|"
    r"fewer\s+than|at\s+least|at\s+most)\s+)?(?P<top>\d+(?:\.\d+)?)\s+(?P<rest>.+)",
    re.IGNORECASE,
)
IN_FRACTION = re.compile(
    r"\b(?:(?P<qualifier>about|roughly|around|approximately|nearly|almost|just\s+over|just\s+under|over|"
    r"under|more\s+than|less\s+than|fewer\s+than|at\s+least|at\s+most)\s+)?"
    r"(?P<top>one|two|three|four|five|six|seven|eight|nine|(?<![\d.,])\d{1,2})\s+"
    r"(?:in(?:\s+every)?|(?:out\s+)?of\s+every)\s+"
    r"(?P<bottom>two|three|four|five|six|seven|eight|nine|ten|\d{1,3})\b(?![.,]\d)",
    re.IGNORECASE,
)
# Units of time a number measures in, never things it counts: "12.64 minutes" with a minute column.
TIME_PLURALS = frozenset({"seconds", "minutes", "hours", "days", "weeks", "months", "years", "decades"})
# Units of money, which a number measures rather than counts: "averaged 18.6 dollars".
MONEY_WORDS = frozenset({"dollars", "euros", "pounds", "cents", "pence", "yen", "rupees", "francs", "reais"})
# Plurals without an s, which count things as "344 penguins" does: "1,835 women applied".
IRREGULAR_PLURALS = frozenset({"men", "women", "people", "children", "persons", "mice", "geese", "teeth"})
# Words a claim is never read without, as leaving them out changes what it says.
KEPT_WORDS = frozenset({
    "not", "no", "never", "without", "except", "excluding", "besides", "other", "non", "only", "neither",
    "nor", "less", "more", "fewer", "than", "under", "over", "above", "below", "before", "after", "per",
    "each", "every", "first", "last", "top", "bottom", "most", "least", "highest", "lowest",
})
# Verbs that stand for a column when a claim names none: "weigh 4000 g" is a mass. "measure" takes
# its kind from the claim's unit.
UNIT_VERBS = {
    "weigh": "mass", "weighs": "mass", "weighed": "mass", "weighing": "mass",
    "lasted": "time", "lasts": "time", "measure": None, "measures": None, "measured": None,
}
# Units a claim or a column name can state plainly, as (dimension, size in the dimension's base unit).
# Short forms that also mean something else ("m" for million, "t", "tons") are left out on purpose.
UNITS = {
    "mg": ("mass", 1e-3), "milligram": ("mass", 1e-3), "milligrams": ("mass", 1e-3),
    "g": ("mass", 1.0), "gram": ("mass", 1.0), "grams": ("mass", 1.0),
    "kg": ("mass", 1e3), "kilogram": ("mass", 1e3), "kilograms": ("mass", 1e3),
    "mm": ("length", 1e-3), "millimetre": ("length", 1e-3), "millimetres": ("length", 1e-3),
    "millimeter": ("length", 1e-3), "millimeters": ("length", 1e-3),
    "cm": ("length", 1e-2), "centimetre": ("length", 1e-2), "centimetres": ("length", 1e-2),
    "centimeter": ("length", 1e-2), "centimeters": ("length", 1e-2),
    "metre": ("length", 1.0), "metres": ("length", 1.0), "meter": ("length", 1.0), "meters": ("length", 1.0),
    "km": ("length", 1e3), "kilometre": ("length", 1e3), "kilometres": ("length", 1e3),
    "kilometer": ("length", 1e3), "kilometers": ("length", 1e3),
    "sec": ("time", 1.0), "secs": ("time", 1.0), "second": ("time", 1.0), "seconds": ("time", 1.0),
    "min": ("time", 60.0), "mins": ("time", 60.0), "minute": ("time", 60.0), "minutes": ("time", 60.0),
    "hr": ("time", 3600.0), "hrs": ("time", 3600.0), "hour": ("time", 3600.0), "hours": ("time", 3600.0),
    "day": ("time", 86400.0), "days": ("time", 86400.0),
    "tonne": ("mass", 1e6), "tonnes": ("mass", 1e6), "lb": ("mass", 453.59237), "lbs": ("mass", 453.59237),
    "ounce": ("mass", 28.349523125), "ounces": ("mass", 28.349523125), "oz": ("mass", 28.349523125),
    "ft": ("length", 0.3048), "foot": ("length", 0.3048), "feet": ("length", 0.3048),
    "inch": ("length", 0.0254), "inches": ("length", 0.0254), "yard": ("length", 0.9144),
    "yards": ("length", 0.9144), "mile": ("length", 1609.344), "miles": ("length", 1609.344),
}
# Name words for a kind of quantity, for a column whose name gives no unit: "height" holds a length.
QUANTITY_NAMES = {
    "length": frozenset({"height", "length", "width", "depth", "distance", "elevation", "altitude",
                         "diameter", "girth", "circumference", "thickness"}),
    "mass": frozenset({"weight", "mass", "wgt", "wt"}),
    "time": frozenset({"duration"}),
}


@dataclass(frozen=True)
class ClaimedNumber:
    """The number a claim states, and the interval of values its writing stands for.

    ``value`` is in the claim's terms: 32 for "32%", 16008872 for "R$16,008,872",
    3.75e10 for "37.5 billion". ``low`` and ``high`` bound the interval;
    ``low_open`` and ``high_open`` say whether a bound itself is excluded.
    """

    text: str
    value: float
    low: float
    high: float
    low_open: bool = False
    high_open: bool = False
    qualifier: str | None = None
    percent: bool = False
    scale: float = 1.0
    scale_word: str | None = None
    unit: str | None = None
    decimals: int = 0

    def holds(self, number: float | None) -> bool:
        """Whether ``number`` lies in the claimed interval."""
        if number is None or not math.isfinite(number):
            return False
        above = number > self.low if self.low_open else number >= self.low - _slack(self.low)
        below = number < self.high if self.high_open else number <= self.high + _slack(self.high)
        return above and below

    def interval(self) -> str:
        """The interval in words: "12.635 to 12.645", "more than 30"."""
        if math.isinf(self.low) and math.isinf(self.high):
            return "any value"
        if math.isinf(self.high):
            return f"{'more than' if self.low_open else 'at least'} {_plain(self.low)}"
        if math.isinf(self.low):
            return f"{'less than' if self.high_open else 'at most'} {_plain(self.high)}"
        if self.low == self.high:
            return f"exactly {_plain(self.low)}"
        return f"{_plain(self.low)} to {_plain(self.high)}"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "value": self.value,
            "interval": [_json_number(self.low), _json_number(self.high)],
            "interval_text": self.interval(),
            "qualifier": self.qualifier,
            "percent": self.percent,
            "scale": self.scale_word,
            "unit": self.unit,
        }


def _slack(bound: float) -> float:
    """Room for floating-point error at an interval's edge, far below any written precision."""
    return abs(bound) * 1e-12 + 1e-12


def _json_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _plain(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.10g}" if abs(value) >= 1 else f"{value:.10g}"


@dataclass(frozen=True)
class Check:
    """One check on a claim: its name, whether it passed and what it found."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class Alternative:
    """Another reading of the claim's words, with the number it gives."""

    reading: str
    question: str
    value: float | None
    agrees: bool
    calculation: str

    def to_dict(self) -> dict:
        return {
            "reading": self.reading, "question": self.question, "value": self.value,
            "agrees": self.agrees, "calculation": self.calculation,
        }


@dataclass(frozen=True)
class ClaimCheck:
    """The result of checking one claim. See the module docstring for ``status`` and ``reason``."""

    claim: str
    status: Literal["verified", "unverified"]
    reason: str | None
    message: str
    claimed: ClaimedNumber | None = None
    question: str | None = None
    answer: QueryAnswer | None = None
    #: The computed number that was compared with the claim, in the claim's terms.
    compared: float | None = None
    agreement: Literal["match", "mismatch", "not_compared"] = "not_compared"
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checks: tuple[Check, ...] = ()
    alternatives: tuple[Alternative, ...] = ()
    #: The smallest question that would let the check go ahead, for the person who made the claim.
    clarification: str | None = None
    source: Source | None = None
    #: Words of the claim that name nothing in the table, which the check read it without.
    words_left_out: tuple[str, ...] = ()

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    def to_dict(self) -> dict:
        """The check as JSON-safe data, in the layout ``CLAIM_SCHEMA_VERSION`` names."""
        answer = self.answer.to_dict() if self.answer is not None else None
        named = None if self.source is None else self.source.to_dict()
        evidence = answer["evidence"] if answer else {"source": named}
        return {
            "schema_version": CLAIM_SCHEMA_VERSION,
            "claim": self.claim,
            "status": self.status,
            "reason": self.reason,
            "message": self.message,
            "claimed": None if self.claimed is None else self.claimed.to_dict(),
            "question": self.question,
            "agreement": self.agreement,
            "result": {
                "value": self.compared,
                **({key: answer["result"][key] for key in ("values", "figures")} if answer else {}),
            },
            "reading": answer["reading"] if answer else None,
            "calculation": answer["calculation"] if answer else None,
            "evidence": evidence,
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
            "checks": [check.to_dict() for check in self.checks],
            "alternatives": [alternative.to_dict() for alternative in self.alternatives],
            "clarification": self.clarification,
            "decisions": answer["decisions"] if answer else [],
            "words_left_out": list(self.words_left_out),
        }

    def __str__(self) -> str:
        lines = [f"{self.status}: {self.claim}", f"  {self.message}"]
        if self.answer is not None:
            lines.append(f"  calc: {self.answer.calculation}")
        for label, items in (("assumes", self.assumptions), ("warning", self.warnings)):
            lines.extend(f"  {label}: {item}" for item in items)
        if self.answer is not None:
            lines.extend(f"  decides: {decision.describe()}" for decision in self.answer.decisions)
        if self.clarification:
            lines.append(f"  to check it: {self.clarification}")
        return "\n".join(lines)


def parse_number(match: re.Match) -> tuple[float, int, str | None, float, bool]:
    """The value, its decimals, its scale word, the scale and whether it is a percent, for one match."""
    digits = match.group("number").replace(",", "")
    value = float(digits)
    if match.group("sign"):
        value = -value
    mantissa = digits.lower().split("e")[0]
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    if "e" in digits.lower():
        decimals -= int(digits.lower().split("e")[1])
    word = (match.group("scale") or match.group("suffix") or "").lower() or None
    scale = SCALES.get(word, 1.0) if word else 1.0
    return value, decimals, word, scale, bool(match.group("percent"))


def _significant_unit(value: float, decimals: int) -> float:
    """The place of the last non-zero digit: 1 for 13, 1000 for 37,000, 0.1 for 10.5."""
    if decimals > 0 or value == 0:
        return 10.0 ** -decimals
    magnitude = abs(value)
    unit = 1.0
    while magnitude >= unit * 10 and (magnitude / (unit * 10)) == int(magnitude / (unit * 10)):
        unit *= 10
    return unit


def _approximate_width(value: float, decimals: int) -> float:
    """How far "about" reaches either side: one significant unit, at most a tenth of the number,
    and never less than the number's own rounding. "about 13" is 12 to 14, "about 100" is 90 to 110.
    """
    rounding = 10.0 ** -decimals / 2
    return max(rounding, min(_significant_unit(value, decimals), abs(value) / 10))


def claimed_number(text: str, match: re.Match, qualifier: str | None, unit: str | None) -> ClaimedNumber:
    """The interval a number stands for, from its decimals and the qualifier before it.

    The bounds are worked out in decimal arithmetic, so "12.64" stands for 12.635
    to 12.645 exactly rather than to 12.645000000000001.
    """
    value, decimals, word, scale, percent = parse_number(match)
    exact = Decimal(match.group("number").replace(",", "")) * (-1 if match.group("sign") else 1)
    half = Decimal(1).scaleb(-decimals) / 2
    widened = _approximate_width(value, decimals)
    low, high, low_open, high_open = float(exact - half), float(exact + half), False, False
    if qualifier == "about":
        low, high = value - widened, value + widened
    elif qualifier == "nearly":
        low, high = value - widened, value
    elif qualifier == "just over":
        low, high, low_open = value, value + widened, True
    elif qualifier == "just under":
        low, high, high_open = value - widened, value, True
    elif qualifier == "more than":
        low, high, low_open = value, math.inf, True
    elif qualifier == "at least":
        low, high = value, math.inf
    elif qualifier == "less than":
        low, high, high_open = -math.inf, value, True
    elif qualifier == "at most":
        low, high = -math.inf, value
    elif qualifier == "exactly":
        low = high = value
    return ClaimedNumber(
        text=text.strip(), value=value * scale, low=low * scale, high=high * scale,
        low_open=low_open, high_open=high_open, qualifier=qualifier, percent=percent,
        scale=scale, scale_word=word, unit=unit, decimals=decimals,
    )


@dataclass(frozen=True)
class _Candidate:
    match: re.Match
    qualifier: str | None
    qualifier_start: int
    cue: str | None
    cue_start: int
    year_like: bool
    filter_like: bool
    ordinal: bool


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _column_words(columns: list[str]) -> set[str]:
    words: set[str] = set()
    for column in columns:
        words.update(word for word in re.findall(r"[a-z]+", str(column).lower()) if len(word) >= 2)
    return words


def _candidates(claim: str, columns: list[str]) -> list[_Candidate]:
    """Every number in the claim, with the qualifier and cue before it, and whether it is part of the
    question instead: a year, a value after a column's name, or an ordinal ("25th percentile").
    """
    found = []
    column_words = _column_words(columns)
    days = [day.span() for day in DAY_PATTERN.finditer(claim)] + [
        age.span() for age in AGE_PHRASE.finditer(claim)] + [
        span.span() for span in YEAR_SPAN.finditer(claim)] + [
        span.span() for span in YEAR_PAIR.finditer(claim)] + [
        span.span() for span in NAMING_NUMBER.finditer(claim)]
    for match in NUMBER_PATTERN.finditer(claim):
        start = match.start()
        before = claim[:start].rstrip()
        lowered = before.lower()
        qualifier, qualifier_start = None, start
        for phrase, kind in QUALIFIERS:
            if not lowered.endswith(phrase):
                continue
            alone = len(lowered) == len(phrase) or not lowered[-len(phrase) - 1].isalnum()
            if alone or not phrase[0].isalnum():
                qualifier, qualifier_start = kind, len(before) - len(phrase)
                before = before[: -len(phrase)].rstrip()
                lowered = before.lower()
                break
        previous = re.search(r"([A-Za-z]+)\W*$", before)
        # "Of the strikes where it was recorded, almost a third": a comma ends the clause the verb is in.
        cued = previous is not None and not re.search(r"[,;:]", before[previous.end(1):])
        cue = previous.group(1).lower() if cued and (
            previous.group(1).lower() in PREDICATE_CUES or previous.group(1).lower() in VERB_NOUNS
        ) else None
        cue_start = previous.start(1) if cue and previous else qualifier_start
        marked = any(match.group(name) for name in ("currency", "scale", "suffix", "percent"))
        plain = not marked and "," not in match.group("number") and "." not in match.group("number")
        number = float(match.group("number").replace(",", ""))
        year_like = plain and YEAR_RANGE[0] <= number <= YEAR_RANGE[1] and number == int(number)
        filter_like = bool(previous) and cue is None and (
            previous.group(1).lower() in column_words or previous.group(1).lower() in POINT_WORDS
        )
        # "At 14, girls averaged 25.1 mm": a number opening the claim after "At" picks rows.
        filter_like = filter_like or bool(
            re.fullmatch(r"\s*at\s*", before, re.IGNORECASE) and re.match(r"\s*,", claim[match.end():])
        )
        # "20 claims per 100 holders": the number after "per" says what the rate is given per.
        filter_like = filter_like or bool(
            re.search(r"\b(?:per|for\s+every|for\s+each)\s*$", before, re.IGNORECASE)
        )
        # "cars with 4 cylinders": a number between "with" and a column's name picks rows.
        following = re.match(r"\s*([A-Za-z]+)", claim[match.end():])
        filter_like = filter_like or bool(
            previous and previous.group(1).lower() in FILTER_WORDS and following
            and any(_abbreviates(following.group(1).lower(), part) for part in column_words)
        )
        ordinal = plain and re.match(r"(?:st|nd|rd|th)\b", claim[match.end():], re.IGNORECASE) is not None
        # "on May 22, 2011", "aged 50 and over" and "from 2023 to 2024" pick rows; "6.5 out of 10" gives
        # the scale.
        filter_like = filter_like or any(low <= start < high for low, high in days) or bool(
            re.search(r"\bout\s+of\s*$", before, re.IGNORECASE)
        )
        found.append(
            _Candidate(match, qualifier, qualifier_start, cue, cue_start, year_like, filter_like, ordinal)
        )
    return found


def _unit_after(claim: str, end: int, columns: list[str], row_noun: str | None) -> tuple[str | None, int]:
    """The unit words right after a number ("minutes", "million tonnes"), and where they end.

    Words that name a column or the rows are part of the question, not a unit,
    unless the first word is a unit of its own ("43.9 mm" for bill_length_mm).
    """
    column_words = _column_words(columns) | ({row_noun} if row_noun else set())
    exact = {str(column).lower() for column in columns}
    words = []
    position = end
    # A unit may hold a slash: "km/h", "mg/dL".
    for match in re.finditer(r"\s*([A-Za-z][A-Za-z.]*(?:/[A-Za-z][A-Za-z0-9.]*)?)", claim[end:]):
        if match.start() != position - end:
            break
        word = match.group(1).rstrip(".")
        if word.lower() in ADJECTIVE_NOUNS:
            break
        # "73 HR" names the HR column, not hours: a column's own name is no unit.
        known_unit = word.lower() in UNITS and not words and word.lower() not in exact
        # "565 boys named Jack": a participle after the noun starts what picks the rows.
        participle = bool(words) and len(word) > 4 and word.lower().endswith("ed")
        stop = word.lower() in UNIT_STOPS or (word.lower() in column_words and not known_unit)
        if stop or len(words) == 3 or participle:
            break
        words.append(word)
        position = end + match.end()
        if match.group(1).endswith("."):
            # A full stop ends the sentence, not the unit: "5076 g."
            position -= len(match.group(1)) - len(word)
            break
    return (" ".join(words) or None), position


def _time_slope(claim: str) -> tuple[str, str, str, bool] | None:
    """In a claim of a change each year ("has grown by about 500 million pounds a year"): the question of the
    trend per that unit, the number's text, the unit and whether it falls."""
    found = TIME_SLOPE.search(claim)
    if not found or len(list(NUMBER_PATTERN.finditer(found.group("number")))) != 1:
        return None
    unit = found.group("unit").lower()
    # A decade is ten years of the yearly trend.
    grain = "year" if unit == "decade" else unit
    question = re.sub(r"\s+", " ", f"{claim[:found.start()]} per {grain} {claim[found.end():]}").strip()
    verb = found.group("verb").lower().split()[-1]
    return question, found.group("number"), unit, verb.startswith(FALLING_VERBS) or verb in LOSING_VERBS


def _slope_claim(claim: str) -> tuple[str, str, str, bool] | None:
    """In a slope claim: what changes, the column it changes with, the number's text, whether it falls."""
    found = SLOPE_PATTERN.match(claim)
    if not found or len(list(NUMBER_PATTERN.finditer(found.group("number")))) != 1:
        return None
    falling = found.group("verb").lower().startswith(FALLING_VERBS)
    return found.group("what"), found.group("per"), found.group("number"), falling


def _says_fall(text: str, number: ClaimedNumber) -> bool:
    """Whether a falling verb comes before the claim's number: "fell by about 19%", "dropped 5 points".

    "halved" says it fell, though the number the check compares (50%) is not written in the claim.
    """
    multiple = MULTIPLE.search(text)
    if multiple and multiple.group("word").lower() == "halved":
        return True
    before = text[:text.find(number.text)] if number.text in text else text
    return bool(re.search(rf"\b(?:{'|'.join(FALLING_VERBS)})\w*(?:\s+\w+){{0,3}}\s*$", before, re.IGNORECASE))


def _fraction_interval(value: float) -> tuple[float, float]:
    """The values nearer ``value`` than any other fraction with a denominator up to 10, as percentages.

    "a quarter" stands for 23.6% to 26.8%: between the midpoints to 2/9 and 2/7.
    """
    fractions = sorted({top / bottom for bottom in range(2, 11) for top in range(1, bottom)})
    below = max((item for item in fractions if item < value - 1e-12), default=0.0)
    above = min((item for item in fractions if item > value + 1e-12), default=1.0)
    return (value + below) / 2 * 100, (value + above) / 2 * 100


def _times(written: str) -> float:
    """The number of times written in digits or words: 200 for "two hundred", 100 for "a hundred"."""
    words = written.lower().replace("-", " ").split()
    if words[0][0].isdigit():
        return float(words[0].replace(",", ""))
    number = 1 if words[0] == "a" else NUMBER_WORDS.get(words[0], 1)
    return float(number * (100 if words[-1] == "hundred" else 1))


def _worded_number(claim: str) -> tuple[str, Any] | None:
    """The claim with a fraction or a whole number in words written in digits, and how to read it back.

    Returns (rewritten claim, function that turns the number read from it back
    into the words' own interval), or None when the claim has none.
    """
    multiple = MULTIPLE.search(claim)
    if multiple:
        # "more than doubled" is a rise of more than 100%, "halved" a fall of about 50%.
        word = multiple.group("word").lower()
        if multiple.group("times"):
            times = _times(multiple.group("times"))
        else:
            times = MULTIPLE_WORDS.get(word) or float(NUMBER_WORDS.get((multiple.group("n") or "").lower())
                                                      or multiple.group("n"))
        change = (times - 1) * 100
        qualifier = (multiple.group("qualifier") or "about ").strip()
        verb = "fell" if change < 0 else "rose"
        written = f"{verb} by {qualifier} {abs(change):g}%"
        return f"{claim[:multiple.start()]}{written}{claim[multiple.end():]}", lambda number: number
    unit_part = FRACTION_OF_UNIT.search(claim)
    if unit_part:
        # "three quarters of a percentage point a year" is 0.75 points, not 75%.
        count = COUNT_WORDS.get((unit_part.group("count") or "a").lower(), 1)
        value = count / FRACTION_PARTS[unit_part.group("part").lower()]
        qualifier = f"{unit_part.group('qualifier')} " if unit_part.group("qualifier") else ""
        written = f"{qualifier}{value:g} {unit_part.group('unit')}s"
        return f"{claim[:unit_part.start()]}{written}{claim[unit_part.end():]}", lambda number: number

    def is_fraction(found: re.Match) -> bool:
        # "the fourth quarter of 2020", "a third quarter" and "the top third of the class" are no shares;
        # a fraction is "a quarter of" or "half".
        ordinal = re.search(r"\b(?:first|second|third|fourth|last|final|1st|2nd|3rd|4th)\s+$",
                            claim[:found.start("part")], re.IGNORECASE)
        return not ordinal and bool(found.group("count") or found.group("part").lower() == "half")

    # "rose by about half" is a change of about 50%, read as "about half of" is.
    fraction = next((found for found in (*BY_FRACTION.finditer(claim), *FRACTION_PATTERN.finditer(claim))
                     if is_fraction(found)), None)
    value, rest = None, claim[fraction.end():] if fraction else ""
    if fraction:
        count = COUNT_WORDS.get((fraction.group("count") or "a").lower(), 1)
        bottom = FRACTION_PARTS[fraction.group("part").lower()]
        value = count / bottom if count < bottom else None
    elif (fraction := IN_FRACTION.search(claim)):
        # "seven in ten of the people surveyed": a share, as "seven tenths" would be.
        words = {**NUMBER_WORDS, "one": 1}
        top, bottom = (int(words.get(part.lower(), part)) if not part.isdigit() else int(part)
                       for part in (fraction.group("top"), fraction.group("bottom")))
        value, rest = (top / bottom if 0 < top < bottom else None), claim[fraction.end():]
        if re.search(r"\bevery\s+\S+$", fraction.group(0), re.IGNORECASE):
            # "35 of every 100 overdose deaths" is 35% of the deaths; "8 in every 100 were" names no noun.
            verb = r"(?:of|were|was|are|is|had|have|has|who|that|[A-Za-z]+ed)\b"
            rest = re.sub(rf"^(\s+)(?!{verb})(?=[A-Za-z])", r"\1of ", rest, flags=re.IGNORECASE)
        if value is not None and bottom > 10:
            # "1 in 20" is 5%, read with a percent's own rounding.
            return f"{claim[:fraction.start()]}{fraction.group('qualifier') or ''} {value * 100:g}%" \
                   f"{rest}".replace("  ", " "), lambda number: number
    if fraction:
        if value is None:
            return None
        low, high = _fraction_interval(value)
        qualifier = re.sub(r"\s+", " ", (fraction.group("qualifier") or "").lower())
        kind = next((kind for phrase, kind in QUALIFIERS if phrase == qualifier), None)
        rewritten = f"{claim[:fraction.start()]}{value * 100:g}%{rest}"

        share = value * 100

        def back(number: ClaimedNumber) -> ClaimedNumber:
            bounds = {
                "more than": (share, math.inf, True, False), "at least": (share, math.inf, False, False),
                "less than": (-math.inf, share, False, True), "at most": (-math.inf, share, False, False),
                "just over": (share, high, True, False), "just under": (low, share, False, True),
                "nearly": (low, share, False, False),
            }.get(kind or "", (low, high, False, False))
            return replace(number, text=fraction.group(0), low=bounds[0], high=bounds[1], low_open=bounds[2],
                           high_open=bounds[3], qualifier=kind, decimals=1)
        return rewritten, back
    if any(not (item.year_like or item.ordinal or _bound_like(claim, item, percent=True))
           for item in _candidates(claim, [])):
        # A number in digits is the one claimed; a year ("in 2020") is not, nor a bound ("topped 60% in
        # nine states").
        return None
    units = {"one": 1, **{word: NUMBER_WORDS[word] for word in list(NUMBER_WORDS)[:8]}}
    worded = re.search(rf"\b({'|'.join(NUMBER_WORDS)})(?:[- ]({'|'.join(units)}))?\b", claim, re.IGNORECASE)
    if not worded:
        return None
    total = NUMBER_WORDS[worded.group(1).lower()]
    if worded.group(2) and total >= 20 and total % 10 == 0:
        total += units[worded.group(2).lower()]
    else:
        worded = re.search(rf"\b{worded.group(1)}\b", claim)
        assert worded is not None
    rewritten = f"{claim[:worded.start()]}{total}{claim[worded.end():]}"
    return rewritten, lambda number: replace(number, text=worded.group(0))


# A comparison that makes the number after it a bound on a column: "deeper than 50 cm", "25 or more days".
COMPARATIVE_THAN = re.compile(r"\b[A-Za-z]+er\s+than\s*$", re.IGNORECASE)
OR_MORE = re.compile(
    r"\s*(?:[A-Za-z]+\s+)?or\s+(?:more|over|above|higher|greater|longer|less|fewer|under|below|lower|younger|"
    r"older|earlier|later|sooner)\b",
    re.IGNORECASE,
)
BOUND_QUALIFIERS = frozenset({"more than", "less than", "at least", "at most"})


def _bound_like(claim: str, item: _Candidate, percent: bool = False) -> bool:
    """Whether a number bounds a condition ("under 25", "deeper than 50 cm", "25 or more days").

    A percentage is a bound only when ``percent`` allows it and a word of comparison comes before it
    ("topped 60% in nine states"), since a percentage is most often the share claimed.
    """
    if item.match.group("percent"):
        return percent and item.qualifier in BOUND_QUALIFIERS
    if item.year_like or item.ordinal:
        return False
    # "declared within 19 days": a span the rows fall inside.
    within = re.search(r"\bwithin\s+$", claim[:item.qualifier_start], re.IGNORECASE)
    return (item.qualifier in BOUND_QUALIFIERS or bool(COMPARATIVE_THAN.search(claim[:item.match.start()]))
            or bool(OR_MORE.match(claim[item.match.end():])) or bool(within))


def _past_bounds(claim: str, usable: list[_Candidate]) -> _Candidate | None:
    """The number claimed when every other number is a bound: 3.9 in "Films that scored under 50 still
    averaged 3.9 stars". None when more than one is left, or a number in words is ("Nine Munros rise above
    4,000 feet, led by Ben Nevis at 4,411 feet")."""
    bounds = [item for item in usable if _bound_like(claim, item, percent=True)]
    rest = [item for item in usable if item not in bounds]
    if bounds and not rest:
        # "more than 1,000 earthquakes of magnitude 4.5 or above": the bound that counts things is the number
        # claimed; the one written "or above" picks the rows.
        rest = [item for item in bounds if item.qualifier and not OR_MORE.match(claim[item.match.end():])
                and re.match(r"\s+[a-z]+s\b", claim[item.match.end():])]
        shares = [item for item in bounds if item.match.group("percent")]
        if not rest and len(shares) == 1 and re.match(r"\s+of\b", claim[shares[0].match.end():]):
            # "More than 85% of the babies born to mothers under 20": the share of a group is the number
            # claimed; the other bounds pick the group.
            rest = shares
        bounds = [item for item in bounds if item not in rest]
    if not bounds or len(rest) != 1:
        return None
    if rest[0].match.group("percent") and any(item.match.group("percent") for item in bounds):
        # "More than 60% of the states had more than 60% of the vote" bounds one share by another.
        return None
    if re.search(rf"\b(?:{'|'.join(NUMBER_WORDS)})\b", claim, re.IGNORECASE):
        return None
    return rest[0]


def _unused_bounds(asked: str, answer: QueryAnswer) -> list[str]:
    """The bounds left in a question ("under 25") that the reading did not use as a condition on a column."""
    plan = answer.plan
    used = [item.value for item in (*plan.conditions, *plan.counted)
            if isinstance(item, Condition) and item.value is not None]
    unused = []
    for item in _candidates(asked, []):
        if not _bound_like(asked, item, percent=True):
            continue
        value, _, _, scale, _ = parse_number(item.match)
        if not any(math.isclose(value * scale, other, rel_tol=1e-9) for other in used):
            unused.append(asked[item.qualifier_start:item.match.end()].strip())
    return unused


def read_claim(
    claim: str, columns: list[str], row_noun: str | None = None
) -> tuple[str | None, ClaimedNumber | None, str]:
    """Split a claim into the question it asks and the number it states.

    Returns (question, number, problem). ``problem`` is "" when the split
    worked, or why it didn't: no number, or several numbers that each look
    like the one being claimed. A change per unit of another column ("rises
    by 10.5 ms per day") asks for the correlation of the two, whose slope the
    check compares. A fraction or a whole number in words ("about a quarter
    of", "Seventeen cases") is read as its number.
    """
    # "6.5 out of 10": the scale says what the number is out of, not a second number.
    claim = re.sub(r"(\d)\s+out\s+of\s+(?:10|5|100)\b", r"\1", claim)
    # "xylazine, a veterinary sedative, turned up in": a description between commas says what a thing is.
    claim = APPOSITIVE.sub("", claim)
    every = EVERY_BASE.match(claim)
    if every:
        # "Of every 100 birds taken in, about 25 were put down" is a share of the birds: 25%.
        share = float(every.group("top")) / float(every.group("bottom").replace(",", "")) * 100
        qualifier = f"{every.group('qualifier')} " if every.group("qualifier") else ""
        rewritten = f"{qualifier}{share:g}% of {every.group('base')} {every.group('rest')}"
        return read_claim(rewritten, columns, row_noun)
    among = AMONG_LEAD.match(claim)
    if among:
        # "Among victims aged 50 and over, 9.5% died" is a share of those victims, and so is "Among
        # victims who died, 79% were men".
        joined = "" if re.match(r"(?:were|was|are|is|had|have|has)\b", among.group("rest"), re.I) else "who "
        rewritten = f"{among.group('share')} of {among.group('base')} {joined}{among.group('rest')}"
        return read_claim(rewritten, columns, row_noun)
    subject = SUBJECT_SHARE.match(claim)
    if subject and not MEASURE_SUBJECTS & {word.lower() for word in subject.group("part").split()}:
        # "Men were 79% of the victims who died" is a share of those victims.
        rewritten = f"{subject.group('share')} of {subject.group('rest')} were {subject.group('part')}"
        return read_claim(rewritten, columns, row_noun)
    reported = REPORTING_CLAUSE.match(claim)
    if reported:
        # "Owners estimated that their cats caught 3.7 prey": who said it is not part of what is checked.
        return read_claim(claim[reported.end():], columns, row_noun)
    worded = _worded_number(claim)
    if worded:
        question, number, problem = read_claim(worded[0], columns, row_noun)
        return question, (worded[1](number) if number is not None else None), problem
    timed = _time_slope(claim)
    if timed:
        # "has grown by about 500 million pounds a year" asks for the trend per year, whose slope is checked.
        question, written, _, _ = timed
        item = _candidates(written, columns)[0]
        if not item.match.group("percent"):
            unit, unit_end = _unit_after(written, item.match.end(), columns, row_noun)
            number = claimed_number(written[item.qualifier_start:unit_end], item.match, item.qualifier, unit)
            named = written[unit_end:].strip()
            if named:
                # "has added about 80,000 visitors a year": words after the number that name a column say
                # what changes.
                question = re.sub(r"\bper (year|month|week|day)\b", rf"{named} per \1", question, count=1)
            return question, number, ""
    slope = _slope_claim(claim)
    if slope:
        what, per, written, _ = slope
        item = _candidates(written, columns)[0]
        if not item.match.group("percent"):
            unit, unit_end = _unit_after(written, item.match.end(), columns, row_noun)
            number = claimed_number(written[item.qualifier_start:unit_end], item.match, item.qualifier, unit)
            what = re.sub(r"^(?:the|a|an)\s+", "", what.strip(), flags=re.IGNORECASE)
            # "per mm of flipper length" is per unit of flipper length, when those words name a column.
            measured = re.match(r"([A-Za-z]+)\s+of\s+(.+)", per)
            unit_word = measured.group(1).lower() if measured else ""
            if measured and (unit_word in UNITS or unit_word in PER_UNIT_WORDS) and set(
                _words(measured.group(2))
            ) & _column_words(columns):
                per = measured.group(2)
            return f"correlation between {per} and {what}", number, ""
    candidates = _candidates(claim, columns)
    if not candidates:
        return None, None, "The claim states no number to check."
    plain = [item for item in candidates if not (item.year_like or item.filter_like or item.ordinal)]
    # The years of a range ("from 2023 to 2024") pick rows and are never the number claimed; nor is a year
    # after "in" or "since" ("rose in 2020").
    spans = [span.span() for span in YEAR_SPAN.finditer(claim)]
    usable = plain or [item for item in candidates if not (item.year_like or item.ordinal)] or [
        item for item in candidates
        if not any(low <= item.match.start() < high for low, high in spans)
        and not (item.year_like and YEAR_PREPOSITION.search(claim[:item.match.start()]))
    ]
    if not usable:
        return None, None, "The claim states no number to check."
    cued = [item for item in usable if item.cue or item.qualifier]
    chosen = cued[0] if cued else usable[-1]
    # Any other number left in the question would be read as part of it ("... and 3700 g" as a filter).
    others = cued if len(cued) > 1 else plain
    rivals = [item for item in others if item.match.group(0) != chosen.match.group(0)]
    claimed = _past_bounds(claim, usable) if rivals else None
    if claimed is not None:
        # "Drivers under 25 accounted for 12% of all claims": the bound picks rows and stays in the question,
        # where the check makes sure it is read as a condition (_unused_bounds).
        chosen, rivals = claimed, []
    if rivals:
        listed = ", ".join(dict.fromkeys(item.match.group(0).strip() for item in [chosen, *rivals]))
        return None, None, (
            f"The claim states more than one number that could be the one claimed ({listed}); "
            "check them one at a time."
        )
    unit, unit_end = _unit_after(claim, chosen.match.end(), columns, row_noun)
    if chosen.match.group("percent"):
        # A percent is its own unit: in "9.5% died" the verb says what is counted.
        unit, unit_end = None, chosen.match.end()
    averaged = re.search(r"\b(?:average|averages|mean|per)\b", claim, re.IGNORECASE)
    measured = bool(unit and unit.split()[0].lower() in UNITS)
    # "recorded 1,950 captures" totals them; a percentage is never a total.
    if chosen.cue in TALLY_VERBS and not averaged and not measured and not chosen.match.group("percent"):
        written = claim[chosen.qualifier_start:chosen.match.end()]
        number = claimed_number(written, chosen.match, chosen.qualifier, None)
        named = f"{unit} " if unit else ""
        question = re.sub(r"\s+", " ", f"total {named}{claim[:chosen.cue_start]} {claim[unit_end:]}")
        return question.strip(" .,;:!?"), number, ""
    number = claimed_number(claim[chosen.qualifier_start:unit_end], chosen.match, chosen.qualifier, unit)
    # "There are 344 penguins", "152 penguins are Adelie": a number before the rows' noun counts them.
    lead = claim[:chosen.qualifier_start].strip()
    following = re.match(r"\s*([A-Za-z]+)(?:\s+([A-Za-z]+))?(?:\s+([A-Za-z]+))?", claim[chosen.match.end():])
    # "68 Chinstrap penguins": the rows' noun may follow a value that names which.
    counted = following is not None and (
        (row_noun in [word.lower() for word in following.groups() if word]
         and not (unit and unit.split()[0].lower() in UNITS))
        or ((unit is None or unit.split()[0].lower() in IRREGULAR_PLURALS) and (
            following.group(1).lower().endswith("s") or following.group(1).lower() in IRREGULAR_PLURALS
        ))
    )
    # "won the Tour 20 times" counts the rows the rest of the claim picks.
    times = re.match(r"\s*times\b", claim[chosen.match.end():], re.IGNORECASE)
    if times and not chosen.match.group("percent"):
        rest = f"{claim[:chosen.qualifier_start]} {claim[chosen.match.end() + times.end():]}"
        rest = re.sub(r"\s+", " ", rest).strip(" .,;:!?")
        written = claim[chosen.qualifier_start:chosen.match.end()]
        number = claimed_number(written, chosen.match, chosen.qualifier, None)
        return f"how many rows {rest}", number, ""
    # "The models came from 28 manufacturers": a plural that names a column counts its distinct values.
    plural = following.group(1).lower() if following else ""
    singulars = {plural[:-1], plural[:-2] if plural.endswith("es") else "", f"{plural[:-3]}y"}
    names = {normalized_name(str(column)) for column in columns}
    measured = plural in UNITS or plural in TIME_PLURALS or re.search(
        r"\b(?:average|averages|averaged|mean|median|total|per)\b", claim, re.IGNORECASE)
    if plural.endswith("s") and not chosen.match.group("percent") and not measured and singulars & names:
        rest = f"{claim[:chosen.qualifier_start]} {claim[chosen.match.end() + following.end(1):]}"
        rest = re.sub(r"\s+", " ", rest).strip(" .,;:!?")
        return f"how many distinct {plural} {rest}", replace(number, unit=None), ""
    # "55 of the cats were neutered" counts the cats that were.
    of_the = OF_THE.match(claim[chosen.match.end():])
    noun = of_the.group(1).lower() if of_the else ""
    if of_the and not lead and (noun.endswith("s") or noun in IRREGULAR_PLURALS):
        rest = re.sub(r"\s+", " ", claim[chosen.match.end() + of_the.end():]).strip(" .,;:!?")
        written = claim[chosen.qualifier_start:chosen.match.end()]
        number = claimed_number(written, chosen.match, chosen.qualifier, None)
        return f"how many {of_the.group(1)} {rest}", number, ""
    # "In 2015, 565 boys were named Jack": a time or place before the count picks rows too.
    led = re.fullmatch(r"(?P<scope>(?:in|during|on|by)\s+[^,]+?)?,?\s*(?:there\s+(?:are|were|is|was))?", lead,
                       flags=re.IGNORECASE)
    if counted and led:
        rest = re.sub(r"\s+", " ", claim[chosen.match.end():]).strip(" .,;:!?")
        written = claim[chosen.qualifier_start:chosen.match.end()]
        number = claimed_number(written, chosen.match, chosen.qualifier, None)
        scope = f" {led.group('scope')}" if led.group("scope") else ""
        return f"how many {rest}{scope}", number, ""
    start = chosen.cue_start if chosen.cue in AGGREGATION_CUES or chosen.cue in TALLY_VERBS or chosen.cue in {
        "is", "was", "were", "are", "be", "been", "equals", "equalled", "equaled", "stood", "amounted",
        "came", "numbered", "at", "by", "of", "reached", "reaches",
    } else chosen.qualifier_start
    before, after = claim[:start], claim[unit_end:]
    adjective = re.match(r"\s*([A-Za-z]+)", after)
    if adjective and adjective.group(1).lower() in ADJECTIVE_NOUNS:
        after = f" {ADJECTIVE_NOUNS[adjective.group(1).lower()]}{after[adjective.end():]}"
    # "Fish farms lost 71.4 million salmon" names `losses` only where the table has such a column.
    noun = VERB_NOUNS.get(chosen.cue or "")
    forms = {noun, noun.rstrip("s"), noun.removesuffix("es")} if noun else set()
    if noun and (chosen.cue in PREDICATE_CUES or any(forms & set(normalized_name(column).split())
                                                     for column in columns)):
        before = re.sub(rf"\b{chosen.cue}\b", noun, before, flags=re.IGNORECASE)
        if not number.percent and not AVERAGING.search(claim) and not AGGREGATION_SAID.search(claim):
            # "Fish farms lost about 71.4 million salmon in 2023": what was lost over the rows, added up.
            before = f"total {before}"
    question = f"{before} {after}"
    if chosen.cue in AGGREGATION_CUES:
        question = f"{AGGREGATION_CUES[chosen.cue]} {question}"
    question = re.sub(r"\bon average\b", "", question, flags=re.IGNORECASE)
    if re.search(r"\bon average\b", claim, re.IGNORECASE):
        question = f"average {question}"
    question = re.sub(r"\s+", " ", question).strip(" .,;:!?")
    question = re.sub(r"^(?:the|a|an)\s+", "", question, flags=re.IGNORECASE)
    question = re.sub(r"\s+(?:is|was|were|are|of|at|by)$", "", question, flags=re.IGNORECASE)
    if not question:
        return None, number, "The claim says nothing about what the number measures."
    if number.percent and re.match(r"of\s", question, re.IGNORECASE):
        # "9.5% of victims aged 50 and over who died" asks for a share of those victims.
        question = f"share {question}"
    return question, number, ""


def _abbreviates(word: str, part: str) -> bool:
    """Whether a question word and a column-name word are the same word, or one shortens the other."""
    if word == part:
        return True
    return (len(part) >= 3 and word.startswith(part)) or (len(word) >= 3 and part.startswith(word))


def _substitute(question: str, column: str) -> str | None:
    """The question with its longest run of words that all match ``column``'s name replaced by the name."""
    parts = re.findall(r"[a-z]+", str(column).lower())
    if not parts:
        return None
    tokens = list(re.finditer(r"[A-Za-z]+", question))
    best: tuple[int, int] | None = None
    i = 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and any(_abbreviates(tokens[j].group(0).lower(), part) for part in parts):
            j += 1
        if j > i and (best is None or j - i > best[1] - best[0]):
            best = (i, j)
        i = max(j, i + 1)
    if best is None:
        return None
    start, end = tokens[best[0]].start(), tokens[best[1] - 1].end()
    return f"{question[:start]}{column}{question[end:]}"


def _counted(reason: str, count: int) -> str:
    """"2 rows with x missing", or "1 row with x missing" for one."""
    if count == 1:
        reason = re.sub(r"^(rows|values|repeats)\b", lambda m: m.group(1)[:-1], reason)
        reason = reason.replace("exact repeats", "exact repeat")
    return f"{count:,} {reason}"


# Words left out of initials, so "in December" is not ID.
INITIALS_SKIP = frozenset({
    "in", "on", "of", "the", "a", "an", "by", "for", "to", "and", "or", "at", "per", "is", "was", "were",
    "are",
})


def _initials(question: str, column: str) -> bool:
    """Whether an all-capitals ``column`` is the initials of words in a row: "home runs" for HR."""
    if not (str(column).isalpha() and str(column).isupper()):
        return False
    words = [word for word in re.findall(r"[a-z]+", question.lower()) if word not in INITIALS_SKIP]
    size = len(str(column))
    return str(column).lower() in {"".join(word[0] for word in words[start:start + size])
                                   for start in range(len(words))}


def _named(question: str, column: str) -> bool:
    """Whether the question names ``column``: one of its words, a word that shortens or lengthens one,
    or its initials."""
    # "demVote" is "dem" and "vote", as everywhere else names are read.
    parts = re.findall(r"[a-z]+", normalized_name(str(column)).lower())
    words = re.findall(r"[a-z]+", question.lower())
    if str(column).lower() in question.lower() or _initials(question, column):
        return True
    if _outcome_named(words, column):
        return True
    return any(_abbreviates(word, part) for word in words for part in parts if len(part) >= 2)


def _outcome_named(words: list[str], column: str) -> bool:
    """Whether a word names a column called for the same outcome: "deaths" names fatal."""
    sides = {word: side for side, forms in OUTCOME_WORDS.items() for word in forms}
    side = sides.get(normalized_name(str(column)))
    return side is not None and any(sides.get(word) == side for word in words)


# Name words that say a column counts or totals something; a claim need not repeat them ("250 parts"
# for num_parts).
COUNTING_WORDS = frozenset({"num", "number", "no", "n", "count", "total", "amount", "qty"})


def _names_head(question: str, column: str, strict: bool = False) -> bool:
    """Whether the question names the last word of ``column``'s name, the kind of quantity it holds.

    The unit and counting words are set aside: "order value" names payment_value
    and "250 parts" names num_parts, "home runs" names HR by its initials, while
    "deaths" names only the first word of death_height_metres, a height. With
    ``strict``, a longer word that only starts with the name word does not
    name it: "seedlings" starts with Seed but is another word.
    """
    lowered = str(column).lower()
    whole = re.search(rf"(?<![a-z0-9]){re.escape(lowered)}(?![a-z0-9])", question.lower())
    words = re.findall(r"[a-z]+", question.lower())
    if (whole if strict else lowered in question.lower()) or _initials(question, column):
        return True
    if _outcome_named(words, column):
        return True
    parts = re.findall(r"[a-z]+", normalized_name(column))
    if len(parts) > 1 and column_unit(column) == parts[-1]:
        parts = parts[:-1]
    parts = [part for part in parts if part not in COUNTING_WORDS] or parts
    if strict:
        return not parts or any(_same_word(word, parts[-1]) for word in words)
    return not parts or any(_abbreviates(word, parts[-1]) for word in words)


# Endings that leave a word the same word: "runs" is run, "weighed" is weigh.
INFLECTIONS = ("s", "es", "ed", "d", "ing", "er", "ers")
# Endings that make a noun of a verb: "production" is what is produced, "collection" what is collected.
NOUN_ENDINGS = ("tion", "ion", "tions", "ions")


def _same_word(word: str, part: str) -> bool:
    """Whether a question word is a column-name word, a shortening of it, it with an inflection, or the noun
    of it as a verb ("production" for produced)."""
    if word == part or (len(word) >= 3 and part.startswith(word)):
        return True
    if word.startswith(part) and word[len(part):] in INFLECTIONS:
        return True
    stems = [word[:-len(ending)] for ending in NOUN_ENDINGS
             if word.endswith(ending) and len(word) > len(ending) + 3]
    return any(part.startswith(stem) and part[len(stem):] in ("e", *INFLECTIONS) for stem in stems)


def _candidate_columns(message: str, columns: list[str]) -> list[str]:
    """Columns named in a refusal's "Did you mean A, B or C?"."""
    asked = re.search(r"Did you mean (.+?)\?", message)
    if not asked:
        return []
    named = re.split(r",\s*|\s+or\s+", asked.group(1))
    known = {str(column): column for column in columns}
    return [known[name.strip()] for name in named if name.strip() in known]


def _comparable(answer: QueryAnswer, number: ClaimedNumber) -> tuple[float | None, list[str]]:
    """The computed number to compare with the claim, in the claim's terms, and how it was taken."""
    figures = dict(answer.figures)
    notes: list[str] = []
    if number.percent:
        if "share_percent" in figures:
            plan = answer.plan
            of = (f"distinct {plan.count_column} values" if plan.count_column
                  else f"the total of {plan.weight}" if plan.weight else "rows")
            notes.append(f"compared with the share of {of}, as a percentage")
            return figures["share_percent"], notes
        if "change_percent" in figures:
            return figures["change_percent"], notes
        if answer.value is not None and abs(answer.value) <= 1 and not number.holds(answer.value):
            notes.append(f"compared as a percentage: {answer.value:.6g} is {answer.value * 100:.6g}%")
            return answer.value * 100, notes
        return answer.value, notes
    if answer.value is not None:
        return answer.value, notes
    if len(answer.values) == 1:
        return answer.values[0][1], notes
    return None, notes


def _scale_note(number: ClaimedNumber, computed: float, column: str | None) -> tuple[str, str] | None:
    """When the claim agrees only if a scale word is read as part of the unit, or the data are in other units.

    Returns (kind, sentence): "unit" when "37,528 million tonnes" agrees with
    37,528 read as the unit's own count; "scale" when it would agree only if the
    column were recorded in thousands, millions or billions.
    """
    name = column or "the value"
    if number.scale_word and number.scale != 1 and number.holds(computed * number.scale):
        unit = " ".join(part for part in (number.scale_word, number.unit) if part)
        return "unit", (
            f'Read "{unit}" as the unit {name} is recorded in, so {_plain(number.value / number.scale)} '
            f"is compared with {name} as recorded."
        )
    if number.percent or math.isinf(number.low) or math.isinf(number.high):
        # A share has no unit to scale, and a bound holds for nearly any number scaled far enough.
        return None
    for factor, word in ((1e3, "thousands"), (1e6, "millions"), (1e9, "billions")):
        readings = ((computed * factor, f"in {word}"), (computed / factor, f"in units {word} smaller"))
        for scaled, recorded in readings:
            if number.holds(scaled):
                return "scale", (
                    f"The claim agrees with {name} only if {name} is recorded {recorded}, and the data "
                    "do not say what unit it is in."
                )
    return None


def column_unit(column: str | None) -> str | None:
    """The unit a column's name ends in: "g" for body_mass_g, "kg" for "Weight (kg)", None for co2.

    A name that is only a unit word ("Days") is a measure of its own, not an annotation.
    """
    if not column:
        return None
    bracketed = re.search(r"[(\[]\s*([A-Za-z]+)\s*[)\]]\s*$", str(column))
    if bracketed and bracketed.group(1).lower() in UNITS:
        return bracketed.group(1).lower()
    parts = [part for part in re.split(r"[^A-Za-z]+", str(column)) if part]
    if len(parts) >= 2 and parts[-1].lower() in UNITS:
        return parts[-1].lower()
    return None


def _row_step(analysis: Any) -> str | None:
    """The hour or day each row of the table stands for, when the rows are one date each, that far apart."""
    date = analysis.roles.date
    if not date or date not in analysis.frame.columns:
        return None
    dates = analysis.frame[date].dropna()
    if dates.dtype.kind != "M" or len(dates) < 3 or dates.nunique() < 0.99 * len(analysis.frame):
        return None
    steps = dates.drop_duplicates().sort_values().diff().dropna().dt.total_seconds()
    usual = float(steps.median())
    if (steps == usual).mean() < 0.9:
        return None
    return {3600.0: "hour", 86400.0: "day"}.get(usual)


def _claim_unit(number: ClaimedNumber) -> str | None:
    if number.percent or number.scale_word or not number.unit:
        return None
    first = number.unit.split()[0].lower()
    return first if first in UNITS else None


def _unit_columns(analysis: Any, asked: str, number: ClaimedNumber) -> list[tuple[str, str]]:
    """Columns whose unit is the kind of quantity the claim states, with the question naming each.

    The kind comes from the claim's unit ("4000 g" is a mass) or from a verb
    ("weigh"). A column whose name gives no unit counts when its name says
    the kind (QUANTITY_NAMES: "height" for "60.3 feet"). The verb, if any, is
    replaced by the column's name; otherwise the name is added to the question.
    """
    verbs = [word for word in re.findall(r"[A-Za-z]+", asked) if word.lower() in UNIT_VERBS]
    claimed = _claim_unit(number)
    kind = UNITS[claimed][0] if claimed else UNIT_VERBS.get(verbs[0].lower()) if verbs else None
    if kind is None:
        return []
    found = []
    for column in analysis.roles.numeric:
        unit = column_unit(str(column))
        words = set(normalized_name(str(column)).split())
        named = not unit and bool(QUANTITY_NAMES.get(kind, frozenset()) & words)
        if (unit and UNITS[unit][0] == kind) or named:
            rewritten = (
                re.sub(rf"\b{re.escape(verbs[0])}\b", str(column), asked, count=1) if verbs
                else f"{asked} {column}"
            )
            found.append((str(column), rewritten))
    return found


def _one_unit_reading(
    analysis: Any, asked: str, number: ClaimedNumber, choices: dict[str, str]
) -> tuple[str, str, QueryAnswer] | None:
    """The column, question and answer when exactly one column fits the claim's unit and answers it."""
    from .api import Refusal  # noqa: PLC0415

    if not _claim_unit(number):
        return None
    readings = []
    for column, rewritten in _unit_columns(analysis, asked, number):
        other = analysis.ask(rewritten, decisions=choices)
        if not isinstance(other, Refusal) and other.plan.measure == column:
            readings.append((column, rewritten, other))
    return readings[0] if len(readings) == 1 else None


# Rows per unit, on average, before a unit counts as measured repeatedly.
MIN_ROWS_PER_UNIT = 1.5


def _repeated_units(analysis: Any, answer: QueryAnswer) -> Check | None:
    """Whether the answer treats rows as independent when a unit (a subject, a site) has several rows."""
    roles = analysis.roles
    unit = roles.identifier
    plan = answer.plan
    if not unit or unit not in analysis.frame.columns or plan.intent not in (
        "aggregate", "compare", "correlation", "correlates", "breakdown", "rank", "trend",
    ):
        return None
    counts = analysis.frame[unit].dropna().value_counts()
    # A few repeated ids (one in 76,519 Himalayan climbers) are not repeated measurements.
    if counts.empty or counts.sum() < MIN_ROWS_PER_UNIT * len(counts):
        return None
    per = plan.per or ()
    if unit in per:
        return Check("repeated_units", True, f"{unit} has several rows each; the answer combines them first.")
    units, rows = len(counts), int(counts.sum())
    if plan.intent in ("compare", "correlation", "correlates"):
        detail = (
            f"{rows:,} rows come from {units:,} {unit} values, so they are not independent; a test or "
            "correlation over rows overstates how much evidence there is."
        )
        return Check("repeated_units", False, detail)
    detail = (
        f"{rows:,} rows come from {units:,} {unit} values; this figure weights each {unit} by its number "
        "of rows."
    )
    return Check("repeated_units", True, detail)


def _source_of(data: Any) -> tuple[Any | None, Source | None, str | None]:
    """An Analysis for ``data``, its Source, and why it could not be read."""
    from .api import AdaError, Analysis, analyze, analyze_file  # noqa: PLC0415

    if isinstance(data, Analysis):
        return data, data.source, None
    if isinstance(data, str | Path) and Path(data).is_dir():
        files = sorted(item.name for item in Path(data).iterdir() if item.suffix.lower() in READABLE)
        listed = ", ".join(files[:12]) + (" and more" if len(files) > 12 else "")
        return None, Source(name=str(data), path=str(data)), (
            f"{data} is a folder. Name the file the claim is about" + (f": {listed}." if files else ".")
        )
    try:
        if isinstance(data, str | Path) and Path(data).is_file():
            return analyze_file(data), None, None
        return analyze(data, row_limit=sys.maxsize), None, None
    except AdaError as error:
        name = str(data) if isinstance(data, str | Path) else type(data).__name__
        return None, Source(name=name, path=name if isinstance(data, str | Path) else None), str(error)


def check_claim(
    data: Any,
    claim: str | None = None,
    *,
    question: str | None = None,
    value: str | float | None = None,
    sheet: str | None = None,
    decisions: Mapping[str, str] | None = None,
    answers: str | Path | Mapping[str, str] | None = None,
) -> ClaimCheck:
    """Check a claim about ``data`` (a path, bytes, a DataFrame or an ``Analysis``).

    Give the claim as a sentence (``claim``), or as ``question`` and ``value``
    when the reading is already known: ``question="average dep_delay",
    value="12.64"``. A string ``value`` keeps its decimals, which set the
    interval; a float is taken at its shortest written form. Every row of the
    file is read: no row limit applies to a check. ``decisions`` pick options
    for the choices the answer rests on (see ``ada.decisions``); an unknown
    key or option raises ``UnknownDecision``. ``answers`` is the answers file
    ``ada scan --template`` writes (or its dict): its ``decide:`` answers are
    used, and ``decisions`` win over them.
    """
    from .folder import load_answers  # noqa: PLC0415
    from .joins import folder_choices  # noqa: PLC0415

    given = load_answers(dict(answers) if isinstance(answers, Mapping) else answers)
    choices = checked_choices(decisions)
    if given and not (isinstance(data, str | Path) and Path(data).is_dir()):
        named = Path(data).name if isinstance(data, str | Path) else None
        choices = folder_choices(given, named, choices)
    text = claim if claim is not None else f"{question} is {value}"
    if sheet is not None and isinstance(data, str | Path):
        from .api import analyze  # noqa: PLC0415

        try:
            data = analyze(data, sheet=sheet, row_limit=sys.maxsize)
        except Exception as error:  # noqa: BLE001 - reported as unreadable
            return ClaimCheck(text, "unverified", "inaccessible_data", f"The data could not be read: {error}")
    if isinstance(data, str | Path) and Path(data).is_dir():
        return _check_folder(Path(data), text, claim, question=question, value=value, choices=choices,
                             given=given)
    analysis, source, problem = _source_of(data)
    if analysis is None:
        reason = "missing_context" if problem and "is a folder" in problem else "inaccessible_data"
        message = problem if reason == "missing_context" else f"The data could not be read: {problem}"
        return ClaimCheck(text, "unverified", reason, message, source=source,
                          clarification=problem if reason == "missing_context" else None)
    source = analysis.source
    columns = [str(column) for column in analysis.frame.columns]
    if question is not None and value is not None:
        written = value if isinstance(value, str) else repr(float(value)).removesuffix(".0")
        found = _candidates(str(written), [])
        if not found:
            problem = f"{written!r} is not a number."
            return ClaimCheck(text, "unverified", "missing_context", problem, source=source)
        # "about 25%" keeps its qualifier.
        item = found[0]
        number = claimed_number(str(written)[item.qualifier_start:], item.match, item.qualifier, None)
        asked = question
    said, ends, direction = text, (), None
    if question is None:
        # "rose by more than 60%, from 29,199 in 1999 to 47,173 in 2017": the change is read without its ends,
        # which are checked after it.
        text, ends = _without_ends(text)
        if not ends:
            text, direction = _as_level(text)
        # "Of the 4,526 applicants, 38.8% were admitted": the lead names the base, checked below.
        body, base = _based(_without_peak(text))
        asked, number, problem = read_claim(body, columns, analysis.roles.row_noun)
        if problem:
            reason = "unsupported_analysis" if "more than one number" in problem else "missing_context"
            return ClaimCheck(
                said, "unverified", reason, problem, claimed=number, question=asked, source=source,
                clarification="Restate the claim with one number, what it measures and over what rows."
                if reason == "missing_context" else "Send each number as its own claim.",
            )
    assert number is not None and asked is not None
    check = _checked_dates(_checked_table(analysis, text, asked, number, source, choices, question), analysis)
    if question is None and number.percent and (by := BY_YEAR.search(asked)) and (
        check.verified or check.reason == "failed_check"
    ):
        check = _by_year(check, analysis, text, asked, by, number, source, choices)
    counts = _counted_ends(check, ends, text) if ends else None
    if counts and not _ends_hold(check, ends):
        # "rose 46%, from 147.7 billion pounds in 1990": ends that count pounds may be totals, as a level in
        # pounds is; the change is one in the total over the rows when the totals give those ends.
        total = _checked_dates(
            _checked_table(analysis, text, f"total {asked}", number, source, choices, question), analysis
        )
        if total.answer is not None and total.answer.plan.intent == "growth" and (
            total.answer.plan.aggregation == "sum"
        ) and _ends_hold(total, ends):
            note = f"Read the change as one in the total, since the values it went between count {counts}."
            check = replace(total, assumptions=(*total.assumptions, note))
    if ends:
        check = _checked_ends(check, ends)
    if direction:
        check = _checked_direction(check, analysis, *direction)
    return replace(check, claim=said)


def _by_year(
    check: ClaimCheck, analysis: Any, text: str, asked: str, by: re.Match, number: ClaimedNumber,
    source: Source | None, choices: dict[str, str],
) -> ClaimCheck:
    """A share "by 2024" read both as the share in 2024 and as the share of every row up to it.

    "48 women had won by 2016" counts up to its end, but "by 2024, xylazine
    turned up in 35 of every 100 deaths" is the share in 2024. When the two
    readings give different verdicts, the claim is ambiguous.
    """
    year = by.group("year")
    within = f"{asked[:by.start()]}in {year}{asked[by.end():]}"
    other = _checked_dates(_checked_table(analysis, text, within, number, source, choices, None), analysis)
    if other.verified == check.verified or not (other.verified or other.reason == "failed_check"):
        return check
    readings = [(f"up to {year}", asked, check), (f"in {year}", within, other)]
    alternatives = tuple(
        Alternative(reading, question, item.compared, item.verified,
                    item.answer.calculation if item.answer is not None else "")
        for reading, question, item in readings
    )
    shown = "; ".join(f"{item.reading} gives {_plain(item.value) if item.value is not None else 'no number'}"
                      for item in alternatives)
    return ClaimCheck(
        text, "unverified", "ambiguous_intent",
        f'"by {year}" may mean every row up to {year} or the rows of {year}, and they give different '
        f"verdicts: {shown}.",
        claimed=number, question=asked, alternatives=alternatives, source=source,
        clarification=f"Is the share over every year up to {year}, or in {year} alone?",
    )


def _checked_table(
    analysis: Any, text: str, asked: str, number: ClaimedNumber, source: Source | None,
    choices: dict[str, str], question: str | None,
) -> ClaimCheck:
    """The check of a claim read as ``asked`` on one table, with its bounds, base, peak and ranking."""
    check = _check(analysis, text, asked, number, source, choices)
    decided = check.verified or check.reason == "failed_check"
    if question is None and check.answer is not None and decided and (
        unused := _unused_bounds(asked, check.answer)
    ):
        # "Drivers under 25" left in the question must pick rows; read any other way the verdict is not the
        # claim's.
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"The claim's {_quoted(unused)} reads as a condition on the rows, but ADA's reading "
            f"({check.answer.calculation}) does not use it as one.",
            claimed=number, question=asked, answer=check.answer, assumptions=check.assumptions, source=source,
            clarification=f"Which column does {_quoted(unused)} bound?",
        )
    without = next((item for item in check.assumptions if item.startswith(LEFT_OUT_NOTE)), None)
    if without:
        check = replace(check, words_left_out=tuple(re.findall(r'"([^"]+)"', without)))
    if question is None and (base := _based(text)[1]) and check.answer is not None:
        return _checked_base(check, *base, frame=analysis.frame)
    if question is None and PEAK_VERB.search(_most_as_peak(text)) and (
        check.verified or check.reason == "failed_check"
    ):
        return _checked_peak(check, analysis, text)
    if question is None and check.verified and check.answer is not None and SUPERLATIVE.search(text) and (
        AGAINST_ALL.search(text) and not _ranked(check.answer)
    ):
        # "had the best score of any town, at 9.4" holds only if no other town scores more; the value alone
        # does not say so.
        return replace(
            check, status="unverified", reason="missing_context",
            message=f"The claim also ranks it against all the others, which ADA did not check: "
            f"{check.message}",
            clarification="Over which groups is it ranked, and by which column?",
        )
    return check


# A change and the values it went between: "rose by more than 60%, from 29,199 in 1999 to 47,173 in 2017".
_END = (
    r"(?:(?:about|nearly|almost|around|roughly|some|just\s+over|just\s+under|more\s+than|less\s+than|over|"
    r"under)\s+)?[$€£]?\d(?:[\d,]*\d)?(?:\.\d+)?(?:\s*(?:%|per\s?cent|percent))?"
    r"(?:\s*(?:million|billion|thousand|bn|m|k)\b)?"
)
CHANGE_WORDS = re.compile(
    r"\b(?:rose|risen|rising|fell|fallen|falling|grew|grown|climbed|dropped|declined|decreased|increased|"
    r"jumped|soared|surged|slumped|plunged|tumbled|went\s+up|went\s+down|doubled|tripled|trebled|"
    r"quadrupled|halved|\w+fold|went(?=\s+from\b))\b",
    re.IGNORECASE,
)
# "from 32 the year before to 61": the first end's year, given against the year of the change.
YEAR_BEFORE = (r"(?:the|a)\s+(?:year|season)\s+(?:before|earlier)|the\s+(?:previous|prior)\s+(?:year|season)|"
               r"last\s+year")
# The words between a value and its year: a unit ("8.4 weeks") and a month ("in December 2007").
_UNIT = r"(?:\s+(?!in\b|to\b|the\b)[a-z]+)?"
_MONTH = (r"(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
          r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+)?")
FROM_TO = re.compile(
    rf",?\s*\bfrom\s+(?P<first>{_END})(?P<unit1>{_UNIT})(?:\s+(?:in\s+{_MONTH}(?P<y1>\d{{4}})|{YEAR_BEFORE}))?"
    rf"\s+to\s+(?:a\s+(?:record|new\s+high|new\s+low|high|low|peak)\s+(?:of\s+)?)?(?P<last>{_END})"
    rf"(?P<unit2>{_UNIT})"
    rf"(?:\s+in\s+{_MONTH}(?P<y2>\d{{4}}))?",
    re.IGNORECASE,
)
TO_ONLY = re.compile(rf",\s*to\s+(?P<last>{_END})(?=\s*[.!;]?\s*$)", re.IGNORECASE)
# "rose to 61", "fell below 40": the level a change reached, not its size. "rose over 30%" is a size.
LEVEL_REACHED = re.compile(
    r"\b(?:(?P<rise>rose|risen|grew|grown|climbed|increased|jumped|soared|surged|went\s+up)|"
    r"(?P<fall>fell|fallen|dropped|declined|decreased|slumped|plunged|tumbled|sank|went\s+down))"
    rf"(?P<size>\s+(?:by\s+)?{_END})?(?P<side>\s+(?:to(?:\s+(?:an?\s+)?(?P<record>(?:new|record|all-time|historic)"
    r"(?:\s+(?:new|record|all-time|historic))*)(?:\s+(?P<extreme>low|high)(?:\s+of)?)?)?|below|beneath|above|"
    r"past|beyond|over|under)\s+)"
    rf"(?=(?P<level>{_END}))",
    re.IGNORECASE,
)
# The qualifier each word before a level stands for.
LEVEL_SIDES = {"to": "", "below": "below ", "beneath": "below ", "under": "under ", "above": "above ",
               "over": "over ", "past": "more than ", "beyond": "more than "}


DURATION = re.compile(
    rf"\b(?:in|over|within)\s+(?:the\s+)?(?:(?P<count>a|an|one|{'|'.join(NUMBER_WORDS)}|\d{{1,3}})\s+)?"
    r"(?P<unit>years?|decades?)(?:\s+to\s+(?P<to>\d{4}))?\b",
    re.IGNORECASE,
)


def _span_years(lasted: re.Match) -> int | None:
    """The years a length of time names: 20 for "twenty years", 10 for "a decade"."""
    count = (lasted.group("count") or "a").lower()
    number = 1 if count in ("a", "an", "one") else NUMBER_WORDS.get(count)
    if number is None and count.isdigit():
        number = int(count)
    if number is None:
        return None
    return number * (10 if lasted.group("unit").lower().startswith("decade") else 1)


def _without_ends(text: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """A change claim without the values it went from and to, and those values as ("previous", "about
    31,500") pairs. The years of the ends name the period when the rest of the claim does not."""
    if not CHANGE_WORDS.search(text):
        return text, ()

    def stated(rest: str) -> bool:
        # "Sales rose from $1.2 million to $2.5 million" states no change besides its ends, which are then
        # the numbers claimed.
        return bool(re.search(r"\d", rest) or MULTIPLE.search(rest) or BY_FRACTION.search(rest))

    # "from 2023 to 2024" is a span of years, not the values the change went between.
    found = next((match for match in FROM_TO.finditer(text) if not all(
        re.fullmatch(r"(?:1[5-9]|20)\d{2}", match.group(side).strip()) for side in ("first", "last"))), None)
    if found:
        # Each end keeps its unit ("147.7 billion pounds"), which says what kind of number it is.
        ends = (("previous", f"{found.group('first')}{found.group('unit1')}".strip()),
                ("latest", f"{found.group('last')}{found.group('unit2')}".strip()))
        # "went from 28,000 to 5.7 million" says no more than its ends.
        rest = re.sub(r"\s*\bwent\s*$", "", text[:found.start()]) + text[found.end():]
        years = found.group("y1"), found.group("y2")
        lasted = DURATION.search(rest)
        if lasted:
            # "in twenty years", "in the twenty years to 2021": the length of the change, which the years
            # of its ends give too.
            span = _span_years(lasted)
            if not all(years) and lasted.group("to") and span:
                years = str(int(lasted.group("to")) - span), lasted.group("to")
            if all(years) and span == int(years[1]) - int(years[0]):
                rest = f"{rest[:lasted.start()]}{rest[lasted.end():]}"
        if not stated(rest):
            return text, ()
        if all(years) and not YEAR_SPAN.search(rest) and not re.search(r"\b(?:1[89]|20)\d{2}\b", rest):
            rest = re.sub(r"\s*([.!]?)\s*$", rf" between {years[0]} and {years[1]}\1", rest, count=1)
        return re.sub(r"\s+", " ", rest).strip(" ,"), ends
    found = TO_ONLY.search(text)
    if found and stated(rest := f"{text[:found.start()]}{text[found.end():]}"):
        return re.sub(r"\s+", " ", rest).strip(), (("latest", found.group("last").strip()),)
    found = _level_reached(text)
    if found:
        # "rose 10% to 61 in 2020": the change is checked, then 61 as its last value.
        end = found.end() + len(found.group("level"))
        rest = f"{text[:found.end('side') - len(found.group('side'))]}{text[end:]}"
        if stated(_without_years(rest)):
            side = LEVEL_SIDES[found.group("side").split()[0].lower()]
            return re.sub(r"\s+", " ", rest).strip(), (("latest", f"{side}{found.group('level').strip()}"),)
    return text, ()


def _without_years(text: str) -> str:
    """The text without its years, which date a claim rather than state its number."""
    return re.sub(r"\b(?:1[5-9]|20)\d{2}\b", "", text)


def _level_reached(text: str) -> re.Match | None:
    """The first level a change is said to reach ("rose to 61", "fell below 40"), or None."""
    for found in LEVEL_REACHED.finditer(text):
        side = found.group("side").split()[0].lower()
        if side in ("over", "under") and re.search(r"%|per\s?cent", found.group("level"), re.IGNORECASE):
            continue
        if re.fullmatch(r"(?:1[5-9]|20)\d{2}", found.group("level").strip()):
            # "rose to 2020 levels" is no level of its own.
            continue
        return found
    return None


def _as_level(text: str) -> tuple[str, tuple[str, str | None] | None]:
    """A claim whose only number is the level a change reached, read as that level, with the direction of the
    change ("rise" or "fall") to check after it and the record it sets, if any ("low" or "high"): "rose to 61
    in 2020" is "were 61 in 2020", having risen; "fell to a record low of 0.72" is also the lowest year."""
    found = _level_reached(text)
    if found is None or found.group("size"):
        return text, None
    end = found.end() + len(found.group("level"))
    others = _without_years(f"{text[:found.start()]}{text[end:]}")
    if re.search(r"\d", others) or MULTIPLE.search(others) or BY_FRACTION.search(others):
        return text, None
    side = LEVEL_SIDES[found.group("side").split()[0].lower()]
    level = f"{text[:found.start()]}was {side}{text[found.end():]}"
    extreme = found.group("extreme").lower() if found.group("extreme") else None
    if found.group("record") and extreme is None:
        # "rose to a record 11.3 million" is a record high, "fell to a record 0.72" a record low.
        extreme = "high" if found.group("rise") else "low"
    return level, ("rise" if found.group("rise") else "fall", extreme)


def _checked_ends(check: ClaimCheck, ends: tuple[tuple[str, str], ...]) -> ClaimCheck:
    """A change checked, then the values the claim says it went from and to.

    Each end must agree with the answer's first or last value. A change that
    agrees with ends that do not fails; a change with no first and last value
    in its answer cannot show its ends, so it is not verified.
    """
    if not (check.verified or check.reason == "failed_check"):
        return check
    figures = dict(check.answer.figures) if check.answer is not None else {}
    notes, wrong = [], []
    for name, written in ends:
        value = figures.get(name)
        found = _candidates(written, [])
        if value is None or not found:
            if check.verified:
                return replace(
                    check, status="unverified", reason="missing_context",
                    message=f"The change agrees, but ADA cannot show the values the claim gives for its ends "
                    f"({_quoted([end for _, end in ends])}): {check.message}",
                    clarification="Check each value as its own claim.",
                )
            return check
        item = found[0]
        stated = claimed_number(written[item.qualifier_start:], item.match, item.qualifier, None)
        side, given = ("first" if name == "previous" else "last"), _plain(float(value))
        if stated.holds(float(value)):
            notes.append(f"The claim's {side} value, {written}, agrees with the data's {given}.")
        else:
            wrong.append(f"the claim's {side} value is {written}, and the data give {given}")
    if wrong and check.verified:
        return replace(
            check, status="unverified", reason="failed_check",
            message=f"The change agrees ({check.message}), but {'; '.join(wrong)}.",
            assumptions=(*check.assumptions, *notes), clarification=None,
        )
    if wrong:
        return replace(check, message=f"{check.message} Also, {'; '.join(wrong)}.",
                       assumptions=(*check.assumptions, *notes))
    return replace(check, assumptions=(*check.assumptions, *notes))


def _ends_hold(check: ClaimCheck, ends: tuple[tuple[str, str], ...]) -> bool:
    """Whether the answer's first and last values agree with each end the claim gives."""
    figures = dict(check.answer.figures) if check.answer is not None else {}
    for name, written in ends:
        found = _candidates(written, [])
        if figures.get(name) is None or not found:
            return False
        item = found[0]
        if not claimed_number(written[item.qualifier_start:], item.match, item.qualifier, None).holds(
            float(figures[name])
        ):
            return False
    return True


def _counted_ends(check: ClaimCheck, ends: tuple[tuple[str, str], ...], text: str) -> str | None:
    """The plural noun the ends of a change count ("147.7 billion pounds"), when the change was read as one in
    an average the claim does not name; None otherwise, or when an end is money or a unit of measure."""
    answer = check.answer
    if answer is None or answer.plan.intent != "growth" or answer.plan.aggregation not in ("mean", "median"):
        return None
    if AVERAGING.search(text):
        return None
    for _, written in ends:
        found = _candidates(written, [])
        noun = re.search(r"([A-Za-z]+)\s*$", written)
        if not found or not noun:
            continue
        word = noun.group(1)
        item = found[0]
        stated = claimed_number(written[item.qualifier_start:], item.match, item.qualifier, word)
        plural = word.lower().endswith("s") or word.lower() in IRREGULAR_PLURALS
        if plural and not stated.percent and not _claim_unit(stated) and word.lower() not in MONEY_WORDS:
            return word
    return None


def _checked_dates(check: ClaimCheck, analysis: Any) -> ClaimCheck:
    """A verdict that rests on a year or a day, made again with the table's other date columns.

    "Ward 1 recorded 15 homicides in 2024" counts 14 by START_DATE and 15 by
    REPORT_DAT. When another date column turns the verdict, the claim does not
    say which date it means, and neither verdict stands.
    """
    from pandas.api.types import is_datetime64_any_dtype  # noqa: PLC0415

    from .nlq import execute_plan  # noqa: PLC0415

    decided = check.verified or check.reason == "failed_check"
    if not decided or check.answer is None or check.claimed is None or check.compared is None:
        return check
    plan, roles, frame = check.answer.plan, analysis.roles, analysis.frame
    date = roles.date
    conditions = [item for item in plan.conditions if isinstance(item, Condition)]
    if date is None or not (plan.year or plan.month or any(item.column == date for item in conditions)):
        return check
    others = [str(column) for column in frame.columns
              if str(column) != date and is_datetime64_any_dtype(frame[column])]
    for other in others:
        swapped = replace(plan, conditions=tuple(
            replace(item, column=other) if isinstance(item, Condition) and item.column == date else item
            for item in plan.conditions
        ))
        try:
            answer = execute_plan(swapped, frame, replace(roles, date=other))
        except (KeyError, ValueError):
            continue
        value, _ = _comparable(answer, check.claimed)
        if value is None:
            continue
        if plan.intent == "growth" and _says_fall(check.claim, check.claimed):
            value = -value
        if check.claimed.holds(value) == check.verified:
            continue
        alternatives = (
            Alternative(f"date={date}", check.question or "", check.compared, check.verified,
                        check.answer.calculation),
            Alternative(f"date={other}", check.question or "", value, not check.verified, answer.calculation),
        )
        return replace(
            check, status="unverified", reason="ambiguous_intent",
            message=f"The verdict rests on which date the claim means: by {date} the data give "
                    f"{_plain(check.compared)}, by {other} {_plain(value)}. {check.message}",
            alternatives=alternatives, clarification=f"Is the date {date} or {other}?",
        )
    return check


def _yearly(analysis: Any, plan: Any) -> tuple[Any, str] | None:
    """The answer's calculation made for each year, with the plan's other filters, as values by year ("2012"),
    and what each value is ("row count", "sum(count)"); None without a date or year to group by, or for a
    count of a share of the rows."""
    from .nlq import _apply_filters, frequency_column  # noqa: PLC0415

    if analysis.roles.date is None or plan.counted or (plan.measure is None and plan.aggregation != "count"):
        return None
    rows, _ = _apply_filters(analysis.frame, replace(plan, year=None), analysis.roles)
    dates = rows[analysis.roles.date]
    years = dates.dt.year if hasattr(dates, "dt") else dates
    grouped = rows.groupby(years.astype("Int64").astype(str))
    if plan.measure is None:
        # The same count the answer makes: distinct values, the rows' frequencies, or the rows.
        weight = plan.weight or frequency_column(analysis.frame)
        if plan.count_column:
            return grouped[plan.count_column].nunique(), f"count distinct {plan.count_column}"
        if weight:
            return grouped[weight].sum(), f"sum({weight})"
        return grouped.size(), "row count"
    return grouped[plan.measure].agg(plan.aggregation).dropna(), f"{plan.aggregation}({plan.measure})"


def _checked_direction(
    check: ClaimCheck, analysis: Any, direction: str, extreme: str | None = None
) -> ClaimCheck:
    """A level a change reached, checked, then the change itself against the year before.

    "Fox rescues rose to 61 in 2020" holds when 2020 gives 61 and 2019 gives
    less. Without one year to compare with the year before, the change cannot
    be shown, so the claim is not verified. A record ("fell to a record low of
    0.72 in 2023") holds only when no year up to it is lower, or higher.
    """
    if not (check.verified or check.reason == "failed_check") or check.answer is None:
        return check
    plan = check.answer.plan
    one_year = plan.year is not None and plan.month is None and not plan.dimension
    yearly = _yearly(analysis, plan) if one_year and plan.intent in ("aggregate", "count") else None
    year, before = (str(plan.year), str(plan.year - 1)) if one_year else ("", "")
    if yearly is None or year not in yearly[0].index or before not in yearly[0].index:
        # A level with no year to set against the one before is not the level the claim means either.
        return replace(
            check, status="unverified", reason="missing_context",
            message=f"The claim gives the level a {direction} reached, and ADA cannot set it against the "
            f"year before, so neither is shown: {check.message}",
            clarification="In which year did it reach that level, and from what?",
        )
    by_year, how = yearly
    latest, previous = float(by_year[year]), float(by_year[before])
    moved = latest > previous if direction == "rise" else latest < previous
    said = "rose" if direction == "rise" else "fell"
    note = f"{how} was {_plain(previous)} in {before} and {_plain(latest)} in {year}."
    # A record is set against the years up to it: 0.78 in 2022 was a record low then, whatever came after.
    upto = by_year[[label for label in by_year.index if label <= year]]
    record = (upto.idxmin() if extreme == "low" else upto.idxmax()) if extreme else year
    if moved and (record == year or upto[record] == latest):
        kept = [f"The claim says it {said} to it: {note}"]
        if extreme:
            kept.append(f"{year} gives the {'lowest' if extreme == 'low' else 'highest'} {how} of the "
                        f"{len(upto)} years up to it.")
        return replace(check, assumptions=(*check.assumptions, *kept))
    if moved:
        wrong = f"the record {extreme} up to {year} is in {record}, at {_plain(float(upto[record]))}"
    else:
        wrong = f"the claim says it {said}, and {note.rstrip('.')}"
    if check.verified:
        return replace(check, status="unverified", reason="failed_check",
                       message=f"The level agrees ({check.message}), but {wrong}.", clarification=None)
    return replace(check, message=f"{check.message} Also, {wrong}.")


# A claim that a measure was at its highest for a group: "Cocaine use peaked at 4.9% among 20-year-olds".
PEAK_VERB = re.compile(
    r"\b(?:peaked|peaks|hit\s+(?:its|their|a)\s+peak|reached\s+(?:its|their|a)\s+peak)(?P<at>\s+at)?\b",
    re.IGNORECASE,
)


# "California had more traffic deaths than any other state in 1988, with 5,390": a peak said as a lead.
MOST_LEAD = re.compile(
    r"^\s*(?P<group>[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3})\s+(?:had|recorded|saw|reported|counted)\s+"
    r"(?:more\s+(?P<more>.+?)(?P<year>\s+in\s+\d{4})?\s+than\s+(?:in\s+)?any\s+other\s+[a-z]+|"
    r"the\s+(?:most|highest|largest)\s+(?P<most>.+?))"
    r"(?P<when>\s+(?:in|on|during)\s+[^,]+)?,\s*(?:with\s+|at\s+)?(?P<value>.+?)\s*[.!]?\s*$",
)


# Nouns for periods, which a peak "than any other year" is over.
TIME_NOUNS = frozenset({
    "year", "years", "month", "months", "day", "days", "week", "weeks", "season", "seasons", "decade",
    "decades", "quarter", "quarters",
})
# "Chorleywood had the best education score of any town in England, at 9.4": a peak said with "of any".
_PLACE = r"[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,2}"
BEST_OF_ANY = re.compile(
    rf"^\s*(?P<group>[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){{0,3}})(?:\s+(?:in|of)\s+{_PLACE})?"
    r"\s+(?:had|has|recorded|saw|scored|got|won)\s+the\s+(?:most|highest|largest|biggest|greatest|best)\s+"
    rf"(?P<what>.+?)\s+(?:of|among)\s+(?:any|all(?:\s+the)?)\s+[a-z]+(?:\s+in\s+{_PLACE})?"
    r",\s*(?:with\s+|at\s+)?(?P<value>.+?)\s*[.!]?\s*$",
)
# Words that rank a group against all the others: "the best of any town", "more than in any other month".
SUPERLATIVE = re.compile(
    r"\b(?:most|highest|largest|biggest|greatest|best|lowest|smallest|fewest|least|worst|wettest|driest|"
    r"hottest|coldest|warmest|sunniest|busiest|longest|shortest|tallest|deepest|fastest|slowest|oldest|"
    r"youngest|strongest|weakest|more|fewer|less)\b",
    re.IGNORECASE,
)
AGAINST_ALL = re.compile(
    r"\b(?:of|among)\s+(?:any|all)\b|\bthan\s+(?:in\s+)?any\b|\bon\s+record\b|\bever\b", re.IGNORECASE
)


def _ranked(answer: QueryAnswer) -> bool:
    """Whether the answer itself ranks the groups, so a claim that one leads them all is checked."""
    plan = answer.plan
    return plan.intent == "rank" or bool(plan.top_n) or (
        plan.aggregation in ("max", "min") and not plan.filters and not plan.conditions
    )


# "Vesuvius had its busiest year on record in 2024, with 1,315 earthquakes": a period's count at its peak.
BUSIEST_PERIOD = re.compile(
    r"^\s*(?P<subject>.+?)\s+had\s+(?:its|their|the)\s+busiest\s+(?:year|month|day|week)(?:\s+on\s+record|"
    r"\s+ever|\s+yet)?\s+(?P<when>(?:in|on)\s+[^,]+),\s*with\s+(?P<number>[^A-Za-z]+?)\s+(?P<noun>[A-Za-z]+)"
    r"\s*[.!]?\s*$",
    re.IGNORECASE,
)


def _most_as_peak(text: str) -> str:
    """"G had more X than any other D, with V" as "X in G peaked at V", which is checked as a peak."""
    busiest = BUSIEST_PERIOD.match(text)
    if busiest and NUMBER_PATTERN.search(busiest.group("number")):
        subject, when, noun = busiest.group("subject"), busiest.group("when"), busiest.group("noun")
        return f"The number of {noun} at {subject} {when} peaked at {busiest.group('number').strip()}."
    found = MOST_LEAD.match(text)
    if found is not None and NUMBER_PATTERN.search(found.group("value")):
        what = found.group("more") or found.group("most")
        when = found.group("when") or found.group("year") or ""
        return f"{what} in {found.group('group')}{when} peaked at {found.group('value')}."
    found = BEST_OF_ANY.match(text)
    if found is not None and NUMBER_PATTERN.search(found.group("value")):
        return f"{found.group('what')} in {found.group('group')} peaked at {found.group('value')}."
    return text


def _without_peak(text: str) -> str:
    """The claim with "peaked" read as "was": the value for the group named, whose peak is checked after."""
    text = _most_as_peak(text)
    found = PEAK_VERB.search(text)
    if found is None:
        return text
    after = text[found.end():]
    joined = "was" if not found.group("at") or NUMBER_PATTERN.match(after.lstrip()) or re.match(
        r"\s*(?:about|nearly|almost|roughly|around|just|over|under|more\s+than|less\s+than)\b", after,
        re.IGNORECASE,
    ) else "was at"
    return f"{text[:found.start()]}{joined}{after}"


def _checked_peak(check: ClaimCheck, analysis: Any, text: str = "") -> ClaimCheck:
    """A value for a group, checked to be the highest of its column's groups too.

    The answer must keep one value of one column ("age = 20"); the same
    calculation is made for every value of that column, with the answer's other
    filters, and the claim holds only if the group named gives the highest. A
    value that disagrees for that group fails as it is. Without one group, the
    value checked is not the peak, so neither verdict stands.
    """
    from .nlq import _apply_filters  # noqa: PLC0415

    answer = check.answer
    plan = answer.plan if answer is not None else None
    # A filter on a column with one value in the whole table ("area = Mount Vesuvius") is no group.
    groups = [
        item for item in plan.filters
        if len(item.values) == 1 and analysis.frame[item.column].nunique(dropna=True) > 1
    ] if plan else []
    combined = plan is not None and plan.measure is not None and plan.aggregation in (
        "mean", "sum", "median", "max", "min")
    # "Rain at Oxford peaked at 984 mm in 2012" peaks over the years: the year comes after the peak.
    # "more deaths than any other state in 1988" names the groups it peaks over.
    rewritten = _most_as_peak(text)
    peak = PEAK_VERB.search(rewritten)
    against = re.search(
        r"\b(?:than\s+(?:in\s+)?any\s+other|of\s+any|among\s+all(?:\s+the)?)\s+([a-z]+)", text
    )
    over_groups = bool(against) and against.group(1) not in TIME_NOUNS
    one_year = plan is not None and plan.year is not None and plan.month is None
    in_year = one_year and not over_groups and bool(
        peak and re.search(rf"\b{plan.year}\b", rewritten[peak.end():]) or against
    )
    yearly = one_year and analysis.roles.date is not None and (
        (plan.measure is None and plan.aggregation == "count") or (combined and in_year)
    )
    if plan is not None and plan.year is not None and groups and not yearly and combined and not over_groups:
        # "Rain at Oxford in 2012 peaked at 984 mm": over the years, or over the stations that year?
        return replace(
            check, status="unverified", reason="missing_context",
            message=f"The claim names a peak, and ADA could not tell whether it is over the years or over "
                    f"{groups[0].column}: {check.message}",
            clarification=f"Is it the highest of the years, or of the {groups[0].column} values in "
                          f"{plan.year}?",
        )
    if plan is None or not (len(groups) == 1 or yearly) or (not yearly and not combined):
        return replace(
            check, status="unverified", reason="missing_context",
            message=f"The claim names a peak, and ADA could not tell for which group: {check.message}",
            clarification="Over which groups is it the highest, and for which one?",
        )
    if check.reason == "failed_check":
        return check
    if yearly:
        # "its busiest year in 2024": the rows of each year are counted.
        found = _yearly(analysis, plan)
        if found is None:
            return replace(
                check, status="unverified", reason="missing_context",
                message=f"The claim names a peak, and ADA could not make the same count for every year: "
                f"{check.message}",
                clarification="Which value peaks: a count of rows, or a share of them?",
            )
        by_group, how = found
        column, named = "year", str(plan.year)
    else:
        group = groups[0]
        others = replace(plan, filters=tuple(item for item in plan.filters if item is not group))
        rows, _ = _apply_filters(analysis.frame, others, analysis.roles)
        by_group = rows.groupby(rows[group.column].astype(str))[plan.measure].agg(plan.aggregation).dropna()
        column, named, how = group.column, str(group.values[0]), f"{plan.aggregation}({plan.measure})"
    if by_group.empty or named not in by_group.index:
        return check
    top = by_group.idxmax()
    if top == named or by_group[named] == by_group.max():
        note = (f"{column} = {named} gives the highest {how} of the {len(by_group)} {column} values "
                f"({_plain(float(by_group.max()))}).")
        return replace(check, assumptions=(*check.assumptions, note))
    return replace(
        check, status="unverified", reason="failed_check",
        message=(f"{check.message} But the peak is not at {column} = {named}: {column} = {top} "
                 f"gives {_plain(float(by_group.max()))}, the highest {how}."),
        clarification=None,
    )


def _share_question(asked: str) -> str:
    """The question asked as a share, with one "of": "share of victims ..." for "of victims ..."."""
    return f"share {asked}" if re.match(r"of\s", asked, re.IGNORECASE) else f"share of {asked}"


def _based(text: str) -> tuple[str, tuple[str, str] | None]:
    """The claim without the base it names, and that base as (count, noun), or None.

    "Of the 4,526 applicants, 38.8% were admitted" and "34 of the 93 cars had no
    airbags" name 4,526 applicants and 93 cars; "The 93 models came from 28
    makers" names 93 models when another number follows.
    """
    if (lead := BASE_LEAD.match(text)):
        return text[lead.end():], (lead.group("count"), lead.group("noun"))
    if (part := PART_OF_BASE.match(text)):
        body = f"{part.group('lead')}{part.group('part')} {part.group('noun')}{text[part.end():]}"
        return body, (part.group("count"), part.group("noun"))
    opening = OPENING_BASE.match(text)
    count = opening and float(opening.group("count").replace(",", ""))
    later = opening and NUMBER_PATTERN.search(text, opening.end())
    year = opening and YEAR_RANGE[0] <= count <= YEAR_RANGE[1] and "," not in opening.group("count")
    if opening and later and not year:
        body = f"{text[:opening.start('count')]}{text[opening.end('count'):].lstrip()}"
        return body, (opening.group("count"), opening.group("noun"))
    shared = SHARE_BASE.search(text)
    if shared and not (
        "," not in shared.group("count") and YEAR_RANGE[0] <= int(shared.group("count")) <= YEAR_RANGE[1]
    ):
        body = f"{text[:shared.start('count')]}{text[shared.end('count'):].lstrip()}"
        return body, (shared.group("count"), shared.group("noun"))
    spanned = SCOPE_BASE.search(text)
    if spanned and "," not in spanned.group("count") and YEAR_RANGE[0] <= int(spanned.group("count")) <= (
        YEAR_RANGE[1]
    ):
        spanned = None
    others = spanned and [
        match for match in NUMBER_PATTERN.finditer(text)
        if not spanned.start("count") <= match.start() < spanned.end()
    ]
    if spanned and others:
        # The span says over which rows the claim holds, all of them: "across the 48 states".
        body = re.sub(r"\s+", " ", f"{text[:spanned.start()]} {text[spanned.end():]}").strip()
        return body, (spanned.group("count"), spanned.group("noun"))
    return text, None


def _checked_base(check: ClaimCheck, count: str, noun: str, frame: Any = None) -> ClaimCheck:
    """A claim that names its base ("Of the 4,526 applicants", "the 93 cars"), with the base checked too.

    When a column is named by the noun ("the 48 states" and a column state),
    the base is its number of distinct values in the rows the answer's
    filters keep. Otherwise it is the share's denominator, or, for any other
    answer, the rows in the answer's scope.
    """
    from .folder import singular  # noqa: PLC0415

    stated = float(count.replace(",", ""))
    named = f"{count} {noun}"
    answer = check.answer
    columns = [
        column for column in (frame.columns if frame is not None else [])
        if normalized_name(str(column)) in (noun.lower(), singular(noun.lower()))
    ]
    if answer is not None and len(columns) == 1:
        rows = frame
        for item in answer.plan.filters:
            rows = rows[rows[item.column].astype(str).isin([str(value) for value in item.values])]
        distinct = int(rows[columns[0]].nunique())
        if distinct == stated:
            note = f"The claim's {named} are the {distinct:,} {columns[0]} values in the answer's rows."
            return replace(check, assumptions=(note, *check.assumptions))
    base = dict(answer.figures).get("denominator") if answer is not None else None
    of, are = "the share is of", "the share is of"
    if base is None and answer is not None and answer.scope is not None:
        base, of, are = float(answer.scope.rows_in_scope), "the answer's scope holds", "rows in scope"
    if base is not None and abs(base - stated) < 0.5:
        note = f"The claim's {named} are the {_plain(base)} {are}."
        return replace(check, assumptions=(note, *check.assumptions))
    found = f"{of} {_plain(base)}" if base is not None else "the answer gives no base to compare"
    return replace(
        check, status="unverified", reason="missing_context",
        message=f"The claim's base is {named}, but {found}; {check.message}",
        clarification=f"Which rows are the {named}?",
    )


def _check_folder(
    folder: Path, text: str, claim: str | None, *, question: str | None, value: str | float | None,
    choices: dict[str, str], given: dict[str, str],
) -> ClaimCheck:
    """Check a claim about a folder against each of its tables, and say which one fits.

    The claim is checked against every readable file. Files whose column the
    claim names by its last word, the kind of quantity it holds, come first.
    When exactly one gives a number for it, that file's check is the result,
    and its assumptions say the file was chosen this way. When several do and they disagree, the claim is
    ambiguous. When none does, the claim may need two files joined: each pair
    of files a link joins (see ``ada.joins``) is checked the same way, and the
    result says which join it used.
    """
    from .folder import scan  # noqa: PLC0415
    from .joins import files_in, folder_choices, join_decision, joined_tables  # noqa: PLC0415

    files = files_in(folder)
    listed = ", ".join(item.name for item in files[:12]) + (" and more" if len(files) > 12 else "")
    folder_source = Source(name=str(folder), path=str(folder))
    if not files:
        return ClaimCheck(text, "unverified", "inaccessible_data", f"{folder} holds no table ADA can read.",
                          source=folder_source)
    fitted: list[tuple[str, ClaimCheck]] = []
    partial: list[tuple[str, ClaimCheck]] = []
    for item in files:
        check = check_claim(item, claim, question=question, value=value,
                            decisions=folder_choices(given, item.name, choices))
        if check.answer is not None and check.compared is not None:
            # A file read without some of the claim's words may lack what another file holds.
            (partial if check.words_left_out else fitted).append((item.name, check))
    if fitted:
        return _pick_fit(fitted, folder, text, folder_source, "file")
    found = scan(folder, answers=given)
    decision = join_decision(folder_choices(given, None, choices))
    joined = []
    for wide in (False, True):
        # Two files first; three or more only when no pair holds what the claim names.
        tables = joined_tables(folder, found, decision.chosen, wide=wide)
        joined += tables
        for table in tables:
            check = check_claim(table.frame, claim, question=question, value=value,
                                decisions=folder_choices(given, table.from_file, choices))
            if check.answer is not None and check.compared is not None and not check.words_left_out:
                answer = replace(check.answer, decisions=(decision, *check.answer.decisions))
                fitted.append((table.name, replace(check, answer=answer, source=table.source(folder),
                                                   assumptions=(table.describe(), *check.assumptions))))
        if fitted:
            return _pick_fit(fitted, folder, text, folder_source, "join")
    if len(partial) == 1:
        # Only one file fits, and only without some of the claim's words; a join read that way adds
        # nothing, and several such files are no evidence of which the claim is about.
        return _pick_fit(partial, folder, text, folder_source, "file")
    links = [link for link in found.links if link.kind in ("many-to-one", "one-to-one")]
    joins = "; ".join(f"{link.label('from')} to {link.label('to')}" for link in links[:6])
    message = f"No single file in {folder.name} holds what the claim names ({listed})."
    if joins:
        tried = f"the {counted(len(joined), 'join')} ADA tried ({decision.key}={decision.chosen})"
        message += f" Nor does {tried}; the keys that could join the files are {joins}."
    return ClaimCheck(
        text, "unverified", "missing_context" if joins else "unsupported_analysis", message,
        source=folder_source,
        clarification="Which file is the claim about, or which files hold the numbers it combines?",
    )


def _pick_fit(
    fitted: list[tuple[str, ClaimCheck]], folder: Path, text: str, folder_source: Source, kind: str
) -> ClaimCheck:
    """The check of the one file or join that fits the claim, or ``ambiguous_intent`` when several disagree.

    One whose column the claim names by its last word, the kind of quantity it
    holds, wins over one read from a lesser word: "member deaths" is
    member_deaths in expeditions.csv, not death_height_metres.
    """
    named = [(name, check) for name, check in fitted
             if _names_head(check.question or "", check.answer.plan.measure or "")]
    lesser = [name for name, _ in fitted if named and name not in {found for found, _ in named}]
    fitted = named or fitted
    if len(fitted) == 1:
        name, check = fitted[0]
        if kind == "join":
            return check
        chosen = f"Checked against {name}, the only file in {folder.name} whose columns fit the claim."
        if lesser:
            chosen = (
                f"Checked against {name}, whose {check.answer.plan.measure} the claim names by what it "
                f"measures; {list_names(lesser)} fit only by a lesser word."
            )
        return replace(check, assumptions=(chosen, *check.assumptions))
    values = {round(check.compared, 12) for _, check in fitted}
    if len(values) == 1 and all(check.verified for _, check in fitted):
        name, check = fitted[0]
        names = [found for found, _ in fitted]
        each = f"{names[0]} and {names[1]} both give" if len(names) == 2 else f"{', '.join(names)} all give"
        same = f"{each} {_plain(check.compared)}."
        return replace(check, assumptions=(same, *check.assumptions))
    alternatives = tuple(
        Alternative(name, check.question or "", check.compared, check.agreement == "match",
                    check.answer.calculation if check.answer else "")
        for name, check in fitted
    )
    shown = "; ".join(f"{item.reading} gives {_plain(item.value)}" for item in alternatives
                      if item.value is not None)
    names = [item.reading for item in alternatives]
    return ClaimCheck(
        text, "unverified", "ambiguous_intent",
        f"The claim fits more than one {kind} in {folder.name}, and they give different answers: {shown}.",
        claimed=fitted[0][1].claimed, question=fitted[0][1].question, alternatives=alternatives,
        source=folder_source,
        clarification=f"Which {kind} is the claim about: {', '.join(names[:-1])} or {names[-1]}?",
    )


def _otherwise(
    analysis: Any, asked: str, number: ClaimedNumber, answer: QueryAnswer, choices: dict[str, str],
    factor: float,
) -> tuple[Decision, str, QueryAnswer, float] | None:
    """Another option of a decision the answer took by default, when it gives a number the claim stands for.

    ``factor`` turns the answer's number into the claim's unit, as it did for
    the default. A decision the caller chose is not second-guessed.
    """
    from .api import Refusal  # noqa: PLC0415

    for decision in answer.decisions:
        if decision.key in choices:
            continue
        for option in decision.options:
            if option == decision.chosen:
                continue
            other = analysis.ask(asked, decisions={**choices, decision.key: option})
            if isinstance(other, Refusal):
                continue
            computed, _ = _comparable(other, number)
            if computed is not None and number.holds(computed * factor):
                return decision, option, other, computed * factor
    return None


# Words that say a claim's number is an average: "averaged", "per", "typical".
AVERAGING = re.compile(
    r"\b(?:average|averages|averaged|averaging|mean|median|typical|per|each)\b", re.IGNORECASE
)
# Words that say how a claim's rows combine: "averaged", "in total".
AGGREGATION_SAID = re.compile(
    r"\b(?:average|averages|averaged|averaging|mean|median|typical|typically|total|totals|totalled|totaled|"
    r"sum|combined|altogether|in\s+all|per|each|a\s+year|a\s+month|a\s+day|annual|monthly|daily)\b",
    re.IGNORECASE,
)


def _typical_other(
    analysis: Any, text: str, asked: str, number: ClaimedNumber, answer: QueryAnswer, choices: dict[str, str],
    factor: float,
) -> tuple[QueryAnswer, float, bool] | None:
    """For a "typical" value read as a mean (or a median), the other one and whether it agrees."""
    from .api import Refusal  # noqa: PLC0415

    plan = answer.plan
    said = re.search(r"\btypical(?:ly)?\b", text, re.IGNORECASE)
    if not said or not plan.measure or plan.aggregation not in ("mean", "median"):
        return None
    word = "median" if plan.aggregation == "mean" else "average"
    bare = re.sub(r"\b(?:typical|typically|average|mean|median)\b", " ", asked, flags=re.IGNORECASE)
    other = analysis.ask(f"{word} {bare}", decisions=choices)
    if isinstance(other, Refusal) or other.plan.measure != plan.measure or other.plan.aggregation == (
        plan.aggregation
    ):
        return None
    value, _ = _comparable(other, number)
    if value is None:
        return None
    return other, value * factor, number.holds(value * factor)


def _other_aggregation(
    analysis: Any, text: str, asked: str, number: ClaimedNumber, answer: QueryAnswer, choices: dict[str, str],
    factor: float, used: int | None,
) -> tuple[str, QueryAnswer, float] | None:
    """The total (or the mean) over the rows, when the claim says neither and that one agrees with it."""
    from .api import Refusal  # noqa: PLC0415

    plan = answer.plan
    if AGGREGATION_SAID.search(text) or not plan.measure or plan.aggregation not in ("mean", "sum") or (
        not used or used < 2
    ):
        return None
    word = "total" if plan.aggregation == "mean" else "average"
    other = analysis.ask(f"{word} {asked}", decisions=choices)
    if isinstance(other, Refusal) or other.plan.measure != plan.measure or other.plan.aggregation == (
        plan.aggregation
    ):
        return None
    value, _ = _comparable(other, number)
    if value is None or not number.holds(value * factor):
        return None
    return word, other, value * factor


# Names of a column that holds a count on each row: "Number" babies of a name, "n" cases in a cell.
TALLY_NAMES = frozenset({"number", "num", "count", "counts", "n", "freq", "frequency", "total", "cases"})


def _tallied(analysis: Any, answer: QueryAnswer, number: ClaimedNumber) -> tuple[str, float, str] | None:
    """The total of the one column of counts over the rows a count of rows kept, when it agrees with the
    claim: the claim may mean the things those rows count."""
    from .nlq import _apply_filters  # noqa: PLC0415

    plan = answer.plan
    if plan.aggregation != "count" or plan.measure or plan.count_column or plan.weight or number.percent:
        return None
    column = _tally_column(analysis.frame)
    if column is None:
        return None
    rows, _ = _apply_filters(analysis.frame, plan, analysis.roles)
    value = float(rows[column].sum())
    if not number.holds(value):
        return None
    return column, value, f"sum({column}) over the {len(rows):,} rows kept"


def _tally_column(frame: Any) -> str | None:
    """The one column named for a count ("count", "n", "cases") that holds whole numbers of zero or more."""
    found = []
    for column in frame.columns:
        values = frame[column]
        if normalized_name(str(column)) not in TALLY_NAMES or values.dtype.kind not in "iuf":
            continue
        numbers = values.dropna()
        if len(numbers) and bool((numbers >= 0).all()) and bool((numbers % 1 == 0).all()):
            found.append(str(column))
    return found[0] if len(found) == 1 else None


# Name words of a column that holds money, for a claim whose number carries a currency symbol.
MONEY_WORDS = frozenset({
    "revenue", "turnover", "sales", "profit", "amount", "income", "spend", "spending", "cost", "expense",
    "price", "balance", "payment", "paid", "fare", "salary", "wage", "wages", "fee", "fees", "charge",
    "value", "earnings", "budget", "tax", "usd", "eur", "gbp", "brl",
})


def _money_column(column: str, values: Any) -> bool:
    """Whether a column holds money: a money word in its name, and values that are not small whole codes."""
    if not MONEY_WORDS & set(normalized_name(str(column)).split()):
        return False
    numbers = values.dropna()
    try:
        whole = bool((numbers % 1 == 0).all())
    except TypeError:
        return False
    return not (whole and numbers.max() <= 100)


def _names_anything(frame: Any, word: str) -> bool:
    """Whether ``word`` (or its singular) is part of a column's name or one of a text column's values."""
    from .folder import singular  # noqa: PLC0415

    forms = {word.lower(), singular(word.lower())}
    if any(forms & set(normalized_name(str(column)).split()) for column in frame.columns):
        return True
    for column in frame.select_dtypes(exclude=["number", "bool", "datetime"]).columns:
        values = frame[column].dropna().astype(str).str.lower().unique()
        if any(value in forms for value in values):
            return True
    return False


def _safe_without(
    left_out: list[str], answer: QueryAnswer, number: ClaimedNumber, used: int | None,
    slope: tuple[str, str, str, bool] | None,
) -> bool:
    """Whether a claim read without words that name nothing can still be verified.

    A bound ("more than 50") holds for far more rows than the claim means, and
    an average, a share, a slope over time or an extreme over every row can be
    near the one over the rows the words pick, so only a lookup of one row, a
    total or a count, or a slope whose left-out words only describe the column
    after "per" ("day of sleep deprivation") may verify.
    """
    if math.isinf(number.low) or math.isinf(number.high):
        return False
    if answer.plan.intent == "trend":
        # A slope over the rows the words pick can differ in size and sign from the slope over every row.
        return False
    if number.percent:
        # A share of every row can be near the share of the rows the words pick.
        return used == 1
    if slope:
        return all(phrase.lower() in slope[1].lower() for phrase in left_out)
    return used == 1 or answer.plan.aggregation in ("sum", "count")


# Units a price or an amount is given in: "a gallon", "per barrel".
MEASURE_UNIT_NOUNS = frozenset({
    "gallon", "gallons", "litre", "litres", "liter", "liters", "pound", "pounds", "kilo", "kilos",
    "kilogram", "kilograms", "barrel", "barrels", "tonne", "tonnes", "ton", "tons", "bushel", "bushels",
    "ounce", "ounces", "dozen", "a", "per",
})
# Words of a folder's name that say nothing about what a table holds.
GENERIC_NAME_WORDS = frozenset({
    "data", "dataset", "datasets", "file", "files", "table", "tables", "csv", "tsv", "xlsx", "raw", "clean",
    "cleaned", "tmp", "temp", "download", "downloads", "export", "final", "new", "old", "copy", "sheet",
    "input", "output", "the", "and", "for",
})


def _table_name_words(source: Source | None) -> set[str]:
    """The words of the table's file and folder names: "road" and "victims" for road-victims/Accident.csv."""
    if source is None or not source.path:
        return set()
    path = Path(source.path)
    words = set(_words(normalized_name(path.stem))) | set(_words(normalized_name(path.parent.name)))
    return {word for word in words if len(word) > 2 and not word.isdigit() and word not in GENERIC_NAME_WORDS}


def _place_scope(phrase: str, frame: Any) -> bool:
    """Whether a phrase only names countries or US states ("Ireland", "England and Wales", "New Zealand's")
    and no column of the table holds countries or states, so the place is the whole table's."""
    from .nlq import DEMONYMS, US_STATES  # noqa: PLC0415

    places = {name for names in DEMONYMS.values() for name in names} | set(DEMONYMS) | set(US_STATES) | {
        "new zealand", "northern ireland", "great britain",
    }
    text = re.sub(r"['’]s\b", "", phrase.lower())
    parts = [part.strip() for part in re.split(r"\band\b|,", text) if part.strip()]
    if not parts or not all(part in places for part in parts):
        return False
    codes = set(US_STATES.values())
    for column in frame.columns:
        series = frame[column]
        if series.dtype.kind in "biufcmM":
            continue
        values = {str(value).strip().lower() for value in series.dropna().unique()[:5_000]}
        if len(values & places) >= 3 or len(values & codes) >= 10:
            return False
    return True


def _names_table(phrase: str, words: set[str]) -> bool:
    """Whether each word of a phrase is in the table's name, as "Marvel's" is in marvel-wikia-data.csv."""
    from .folder import singular  # noqa: PLC0415

    parts = _words(re.sub(r"['’]s\b", "", phrase))
    # "died" names Accidental_Drug_Related_Deaths.csv: a word for dying names a table of deaths.
    dying = OUTCOME_WORDS["died"] if words & OUTCOME_WORDS["died"] else frozenset()
    return bool(parts) and all(
        part in words or singular(part) in words or f"{part}s" in words or part in dying for part in parts
    )


def _key_filter(frame: Any, answer: QueryAnswer) -> str | None:
    """A filter column of the answer whose values are each on one row only: a key, such as County."""
    for item in answer.plan.filters:
        values = frame[item.column].dropna() if item.column in frame.columns else None
        if values is not None and len(values) > 1 and values.is_unique:
            return item.column
    return None


# Words in a column's name that say each row holds a statistic of a group, which do not combine.
SUMMARY_WORDS = {"median": "median", "mean": "mean", "average": "average", "avg": "average"}


def _summary_word(column: str | None) -> str | None:
    """The statistic a column's name says each row holds: "median" for median_weekly_earn."""
    words = normalized_name(str(column)).split() if column else []
    return next((SUMMARY_WORDS[word] for word in words if word in SUMMARY_WORDS), None)


# Fewest and most distinct values of a column searched for other spellings of a value a filter picked.
SPELLING_MIN_VALUES, MAX_SPELLING_VALUES = 50, 20_000


def _other_spellings(frame: Any, answer: QueryAnswer) -> tuple[str, str, list[str]] | None:
    """A value a filter picked that the same column also writes another way, with those ways.

    Another way is a longer value holding all its words in order ("Sen. John
    McCain" for "John McCain"); a filter already matches the value in any
    capitals. A value of one word counts only where it starts the longer value
    or an item of a list in it ("Norovirus Genogroup II" and "Clostridium
    difficile;Norovirus unknown" for "Norovirus"), as "Smith" is not "John
    Smith". Only a column of more than SPELLING_MIN_VALUES values is searched,
    as a list of names is: in a column of a few categories, "Wheelchair Men" is
    a category of its own, not "Men" written another way. Returns (column,
    value, the other ways) for the first such value.
    """
    picked = [item for item in (*answer.plan.filters, *answer.plan.counted) if isinstance(item, ValueFilter)]
    for item in picked:
        if item.column not in frame.columns:
            continue
        uniques = frame[item.column].dropna().astype(str).unique()
        if not SPELLING_MIN_VALUES < len(uniques) <= MAX_SPELLING_VALUES:
            continue
        for value in item.values:
            words = re.findall(r"[a-z0-9]+", value.lower())
            if not words:
                continue
            # "non-USA" is not another way to write "USA".
            pattern = re.compile(r"(?<!non-)(?<!non )(?<!not )(?<!no )\b" + r"\W+".join(map(re.escape, words))
                                 + r"\b")
            if len(words) == 1:
                pattern = re.compile(rf"(?:^|[;,/|]\s*){re.escape(words[0])}\b")
            lowered = value.lower()
            others = [
                other for other in uniques
                if other.lower() != lowered and other not in item.values and pattern.search(other.lower())
            ]
            if others:
                # The ways that start with the value come first: "Norovirus Genogroup II" before
                # "Adenovirus;Norovirus Genogroup I".
                return str(item.column), value, sorted(others, key=lambda other: (
                    not other.lower().startswith(lowered), other))
    return None


def _names_rows(phrases: Sequence[str], row_noun: str | None) -> bool:
    """Whether any of the phrases holds the file's own noun for its rows ("characters" in marvel.csv)."""
    from .folder import singular  # noqa: PLC0415

    if not row_noun:
        return False
    forms = {row_noun.lower(), singular(row_noun.lower())}
    words = {form for phrase in phrases for word in _words(phrase)
             for form in (word.lower(), singular(word.lower()))}
    return bool(forms & words)


def _quoted(phrases: Sequence[str]) -> str:
    """"a", "a" and "b", or "a", "b" and "c", each in quotes."""
    quoted = [f'"{phrase}"' for phrase in phrases]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _unknown_phrases(message: str) -> list[str]:
    """The words a refusal says name nothing in the table: "lost" and "forest"."""
    found = re.search(r"found no column or value called (.+?) in this table", message)
    return re.findall(r'"([^"]+)"', found.group(1)) if found else []


def _check(
    analysis: Any, text: str, asked: str, number: ClaimedNumber, source: Source | None,
    choices: dict[str, str] | None = None, left: tuple[str, ...] = (),
) -> ClaimCheck:
    from .api import Refusal  # noqa: PLC0415
    from .nlq import _kind_value  # noqa: PLC0415

    choices = choices or {}
    noun = (number.unit or "").split()
    unsaid = len(noun) == 1 and not re.search(rf"\b{re.escape(noun[0])}\b", asked, re.IGNORECASE)
    if unsaid and not _claim_unit(number) and _kind_value(noun[0].lower(), analysis.frame):
        # "565 boys were named Jack": the boys are Sex = Boy, so the noun picks rows as well as naming them.
        asked = f"{asked} {noun[0]}"
    answer = analysis.ask(asked, decisions=choices)
    table_words = _table_name_words(source)
    for _ in range(3):
        spread = re.search(r'"([^"]+)" (?:is a value of|matches several)', answer.message) if isinstance(
            answer, Refusal) else None
        if not spread or not _names_table(spread.group(1), table_words) or any(
            _names_anything(analysis.frame, word) for word in _words(re.sub(r"['’]s\b", "", spread.group(1)))
        ):
            break
        # "the shelter" in long-beach-shelter/longbeach.csv is only part of "at shelter" and "shelter,
        # neuter, return": words of the table's name that no value is name the table.
        asked = re.sub(r"\s+", " ", re.sub(rf"(?<!\w){re.escape(spread.group(1))}(?!\w)", " ", asked)).strip()
        left = (*left, spread.group(1))
        answer = analysis.ask(asked, decisions=choices)
    alternatives: list[Alternative] = []
    assumptions: list[str] = [f"{LEFT_OUT_NOTE} {_quoted(left)}, which name no column or value in the table."
                              ] if left else []
    counted_rows = re.match(r"how many (\w+) (.+)$", asked)
    if counted_rows and not number.percent and OF_THE.search(text) and (
        isinstance(answer, Refusal) or answer.plan.measure or answer.plan.count_column
    ):
        # "55 of the cats were neutered" read "cats" as n_cats; the cats are the rows when no unit repeats.
        from .compare import repeated_unit  # noqa: PLC0415

        rows = analysis.ask(f"how many rows {counted_rows.group(2)}", decisions=choices)
        if (repeated_unit(analysis.frame, analysis.roles) is None and not isinstance(rows, Refusal)
                and rows.plan.intent == "count" and not rows.plan.measure and not rows.plan.count_column):
            assumptions.append(f'Read "{counted_rows.group(1)}" as the rows of the table, one each.')
            answer, asked = rows, f"how many rows {counted_rows.group(2)}"
        elif not isinstance(answer, Refusal):
            return ClaimCheck(
                text, "unverified", "missing_context",
                f'The claim counts {counted_rows.group(1)}, but ADA read the words as {answer.calculation}, '
                f"and the rows are not one {counted_rows.group(1)} each.",
                claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
                clarification=f"Which column or rows are the {counted_rows.group(1)}?",
            )
    subject = re.match(r"\s*(?:(?:total|average|mean)\s+)?([A-Za-z]+)\s+", asked, re.IGNORECASE)
    if isinstance(answer, Refusal) and subject and subject.group(1).lower() in PEOPLE_NOUNS and (
        not _names_anything(analysis.frame, subject.group(1))
    ):
        # "Customers paid R$16M": a subject that names nothing in the table is every row of it.
        shorter = f"{asked[:subject.start(1)]}{asked[subject.end(1):]}"
        retried = analysis.ask(shorter, decisions=choices)
        assumptions.append(f'Read "{subject.group(1)}" as every row of the table, since no column names '
                           "them.")
        answer, asked = retried, shorter
    from .folder import singular  # noqa: PLC0415

    row_noun = analysis.roles.row_noun
    if isinstance(answer, Refusal) and row_noun:
        # "The average Lego set": a word before the rows' own noun that names nothing only describes them.
        forms = {row_noun, singular(row_noun)}
        modifier = next((match for match in re.finditer(r"\b([A-Za-z]+)\s+(?=([A-Za-z]+)\b)", asked)
                         if match.group(2).lower() in forms and match.group(1)[0].isupper()
                         and not _names_anything(analysis.frame, match.group(1))), None)
        if modifier:
            shorter = f"{asked[:modifier.start(1)]}{asked[modifier.end(2):]}"
            retried = analysis.ask(shorter, decisions=choices)
            if not isinstance(retried, Refusal):
                assumptions.append(f'Read "{modifier.group(1)} {modifier.group(2)}" as every '
                                   f"{singular(row_noun)} of the table, since no column or value holds "
                                   f"{modifier.group(1)}.")
                answer, asked = retried, shorter
    left_out: list[str] = list(left)
    if isinstance(answer, Refusal):
        readings = []
        columns = _candidate_columns(answer.message, list(analysis.frame.columns))
        named = [(column, _substitute(asked, column)) for column in columns]
        by_unit = not named
        for column, rewritten in named or _unit_columns(analysis, asked, number):
            if not rewritten:
                continue
            other = analysis.ask(rewritten, decisions=choices)
            if isinstance(other, Refusal):
                continue
            computed, _ = _comparable(other, number)
            readings.append((column, rewritten, other, computed))
        written = NUMBER_PATTERN.search(number.text)
        money = [reading for reading in readings
                 if reading[3] is not None and _money_column(reading[0], analysis.frame[reading[0]])]
        if written and written.group("currency") and len(readings) > 1 and len(money) == 1:
            column, asked, answer, _ = money[0]
            assumptions.append(f"Read the claim as being about {column}, the only money column it fits, as "
                               f"the number is written in {written.group('currency').strip()}.")
        elif len(readings) == 1:
            column, asked, answer, _ = readings[0]
            assumptions.append(
                f"Read the claim as being about {column}, the only column recorded in a unit of that kind."
                if by_unit else f"Read the claim's words as {column}, the only column the question fits."
            )
        elif readings:
            alternatives = [
                Alternative(column, rewritten, computed, number.holds(computed), other.calculation)
                for column, rewritten, other, computed in readings
            ]
            shown = "; ".join(
                f"{item.reading} gives {_plain(item.value) if item.value is not None else 'no single number'}"
                for item in alternatives
            )
            fits = "The claim's unit fits" if by_unit else "The claim's words fit"
            differ = "they give different answers" if len({a.agrees for a in alternatives}) > 1 else (
                "each agrees with the claim" if alternatives[0].agrees else "none agrees with the claim"
            )
            names = [a.reading for a in alternatives]
            listed = f"{', '.join(names[:-1])} or {names[-1]}"
            message = f"{fits} more than one column, and {differ}: {shown}."
            return ClaimCheck(
                text, "unverified", "ambiguous_intent", message,
                claimed=number, question=asked, alternatives=tuple(alternatives), source=source,
                clarification=f"Which column is meant: {listed}?",
            )
        else:
            # "Brazil lost 2,564,000 hectares of forest to pasture": words that name nothing in the table
            # are left out. The claim is verified only when the rest agrees and the reading is one a
            # left-out word could not change (_safe_without); a mismatch proves nothing.
            unknown = _unknown_phrases(answer.message)
            retried, shorter = None, asked
            if unknown and not any(KEPT_WORDS & set(_words(phrase)) for phrase in unknown):
                for phrase in unknown:
                    shorter = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", " ", shorter, flags=re.IGNORECASE)
                shorter = re.sub(r"\s+", " ", shorter).strip(" .,;:!?")
                retried = analysis.ask(shorter, decisions=choices) if shorter else None
            if retried is None or isinstance(retried, Refusal):
                unsupported = answer.code == "unsupported_phrasing"
                return ClaimCheck(
                    text, "unverified", "unsupported_analysis" if unsupported else "missing_context",
                    f"ADA could not read the claim as a calculation on this table: {answer.message}",
                    claimed=number, question=asked, source=source,
                    clarification="Say which column the number is about, and over which rows (a group, a "
                    "year).",
                )
            left_out = [*left_out, *unknown]
            assumptions.append(
                f"{LEFT_OUT_NOTE} {_quoted(unknown)}, which name no column or value in the table."
            )
            answer, asked = retried, shorter
    unnamed = [
        column for column in (answer.plan.measure, answer.plan.partner)
        if column and not _named(asked, column)
    ]
    tally = _tally_column(analysis.frame) if re.search(r"\bcases\b", asked, re.IGNORECASE) else None
    if tally and unnamed == [tally]:
        # "polio cases peaked at almost 50,000" in a table with a count on each row: the cases are the count.
        unnamed = []
        assumptions.append(f'Read "cases" as {tally}, the one column that holds a count on each row.')
    elif tally and not (_named(asked, tally) or number.percent) and answer.plan.aggregation == "count" and (
        not (answer.plan.measure or answer.plan.count_column or answer.plan.weight)
    ):
        totalled = analysis.ask(f"total {tally} {asked}", decisions=choices)
        if not isinstance(totalled, Refusal) and totalled.plan.measure == tally:
            answer, asked = totalled, f"total {tally} {asked}"
            assumptions.append(f'Read "cases" as {tally}, the one column that holds a count on each row, '
                               "rather than as the rows.")
    if unnamed and number.percent:
        # "32.3% survived" gives a share even with no word for one.
        share = analysis.ask(_share_question(asked), decisions=choices)
        if not isinstance(share, Refusal) and "share_percent" in dict(share.figures):
            answer, asked, unnamed = share, _share_question(asked), []
            assumptions.append("Read the percentage as a share, which the claim names no column for.")
    if unnamed and (unit_read := _one_unit_reading(analysis, asked, number, choices)):
        # "the pines averaged 60.3 feet": the one column of lengths is what the claim measures.
        column, asked, answer = unit_read
        unnamed = []
        assumptions.append(f"Read the claim as being about {column}, the only column recorded in a unit of "
                           f"that kind or named for it, as the number is in {number.unit}.")
    after = re.search(rf"{re.escape(number.text)}\s+([A-Za-z]+)\b", text)
    noun = (number.unit or "").split() or ([after.group(1)] if after else [])
    averaging = AVERAGING.search(text)
    if unnamed and len(noun) == 1 and not _claim_unit(number) and not averaging and (
        noun[0].lower() not in MONEY_WORDS
    ) and (noun[0].lower().endswith("s") or noun[0].lower() in IRREGULAR_PLURALS):
        # "The survey recorded 1,950 captures in 2007": a plural that names no column counts rows. An
        # average ("averaged 18.6 dollars") or an amount of money is never a count of rows.
        rest = re.sub(r"^(?:total|average)\s+", "", asked, flags=re.IGNORECASE)
        tally = analysis.ask(f"how many {noun[0]} {rest}", decisions=choices)
        if not isinstance(tally, Refusal) and tally.plan.intent == "count" and not tally.plan.measure:
            answer, asked, unnamed = tally, f"how many {noun[0]} {rest}", []
            if any(phrase.lower() == noun[0].lower() for phrase in left_out):
                # The noun is what is counted now, not a word left out.
                left_out = [phrase for phrase in left_out if phrase.lower() != noun[0].lower()]
                assumptions = [item for item in assumptions if not item.startswith(LEFT_OUT_NOTE)]
                if left_out:
                    assumptions.append(
                        f"{LEFT_OUT_NOTE} {_quoted(left_out)}, which name no column or value in the table."
                    )
            assumptions.append(f'Read "{number.text}" as a count of rows, as no column names {noun[0]}.')
    changed = CHANGE_WORDS.search(asked) if unnamed and answer.plan.intent == "growth" else None
    subject = re.search(r"([A-Za-z]+)\s*$", asked[:changed.start()]) if changed else None
    row_noun = (analysis.roles.row_noun or "").lower()
    if subject and not averaging and row_noun and subject.group(1).lower() in (row_noun, f"{row_noun}s"):
        # "Fox rescues nearly doubled" in animal_rescues.csv: the file's noun for its rows, before the change,
        # counts them.
        rows_changed = analysis.ask(f"number of {asked}", decisions=choices)
        if not isinstance(rows_changed, Refusal) and rows_changed.plan.intent == "growth" and (
            not rows_changed.plan.measure
        ):
            answer, asked, unnamed = rows_changed, f"number of {asked}", []
            assumptions.append(f"Read the change as one in the number of {subject.group(1)}, the file's "
                               "rows.")
    indicator = next((item for item in answer.plan.filters
                      if set(normalized_name(str(item.column)).split()) & INDICATOR_WORDS), None)
    if unnamed == [answer.plan.measure] and indicator and normalized_name(str(unnamed[0])) in VALUE_NAMES:
        # "Adult smoking in England" in a long table: Indicator Name = Smoking Prevalence says what Value
        # holds on the rows it keeps.
        unnamed = []
        assumptions.append(f"Read {answer.plan.measure} as the value of {indicator.column} = "
                           f"{', '.join(map(str, indicator.values))}, which the claim names.")
    if unnamed:
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"The claim does not say what the number measures; ADA would have used {unnamed[0]}, the column "
            "it picks when none is named, and a check should not rest on that guess.",
            claimed=number, question=asked, source=source,
            clarification="Which column is the number about?",
        )
    items = [*answer.plan.filters, *answer.plan.conditions]
    if number.percent and answer.plan.intent == "count" and not answer.plan.counted and (
        len(items) > 1 or answer.plan.count_column
    ):
        # "Men were admitted at a rate of 44.5%": the share is of the men, so it is asked as a share;
        # so is "half of the squirrels", which counts distinct squirrels rather than giving a share.
        share = analysis.ask(_share_question(asked), decisions=choices)
        if not isinstance(share, Refusal) and share.plan.counted:
            answer, asked = share, _share_question(asked)
    averaged = AVERAGING.search(text)
    noun = (number.unit or "").split()[-1:] or [""]
    # "0.75 percentage points" or "3 degrees" measure a level; they count nothing.
    measured = noun[0].lower() in ("points", "degrees") or (
        (number.unit or "").lower().startswith("percentage"))
    if answer.plan.aggregation in ("mean", "median") and not averaged and not _claim_unit(number) and (
        noun[0].lower().endswith("s") and not number.percent and not measured
    ):
        # "1,524 Kenyans were recorded as intersex" counts people: a total, never an average.
        total = analysis.ask(f"total {asked}", decisions=choices)
        if isinstance(total, Refusal) or total.plan.aggregation != "sum" or (
            total.value is None and total.plan.intent != "trend"
        ):
            return ClaimCheck(
                text, "unverified", "missing_context",
                f'"{number.text}" counts {noun[0]}, but ADA read the claim as an average '
                f"({answer.calculation}), and found no total to check it against.",
                claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
                clarification="Which column holds the number?",
            )
        assumptions.append(f'Read "{number.text}" as a total, since it counts {noun[0]}.')
        answer, asked = total, f"total {asked}"
    summary = _summary_word(answer.plan.measure)
    used_rows = answer.scope.rows_used if answer.scope is not None else None
    if summary and answer.plan.intent == "aggregate" and used_rows is not None and used_rows > 1:
        # "women's median weekly earnings were $894": each row holds a group's median; averaging the
        # medians of 32 groups of women is not the median of the women.
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"{answer.plan.measure} holds a {summary} for each row's group, and {used_rows:,} rows match; "
            f"combining their {summary}s ({answer.calculation}) does not give the {summary} of all of them, "
            "so the claim is about one row the question does not pick out.",
            claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
            clarification="Which single group (row) is the number about?",
        )
    spellings = _other_spellings(analysis.frame, answer)
    if spellings:
        # "John McCain" is 5 rows and "Sen. John McCain" 6 more: the claim may count both.
        column, value, others = spellings
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"{column} also writes {value!r} as {_quoted(others[:3])}"
            + (f" and {len(others) - 3} others" if len(others) > 3 else "")
            + f"; the answer keeps only {value!r}, and the claim may mean them all: {answer.answer}",
            claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
            clarification=f"Should {_quoted(others[:3])} count as {value!r}?",
        )
    if number.percent and (answer.plan.intent == "count" or answer.plan.aggregation == "sum") and not (
        {"share_percent", "change_percent"} & set(dict(answer.figures))
    ):
        # A percentage compared with a plain count fails a true claim ("half of the squirrels" with 1,434).
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"The claim gives a percentage, but ADA read the words as a count ({answer.calculation}), not as "
            "a share of a base it could name.",
            claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
            clarification="A percentage of which rows?",
        )
    computed, notes = _comparable(answer, number)
    slope = _slope_claim(text) if answer.plan.intent == "correlation" else None
    measure, sloped = answer.plan.measure, ""
    if slope:
        # The change for each unit of the column after "per" is the slope of the other column on it.
        x = next((column for column in (answer.plan.measure, answer.plan.partner)
                  if column and _named(slope[1], column)), None)
        y = next((column for column in (answer.plan.measure, answer.plan.partner) if column and column != x),
                 None)
        fitted = dict(answer.figures).get(f"slope of {y} on {x}")
        if fitted is None:
            return ClaimCheck(
                text, "unverified", "unsupported_analysis",
                f"The claim states a change per unit of another column, and ADA found no slope for it: "
                f"{answer.answer}",
                claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
                clarification="Which column changes, and per unit of which column?",
            )
        computed, measure, sloped = (-fitted if slope[3] else fitted), y, f"the slope of {y} on {x}, from "
        notes = [f"Compared with the slope of {y} on {x}, {_plain(fitted)} {y} per {x}"
                 + ("; the claim says it falls, so the slope's sign is reversed." if slope[3] else ".")]
    # "about two percentage points a year": the slope is read from the claim with its number in digits.
    worded = _worded_number(text)
    timed = _time_slope(worded[0] if worded else text) if answer.plan.intent == "trend" else None
    if timed:
        # "grew by about 500 million pounds a year": the slope of the yearly values, by least squares and by
        # Theil-Sen; when only one of them agrees, the claim does not say which it means.
        _, _, unit, falling = timed
        found = dict(answer.figures)
        keys = (("least squares", "least_squares_per_year"), ("Theil-Sen", "slope_per_year"))
        fits = [(name, found[key] / TIME_SLOPE_UNITS[unit] * (-1 if falling else 1))
                for name, key in keys if key in found]
        if not fits:
            return ClaimCheck(
                text, "unverified", "unsupported_analysis",
                f"The claim states a change each {unit}, and ADA found no yearly slope for it: "
                f"{answer.answer}",
                claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
                clarification="Which column changes, and over which years?",
            )
        held = [number.holds(value) for _, value in fits]
        if len(set(held)) > 1:
            alternatives = tuple(Alternative(f"{name} slope", asked, value, agrees, answer.calculation)
                                 for (name, value), agrees in zip(fits, held, strict=True))
            shown = " and ".join(f"by {name} {_plain(value)}" for name, value in fits)
            return ClaimCheck(
                text, "unverified", "ambiguous_intent",
                f"The change each {unit} depends on how the line is fitted: {shown}, and only one is in "
                f"{number.interval()}. {answer.answer}",
                claimed=number, question=asked, answer=answer, compared=fits[0][1],
                assumptions=tuple(assumptions), alternatives=alternatives, source=source,
                clarification="Is the change the least-squares slope or the Theil-Sen slope?",
            )
        computed = fits[0][1]
        sloped = f"the least-squares slope per {unit} of the yearly values, from "
        same = f"; the Theil-Sen slope, {_plain(fits[1][1])}, gives the same verdict" if len(fits) > 1 else ""
        notes = [f"Compared with the least-squares slope of the yearly values, {_plain(computed)} per {unit}"
                 + same
                 + ("; the claim says it falls, so the slope's sign is reversed." if falling else ".")]
    figures = dict(answer.figures)
    if answer.plan.intent == "growth" and number.percent and "change_percent" not in figures and (
        figures.get("previous") and "latest" in figures
    ):
        # A rate's change is in points; a change in percent may be of the first value instead.
        previous, latest = float(figures["previous"]), float(figures["latest"])
        relative = (latest - previous) / abs(previous) * 100
        if MULTIPLE.search(text) or BY_FRACTION.search(text):
            # "halved", "fell by a third": a multiple or a fraction is of the first value.
            computed = relative
            notes.append(f"A multiple or a fraction compares the values themselves, so the change is taken "
                         f"in percent of the first value ({_plain(relative)}%), not in points.")
        else:
            alternatives = (
                Alternative("points", asked, computed, False, answer.calculation),
                Alternative("percent of the first value", asked, relative, False,
                            f"({_plain(latest)} − {_plain(previous)}) ÷ {_plain(abs(previous))} × 100"),
            )
            return ClaimCheck(
                text, "unverified", "ambiguous_intent",
                f"The claim gives a change in percent of a rate, which may be in points "
                f"({_plain(computed)}) or in percent of the first value ({_plain(relative)}%): "
                f"{answer.answer}",
                claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions),
                alternatives=alternatives, source=source,
                clarification="Is the change in percentage points, or in percent of the first value?",
            )
    if answer.plan.intent == "growth" and computed is not None and _says_fall(text, number):
        # "fell by about 19%" states the size of a fall: the change is compared with its sign reversed.
        computed = -computed
        notes.append(f"The claim says it fell, so the change ({_plain(-computed)}) is compared by its size.")
    assumptions.extend(notes)
    scope = answer.scope
    if scope is not None:
        assumptions.extend(scope.readings)
        assumptions.extend(f"Left out {_counted(reason, count)}." for reason, count in scope.left_out)
    if number.qualifier:
        assumptions.append(f'Read "{number.qualifier}" as {number.interval()}.')
    claimed_unit, recorded_unit = _claim_unit(number), column_unit(measure)
    key = _key_filter(analysis.frame, answer)
    if key and answer.plan.aggregation == "count" and not measure and not answer.plan.count_column:
        # "Nairobi has 5.2 million residents": County = Nairobi City is one row, and counting it says
        # nothing about the number, which is one of that row's values.
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"ADA read the claim as a count of rows, but {key} names one row each, so the count is 1 "
            "whatever the claim says; the number is likely one of that row's columns, and the claim names "
            "none of them.",
            claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
            clarification="Which column holds the number?",
        )
    if claimed_unit and answer.plan.aggregation == "count" and not measure and (
        step := _row_step(analysis)
    ) and UNITS[claimed_unit] == UNITS[step]:
        # "negative prices for more than 500 hours" in a table of one reading an hour: each row is an hour.
        assumptions.append(f"Read the {number.unit} as rows, since each row of the table is one {step} of "
                           f"{analysis.roles.date}.")
    elif claimed_unit and answer.plan.aggregation == "count" and not measure:
        # A count of rows has no unit; a number in hours or grams is about some column the reading missed.
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"The claim gives the number in {number.unit}, but ADA read it as a count of rows "
            f"({answer.calculation}), which has no unit.",
            claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
            clarification="Which column is the number about?",
        )
    if claimed_unit and recorded_unit and UNITS[claimed_unit][0] != UNITS[recorded_unit][0]:
        return ClaimCheck(
            text, "unverified", "conflicting_evidence",
            f'The claim gives the number in {number.unit}, but {measure} is recorded in {recorded_unit}, '
            f"which measures {UNITS[recorded_unit][0]}, not {UNITS[claimed_unit][0]}.",
            claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
            clarification=f"Is the number about {measure}, or about another column?",
        )
    factor = 1.0
    if claimed_unit and recorded_unit and computed is not None and claimed_unit != recorded_unit:
        factor = UNITS[recorded_unit][1] / UNITS[claimed_unit][1]
        converted = computed * factor
        assumptions.append(
            f"Converted {measure} from {recorded_unit}, the unit in its name, to {claimed_unit}: "
            f"{_plain(computed)} {recorded_unit} is {_plain(converted)} {claimed_unit}."
        )
        computed = converted
    elif number.unit and not number.scale_word and not recorded_unit:
        assumptions.append(f'The unit "{number.unit}" comes from the claim; the check compares numbers only.')
    warnings = list(scope.cautions) if scope is not None else []
    checks: list[Check] = []
    repeated = _repeated_units(analysis, answer)
    if repeated is not None:
        checks.append(repeated)
        if not repeated.passed or "weights each" in repeated.detail:
            warnings.append(repeated.detail)
    if computed is None:
        groups = len(answer.values)
        dated = answer.plan.year is not None or answer.plan.month is not None
        if groups > 1:
            reason, clarification = "missing_context", "Which group is the number about?"
            message = f"The answer gives {groups} numbers, one per group, and the claim does not say which."
        elif dated and not groups:
            reason, clarification = "missing_context", "Which column holds the date or year?"
            message = f"ADA computed no number for it: {answer.answer}"
        else:
            reason, clarification = "unsupported_analysis", "Say which single number the claim is about."
            message = f"The calculation the claim describes gives no single number: {answer.answer}"
        return ClaimCheck(
            text, "unverified", reason, message, claimed=number, question=asked, answer=answer,
            assumptions=tuple(assumptions), warnings=tuple(warnings), checks=tuple(checks),
            clarification=clarification, source=source,
        )
    agreement: Literal["match", "mismatch"] = "match" if number.holds(computed) else "mismatch"
    reason = None
    if agreement == "mismatch" and not recorded_unit:
        scaled = _scale_note(number, computed, measure)
        if scaled and scaled[0] == "unit":
            agreement = "match"
            assumptions.append(scaled[1])
        elif scaled:
            return ClaimCheck(
                text, "unverified", "missing_context", scaled[1], claimed=number, question=asked,
                answer=answer, compared=computed, agreement="mismatch", assumptions=tuple(assumptions),
                warnings=tuple(warnings),
                checks=tuple(checks), source=source,
                clarification=f"What unit is {measure or 'the value'} recorded in?",
            )
    checks.insert(0, Check(
        "numeric_agreement", agreement == "match",
        f"computed {_plain(computed)}; the claim stands for {number.interval()}",
    ))
    failed = [check for check in checks if not check.passed]
    described = f"{sloped}{answer.calculation}"
    used = scope.rows_used if scope is not None else None
    rows = f" ({counted(used, 'row')})" if used is not None else ""
    if number.percent and "change_percent" in dict(answer.figures):
        # "the price of a gallon fell by 11%": a change in percent is the same in any unit.
        units = [phrase for phrase in left_out if set(_words(phrase)) <= MEASURE_UNIT_NOUNS]
        if units:
            left_out = [phrase for phrase in left_out if phrase not in units]
            assumptions = [item for item in assumptions if not item.startswith(LEFT_OUT_NOTE)]
            if left_out:
                assumptions.append(
                    f"{LEFT_OUT_NOTE} {_quoted(left_out)}, which name no column or value in the table."
                )
            assumptions.append(f"Read {_quoted(units)} as the unit the measure is in; a change in percent is "
                               "the same in any unit.")
    measure_values = analysis.frame[measure] if measure in analysis.frame.columns else None
    spent = [phrase for phrase in left_out if phrase.lower() in VERB_NOUNS] if (
        measure_values is not None and _money_column(str(measure), measure_values)
    ) else []
    if spent:
        # "nurses earned a median salary of $33,150": a verb of money says what a money column holds.
        left_out = [phrase for phrase in left_out if phrase not in spent]
        assumptions = [item for item in assumptions if not item.startswith(LEFT_OUT_NOTE)]
        if left_out:
            assumptions.append(
                f"{LEFT_OUT_NOTE} {_quoted(left_out)}, which name no column or value in the table."
            )
        assumptions.append(f"Read {_quoted(spent)} as what {measure} holds, a sum of money.")
    named = [phrase for phrase in left_out if _names_table(phrase, _table_name_words(source))]
    if named:
        # "Marvel's characters" in marvel-wikia-data.csv: the words name the whole table, so leaving
        # them out picks no rows.
        left_out = [phrase for phrase in left_out if phrase not in named]
        assumptions = [item for item in assumptions if not item.startswith(LEFT_OUT_NOTE)]
        if left_out:
            assumptions.append(
                f"{LEFT_OUT_NOTE} {_quoted(left_out)}, which name no column or value in the table."
            )
        assumptions.append(
            f"Read {_quoted(named)} as the name of the table ({source.name}), not of rows in it."
        )
    scope = [phrase for phrase in left_out if _place_scope(phrase, analysis.frame)]
    if scope:
        # "babies born in Ireland" in a table of Irish births with no column of places: the place is the
        # whole table's.
        left_out = [phrase for phrase in left_out if phrase not in scope]
        assumptions = [item for item in assumptions if not item.startswith(LEFT_OUT_NOTE)]
        if left_out:
            assumptions.append(
                f"{LEFT_OUT_NOTE} {_quoted(left_out)}, which name no column or value in the table."
            )
        assumptions.append(f"Read {_quoted(scope)} as the place the whole table covers, since no column of "
                           "the table holds countries or states.")
    plan = answer.plan
    counts_rows = plan.aggregation == "count" and not (plan.measure or plan.count_column or plan.weight)
    if named and counts_rows and agreement == "mismatch" and not _names_rows(named, analysis.roles.row_noun):
        # "Visits to the park rose to 11.3 million" in national-park-visits/national_parks.csv: the visits
        # are a column of counts, not the park-years the rows are, so a count of rows proves nothing.
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"The data give {_plain(computed)}{rows} for {described}, outside {number.interval()}, but "
            f"{_quoted(named)} name the table, not its rows, so the claim may count something the rows hold "
            "rather than the rows.",
            claimed=number, question=asked, answer=answer, compared=computed, agreement="mismatch",
            assumptions=tuple(dict.fromkeys(assumptions)), warnings=tuple(dict.fromkeys(warnings)),
            checks=tuple(checks), source=source,
            clarification="Which column holds what the claim counts?",
        )
    if left_out and agreement == "match" and not _safe_without(left_out, answer, number, used, slope):
        # An average or a bound over every row may agree by chance when the words left out pick some.
        words = " and ".join(f'"{phrase}"' for phrase in left_out)
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"Without {words}, which name nothing in the table, the data give {_plain(computed)}{rows} for "
            f"{described}, which agrees with {number.interval()}; but the words left out may pick out rows "
            "the table cannot, and over a subset the number could differ, so the check does not rest on it.",
            claimed=number, question=asked, answer=answer, compared=computed, agreement="match",
            assumptions=tuple(dict.fromkeys(assumptions)), warnings=tuple(dict.fromkeys(warnings)),
            checks=tuple(checks), source=source,
            clarification=f"What do {words} stand for in the table?",
        )
    if agreement == "mismatch" and left_out:
        words = " and ".join(f'"{phrase}"' for phrase in left_out)
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"Without {words}, which name nothing in the table, the data give {_plain(computed)}{rows} for "
            f"{described}, outside {number.interval()}; the words left out may pick other rows or another "
            "column.",
            claimed=number, question=asked, answer=answer, compared=computed, agreement="mismatch",
            assumptions=tuple(dict.fromkeys(assumptions)), warnings=tuple(dict.fromkeys(warnings)),
            checks=tuple(checks), source=source,
            clarification=f"What do {words} stand for in the table?",
        )
    guessed = next((match.group(1) for item in assumptions
                    if (match := re.fullmatch(rf'Read "(.+)" as {re.escape(str(measure))}\.', item))), None)
    if agreement == "mismatch" and guessed and not _names_head(asked, measure, strict=True):
        # "the seedlings averaged 4.2 feet": Seed was guessed from "seedlings"; the one column of lengths
        # may be what the claim measures.
        swapped = re.sub(rf"(?<!\w){re.escape(guessed)}(?!\w)", "", asked, count=1, flags=re.IGNORECASE)
        unit_read = _one_unit_reading(analysis, re.sub(r"\s+", " ", swapped).strip(), number, choices)
        if unit_read and unit_read[0] != measure:
            check = _check(analysis, text, unit_read[1], number, source, choices, left=(*left_out, guessed))
            note = (f"Read the claim as being about {unit_read[0]}, the only column of the kind its unit "
                    f'gives, rather than {measure}, which "{guessed}" only resembles.')
            return replace(check, assumptions=(note, *check.assumptions))
    if agreement == "mismatch" and measure and not _names_head(" ".join([asked, *named]), measure,
                                                                  strict=True):
        # A column read from a word that does not say what it measures may be the wrong one; a
        # mismatch there proves nothing. Words that also name the table still name the column:
        # "milk production" in state_milk_production.csv names milk_produced.
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"The data give {_plain(computed)}{rows} for {described}, outside {number.interval()}, but the "
            f"claim does not name what {measure} measures, so it may be about another column.",
            claimed=number, question=asked, answer=answer, compared=computed, agreement="mismatch",
            assumptions=tuple(dict.fromkeys(assumptions)), warnings=tuple(dict.fromkeys(warnings)),
            checks=tuple(checks), source=source,
            clarification=f"Is the number about {measure}, or about another column?",
        )
    typical = _typical_other(analysis, text, asked, number, answer, choices, factor)
    if typical and typical[2] != (agreement == "match"):
        # "the typical strike cost about $27,000": the median is 26,783 and the mean 242,388; "typical" is
        # said of either.
        other, value, agrees = typical
        alternatives = (
            Alternative(answer.plan.aggregation, asked, computed, agreement == "match", described),
            Alternative(other.plan.aggregation, asked, value, agrees, other.calculation),
        )
        return ClaimCheck(
            text, "unverified", "ambiguous_intent",
            f'"Typical" may mean the mean or the median: the {answer.plan.aggregation} is {_plain(computed)} '
            f"and the {other.plan.aggregation} {_plain(value)}, and only one is in {number.interval()}.",
            claimed=number, question=asked, answer=answer, compared=computed, agreement=agreement,
            assumptions=tuple(dict.fromkeys(assumptions)), warnings=tuple(dict.fromkeys(warnings)),
            checks=tuple(checks), alternatives=alternatives, source=source,
            clarification="Is the typical value the mean or the median?",
        )
    otherwise = None
    if agreement == "mismatch":
        otherwise = _otherwise(analysis, asked, number, answer, choices, factor)
    aggregated = _other_aggregation(analysis, text, asked, number, answer, choices, factor, used) if (
        agreement == "mismatch" and not otherwise) else None
    if aggregated:
        # "Oxford got 984 mm of rain in 2012": twelve monthly rows give a mean of 82 and a total of 984,
        # and the claim says neither.
        word, other, value = aggregated
        alternatives = (
            Alternative(answer.plan.aggregation, asked, computed, False, described),
            Alternative(word, f"{word} {asked}", value, True, other.calculation),
        )
        return ClaimCheck(
            text, "unverified", "ambiguous_intent",
            f"The claim does not say how the {used:,} rows combine: by default the data give "
            f"{_plain(computed)} ({described}); the {word} is {_plain(value)}, which {number.text} stands "
            "for.",
            claimed=number, question=asked, answer=answer, compared=computed, agreement="mismatch",
            assumptions=tuple(dict.fromkeys(assumptions)), warnings=tuple(dict.fromkeys(warnings)),
            checks=tuple(checks), alternatives=alternatives, source=source,
            clarification=f"Is the number the {word} over those rows, or the {answer.plan.aggregation}?",
        )
    tallied = _tallied(analysis, answer, number) if agreement == "mismatch" and not otherwise else None
    if tallied:
        # "565 boys were named Jack" in a table of names with a Number of babies each: one row, 565 babies.
        column, value, how = tallied
        alternatives = (
            Alternative("rows", asked, computed, False, described),
            Alternative(column, f"total {column} {asked}", value, True, how),
        )
        return ClaimCheck(
            text, "unverified", "ambiguous_intent",
            f"The claim may count the rows ({_plain(computed)}, {described}) or add up {column}, which holds "
            f"a count on each row: {_plain(value)} ({how}), which {number.text} stands for.",
            claimed=number, question=asked, answer=answer, compared=computed, agreement="mismatch",
            assumptions=tuple(dict.fromkeys(assumptions)), warnings=tuple(dict.fromkeys(warnings)),
            checks=tuple(checks), alternatives=alternatives, source=source,
            clarification=f"Does the claim count the rows, or the {column} on them?",
        )
    if otherwise:
        # The claim agrees once a choice goes the other way; the data alone cannot say which way it should go.
        decision, option, other, value = otherwise
        alternatives = (
            Alternative(f"{decision.key}={decision.chosen}", asked, computed, False, described),
            Alternative(f"{decision.key}={option}", asked, value, True, other.calculation),
        )
        return ClaimCheck(
            text, "unverified", "ambiguous_intent",
            f"The answer rests on a choice: {decision.question} By default ({decision.chosen}) the data give "
            f"{_plain(computed)}{rows}; with {option} they give {_plain(value)}, which {number.text} stands "
            "for.",
            claimed=number, question=asked, answer=answer, compared=computed, agreement="mismatch",
            assumptions=tuple(dict.fromkeys(assumptions)), warnings=tuple(dict.fromkeys(warnings)),
            checks=tuple(checks), alternatives=alternatives, source=source,
            clarification=f"{decision.question} Choose {decision.key} {decision.chosen} or {option}.",
        )
    counting = answer.plan.aggregation == "count" or "share_percent" in dict(answer.figures)
    if agreement == "mismatch" and computed == 0 and number.low > 0 and counting:
        # "24% of cats were adopted" read as intake_type = adopted animal return, which no cat has: no row
        # fits the words as read, which says more about the reading than about the claim.
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"No row fits the claim as ADA read it ({described}), so the data give 0; a claim of "
            f"{number.text} about rows that are not there more likely means other rows or another column.",
            claimed=number, question=asked, answer=answer, compared=computed, agreement="mismatch",
            assumptions=tuple(dict.fromkeys(assumptions)), warnings=tuple(dict.fromkeys(warnings)),
            checks=tuple(checks), source=source,
            clarification="Which rows does the claim count?",
        )
    if agreement == "mismatch":
        reason = "failed_check"
        message = (
            f"The data give {_plain(computed)}{rows} for {described}; the claim's {number.text} stands for "
            f"{number.interval()}."
        )
        per_row = dict(answer.figures).get("per_row")
        if per_row is not None and number.holds(per_row) and answer.plan.per:
            message += (
                f" The claim matches the average taken row by row ({_plain(per_row)}), which counts each row "
                f"rather than each {answer.plan.per[0]}."
            )
    elif failed:
        reason = "failed_check"
        message = (
            f"The number agrees ({_plain(computed)}{rows}, {described}), but a check on the analysis failed: "
            f"{failed[0].detail}"
        )
    else:
        message = (
            f"{_plain(computed)}{rows} from {described} lies in {number.interval()}, "
            f"which {number.text} stands for."
        )
    return ClaimCheck(
        text, "verified" if reason is None else "unverified", reason, message, claimed=number, question=asked,
        answer=answer, compared=computed, agreement=agreement, assumptions=tuple(dict.fromkeys(assumptions)),
        warnings=tuple(dict.fromkeys(warnings)), checks=tuple(checks), source=source,
    )


__all__ = [
    "CLAIM_SCHEMA_VERSION",
    "REASONS",
    "Alternative",
    "Check",
    "ClaimCheck",
    "ClaimedNumber",
    "check_claim",
    "read_claim",
]
