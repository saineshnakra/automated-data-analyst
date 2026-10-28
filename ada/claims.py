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

from .aggregates import total_levels
from .decisions import Decision, checked_choices
from .file_io import SUPPORTED_SUFFIXES as READABLE
from .formatting import counted, holds_undated, list_names, normalized_name
from .nlq import (
    AGE_PHRASE,
    DAY_PATTERN,
    LEADING_UNDER,
    MONTH_WORD,
    OUTCOME_WORDS,
    Condition,
    QueryAnswer,
    ValueFilter,
    counts_column,
    cube_measure,
    days_in_words,
)
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
    # Digits right after a letter are part of a word or a code ("CO2", "bondsba01", "I-94"), not a number.
    rf"(?<![A-Za-z0-9_])(?<![A-Za-z]-)(?P<currency>{_CURRENCY})?\s?(?P<sign>[-−])?(?P<number>{_DIGITS})"
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
    "lost": "losses", "loses": "losses", "raised": "receipt", "raises": "receipt",
}
# Nouns for what an amount of money is: "the average price of a dozen eggs" is the value, in dollars.
PRICE_WORDS = frozenset({"price", "prices"})
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
# "Most of Ballarat's parking tickets", "the majority of deaths": more than half of them.
MOST_SHARE = re.compile(r"^\s*(?:(?:the\s+)?majority\s+of|most(?:\s+of)?)\s+(?=[A-Za-z])", re.IGNORECASE)
# A time of day, which picks rows and is never the number claimed: "before 10 am", "between 22:00 and 23:00".
CLOCK_PHRASE = re.compile(r"\b\d{1,2}(?::[0-5]\d)?\s*[ap]\.?\s?m\b\.?|\b\d{1,2}:[0-5]\d(?::[0-5]\d)?\b",
                          re.IGNORECASE)
# A span back from now, which picks rows and is never the number claimed: "over the past ten years".
RELATIVE_SPAN = re.compile(r"\b(?:past|last|previous)\s+[\w-]+\s+(?:years|months|weeks|days|decades)\b",
                           re.IGNORECASE)
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
    r"\s+)?(?:(?P<count>a|one|two|three|four|nine)\s+)?(?P<part>half|thirds?|quarters?|fifths?|tenths?)\b"
    r"(?!\s+of\b)",
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
# the riders", "People aged 25 to 44 made up 52% of immigrants". The share is of what follows "of".
SUBJECT_SHARE = re.compile(
    r"^\s*(?P<part>[A-Za-z'-]+(?:\s+[A-Za-z'-]+){0,3}?"
    r"(?:\s+(?:aged|ages|age|under|over)\s+\d{1,3}(?:\s*(?:-|–|to|and)\s*\d{1,3})?"
    r"(?:\s+(?:and|or)\s+(?:over|older|under|younger))?)?)"
    r"\s+(?P<verb>were|was|are|is|made\s+up|make\s+up|"
    r"makes\s+up|accounted\s+for|account\s+for|accounts\s+for|represented|represent|represents|"
    r"comprised|comprise|comprises|formed|form|forms|won|received|got|took|earned|captured|caused|"
    r"cause|causes)\s+"
    r"(?P<share>(?:(?:about|nearly|almost|roughly|around|only|just|over|under|more\s+than|less\s+than)\s+)?"
    r"\d[\d,]*(?:\.\d+)?\s*%)\s+of\s+(?P<rest>.+?)[.!?]?\s*$",
    re.IGNORECASE,
)
# A share as a percentage, with the words that bound it: "more than 90%".
_SHARE = (r"(?:(?:about|nearly|almost|roughly|around|only|just|over|under|more\s+than|less\s+than|"
          r"fewer\s+than)\s+)?\d[\d,]*(?:\.\d+)?\s*%")
# A place's name, each word in capitals: "Maricopa County".
_PLACE = r"(?-i:[A-Z])[\w'.-]*(?:\s+(?-i:[A-Z])[\w'.-]*){0,4}"
# A state on a share of occasions: "Air quality was good on more than 90% of the days monitored", "In Maricopa
# County, air was good on ...", "Air quality in Maricopa County was good on ...".
STATE_SHARE = re.compile(
    rf"^\s*(?:in\s+(?P<lead>{_PLACE}),\s+)?(?P<subject>[A-Za-z'-]+(?:\s+[A-Za-z'-]+){{0,3}}?)"
    rf"(?:\s+in\s+(?P<place>{_PLACE}))?\s+(?:was|were|is|are)\s+"
    rf"(?P<state>[A-Za-z-]+(?:\s+[A-Za-z-]+){{0,2}}?)\s+(?:on|in|for|during)\s+"
    rf"(?P<share>{_SHARE})\s+of\s+(?P<rest>.+?)[.!?]?\s*$",
    re.IGNORECASE,
)
# A place that had a state on a share of occasions: "Maricopa County had good air on 8.8% of days".
HAD_STATE_SHARE = re.compile(
    rf"^\s*(?P<place>{_PLACE}),?\s+had\s+(?P<state>[a-z-]+)\s+[a-z]+\s+(?:on|in|for|during)\s+"
    rf"(?P<share>{_SHARE})\s+of\s+(?P<rest>.+?)[.!?]?\s*$",
    re.IGNORECASE,
)
# "one day in ten", "two people in five": a share of the things the noun names, unless another noun follows
# ("one goal in ten games").
NOUN_IN_FRACTION = re.compile(
    r"\b(?P<top>one|two|three|four|five|six|seven|eight|nine)\s+(?!in\b|out\b|of\b)(?P<noun>[a-z]+)\s+in\s+"
    r"(?:every\s+)?(?P<bottom>two|three|four|five|six|seven|eight|nine|ten)\b"
    r"(?!\s+(?!(?:was|is|has|does|as|its|his|this|thus|plus|less)\b)[a-z]+s\b)",
    re.IGNORECASE,
)
# Plurals not made by adding s.
PLURAL_OF = {"person": "people", "child": "children", "man": "men", "woman": "women"}
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
# "averaged five times their 2019 level": the later level as a multiple of an earlier year's.
TIMES_LEVEL = re.compile(
    r"\b(?P<verb>averaged|was|were|stood\s+at|came\s+to|reached|hit)\s+"
    r"(?:(?P<qualifier>about|almost|nearly|roughly|more\s+than|over|less\s+than)\s+)?"
    r"(?P<times>\d+(?:\.\d+)?|two|three|four|five|six|seven|eight|nine|ten)\s+times\s+(?:its|their|the)\s+"
    r"(?P<earlier>(?:19|20)\d{2})\s+(?:level|value|figure|average|total|rate|price)s?\b",
    re.IGNORECASE,
)
# A description set off by commas after a name: ", a veterinary sedative,", ", the category that takes in
# SUVs and pickups,".
APPOSITIVE = re.compile(
    r",\s+(?:(?:a|an)\s+[A-Za-z][A-Za-z' -]{0,40}|the\s+[a-z]+\s+(?:that|which|who)\s+[A-Za-z' -]{1,60}|"
    r"home\s+to\s+[A-Za-z][A-Za-z' -]{0,40})(?=,)",
    re.IGNORECASE,
)
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
    # Degrees Celsius and Fahrenheit differ by an offset as well as a size, so they are never converted.
    "°c": ("degrees Celsius", 1.0), "ºc": ("degrees Celsius", 1.0), "celsius": ("degrees Celsius", 1.0),
    "°f": ("degrees Fahrenheit", 1.0), "ºf": ("degrees Fahrenheit", 1.0),
    "fahrenheit": ("degrees Fahrenheit", 1.0),
    # Money in cents and dollars; "pounds" is a weight as often as money, so it is left out.
    "cent": ("money", 0.01), "cents": ("money", 0.01), "dollar": ("money", 1.0), "dollars": ("money", 1.0),
}
# Name words of a column of temperatures, whose bracketed "(C)" or "(F)" is a unit.
TEMPERATURE_NAMES = frozenset({"temp", "temperature", "temperatures", "tmax", "tmin", "tavg"})
# Words of a claim that pick a day's maximum, minimum or mean among columns of temperatures.
TEMPERATURE_SIDES = {
    "maximum": frozenset({"high", "highs", "highest", "hottest", "warmest", "hot", "reach", "reached",
                          "reaches", "hit", "hits", "topped", "tops", "maximum", "max"}),
    "minimum": frozenset({"low", "lows", "lowest", "coldest", "cold", "fell", "falls", "dropped", "drops",
                          "dipped", "plunged", "minimum", "min"}),
    "mean": frozenset({"mean"}),
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
        span.span() for span in NAMING_NUMBER.finditer(claim)] + [
        span.span() for span in CLOCK_PHRASE.finditer(claim)] + [
        span.span() for span in RELATIVE_SPAN.finditer(claim)]
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
    for match in re.finditer(r"\s*([°º]?[A-Za-z][A-Za-z.]*(?:/[A-Za-z][A-Za-z0-9.]*)?)", claim[end:]):
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


def _fraction_below(value: float) -> float:
    """The next fraction below ``value`` with a denominator up to 10: 4/9 below a half."""
    fractions = {top / bottom for bottom in range(2, 11) for top in range(1, bottom)}
    return max((item for item in fractions if item < value - 1e-12), default=0.0)


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
    scaled = re.search(r"\b(?:a|one)\s+(?P<scale>hundred|thousand|million|billion)\b", claim, re.IGNORECASE)
    rest = claim[:scaled.start()] + claim[scaled.end():] if scaled else ""
    if scaled and all(item.year_like for item in _candidates(rest, [])):
        # "more than a thousand hours" is more than 1,000.
        written = {"hundred": "100", "thousand": "1,000"}.get(scaled.group("scale").lower(),
                                                             f"1 {scaled.group('scale')}")
        return f"{claim[:scaled.start()]}{written}{claim[scaled.end():]}", lambda number: number
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
        lead = claim[:fraction.start()]
        # "made up half of": the match may start at the space before the fraction, which stays.
        rewritten = f"{lead}{' ' if lead and not lead[-1].isspace() else ''}{value * 100:g}%{rest}"

        share = value * 100

        def back(number: ClaimedNumber) -> ClaimedNumber:
            bounds = {
                "more than": (share, math.inf, True, False), "at least": (share, math.inf, False, False),
                "less than": (-math.inf, share, False, True), "at most": (-math.inf, share, False, False),
                "just over": (share, high, True, False), "just under": (low, share, False, True),
                # "nearly half" of 46.9%: nearer half than 4/9, the next fraction below it. No fraction lies
                # below a tenth, so "nearly one in ten" is from nine tenths of it, as "nearly 100" is from 90.
                "nearly": ((_fraction_below(value) or value * 0.9) * 100, share, True, False),
            }.get(kind or "", (low, high, False, False))
            return replace(number, text=fraction.group(0), low=bounds[0], high=bounds[1], low_open=bounds[2],
                           high_open=bounds[3], qualifier=kind, decimals=1)
        return rewritten, back
    if any(not (item.year_like or item.ordinal or item.filter_like or _bound_like(claim, item, percent=True))
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
    if any(span.start() <= worded.start() < span.end() for span in RELATIVE_SPAN.finditer(claim)):
        # "over the past ten years" says which rows; its number is not the one claimed.
        return None
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
    # "aged 35 or older" read as the bands 35 to 39, 40 to 44 and 45 to 49 is used, though as values.
    readings = [*plan.notes, *(answer.scope.readings if answer.scope is not None else ())]
    banded = [match.span() for match in (*AGE_PHRASE.finditer(asked), *LEADING_UNDER.finditer(asked))
              if any(note.lower().startswith(f'read "{match.group().strip().lower()}" as')
                     for note in readings)]
    unused = []
    for item in _candidates(asked, []):
        if not _bound_like(asked, item, percent=True) or any(low <= item.match.start() < high
                                                              for low, high in banded):
            continue
        value, _, _, scale, _ = parse_number(item.match)
        if not any(math.isclose(value * scale, other, rel_tol=1e-9) for other in used):
            unused.append(asked[item.qualifier_start:item.match.end()].strip())
    return unused


def _noun_share(found: re.Match) -> str:
    """The share "one day in ten" names, written as "10% of days"."""
    words = {**NUMBER_WORDS, "one": 1}
    top, bottom = words[found.group("top").lower()], words[found.group("bottom").lower()]
    if not 0 < top < bottom:
        return found.group(0)
    noun = found.group("noun").lower()
    if noun in PLURAL_OF:
        plural = PLURAL_OF[noun]
    elif noun in IRREGULAR_PLURALS:
        # "two men in five": the noun is already a plural.
        plural = noun
    elif re.search(r"[^aeiou]y$", noun):
        plural = f"{noun[:-1]}ies"
    elif noun.endswith(("s", "x", "ch", "sh")):
        plural = f"{noun}es"
    else:
        plural = f"{noun}s"
    return f"{top / bottom * 100:g}% of {plural}"


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
    most = MOST_SHARE.match(claim)
    numbered = [item for item in _candidates(claim, columns)
                if not (item.year_like or item.filter_like or item.ordinal)]
    if most and not numbered and not re.match(
        r"(?:the\s+)?(?:most|least|common)\b", claim[most.end():], re.IGNORECASE
    ):
        # With a number of its own ("Most of the 20,071 tickets cost $55"), "most" is not the number claimed.
        claim = f"More than half of {claim[most.end():]}"
    claim = NOUN_IN_FRACTION.sub(_noun_share, claim)
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
    state = STATE_SHARE.match(claim) or HAD_STATE_SHARE.match(claim)
    verb = r"[a-z]+ed|found|seen|known|shown|given|taken|held|made"
    if state and state.re is STATE_SHARE and (
            set(_words(state.group("subject"))) & _column_words(columns)
            or re.fullmatch(verb, state.group("state").split()[0], re.IGNORECASE)):
        # "Fentanyl was found in 68% of the deaths": a column the claim is about, or a verb, is no state.
        state = None
    if state:
        # "Air quality was good on 75% of the days" is a share of the days: "75% of the days were good". A
        # place it names picks its rows.
        places = [state.groupdict().get(name) for name in ("lead", "place")]
        where = "".join(f" in {place}" for place in places if place)
        rewritten = f"{state.group('share')} of {state.group('rest')} were {state.group('state')}{where}"
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
    """Whether ``column`` is the initials of words in a row: "home runs" for HR, "hit points" for hp."""
    from .nlq import initials_of  # noqa: PLC0415

    words = [word for word in re.findall(r"[a-z]+", question.lower()) if word not in INITIALS_SKIP]
    for size in range(1, 6):
        initials = initials_of(str(column), size)
        phrases = {"".join(word[0] for word in words[start:start + size])
                   for start in range(len(words) - size + 1)}
        if initials and len(initials) == size and initials in phrases:
            return True
        # "sea level" for MSL: the M of a name in capitals stands for mean.
        if size >= 2 and initials and len(initials) == size + 1 and initials[0] == "m" and (
                initials[1:] in phrases):
            return True
    return False


# The currency a sign before a number writes: "$1.18" is in dollars.
CURRENCY_SIGNS = {"$": "dollar", "us$": "dollar", "£": "pound", "€": "euro", "¥": "yen"}


def _currency_named(frame: Any, number: ClaimedNumber) -> str | None:
    """The currency a claim's number is in ("10 yen", "$1.18") when one column of the table, titled by its
    ISO code, holds that currency's rate."""
    from .nlq import CURRENCY_NAMES  # noqa: PLC0415

    written = NUMBER_PATTERN.search(number.text)
    sign = (written.group("currency") or "").strip().lower() if written else ""
    unit = (number.unit or "").split()[0].lower() if number.unit else ""
    for word in (CURRENCY_SIGNS.get(sign), unit, unit.removesuffix("s")):
        holding = [column for column in frame.columns
                   if word and word in CURRENCY_NAMES.get(str(column).strip(), ())]
        if len(holding) == 1:
            return word
    return None


def _times_level(text: str) -> str:
    """A claim of a later level as a multiple of an earlier year's, written as the change between the years.

    "Prices in 2023 averaged five times their 2019 level" is "average Prices rose by about 400% between
    2019 and 2023"; the later year is the claim's one other year.
    """
    found = TIMES_LEVEL.search(text)
    if not found:
        return text
    before, after = text[:found.start()], text[found.end():]
    years = re.findall(r"\b(?:19|20)\d{2}\b", before + after)
    later = [year for year in years if year != found.group("earlier")]
    if len(later) != 1:
        return text
    times = found.group("times").lower()
    factor = float(NUMBER_WORDS.get(times, times))
    qualifier = re.sub(r"\s+", " ", (found.group("qualifier") or "about").lower())
    subject = re.sub(rf"\s*\b(?:in\s+)?{later[0]}\b,?", "", before, flags=re.IGNORECASE).strip()
    tail = re.sub(rf"\s*\b(?:in\s+)?{later[0]}\b", "", after, flags=re.IGNORECASE)
    lead = "average " if found.group("verb").lower() == "averaged" else ""
    change = f"rose by {qualifier} {(factor - 1) * 100:g}%"
    return f"{lead}{subject} {change} between {found.group('earlier')} and {later[0]}{tail}"


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
    from .nlq import CURRENCY_NAMES  # noqa: PLC0415

    # "the dollar" names USD in a table of exchange rates.
    if any(re.search(rf"\b{re.escape(name)}s?\b", question, re.IGNORECASE)
           for name in CURRENCY_NAMES.get(str(column).strip(), ())):
        return True
    return any(_abbreviates(word, part) for word in words for part in parts if len(part) >= 2)


def _outcome_named(words: list[str], column: str) -> bool:
    """Whether a word names a column called for the same outcome: "deaths" names fatal; or a word for new
    diagnoses names a column of incidences: "diagnoses" names IncidencesAllAges."""
    from .nlq import DIAGNOSIS_WORDS  # noqa: PLC0415

    if set(words) & DIAGNOSIS_WORDS and {"incidence", "incidences"} & set(
            re.findall(r"[a-z]+", normalized_name(str(column)).lower())):
        return True
    sides = {word: side for side, forms in OUTCOME_WORDS.items() for word in forms}
    side = sides.get(normalized_name(str(column)))
    return side is not None and any(sides.get(word) == side for word in words)


# Name words that say a column counts or totals something; a claim need not repeat them ("250 parts"
# for num_parts).
COUNTING_WORDS = frozenset({"num", "number", "no", "n", "count", "total", "amount", "amt", "qty"})


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
    # "Price (EUR/MWhe)": what is in brackets is a unit or a note, not the kind of quantity.
    parts = re.findall(r"[a-z]+", normalized_name(re.sub(r"\([^)]*\)", " ", str(column)).strip() or column))
    if len(parts) > 1 and column_unit(column) == parts[-1]:
        parts = parts[:-1]
    parts = [part for part in parts if part not in COUNTING_WORDS] or parts
    sides = {word: side for side, forms in OUTCOME_WORDS.items() for word in forms}
    if parts and parts[-1] in sides and any(sides.get(word) == sides[parts[-1]] for word in words):
        # "died" names the deaths of "COVID-19 Deaths".
        return True
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


# A refusal naming the columns that hold a value: '"broken-down" is a value of cause_en and cause_group'.
VALUE_IN_COLUMNS = re.compile(r'"(?P<phrase>[^"]+)" is a value of (?P<columns>[^;]+);')


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
                  else f"the total of {plan.weight}" if plan.weight
                  else f"the total of {plan.partner}" if plan.aggregation == "ratio" and plan.partner
                  else "rows")
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
    # Only both ends of a claim say a scale: "more than 25 million" holds for any count of drivers times a
    # million.
    bounded = not (math.isinf(number.low) or math.isinf(number.high))
    if number.scale_word and number.scale != 1 and bounded and number.holds(computed * number.scale):
        unit = " ".join(part for part in (number.scale_word, number.unit) if part)
        return "unit", (
            f'Read "{unit}" as the unit {name} is recorded in, so {_plain(number.value / number.scale)} '
            f"is compared with {name} as recorded."
        )
    if number.percent:
        # A share has no unit to scale.
        return None
    if column is not None and (other := _other_unit(number, computed)) is not None:
        # "about 40 centimetres" or "more than 30 centimetres" of a column in metres: the claim holds once the
        # column is read in another unit of the claim's kind. A count of rows has no unit to read.
        return "scale", (f"The claim agrees with {name} only if {name} is recorded in {other}, and the data "
                         "do not say what unit it is in.")
    if math.isinf(number.low) or math.isinf(number.high):
        # A bound holds for nearly any number scaled far enough.
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


def _other_unit(number: ClaimedNumber, computed: float) -> str | None:
    """Another unit of the claim's kind that, were the column recorded in it, would make the claim agree:
    "metres" for "about 40 centimetres" against 0.403."""
    unit = _claim_unit(number)
    if unit is None:
        return None
    kind, size = UNITS[unit]
    names: dict[float, list[str]] = {}
    for name, (other_kind, other_size) in UNITS.items():
        if other_kind == kind and other_size != size:
            names.setdefault(other_size, []).append(name)
    for other_size, written in names.items():
        if number.holds(computed * other_size / size):
            # The unit's plural word ("metres") rather than its symbol.
            return next((name for name in written if len(name) > 3 and name.endswith("s")), written[0])
    return None


def column_unit(column: str | None) -> str | None:
    """The unit a column's name ends in: "g" for body_mass_g, "kg" for "Weight (kg)", None for co2.

    A name that is only a unit word ("Days") is a measure of its own, not an annotation.
    """
    if not column:
        return None
    bracketed = re.search(r"[(\[]\s*([°º]?\s*[A-Za-z]+)\s*[)\]]\s*$", str(column))
    if bracketed and bracketed.group(1).lower() in UNITS:
        return bracketed.group(1).lower()
    degrees = {"c": "°c", "f": "°f", "degc": "°c", "degf": "°f"}
    written = re.sub(r"[°º\s]", "", bracketed.group(1)).lower() if bracketed else ""
    if written in degrees and TEMPERATURE_NAMES & set(normalized_name(str(column)).split()):
        # "Maximum Temperature (C)" is in degrees Celsius; "Heating Degree Days (C)" is no temperature.
        return degrees[written]
    parts = [part for part in re.split(r"[^A-Za-z]+", str(column)) if part]
    if len(parts) >= 2 and parts[-1].lower() in UNITS:
        return parts[-1].lower()
    return None


# Words of a unit column that give the scale the values are recorded in: UNIT = Thousand.
RECORDED_SCALES = {"thousand": 1e3, "thousands": 1e3, "000": 1e3, "000s": 1e3, "million": 1e6,
                   "millions": 1e6, "billion": 1e9, "billions": 1e9}


def _recorded_scale(analysis: Any, answer: QueryAnswer | None) -> tuple[float, str, str] | None:
    """The scale a unit column gives the answer's column in, when it gives one for every row.

    In a CSO table whose UNIT is "Thousand" on every row, VALUE 149.2 is
    149,200. Returns (factor, the unit as written, the unit column).
    """
    from .nlq import UNIT_COLUMN_NAMES  # noqa: PLC0415

    if answer is None:
        return None
    measure = answer.plan.measure or answer.plan.weight
    frame = analysis.frame
    if measure is None or measure != cube_measure(frame) and normalized_name(str(measure)) not in VALUE_NAMES:
        return None
    for column in frame.columns:
        if normalized_name(str(column)) not in UNIT_COLUMN_NAMES:
            continue
        units = frame[column].dropna().astype(str).str.strip().unique()
        words = re.findall(r"[a-z0-9]+", units[0].lower()) if len(units) == 1 else []
        factor = next((RECORDED_SCALES[word] for word in words if word in RECORDED_SCALES), None)
        if factor:
            return factor, str(units[0]), str(column)
    return None


def _row_unit(analysis: Any, answer: QueryAnswer | None) -> tuple[str, str] | None:
    """The unit a unit column gives every row the answer uses, when it is one of UNITS: UNITS = Dollars for
    the price of a dozen eggs. Returns (unit, the unit column)."""
    from .nlq import UNIT_COLUMN_NAMES, _apply_filters  # noqa: PLC0415

    if answer is None or not (answer.plan.measure or answer.plan.weight):
        return None
    frame = analysis.frame
    for column in frame.columns:
        if normalized_name(str(column)) not in UNIT_COLUMN_NAMES:
            continue
        rows, _ = _apply_filters(frame, answer.plan, analysis.roles)
        units = rows[column].dropna().astype(str).str.strip().str.lower().unique()
        if len(units) == 1 and units[0] in UNITS:
            return str(units[0]), str(column)
    return None


def _unit_factor(analysis: Any, answer: QueryAnswer, number: ClaimedNumber, measure: str | None
                 ) -> tuple[float, str, str] | None:
    """The factor from the unit the data are recorded in to the claim's, with both units and where the data's
    comes from: (factor, recorded unit, where)."""
    claimed = _claim_unit(number)
    recorded, where = column_unit(measure), "the unit in its name"
    if claimed and not recorded and (row_unit := _row_unit(analysis, answer)):
        # "29 cents" of a price whose rows say UNITS = Dollars.
        recorded, where = row_unit[0], f"the unit {row_unit[1]} gives for these rows"
    if not (claimed and recorded) or claimed == recorded or UNITS[claimed][0] != UNITS[recorded][0]:
        return None
    return UNITS[recorded][1] / UNITS[claimed][1], recorded, where


def _row_step(analysis: Any, plan: Any = None) -> str | None:
    """The hour or day each row of the table stands for, when the rows are one date each, that far apart.

    With a plan, the rows are those its values pick: one station's rows in a table of several stations.
    """
    date = analysis.roles.date
    if not date or date not in analysis.frame.columns:
        return None
    rows = analysis.frame
    for item in (plan.filters if plan is not None else ()):
        if isinstance(item, ValueFilter) and item.column in rows.columns:
            rows = rows[rows[item.column].isin(item.values)]
    dates = rows[date].dropna()
    if dates.dtype.kind != "M" or len(dates) < 3 or dates.nunique() < 0.99 * len(rows):
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
        # A plural unit word inside the name holds the kind too: hours_mission, total_hrs_sum. A singular one
        # is often something else: temp_min, day of the month, hour of the day.
        spelled = {word for word in words if word in UNITS and UNITS[word][0] == kind and word.endswith("s")
                   and len(word) > 2} if not unit else set()
        if claimed and spelled and any(UNITS[word][1] != UNITS[claimed][1] for word in spelled):
            # hours_mission beside a claim in minutes: a column of the kind in another unit, which the claim's
            # number cannot be compared with as it is, so no one column is read.
            return []
        if (unit and UNITS[unit][0] == kind) or named or spelled:
            rewritten = (
                re.sub(rf"\b{re.escape(verbs[0])}\b", str(column), asked, count=1) if verbs
                else f"{asked} {column}"
            )
            found.append((str(column), rewritten))
    if kind.startswith("degrees") and len(found) > 1:
        # "fell to -46.6 °C" among a day's maximum, minimum and mean: the minimum.
        said = set(re.findall(r"[a-z]+", asked.lower()))
        sides = [side for side, cues in TEMPERATURE_SIDES.items() if cues & said]
        found = [item for item in found if len(sides) == 1 and sides[0] in normalized_name(item[0]).split()]
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
    if question is None and (pair := EACH_PAIR.match(text)):
        return _each_checked(analysis, text, pair, choices)
    if question is None and (both := CHANGE_AND_LEVEL.match(text)):
        return _change_and_level(analysis, text, both, choices)
    check = _checked_claim(analysis, text, question, value, choices)
    if not (check.verified or check.reason == "failed_check") or not analysis.frame.duplicated().any() or (
        counts_column(analysis.frame, analysis.roles) is None
    ):
        # Rows that are records may repeat as separate records; rows that each hold a count may not.
        return check
    once = replace(analysis, frame=analysis.frame.drop_duplicates(ignore_index=True))
    return _repeated_rows(check, analysis, _checked_claim(once, text, question, value, choices))


# "Pennsylvania and Texas each have 1,477 agencies": the same number for two names, each checked as its own
# claim.
EACH_PAIR = re.compile(
    r"^\s*(?P<first>[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3})\s+and\s+(?P<second>[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)"
    r"{0,3})\s+each\s+(?P<verb>[a-z]+)\s+(?P<rest>.+)$"
)
# Plural verbs and their singular, for one of the two names: "each have" is "has".
SINGULAR_VERBS = {"have": "has", "are": "is", "were": "was", "do": "does", "hold": "holds", "count": "counts",
                  "record": "records", "report": "reports", "get": "gets", "make": "makes"}


def _each_checked(analysis: Any, text: str, pair: re.Match, choices: dict[str, str]) -> ClaimCheck:
    """A claim that two names each have a number, checked for each: it holds when both do, fails when either
    does, and otherwise says what the one left unsettled says."""
    verb = SINGULAR_VERBS.get(pair.group("verb"), pair.group("verb"))
    names = (pair.group("first"), pair.group("second"))
    checks = [_checked_claim(analysis, f"{name} {verb} {pair.group('rest')}", None, None, choices)
              for name in names]
    if all(check.verified for check in checks):
        values = " and ".join(_plain(check.compared) for check in checks if check.compared is not None)
        note = f"Checked for {names[0]} and for {names[1]} on their own: {values}."
        return replace(checks[0], claim=text, assumptions=(*checks[0].assumptions, note))
    failed = next((check for check in checks if check.reason == "failed_check"), None)
    picked = failed or next(check for check in checks if not check.verified)
    name = names[checks.index(picked)]
    return replace(picked, claim=text, message=f"For {name}: {picked.message}")


# "grew by about a quarter in 2024-25, to 34.8 million tonnes": the size of a change and the level it reached,
# each checked as its own claim.
CHANGE_AND_LEVEL = re.compile(
    r"^(?P<head>.*?\b(?:rose|fell|grew|dropped|declined|increased|decreased|climbed|jumped|slipped|shrank|"
    r"slumped|plunged|soared|surged)\s+)by\s+(?P<size>[^,]+?)(?P<scope>\s+(?:in|between|from|over|since|during)\s+"
    r"[^,]+?)?\s*,?\s+to\s+(?P<level>(?:(?:about|around|nearly|almost|roughly|some|over|under|more\s+than|"
    r"less\s+than|just\s+over|just\s+under)\s+)?[$€£]?\d[\d,.]*(?:\s*(?:%|million|billion|thousand|bn|m|k)\b)?)"
    r"(?P<rest>[^,;]*?)\s*[.!]?\s*$",
    re.IGNORECASE,
)


def _change_and_level(analysis: Any, text: str, both: re.Match, choices: dict[str, str]) -> ClaimCheck:
    """A claim that gives a change and the level it reached, checked as two claims: it holds when both do,
    fails when either does, and otherwise says what the one left unsettled says."""
    head, scope, rest = both.group("head"), both.group("scope") or "", both.group("rest")
    # "rose by 12% to 1,200 in 2020": the year after the level is the change's too.
    when = re.search(r"\s+(?:in|between|from|over|since|during)\s+.*$", rest)
    parts = (("The change", f"{head}by {both.group('size')}{scope}{when.group(0) if when else ''}."),
             ("The level", f"{head}to {both.group('level')}{rest}{scope}."))
    checks = [_checked_claim(analysis, part, None, None, choices) for _, part in parts]
    if all(check.verified for check in checks):
        values = " and ".join(_plain(check.compared) for check in checks if check.compared is not None)
        note = f"Checked the change and the level it reached on their own: {values}."
        return replace(checks[0], claim=text, assumptions=(*checks[0].assumptions, note))
    failed = next((check for check in checks if check.reason == "failed_check"), None)
    picked = failed or next(check for check in checks if not check.verified)
    label = parts[checks.index(picked)][0]
    return replace(picked, claim=text, message=f"{label}: {picked.message}")


def _checked_claim(
    analysis: Any, text: str, question: str | None, value: str | float | None, choices: dict[str, str]
) -> ClaimCheck:
    """Check a claim on a table that was read: the body of check_claim."""
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
    said, ends, direction, every, worst, compared, rate_base = text, (), None, None, None, None, None
    own_name = started = None
    if question is None and (wrong := _wrong_weekday(text)):
        return ClaimCheck(text, "unverified", "failed_check", f"The claim's date and weekday disagree: "
                          f"{wrong}", source=analysis.source, clarification="Which day is meant?")
    if question is None:
        text = _times_level(days_in_words(text))
        text, every = _every_value(text, analysis.frame)
        text = _plain_change(_paid_rate(_ranked_lead(text)))
        text = _half_within(_records_began(_temperature_words(text, analysis.frame)))
        text, worst = _worst_day(text)
        text, compared = _compared_tail(text)
        text, rate_base = _table_rate_base(text, analysis.frame)
        text, own_name = _own_name(text, analysis.source, analysis.roles.row_noun)
        text, started = _start_dates(text, analysis.frame)
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
    if question is None:
        asked = _without_closing_rank(analysis, asked, choices)
    check = _checked_dates(_checked_table(analysis, text, asked, number, source, choices, question), analysis)
    check = _checked_record_day(_checked_unnamed_year(check, analysis), analysis, said)
    check = _checked_record_year(check, analysis, _any_year_record(said, every=True))
    check = _checked_clock(check, analysis)
    check = _checked_namesakes(check, analysis)
    check = _checked_occasions(check, analysis)
    recheck = lambda other_question: _checked_dates(  # noqa: E731
        _checked_table(analysis, text, other_question, number, source, choices, question), analysis)
    check = _checked_net_sums(check, analysis)
    check = _checked_kin_columns(check, analysis, recheck)
    check = _checked_other_meanings(check, recheck)
    check = _checked_country_scope(_checked_wider_values(check, analysis), analysis)
    check = _mixed_breakdowns(check, analysis)
    if question is None and number.percent and (by := BY_YEAR.search(asked)) and (
        check.verified or check.reason == "failed_check"
    ):
        check = _by_year(check, analysis, text, asked, by, number, source, choices)
    if question is None and (check.verified or check.reason == "failed_check"):
        check = _repeated_dates(check, analysis, text, asked, number, source, choices)
    counts = _counted_ends(check, ends, text) if ends else None
    factor = (_recorded_scale(analysis, check.answer) or (1.0,))[0]
    if counts and not _ends_hold(check, ends, factor):
        # "rose 46%, from 147.7 billion pounds in 1990": ends that count pounds may be totals, as a level in
        # pounds is; the change is one in the total over the rows when the totals give those ends.
        total = _checked_dates(
            _checked_table(analysis, text, f"total {asked}", number, source, choices, question), analysis
        )
        if total.answer is not None and total.answer.plan.intent == "growth" and (
            total.answer.plan.aggregation == "sum"
        ) and _ends_hold(total, ends, factor):
            note = f"Read the change as one in the total, since the values it went between count {counts}."
            check = replace(total, assumptions=(*total.assumptions, note))
    if ends:
        check = _ends_other_readings(check, _checked_ends(check, ends, factor), analysis, ends, factor)
    if direction:
        check = _checked_direction(check, analysis, *direction)
    if every:
        check = replace(check, assumptions=(every, *check.assumptions))
    if worst:
        adjective, period = worst
        checker = _checked_worst_month if period == "month" else _checked_worst_day
        check = checker(check, analysis, adjective)
    if compared:
        check = _checked_compared(check, compared)
    if rate_base:
        check = replace(check, assumptions=(rate_base, *check.assumptions))
    if own_name:
        check = replace(check, assumptions=(own_name, *check.assumptions))
    if started:
        check = replace(check, assumptions=(started, *check.assumptions))
    return replace(_guessed_failure(check, analysis.roles.date), claim=said)


def _repeated_rows(check: ClaimCheck, analysis: Any, other: ClaimCheck) -> ClaimCheck:
    """A claim on a table with rows repeated exactly, checked with each such row counted once too (``other``).

    Only a table whose rows each hold a count (``counts_column``) is checked so: there a repeat is the same
    count given twice, where a table of records may hold two records alike. NYC's baby names give every 2014
    row twice: "578 girls named Sophia in 2014" holds once the repeats are dropped and 1,156 is the sum of
    every row. When the two readings give different verdicts the claim is
    ambiguous; when only the number moves, the check says so. Repeats outside the rows the claim uses
    change nothing and are not mentioned.
    """
    if other.compared == check.compared and (other.verified, other.reason) == (check.verified, check.reason):
        return check
    repeats = int(analysis.frame.duplicated().sum())
    shown = _counted("rows repeated exactly", repeats)
    if (other.verified, other.reason) == (check.verified, check.reason):
        note = (f"The table has {shown}; counting each once gives the same verdict "
                f"({_plain(other.compared) if other.compared is not None else 'no number'}).")
        return replace(check, assumptions=(*check.assumptions, note))
    alternatives = tuple(
        Alternative(reading, item.question or "", item.compared, item.verified,
                    item.answer.calculation if item.answer is not None else "")
        for reading, item in (("every row", check), ("each repeated row once", other))
    )
    values = "; ".join(f"{item.reading} gives {_plain(item.value) if item.value is not None else 'no number'}"
                       for item in alternatives)
    return ClaimCheck(
        check.claim, "unverified", "ambiguous_intent",
        f"The table has {shown}, and the verdict rests on whether they count: {values}.",
        claimed=check.claimed, question=check.question, alternatives=alternatives, source=check.source,
        clarification="Are the repeated rows separate records, or one record given twice?",
    )


# The last word of the noun phrase a question starts with, before a preposition, a participle, a relative
# word or a comma: "stops" in "stops in Seattle fell between 2018 and 2025", "Notices" in "Notices citing
# non-payment rose".
LEADING_HEAD = re.compile(
    r"\s*(?:the\s+)?(?:[A-Za-z]+\s+){0,3}?([A-Za-z]+)(?=\s+(?:in|at|for|across|from|on|within|among|during|"
    r"since|between|of|with|without|who|that|which|whose|where|[a-z]+ing|[a-z]+ed)\b|\s*[,;]|\s*$)",
    re.IGNORECASE)


# Notes that say a reading rests on a column ADA picked because nothing else fit, not one the claim names.
GUESSED_NOTES = ("the only column the question fits",)


def _guessed_failure(check: ClaimCheck, date: str | None = None) -> ClaimCheck:
    """A failed check whose reading rests on a guessed column, made missing_context.

    "Washington has had 223 federal disaster declarations" read through declarationTitle, the only column its
    words fit, gives 1 row of EASTERN WASHINGTON FIRES: a number that far off says the guess is wrong more
    often than the claim is. A guess that agrees is kept, as it rarely agrees by chance. So is the table's
    ``date`` guessed from "election" in "Sweden's 2022 election": it only places the year, as the date would
    whatever word named it.
    """
    if check.reason != "failed_check":
        return check
    if check.claimed is not None and check.compared is not None and check.claimed.holds(check.compared):
        # The guess gave the claim's number; what failed is a rank or a record beside it.
        return check
    guessed = next((note for note in check.assumptions if any(cue in note for cue in GUESSED_NOTES)), None)
    if guessed is None:
        return check
    plan = check.answer.plan if check.answer is not None else None
    dated = date is not None and guessed.startswith(f"Read the claim's words as {date}, ")
    if dated and plan is not None and date not in (
            plan.measure, plan.count_column, plan.dimension, plan.weight):
        return check
    return replace(
        check, reason="missing_context",
        message=f"{guessed} The claim names no column, so a number that disagrees may only show the column "
                f"is the wrong one: {check.message}",
        clarification="Which column is the number about?",
    )


# Words for a span of time, which change what a figure adds up to when left out of a claim.
TIME_UNIT_WORDS = frozenset({"hour", "hours", "day", "days", "week", "weeks", "month", "months", "year",
                             "years", "minute", "minutes"})


def _beyond_every_row(
    answer: QueryAnswer, number: ClaimedNumber, computed: float, measure: str | None, asked: str, frame: Any
) -> bool:
    """Whether a claim lies beyond what any narrower set of rows could give: above the highest or the total
    of a column of non-negative values over every row it keeps, or below the lowest."""
    plan = answer.plan
    if measure is None or plan.measure != measure or not _named(asked, measure) or plan.dimension:
        return False
    if plan.intent != "aggregate" or plan.aggregation not in ("max", "min", "sum"):
        return False
    if plan.aggregation == "min":
        return number.high < computed
    if plan.aggregation == "sum" and bool((frame[measure].dropna() < 0).any()):
        return False
    return number.low > computed


def _described_as(frame: Any, noun: str) -> str | None:
    """The one text every row of the table gives that names the noun, which says what its figures are of:
    "Passenger cars - per thousand inhabitants" for "cars"."""
    from .folder import singular  # noqa: PLC0415

    forms = {noun.lower(), singular(noun.lower())}
    for column in frame.columns:
        values = frame[column].dropna()
        if values.dtype != object or values.nunique() != 1:
            continue
        text = str(values.iloc[0])
        if forms & set(re.findall(r"[a-z]+", text.lower())):
            return text
    return None


# A rate's base in a claim: "per 1,000 inhabitants", "for every 1,000 residents".
RATE_BASE = re.compile(
    r"\s*\b(?:per|for\s+every|for\s+each|in\s+every)\s+(?P<base>1,000|1000|one\s+thousand|a\s+thousand|thousand|"
    r"100,000|100000|hundred\s+thousand|100|one\s+hundred|a\s+hundred|1,000,000|million|a\s+million)\s+"
    r"(?:inhabitants|people|residents|persons|population|head|capita)\b",
    re.IGNORECASE,
)
# The words a table may write a base in.
BASE_WORDS = {"1,000": "thousand", "1000": "thousand", "100,000": "hundred thousand",
              "100000": "hundred thousand", "100": "hundred", "1,000,000": "million"}


# Verbs of beginning, and the name words of a date column that says when each row began something.
START_VERBS = re.compile(r"\b(?:began|begun|begin|begins|started|starts|start|joined|joins)\b", re.IGNORECASE)
START_NAME_WORDS = frozenset({"start", "begin", "began", "first", "since", "joined"})


def _start_dates(text: str, frame: Any) -> tuple[str, str | None]:
    """The claim with "began reporting to NIBRS" written as the date column for it, nibrs_start_date.

    A verb of beginning followed, within four words, by a word of a date column's name that also says start
    ("nibrs" of nibrs_start_date) names that column: "in 2021" is then the year it began.
    """
    from pandas.api.types import is_datetime64_any_dtype  # noqa: PLC0415

    verb = START_VERBS.search(text)
    if verb is None:
        return text, None
    after = list(re.finditer(r"[A-Za-z]+", text[verb.end():]))[:4]
    for column in frame.columns:
        words = set(normalized_name(str(column)).split())
        if not is_datetime64_any_dtype(frame[column]) or not words & START_NAME_WORDS:
            continue
        named = [match for match in after if match.group().lower() in words - START_NAME_WORDS - {"date"}]
        if not named:
            continue
        end = verb.end() + named[-1].end()
        note = f'Read "{text[verb.start():end]}" as {column}, the date each row began it.'
        return f"{text[:verb.start()]}{column}{text[end:]}", note
    return text, None


def _own_name(text: str, source: Any, row_noun: str | None) -> tuple[str, str | None]:
    """The claim without the table's own name before its rows' noun, and the note saying so.

    "There were 5,034 film permits in 2025" in Film_Permits.csv: "film permits" is what every row is, so
    "film" picks no rows, though Category holds Film.
    """
    from .folder import singular  # noqa: PLC0415

    if source is None or not row_noun or not getattr(source, "name", None):
        return text, None
    words = re.findall(r"[a-z]+", Path(str(source.name)).stem.lower())
    if len(words) < 2 or singular(words[-1]) != singular(row_noun.lower()):
        return text, None
    joined = r"[\s_-]+".join(words[:-1])
    found = re.search(rf"\b({joined})\s+(?={words[-1]}\b)", text, re.IGNORECASE)
    if found is None:
        return text, None
    note = (f'Read "{found.group(1)} {words[-1]}" as the table\'s own name ({source.name}), so '
            f'"{found.group(1)}" picks no rows.')
    return f"{text[:found.start()]}{text[found.end():]}", note


def _table_rate_base(text: str, frame: Any) -> tuple[str, str | None]:
    """The claim without a rate's base the table already gives its figures in ("per 1,000 inhabitants"
    beside "Passenger cars - per thousand inhabitants"), and a note saying so."""
    found = RATE_BASE.search(text)
    if not found:
        return text, None
    base = re.sub(r"^(?:one|a)\s+", "", found.group("base").lower())
    word = BASE_WORDS.get(base, base)
    pattern = re.compile(rf"\bper\s+(?:{re.escape(word)}|{re.escape(base)})\b", re.IGNORECASE)
    named = [str(column) for column in frame.columns if pattern.search(str(column))]
    for column in frame.columns:
        values = frame[column].dropna()
        constant = values.dtype == object and values.nunique() == 1
        if not named and constant and pattern.search(str(values.iloc[0])):
            named.append(str(values.iloc[0]))
    if not named:
        return text, None
    note = f'Read "{found.group().strip()}" as the base the table gives its figures in: "{named[0]}".'
    return f"{text[:found.start()]}{text[found.end():]}", note


# "since records began in 1856": the year the rows start from.
RECORDS_BEGAN = re.compile(
    r"\bsince\s+(?:the\s+)?(?:records?|record-keeping|recordkeeping|measurements?|observations?|the\s+series)\s+"
    r"(?:began|started|start|begin)(?:\s+in\s+(?P<year>(?:1[6-9]|20)\d{2}))?",
    re.IGNORECASE,
)


# A reading in degrees: "-46.6 °C", "30 degrees Celsius".
DEGREES = r"-?\d[\d,]*(?:\.\d+)?\s*(?:[°º]\s*[CF]\b|degrees?\s+(?:Celsius|Fahrenheit|C|F)\b)"
# "fell to -46.6 °C" is the lowest reading, not a fall from one period to the next; "hit 37 °C" the highest.
TEMPERATURE_EXTREMES = (
    (re.compile(rf"\b(?:fell|dropped|dipped|plunged|plummeted|sank)\s+to\s+(?={DEGREES})", re.IGNORECASE),
     "had a lowest temperature of "),
    (re.compile(rf"\b(?:rose|climbed|soared|hit|reached|topped\s+out\s+at|peaked\s+at)\s+(?:to\s+)?"
                rf"(?={DEGREES})", re.IGNORECASE),
     "had a highest temperature of "),
)
# "fell below -40 °C on 5 days" is a threshold for the day's low, not a fall from one period to the next.
THRESHOLD_DEGREES = (r"(?P<number>-?\d[\d,]*(?:\.\d+)?)\s*"
                     r"(?P<unit>[°º]\s*(?P<scale>[CF])\b|degrees?\s+(?P<word>Celsius|Fahrenheit|C|F)\b)")
TEMPERATURE_THRESHOLDS = (
    (re.compile(rf"\b(?:fell|dropped|dipped|plunged|sank|falls|drops|dips)\s+(?P<bound>below|under)\s+"
                rf"{THRESHOLD_DEGREES}", re.IGNORECASE), "low"),
    (re.compile(rf"\b(?:rose|climbed|soared|rises|climbs)\s+(?P<bound>above|over|past)\s+{THRESHOLD_DEGREES}",
                re.IGNORECASE), "high"),
)
# "a high of 24 °C": the day's high temperature; "averages a high of" its average.
HIGH_OR_LOW_OF = re.compile(
    rf"\b(?P<average>averag(?:e|es|ed|ing)\s+)?(?:an?\s+)?(?P<side>highs?|lows?)\s+of\s+"
    rf"(?P<qualifier>(?:about|around|nearly|almost|over|under|roughly|some)\s+)?(?={DEGREES})",
    re.IGNORECASE,
)


def _temperature_words(text: str, frame: Any) -> str:
    """Words for a reading in degrees as a question can say them: "fell to -46.6 °C" as "had a lowest
    temperature of -46.6 °C", and "a high of 24 °C" as "a high temperature of 24 °C". A threshold ("fell
    below -40 °C") loses its unit when the table's temperatures are in it: it is then no word to look up."""
    units = {column_unit(str(column)) for column in frame.columns}

    def threshold(found: re.Match, side: str) -> str:
        scale = (found.group("scale") or found.group("word") or "").upper()[:1]
        unit = "" if f"°{scale.lower()}" in units else f" {found.group('unit')}"
        return f"had a {side} temperature {found.group('bound')} {found.group('number')}{unit}"

    for pattern, side in TEMPERATURE_THRESHOLDS:
        text = pattern.sub(lambda found, side=side: threshold(found, side), text)
    for pattern, said in TEMPERATURE_EXTREMES:
        text = pattern.sub(said, text)
    def said(found: re.Match) -> str:
        average = "average " if found.group("average") else "a "
        return f"{average}{found.group('side')} temperature of {found.group('qualifier') or ''}"

    return HIGH_OR_LOW_OF.sub(said, text)


def _records_began(text: str) -> str:
    """"since records began in 1856" as "since 1856"; with no year, every row, so the words go."""
    def since(found: re.Match) -> str:
        return f"since {found.group('year')}" if found.group("year") else ""

    return RECORDS_BEGAN.sub(since, _any_year_record(text))


# A verb of paying or carrying before a rate: "TIPS paid an average interest rate of 1.1%" and "bonds still
# carried an average interest rate above 6%" had that rate; a rate is not a payment, which "paid" names before
# an amount.
PAID_RATE = re.compile(
    r"\b(?:still\s+)?(?:paid|pays|pay|offered|offers|yielded|yields|carried|carries|carry)\s+(?=(?:an?\s+|"
    r"the\s+)?(?:average\s+|mean\s+)?(?:[a-z-]+\s+){0,2}rates?\b)", re.IGNORECASE)


def _paid_rate(text: str) -> str:
    """"paid an average interest rate" or "still carried an average interest rate" as "had an average
    interest rate"."""
    return PAID_RATE.sub("had ", text)


# A strong verb of change before "by", "to" or a percent: "collapsed by more than 80%" fell by it; "soared
# 40%" rose by it. "jumped 8.95 metres" is no change.
_ABOUT = r"(?:(?:about|nearly|almost|around|roughly|some|over|under|more\s+than|less\s+than|just\s+over|" \
         r"just\s+under)\s+)?"
STRONG_CHANGE = re.compile(
    r"\b(?:(?P<fall>plunged|plummeted|slumped|tumbled|collapsed)|(?P<rise>soared|surged|jumped|rocketed|"
    rf"skyrocketed))(?=\s+(?:(?:by|to)\s+{_ABOUT}[$€£]?\d|{_ABOUT}\d[\d,.]*\s*(?:%|per\s*cent\b|percent\b)))",
    re.IGNORECASE,
)


def _plain_change(text: str) -> str:
    """Strong verbs of change before a number as "fell" or "rose", and "hit a record high of" as
    "reached"."""
    return RECORD_REACHED.sub("reached ", STRONG_CHANGE.sub(
        lambda found: "fell" if found.group("fall") else "rose", text))


# "hit a record high of more than 63,000": the level reached. The record itself is checked against the other
# years from the claim as written (``_checked_record_year``).
RECORD_REACHED = re.compile(
    r"\b(?:reached|hit|set|touched)\s+(?:a\s+|an\s+)?(?:new\s+)?(?:all-time\s+)?record(?:\s+(?:high|low))?"
    r"(?:\s+of)?\s+(?=(?:(?:about|nearly|almost|around|roughly|some|over|under|more\s+than|less\s+than|"
    r"just\s+over|just\s+under)\s+)?[$€£]?\d)",
    re.IGNORECASE,
)


# "the fewest of any year", alone or "since records began": a record low over every year, as "a record low"
# is.
ANY_YEAR_RECORD = re.compile(
    r"\bthe\s+(?:(?P<low>fewest|lowest|least)|(?P<high>most|highest|largest|greatest))\s+(?:of|in|for)\s+"
    r"any\s+(?:other\s+)?year(?:\s+since\s+(?:records\s+began(?:\s+in\s+(?:1[89]|20)\d{2})?|"
    r"(?P<since>(?:1[89]|20)\d{2})))?(?=\s*[,.;]|\s*$)",
    re.IGNORECASE,
)


def _any_year_record(text: str, every: bool = False) -> str:
    """"fell to 778 in 2020, the fewest of any year since records began in 1997" as "..., a record low"; with
    ``every``, as "an every-year record low", which the record check sets against every year. A year after
    "since" ("the most of any year since 2016") keeps the years from it: "an every-year record high since
    2016"."""
    record = "an every-year record" if every else "a record"

    def written(found: re.Match) -> str:
        since = f" since {found.group('since')}" if every and found.group("since") else ""
        return f"{record} {'low' if found.group('low') else 'high'}{since}"

    return ANY_YEAR_RECORD.sub(written, text)


# ", three times the long-run pace": a clause after the number that compares it with another figure.
COMPARED_TAIL = re.compile(
    r",\s*(?P<tail>(?:about\s+|nearly\s+|almost\s+|roughly\s+|more\s+than\s+)?(?:twice|double|triple|half|"
    r"(?:two|three|four|five|ten|\d+(?:\.\d+)?)\s+times)\s+(?:as\s+\w+\s+as\s+|the\s+|that\s+|its\s+)"
    r"[^,.;]*?)\s*[.!]?\s*$",
    re.IGNORECASE,
)


def _compared_tail(text: str) -> tuple[str, str | None]:
    """The claim without a closing clause that compares its number with another ("three times the long-run
    pace"), and the clause."""
    found = COMPARED_TAIL.search(text)
    if not found:
        return text, None
    return f"{text[:found.start()].rstrip()}.", found.group("tail")


def _checked_compared(check: ClaimCheck, compared: str) -> ClaimCheck:
    """A claim that also compares its number with another figure: a wrong number fails it, but a right one
    does not verify the comparison, which ADA does not check."""
    note = f'Set aside "{compared}", which compares the number with another figure.'
    if check.reason == "failed_check":
        return replace(check, assumptions=(*check.assumptions, note))
    if not check.verified:
        return check
    return replace(
        check, status="unverified", reason="missing_context",
        clarification="Send the comparison as its own claim.",
        message=f"{check.message} The claim also says \"{compared}\", a comparison with another figure "
                f"that ADA does not check.",
    )


# "The first of July 2025 was the worst day of the year on the Dutch railways, with fifty-one disruptions".
WORST_DAY = re.compile(
    r"^\s*(?:the\s+)?(?P<when>.+?)\s+was\s+(?:the\s+)?(?P<adjective>worst|busiest|biggest|heaviest|deadliest|"
    r"quietest|calmest)\s+day(?:\s+of\s+the\s+year|\s+on\s+record|\s+ever|\s+of\s+(?:19|20)\d{2})?"
    r"(?P<scope>[^,]*),\s*with\s+(?P<number>.+?)\s+(?P<noun>[A-Za-z][A-Za-z-]*)\s*[.!]?\s*$",
    re.IGNORECASE,
)


# A weekday written with a date: "Thursday 12 March 2015", "12 March 2015, a Thursday".
_A_DAY = re.sub(r"\(\?P<\w+>", "(?:", DAY_PATTERN.pattern)
_WEEKDAY = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
WEEKDAY_WITH_DATE = re.compile(
    rf"\b(?P<before>{_WEEKDAY}),?\s+(?P<date>{_A_DAY})"
    rf"|(?P<date2>{_A_DAY}),?\s+(?:a|on\s+a)\s+(?P<after>{_WEEKDAY})\b",
    re.IGNORECASE,
)


def _wrong_weekday(text: str) -> str | None:
    """Why the weekday a claim gives a date is not the date's own ("12 March 2015 was a Thursday, not a
    Friday"), or None."""
    from .nlq import day_named  # noqa: PLC0415

    for found in WEEKDAY_WITH_DATE.finditer(text):
        written = found.group("before") or found.group("after")
        date = DAY_PATTERN.search(found.group("date") or found.group("date2"))
        day = day_named(date) if date else None
        if day is not None and day.day_name().lower() != written.lower():
            return f"{day:%-d %B %Y} was a {day.day_name()}, not a {written.capitalize()}."
    return None


# "The busiest day of the year was Thursday 12 March 2015, when 159 tickets were written": the day named last.
WORST_DAY_LAST = re.compile(
    r"^\s*the\s+(?P<adjective>worst|busiest|biggest|heaviest|deadliest|quietest|calmest)\s+day"
    r"(?:\s+of\s+the\s+year|\s+on\s+record|\s+in\s+the\s+records|\s+ever)?(?P<scope>[^,]*?)\s+was\s+"
    r"(?P<when>[^,]+?),\s*(?:when|with)\s+(?P<number>.+?)\s+(?P<noun>[A-Za-z][A-Za-z-]*)"
    r"(?:\s+(?:were|was)\s+[a-z]+)?\s*[.!]?\s*$",
    re.IGNORECASE,
)
# "March was the busiest month for Ballarat's parking officers in 2015, with 2,259 tickets".
WORST_MONTH = re.compile(
    rf"^\s*(?P<when>{MONTH_WORD})\s+was\s+(?:the\s+)?(?P<adjective>worst|busiest|biggest|heaviest|deadliest|"
    r"quietest|calmest)\s+month(?:\s+of\s+the\s+year)?(?P<scope>[^,]*),\s*with\s+(?P<number>.+?)\s+"
    r"(?P<noun>[A-Za-z][A-Za-z-]*)\s*[.!]?\s*$",
    re.IGNORECASE,
)
WEEKDAY_FIRST = re.compile(r"^\s*(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday),?\s+",
                           re.IGNORECASE)


def _worst_day(text: str) -> tuple[str, tuple[str, str] | None]:
    """"D was the worst day, with N things" as "There were N things on D", with the word for most or least and
    the period ("day" or "month"). A weekday before the date ("Thursday 12 March 2015") goes when it is the
    date's own."""
    from .nlq import day_named  # noqa: PLC0415

    found = WORST_DAY.match(text) or WORST_DAY_LAST.match(text)
    if found and (day := DAY_PATTERN.search(found.group("when"))):
        when = found.group("when").strip()
        weekday = WEEKDAY_FIRST.match(when)
        named = day_named(day)
        if weekday and named is not None and named.day_name().lower() == weekday.group("weekday").lower():
            when = when[weekday.end():]
        scope = found.group("scope").rstrip()
        rewritten = f"There were {found.group('number')} {found.group('noun')} on {when}{scope}."
        return rewritten, (found.group("adjective").lower(), "day")
    found = WORST_MONTH.match(text)
    if found:
        scope = found.group("scope").rstrip()
        number, noun, when = found.group("number"), found.group("noun"), found.group("when")
        # "March was the busiest month in 2015": the year goes with the month.
        if year := re.search(r"\s+in\s+((?:19|20)\d{2})\b", scope):
            when, scope = f"{when} {year.group(1)}", scope[:year.start()] + scope[year.end():]
        return f"There were {number} {noun} in {when}{scope}.", (found.group("adjective").lower(), "month")
    return text, None


def _checked_worst_month(check: ClaimCheck, analysis: Any, adjective: str) -> ClaimCheck:
    """A month the claim calls the worst or busiest, checked to have the most of the claim's rows of any month
    of its year (the fewest for the quietest)."""
    import pandas as pd  # noqa: PLC0415

    from .nlq import _apply_filters  # noqa: PLC0415

    if not check.verified or check.answer is None:
        return check
    plan, date = check.answer.plan, analysis.roles.date
    working, _ = _apply_filters(analysis.frame, replace(plan, month=None), analysis.roles) if (
        date is not None and plan.month is not None) else (None, [])
    dated = working.loc[working[date].notna()] if working is not None else None
    if dated is None or (plan.year is None and dated[date].dt.year.nunique() != 1):
        message = (f"The claim calls a month the {adjective}, and ADA could not count every month of one "
                   f"year the same way: {check.message}")
        return replace(check, status="unverified", reason="missing_context",
                       clarification="Which year is meant?", message=message)
    by_month = dated.groupby(dated[date].dt.month)
    counts = by_month[plan.weight].sum() if plan.weight else by_month.size()
    fewest = adjective in ("quietest", "calmest")
    best_month = int(counts.idxmin() if fewest else counts.idxmax())
    best, claimed = float(counts.loc[best_month]), float(counts.get(plan.month, 0))
    word, year = ("fewest" if fewest else "most"), int(dated[date].dt.year.iloc[0])
    names = pd.Timestamp(year, best_month, 1).month_name(), pd.Timestamp(year, plan.month, 1).month_name()
    if claimed == best:
        note = f"Checked that {names[1]} has the {word} of any month of {year}, {_plain(best)}."
        return replace(check, assumptions=(*check.assumptions, note))
    return replace(
        check, status="unverified", reason="failed_check", clarification=None,
        message=f"{check.message} But {names[0]} has {_plain(best)}, so {names[1]} is not the {adjective} "
                f"month of {year}.",
    )


def _checked_worst_day(check: ClaimCheck, analysis: Any, adjective: str) -> ClaimCheck:
    """A day the claim calls the worst or busiest, checked to have the most of the claim's rows of any day of
    its year (the fewest for the quietest)."""
    import pandas as pd  # noqa: PLC0415

    from .nlq import _apply_filters  # noqa: PLC0415

    if not check.verified or check.answer is None:
        return check
    plan, date = check.answer.plan, analysis.roles.date
    days = [item for item in plan.conditions if isinstance(item, Condition) and item.column == date
            and item.part == "date" and item.operator == "=" and item.value is not None]
    if date is None or len(days) != 1:
        return replace(check, status="unverified", reason="missing_context",
                       clarification="Which day is meant?",
                       message=f"The claim calls a day the {adjective}, and ADA could not count every day "
                               f"the same way: {check.message}")
    day = pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(days[0].value))
    rest = replace(plan, conditions=tuple(item for item in plan.conditions if item is not days[0]), year=None,
                   month=None)
    working, _ = _apply_filters(analysis.frame, rest, analysis.roles)
    dated = working.loc[working[date].notna()]
    dated = dated.loc[dated[date].dt.year == day.year]
    by_day = dated.groupby(dated[date].dt.normalize())
    counts = by_day[plan.weight].sum() if plan.weight else by_day.size()
    fewest = adjective in ("quietest", "calmest")
    best_day = counts.idxmin() if fewest else counts.idxmax()
    best = float(counts.loc[best_day])
    claimed = float(counts.get(day, 0))
    word = "fewest" if fewest else "most"
    if claimed == best:
        note = f"Checked that {day:%d %B %Y} has the {word} of any day of {day.year}, {_plain(best)}."
        return replace(check, assumptions=(*check.assumptions, note))
    return replace(
        check, status="unverified", reason="failed_check", clarification=None,
        message=f"{check.message} But {best_day:%d %B %Y} has {_plain(best)}, so {day:%d %B %Y} is not the "
                f"{adjective} day of {day.year}.",
    )


# "Half of all rail disruptions were over within 51 minutes": the median, said as half the rows under it.
HALF_WITHIN = re.compile(
    r"^\s*half\s+(?:of\s+)?(?:all\s+)?(?:the\s+)?(?P<what>.+?)\s+(?:were|was|are|is|took|lasted)\s+"
    r"(?:(?:over|done|finished|resolved|cleared|seen|treated|answered)\s+)?"
    r"(?:within|in\s+under|in\s+less\s+than|under|less\s+than|below)\s+(?P<number>\d[\d,.]*)\s+"
    r"(?P<unit>[A-Za-z]+)(?P<rest>.*)$",
    re.IGNORECASE,
)


def _half_within(text: str) -> str:
    """"Half of the X were over within 51 minutes" as "The median of the X was 51 minutes"."""
    found = HALF_WITHIN.match(text)
    if not found:
        return text
    number, unit = found.group("number"), found.group("unit")
    return f"The median of {found.group('what')} was {number} {unit}{found.group('rest')}"


# "the 15 parks", "America's 15 most visited national parks": a count of things the claim spans.
EVERY_VALUE = re.compile(
    r"\b(?P<lead>the|all|all\s+the|these|those|[A-Z][A-Za-z]+['’]s)\s+(?P<count>\d{1,3}(?:,\d{3})*)\s+"
    r"(?P<words>(?:[A-Za-z-]+\s+){0,3}?)(?P<noun>[A-Za-z]+s)\b"
)


def _every_value(text: str, frame: Any) -> tuple[str, str | None]:
    """The claim without a count that is every value of a column, and a note saying so.

    "Between them, the 15 parks list 886 bird species": ParkName holds 15
    values, so "the 15 parks" spans the table rather than giving a second
    number. A count that is not the column's number of values stays, and the
    claim is read with both numbers, as before.
    """
    from .folder import singular  # noqa: PLC0415

    for found in EVERY_VALUE.finditer(text):
        noun = singular(found.group("noun").lower())
        count = int(found.group("count").replace(",", ""))
        named = [column for column in frame.columns
                 if noun in normalized_name(str(column)).split() and frame[column].nunique() == count]
        if count < 2 or not named:
            continue
        note = (f'Read "{found.group(0)}" as every value of {named[0]}, which holds {count:,}, so it picks '
                "no rows and is no number of the claim's own.")
        # The words go with the preposition before them: "among the 15 parks" spans them all.
        before = re.search(r"\b(?:among|across|for|in|of|from|at)\s+$", text[:found.start()], re.IGNORECASE)
        start = before.start() if before else found.start()
        rest = f"{text[:start].rstrip()} {text[found.end():].lstrip()}"
        return re.sub(r"\s+([,.;:])", r"\1", rest).strip(), note
    return text, None


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
    if (other.verified, other.reason) == (check.verified, check.reason):
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


def _repeated_dates(
    check: ClaimCheck, analysis: Any, text: str, asked: str, number: ClaimedNumber, source: Source | None,
    choices: dict[str, str],
) -> ClaimCheck:
    """A claim on a table of one row per date where a date or two repeat, checked with each counted once.

    BTS's monthly table gives April 2023 twice; a yearly total counts it twice
    unless the repeat is left out. When the two readings give different
    verdicts, the claim is ambiguous; when they agree, the check says so.
    """
    date = analysis.roles.date
    if not date or date not in analysis.frame.columns:
        return check
    counts = analysis.frame[date].value_counts()
    repeated = counts[counts > 1]
    if repeated.empty or len(counts) < 20 or (counts == 1).mean() < 0.95 or int(counts.max()) > 2:
        return check
    once = replace(analysis, frame=analysis.frame.drop_duplicates(subset=[date], keep="first"))
    other = _checked_dates(_checked_table(once, text, asked, number, source, choices, None), once)
    shown = ", ".join(str(value)[:10] for value in repeated.index[:3])
    if (other.verified, other.reason) == (check.verified, check.reason):
        note = (f"{date} repeats {shown}; counting each date once gives the same verdict "
                f"({_plain(other.compared) if other.compared is not None else 'no number'}).")
        return replace(check, assumptions=(*check.assumptions, note))
    alternatives = tuple(
        Alternative(reading, asked, item.compared, item.verified,
                    item.answer.calculation if item.answer is not None else "")
        for reading, item in ((f"every row, {shown} counted twice", check), ("each date once", other))
    )
    values = "; ".join(f"{item.reading} gives {_plain(item.value) if item.value is not None else 'no number'}"
                       for item in alternatives)
    return ClaimCheck(
        text, "unverified", "ambiguous_intent",
        f"{date} repeats {shown} in a table of one row per date, and the verdict rests on whether the repeat "
        f"counts: {values}.",
        claimed=number, question=asked, alternatives=alternatives, source=source,
        clarification=f"Is {shown} one row or two?",
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
        recorded = _recorded_scale(analysis, check.answer)
        return _checked_base(check, *base, frame=analysis.frame, factor=recorded[0] if recorded else 1.0,
                             text=text)
    if question is None and PEAK_VERB.search(_most_as_peak(text)) and (
        check.verified or check.reason == "failed_check"
    ):
        return _checked_peak(check, analysis, text)
    if question is None and check.verified and check.answer is not None and RECORD_EXTREME.search(text) and (
        check.answer.plan.measure
    ):
        return _checked_extreme(check, analysis, RECORD_EXTREME.search(text))
    if question is None and check.verified and check.answer is not None and SUPERLATIVE.search(text) and (
        AGAINST_ALL.search(text) and not _ranked(check.answer)
    ):
        # "raised about $96.5 million, the most of any Senate candidate": the same total for every candidate
        # the closing words keep.
        ranked = _checked_rank_tail(check, analysis, text)
        if ranked is not None:
            return ranked
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
        about = re.match(r"(?:about|around|roughly|approximately|some|nearly|almost)\s+", ends[0][1], re.I)
        if about and not re.match(r"[A-Za-z]", ends[1][1]):
            # "from about 74,000 to 149,000": "about" is said of both ends.
            ends = (ends[0], ("latest", f"{about.group(0)}{ends[1][1]}"))
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


def _checked_ends(check: ClaimCheck, ends: tuple[tuple[str, str], ...], factor: float = 1.0) -> ClaimCheck:
    """A change checked, then the values the claim says it went from and to.

    Each end must agree with the answer's first or last value, times
    ``factor`` when a unit column gives the values in thousands. A change that
    agrees with ends that do not fails; a change with no first and last value
    in its answer cannot show its ends, so it is not verified.
    """
    if not (check.verified or check.reason == "failed_check"):
        return check
    figures = dict(check.answer.figures) if check.answer is not None else {}
    notes, wrong, far = [], [], 0
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
        side, given = ("first" if name == "previous" else "last"), _plain(float(value) * factor)
        if stated.holds(float(value) * factor):
            notes.append(f"The claim's {side} value, {written}, agrees with the data's {given}.")
        else:
            wrong.append(f"the claim's {side} value is {written}, and the data give {given}")
            low, high = sorted((abs(float(value) * factor), abs(stated.value)))
            far += high >= FAR_ENDS * low
    if ends and far == len(ends):
        # "fell by more than half, from 280 to 123" read as deaths from Duty related illness, 2 and 4: ends
        # that far off say the reading counts something else, not that the claim is false.
        return replace(
            check, status="unverified", reason="missing_context",
            message=f"The claim's ends are far from the reading's ({'; '.join(wrong)}), so the reading may "
                    f"not count what the claim does: {check.message}",
            clarification="Which rows or column does the claim count?",
        )
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


# How many times larger one end must be than the other to be far from it.
FAR_ENDS = 10


def _ends_hold(check: ClaimCheck, ends: tuple[tuple[str, str], ...], factor: float = 1.0) -> bool:
    """Whether the answer's first and last values agree with each end the claim gives."""
    figures = dict(check.answer.figures) if check.answer is not None else {}
    for name, written in ends:
        found = _candidates(written, [])
        if figures.get(name) is None or not found:
            return False
        item = found[0]
        if not claimed_number(written[item.qualifier_start:], item.match, item.qualifier, None).holds(
            float(figures[name]) * factor
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


# Words of a date column's name for when a thing began, and for when it ended.
START_WORDS = frozenset({"start", "started", "begin", "began", "onset", "opened", "from"})
END_WORDS = frozenset({"end", "ended", "finish", "finished", "stop", "stopped", "closed", "resolved", "until",
                       "to"})
END_CLAIMED = re.compile(r"\b(?:ended|finished|stopped|closed|resolved|were\s+over|was\s+over)\b", re.I)


# Name words of a date the figures were published or fetched on.
PUBLISHED_WORDS = frozenset({
    "release", "released", "published", "publication", "vintage", "revision", "revised", "extracted",
    "uploaded", "downloaded", "retrieved", "refreshed", "loaded",
})


# The least share of the rows the date holds that another date column must hold to date the same rows.
OTHER_DATES_SHARE = 0.5


def _other_dates(frame: Any, date: str, claim: str) -> list[str]:
    """The table's other date columns a claim's year or day may mean, besides ``date``.

    A date the figures were published on (month_of_release) is never one, nor is a time of day with no
    date (a crash's Time), nor the date the data are as of (data_as_of), nor a date only some rows have
    (Constraints Date, on 9% of San Francisco's eviction notices).
    """
    from pandas.api.types import is_datetime64_any_dtype  # noqa: PLC0415

    dated = int(frame[date].notna().sum())
    others = [str(column) for column in frame.columns
              if str(column) != date and is_datetime64_any_dtype(frame[column])
              and not holds_undated(frame[column])
              and not set(normalized_name(str(column)).split()) & PUBLISHED_WORDS
              and not re.search(r"\bas of\b", normalized_name(str(column)))
              and frame[column].notna().sum() >= OTHER_DATES_SHARE * dated]
    if set(normalized_name(date).split()) & START_WORDS and not END_CLAIMED.search(claim):
        # "51 disruptions on 1 July 2025" counts those that started that day, not those that ended then.
        others = [other for other in others if not set(normalized_name(other).split()) & END_WORDS]
    return others


def _ends_other_readings(
    check: ClaimCheck, ended: ClaimCheck, analysis: Any, ends: tuple[tuple[str, str], ...], factor: float
) -> ClaimCheck:
    """A change that agrees with ends that do not (``ended``), made again under the other readings its
    words allow: by each other date column, and, when rows are counted and the table's ID repeats, with each
    ID once.

    "Terry stops in Seattle fell by more than half between 2018 and 2025, from 8,798 to 3,651": by Occurred
    Date the rows give 3,670 in 2025; each Terry Stop ID once, by Reported Date, gives 3,651. When one
    reading gives the change and both ends, the claim does not say which it means, and gets
    `ambiguous_intent`.
    """
    from .nlq import execute_plan  # noqa: PLC0415

    if ended.reason != "failed_check" or not check.verified or check.answer is None or check.claimed is None:
        return ended
    plan, roles, frame = check.answer.plan, analysis.roles, analysis.frame
    if plan.intent != "growth" or not roles.date or roles.date not in frame.columns:
        return ended
    dates = [roles.date, *_other_dates(frame, roles.date, check.claim)]
    identifier = roles.identifier
    counted = plan.measure is None and plan.aggregation == "count" and not plan.count_column
    repeats = counted and identifier in frame.columns and frame[identifier].dropna().duplicated().any()
    readings = [(date, once) for date in dates for once in ((False, True) if repeats else (False,))][1:]
    for date, once in readings:
        swapped = replace(plan, conditions=tuple(
            replace(item, column=date) if isinstance(item, Condition) and item.column == roles.date else item
            for item in plan.conditions
        ))
        rows = frame.drop_duplicates(subset=[identifier]) if once else frame
        try:
            answer = execute_plan(swapped, rows, replace(roles, date=date))
        except (KeyError, ValueError):
            continue
        value, _ = _comparable(answer, check.claimed)
        if value is None:
            continue
        if _says_fall(check.claim, check.claimed):
            value = -value
        other = replace(check, answer=answer, compared=value)
        if not check.claimed.holds(value) or not _ends_hold(other, ends, factor):
            continue
        reading = f"by {date}" + (f", each {identifier} once" if once else "")
        alternatives = (
            Alternative(f"by {roles.date}", check.question or "", check.compared, False,
                        check.answer.calculation),
            Alternative(reading, check.question or "", value, True, answer.calculation),
        )
        return ClaimCheck(
            check.claim, "unverified", "ambiguous_intent",
            f"{ended.message} Read {reading}, the change and both ends agree, and the claim does not say "
            "which it means.",
            claimed=check.claimed, question=check.question, alternatives=alternatives, source=check.source,
            assumptions=check.assumptions,
            clarification=f"Which date counts, and is each {identifier} counted once?" if repeats
            else "Which date counts?",
        )
    return ended


def _checked_dates(check: ClaimCheck, analysis: Any) -> ClaimCheck:
    """A verdict that rests on a year or a day, made again with the table's other date columns.

    "Ward 1 recorded 15 homicides in 2024" counts 14 by START_DATE and 15 by
    REPORT_DAT. When another date column turns the verdict, the claim does not
    say which date it means, and neither verdict stands.
    """
    from .nlq import execute_plan  # noqa: PLC0415

    decided = check.verified or check.reason == "failed_check"
    if not decided or check.answer is None or check.claimed is None or check.compared is None:
        return check
    plan, roles, frame = check.answer.plan, analysis.roles, analysis.frame
    date = roles.date
    conditions = [item for item in plan.conditions if isinstance(item, Condition)]
    if date is None or not (plan.year or plan.month or any(item.column == date for item in conditions)):
        return check
    for other in _other_dates(frame, date, check.claim):
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


# Words that make a day's reading a record: "the coldest reading in the records came on 12 January 2024".
RECORD_WORDS = re.compile(
    r"\b(?:coldest|hottest|warmest|wettest|driest|windiest|snowiest|highest|lowest|largest|smallest|biggest|"
    r"record|ever)\b",
    re.IGNORECASE,
)


def _checked_record_day(check: ClaimCheck, analysis: Any, said: str) -> ClaimCheck:
    """A verified lowest or highest reading on a named day that the claim, as written (``said``), calls a
    record, checked against every day: the day's reading must be the extreme of all of them too."""
    from .nlq import execute_plan  # noqa: PLC0415

    if not check.verified or check.answer is None or check.claimed is None or check.compared is None:
        return check
    plan = check.answer.plan
    days = [item for item in plan.conditions if isinstance(item, Condition) and item.part == "date"]
    if plan.aggregation not in ("min", "max") or not days or not RECORD_WORDS.search(said):
        return check
    every = replace(plan, conditions=tuple(item for item in plan.conditions if item not in days))
    try:
        answer = execute_plan(every, analysis.frame, analysis.roles)
    except (KeyError, ValueError):
        return check
    value, _ = _comparable(answer, check.claimed)
    side = "lowest" if plan.aggregation == "min" else "highest"
    if value is None:
        message = (f"The claim calls the day's reading a record, and ADA could not find the {side} over "
                   f"every day to compare it with. {check.message}")
        return replace(check, status="unverified", reason="missing_context", message=message)
    if check.claimed.holds(value):
        held = f"The {side} over every day is {_plain(value)} too, so the day holds the record."
        return replace(check, assumptions=(*check.assumptions, held))
    return replace(
        check, status="unverified", reason="failed_check",
        message=f"The day gives {_plain(check.compared)}, but the {side} over every day is {_plain(value)}, "
                f"so the day does not hold the record. {check.message}",
        compared=value, agreement="mismatch",
    )


# "hit a record of about 126,000 in 2023": the year's level must be the highest of every year, or the
# lowest for "a record low".
RECORD_YEAR = re.compile(
    r"\b(?:a\s+|an\s+)?(?:new\s+)?(?:all-time\s+)?(?P<every>every-year\s+)?record(?:(?P<low>\s+low)|"
    r"(?P<high>\s+high))?(?:\s+since\s+(?P<since>(?:1[89]|20)\d{2}))?\b",
    re.IGNORECASE,
)


def _checked_record_year(check: ClaimCheck, analysis: Any, said: str) -> ClaimCheck:
    """A verified level in one year that the claim, as written, calls a record, checked against the years up
    to it: 3.0 in 1953 was a record low then, whatever came after. "fell to a record 2" is a record low."""
    if not check.verified or check.answer is None or check.claimed is None:
        return check
    plan = check.answer.plan
    found = RECORD_YEAR.search(said)
    if found is None or plan.year is None or plan.month is not None or plan.dimension:
        return check
    yearly = _yearly(analysis, plan)
    if yearly is None or str(plan.year) not in yearly[0].index or len(yearly[0]) < 2:
        message = (f"The claim calls {plan.year}'s level a record, and ADA could not set it against the "
                   f"other years. {check.message}")
        return replace(check, status="unverified", reason="missing_context", message=message)
    by_year, how = yearly
    # "the fewest of any year" is set against every year; a record, against the years up to it.
    every = bool(found.group("every"))
    by_year = by_year if every else by_year[[int(year) <= plan.year for year in by_year.index]]
    if found.group("since"):
        by_year = by_year[[int(year) >= int(found.group("since")) for year in by_year.index]]
    if len(by_year) < 2:
        # The first year of the table is a record of nothing.
        message = (f"The claim calls {plan.year}'s level a record, and the table has no year before it to "
                   f"set it against. {check.message}")
        return replace(check, status="unverified", reason="missing_context", message=message)
    low = bool(found.group("low")) or (not found.group("high") and _says_fall(said, check.claimed))
    record = by_year.idxmin() if low else by_year.idxmax()
    if by_year[record] == by_year[str(plan.year)]:
        side = "lowest" if low else "highest"
        since = f" from {found.group('since')}" if found.group("since") else ""
        upto = "" if every else " up to it"
        held = f"{how} is {side} in {plan.year} of the {len(by_year)} years{since}{upto}."
        return replace(check, assumptions=(*check.assumptions, held))
    return replace(
        check, status="unverified", reason="failed_check", agreement="mismatch",
        message=f"The level agrees ({check.message}), but the record is {record}, at "
                f"{_plain(float(by_year[record]))}.",
    )


def _only_number(analysis: Any, column: str) -> bool:
    """Whether ``column`` is the table's one column of numbers, its date and years aside."""
    from pandas.api.types import is_bool_dtype, is_numeric_dtype  # noqa: PLC0415

    frame = analysis.frame
    numbers = [str(name) for name in frame.columns
               if is_numeric_dtype(frame[name]) and not is_bool_dtype(frame[name])
               and name != analysis.roles.date and "year" not in normalized_name(str(name)).split()]
    return numbers == [str(column)]


# Name words that say how a column counts, not what: Total_Receipt and Total_Disbursement share no thing.
KIN_GENERIC_WORDS = frozenset({"total", "number", "count", "amount", "value", "values", "data", "mean",
                                "average", "median", "percent", "share", "rate", "change", "other", "year"})
# Most other columns named for the same thing that a failed check is made again with.
KIN_COLUMNS_MAX = 3


def _checked_kin_columns(check: ClaimCheck, analysis: Any, recheck: Any) -> ClaimCheck:
    """A failed check made again with each other column named for the same thing as one the reading used.

    "Deaths from heart disease fell by more than a third" fails on Deaths (10.7%) and holds on Age-adjusted
    Death Rate (38.1%); "Amphetamines were found in nearly 4% of the deaths" fails on Meth/Amphetamine and
    holds on Amphet. The claim's words fit both, and it does not say which it means. ``recheck`` checks the
    claim for a question, which names the other column in place of the words that named the first.
    """
    from pandas.api.types import is_bool_dtype, is_numeric_dtype  # noqa: PLC0415

    if check.reason != "failed_check" or check.answer is None or check.claimed is None or not check.question:
        return check
    plan, frame = check.answer.plan, analysis.frame
    used = [(str(plan.measure), "measure")] if plan.measure in frame.columns and plan.intent in (
        "aggregate", "growth") else []
    used += [(item.column, "filter") for item in plan.filters if item.column in frame.columns]
    for named, role in used:
        said = [match for match in re.finditer(r"[A-Za-z]+", check.question)
                if _kin_words({match.group().lower()}, _name_words_of(named))]
        if not said:
            continue
        # Only the first run of words that name the column, side by side, is replaced.
        said = [match for index, match in enumerate(said)
                if index == 0 or not check.question[said[index - 1].end():match.start()].strip(" _-'/")
                and all(not check.question[said[k - 1].end():said[k].start()].strip(" _-'/")
                        for k in range(1, index + 1))]
        flag = frame[named].nunique() <= 3
        kin = [str(column) for column in frame.columns
               if column != named and str(column) not in {analysis.roles.date, analysis.roles.identifier}
               and _kin_words(_name_words_of(str(column)), _name_words_of(named))
               and (is_numeric_dtype(frame[column]) and not is_bool_dtype(frame[column]) if role == "measure"
                    else (frame[column].nunique() <= 3) == flag)][:KIN_COLUMNS_MAX]
        for column in kin:
            asked = f"{check.question[:said[0].start()]}{column}{check.question[said[-1].end():]}"
            other = recheck(asked)
            reads = other.answer is not None and (
                other.answer.plan.measure == column if role == "measure"
                else column in {item.column for item in other.answer.plan.filters})
            if not other.verified or not reads:
                continue
            alternatives = (
                Alternative(f"{role}={named}", check.question, check.compared, False,
                            check.answer.calculation),
                Alternative(f"{role}={column}", asked, other.compared, True, other.answer.calculation),
            )
            return replace(
                check, reason="ambiguous_intent", alternatives=alternatives,
                message=f"The claim's words fit {named} and {column}: {named} gives "
                        f"{_plain(check.compared)}, which disagrees, and {column} {_plain(other.compared)}, "
                        f"which agrees. {check.message}",
                clarification=f"Is the claim about {named} or {column}?",
            )
    return check


def _checked_net_sums(check: ClaimCheck, analysis: Any) -> ClaimCheck:
    """A verdict on a sum over a column that holds Net beside the flows it comes from, with none chosen.

    "Migration to New Zealand fell by more than half" adds up direction = Arrivals, Departures and Net, and
    Net is Arrivals less Departures: the sum counts nothing. The claim gets missing_context with each
    direction's number.
    """
    from .nlq import execute_plan  # noqa: PLC0415

    decided = check.verified or check.reason == "failed_check"
    if not decided or check.answer is None or check.claimed is None:
        return check
    plan, frame = check.answer.plan, analysis.frame
    summed = plan.aggregation == "sum" or bool(plan.weight)
    if not summed or plan.intent not in ("aggregate", "growth", "count"):
        return check
    picked = {item.column for item in plan.filters}
    for column in frame.columns:
        if column in picked or frame[column].dtype != object:
            continue
        values = [str(value) for value in frame[column].dropna().unique()]
        flows = [value for value in values if value.strip().lower() != "net"]
        if not 3 <= len(values) <= NET_VALUES_MAX or len(flows) == len(values):
            continue
        readings = []
        for value in sorted(values):
            try:
                split = replace(plan, filters=(*plan.filters, ValueFilter(str(column), (value,))))
                answer = execute_plan(split, frame, analysis.roles)
            except (KeyError, ValueError):
                continue
            number, _ = _comparable(answer, check.claimed)
            if number is None:
                continue
            if plan.intent == "growth" and _says_fall(check.claim, check.claimed):
                number = -number
            readings.append(Alternative(f"{column} = {value}", check.question or "", number,
                                        check.claimed.holds(number), answer.calculation))
        shown = "; ".join(f"{item.reading} gives {_plain(item.value)}" for item in readings)
        # When one direction agrees, the claim fits it and not the others: ambiguous_intent. When none does,
        # a failed claim fails whichever is meant; a sum that agreed did so by chance.
        agreeing = any(item.agrees for item in readings)
        failed = not agreeing and check.reason == "failed_check" and len(readings) == len(values)
        reason = "ambiguous_intent" if agreeing else "failed_check" if failed else "missing_context"
        return replace(
            check, status="unverified", reason=reason, alternatives=tuple(readings),
            message=f"{column} holds Net beside {_and_names(flows)}, and the claim names none, so the answer "
                    f"adds Net to the flows it comes from: {shown}. {check.message}",
            clarification=f"Which {column} is the claim about?",
        )
    return check


# Most values a column holding Net may have: arrivals, departures and net, by a kind or two.
NET_VALUES_MAX = 6
# Values a word for another value may cover as well: "neutered" is said of spayed females too, and a character
# with "no hair" may be one recorded as Bald.
WIDER_VALUES = {"neutered": ("spayed",), "no hair": ("bald",), "bald": ("no hair",)}


def _checked_wider_values(check: ClaimCheck, analysis: Any) -> ClaimCheck:
    """A check on a value a column also holds under another name (WIDER_VALUES), made with both.

    "55 of the cats were neutered": animal_reproductive_condition holds Neutered (55) and Spayed (41), and
    neutering is said of both. When the rows of both give another verdict, the claim does not say which is
    meant, and it gets `ambiguous_intent` with both numbers.
    """
    from .nlq import _and_listed, execute_plan  # noqa: PLC0415

    settled = check.verified or check.reason == "failed_check"
    if not settled or check.answer is None or check.claimed is None:
        return check
    plan, frame = check.answer.plan, analysis.frame
    for item in plan.filters:
        if len(item.values) != 1 or item.column not in frame.columns:
            continue
        wider = WIDER_VALUES.get(str(item.values[0]).lower(), ())
        held = [str(value) for value in frame[item.column].dropna().unique() if str(value).lower() in wider]
        if not held:
            continue
        both = replace(plan, filters=tuple(
            ValueFilter(other.column, (*other.values, *held)) if other is item else other
            for other in plan.filters))
        try:
            answer = execute_plan(both, frame, analysis.roles)
        except (KeyError, ValueError):
            continue
        number, _ = _comparable(answer, check.claimed)
        if number is None or check.claimed.holds(number) == check.verified:
            continue
        joined = " or ".join([str(item.values[0]), *held])
        alternatives = (
            Alternative(f"{item.column} = {item.values[0]}", check.question or "", check.compared,
                        check.verified, check.answer.calculation),
            Alternative(f"{item.column} = {joined}", check.question or "", number, not check.verified,
                        answer.calculation),
        )
        return ClaimCheck(
            check.claim, "unverified", "ambiguous_intent",
            f"{item.values[0]} may also take in {_and_listed(held)}, which {item.column} holds too, and the "
            f"verdict rests on it: {item.values[0]} gives {_plain(check.compared)} and {joined} "
            f"{_plain(number)}.",
            claimed=check.claimed, question=check.question, alternatives=alternatives, source=check.source,
            clarification=f"Does the claim count {_and_listed(held)} too?",
        )
    return check
def _checked_country_scope(check: ClaimCheck, analysis: Any) -> ClaimCheck:
    """A check that picked a country's name in a column of smaller places, made over the whole table too.

    "Colombia recorded more than 7,500 road deaths in 2025": MUNICIPIO holds Colombia, a town in Huila with
    one death that year, and the table is of the whole country's. When the two give other verdicts, the claim
    gets `ambiguous_intent` with both numbers.
    """
    from .nlq import execute_plan  # noqa: PLC0415

    settled = check.verified or check.reason == "failed_check"
    if not settled or check.answer is None or check.claimed is None:
        return check
    plan, frame, countries = check.answer.plan, analysis.frame, _country_names()
    for item in (*plan.filters, *plan.counted):
        if not isinstance(item, ValueFilter) or len(item.values) != 1 or item.column not in frame.columns:
            continue
        named = str(item.values[0]).strip().lower()
        if named not in countries or _holds_places(frame[item.column], countries):
            continue
        whole = replace(plan, filters=tuple(other for other in plan.filters if other is not item),
                        counted=tuple(other for other in plan.counted if other is not item))
        try:
            answer = execute_plan(whole, frame, analysis.roles)
        except (KeyError, ValueError):
            continue
        number, _ = _comparable(answer, check.claimed)
        if number is None or check.claimed.holds(number) == check.verified:
            continue
        place = item.values[0]
        alternatives = (
            Alternative(f"{item.column} = {place}", check.question or "", check.compared, check.verified,
                        check.answer.calculation),
            Alternative(f"the whole table as {place}", check.question or "", number, not check.verified,
                        answer.calculation),
        )
        return ClaimCheck(
            check.claim, "unverified", "ambiguous_intent",
            f"{place} is a value of {item.column}, which gives {_plain(check.compared)}, and may also be the "
            f"country the whole table covers, which gives {_plain(number)}; the verdict rests on which.",
            claimed=check.claimed, question=check.question, alternatives=alternatives, source=check.source,
            clarification=f"Is {place} the {item.column} value or the whole table?",
        )
    return check


# Most labels a column may have to be a breakdown (Sex, Age, Category) rather than names or notes.
MAX_BREAKDOWN_LABELS = 30


def _mixed_breakdowns(check: ClaimCheck, analysis: Any) -> ClaimCheck:
    """A failed check that averages a column named Value over rows that differ in a breakdown, made
    missing_context.

    "Adult smoking in England has fallen by about three quarters of a percentage point a year" averaged Value
    over Persons, Male and Female, every age band and every category each year; the number mixes them, so it
    says nothing of the claim.
    """
    from .nlq import _apply_filters  # noqa: PLC0415

    if check.reason != "failed_check" or check.answer is None:
        return check
    plan, roles = check.answer.plan, analysis.roles
    measure = normalized_name(str(plan.measure)) if plan.measure else ""
    if plan.aggregation != "mean" or measure not in VALUE_NAMES:
        return check
    try:
        rows, _ = _apply_filters(analysis.frame, plan, roles)
    except (KeyError, ValueError):
        return check
    fixed = {item.column for item in plan.filters} | {plan.dimension, roles.date}
    mixed = [column for column in rows.columns
             if column not in fixed and rows[column].dtype == object
             and 1 < rows[column].nunique() <= MAX_BREAKDOWN_LABELS
             and not set(normalized_name(str(column)).split()) & {"note", "notes", "comment", "comments"}]
    if not mixed:
        return check
    shown = "; ".join(f"{column} ({_and_names([str(value) for value in rows[column].dropna().unique()[:3]])}"
                      f"{' and others' if rows[column].nunique() > 3 else ''})" for column in mixed[:2])
    return replace(
        check, reason="missing_context",
        message=f"The rows averaged differ in {shown}, so the number mixes them and may not be what the "
                f"claim measures: {check.message}",
        clarification=f"Which {mixed[0]} is meant?",
    )


def _country_names() -> set[str]:
    """Countries by name, lower case, as the question reader knows them from their demonyms."""
    from .nlq import DEMONYMS  # noqa: PLC0415

    return {name for names in DEMONYMS.values() for name in names} | {
        "new zealand", "northern ireland", "great britain"}


def _holds_places(series: Any, places: set[str]) -> bool:
    """Whether a column is one of the places named: three or more of them, and a fifth of its values.
    Colombia's 994 municipalities hold Colombia, California and Florida, and are not."""
    values = {str(value).strip().lower() for value in series.dropna().unique()[:5_000]}
    shared = len(values & places)
    return shared >= 3 and shared >= PLACE_COLUMN_SHARE * len(values)


# The share of a column's values that must be places for it to be a column of them.
PLACE_COLUMN_SHARE = 0.2


# Words often used for a narrower thing than they mean: "casualties" for the dead alone.
OTHER_MEANINGS = {"casualties": "deaths", "casualty": "death"}


def _checked_other_meanings(check: ClaimCheck, recheck: Any) -> ClaimCheck:
    """A failed check made again with the narrower meaning of a word the claim uses.

    "Pedestrians accounted for 22% of the casualties": of everyone killed or injured they are 15.2%, of the
    dead 22.2%. When the narrower reading agrees, the claim does not say which it means.
    """
    if check.reason != "failed_check" or not check.question or check.answer is None:
        return check
    for word, narrower in OTHER_MEANINGS.items():
        found = re.search(rf"\b{word}\b", check.question, re.IGNORECASE)
        if found is None:
            continue
        asked = f"{check.question[:found.start()]}{narrower}{check.question[found.end():]}"
        other = recheck(asked)
        if not other.verified or other.answer is None:
            continue
        alternatives = (
            Alternative(word, check.question, check.compared, False, check.answer.calculation),
            Alternative(narrower, asked, other.compared, True, other.answer.calculation),
        )
        return replace(
            check, reason="ambiguous_intent", alternatives=alternatives,
            message=f'"{word}" may mean {narrower} alone: read that way the data give '
                    f"{_plain(other.compared)}, which agrees, and as written {_plain(check.compared)}. "
                    f"{check.message}",
            clarification=f'Does "{word}" mean {narrower} alone?',
        )
    return check


def _name_words_of(column: str) -> set[str]:
    """A column's name words of four letters or more, singular: "death" and "rate" for "Age-adjusted Death
    Rate"."""
    from .folder import singular  # noqa: PLC0415

    return {singular(word) for word in re.findall(r"[a-z]+", normalized_name(column))
            if len(word) >= 4 and singular(word) not in KIN_GENERIC_WORDS}


def _kin_words(some: set[str], others: set[str]) -> bool:
    """Whether a word of one set is a word of the other, or the start of one: "amphet" of "amphetamine"."""
    from .folder import singular  # noqa: PLC0415

    some = {singular(word) for word in some if len(word) >= 4}
    return any(word == other or other.startswith(word) or word.startswith(other)
               for word in some for other in others)


# Name words of a column that says when each subject was measured: age 8, 10, 12 and 14; day 1 to 64.
OCCASION_WORDS = frozenset({"age", "time", "day", "days", "week", "weeks", "visit", "month", "occasion",
                            "wave", "session", "trial", "round"})
# Most occasions a subject may be measured at for each to be tried.
OCCASIONS_MAX = 30


def _checked_occasions(check: ClaimCheck, analysis: Any) -> ClaimCheck:
    """A failed average over every occasion each subject was measured at, made again at each occasion.

    "The average distance was 23.2 mm" of children measured at 8, 10, 12 and 14 pools four ages into 24.0;
    at age 10 the mean is 23.17. When one occasion agrees, the claim names none, and neither verdict stands.
    """
    from pandas.api.types import is_numeric_dtype  # noqa: PLC0415

    from .nlq import execute_plan  # noqa: PLC0415

    if check.reason != "failed_check" or check.answer is None or check.claimed is None:
        return check
    plan, frame = check.answer.plan, analysis.frame
    if plan.intent != "aggregate" or plan.aggregation not in ("mean", "median"):
        return check
    used = {item.column for item in (*plan.filters, *plan.conditions)} | {plan.measure}
    for column in frame.columns:
        if column in used or not OCCASION_WORDS & set(normalized_name(str(column)).split()):
            continue
        values = frame[column].dropna()
        if not 2 <= values.nunique() <= OCCASIONS_MAX:
            continue
        unit = _measured_unit(frame, str(column), used)
        if unit is None:
            continue
        readings = []
        for occasion in sorted(values.unique()):
            scope = (Condition(str(column), "=", float(occasion)),) if is_numeric_dtype(values) else ()
            filters = () if scope else (ValueFilter(str(column), (str(occasion),)),)
            at = replace(plan, conditions=(*plan.conditions, *scope), filters=(*plan.filters, *filters))
            try:
                answer = execute_plan(at, frame, analysis.roles)
            except (KeyError, ValueError):
                continue
            value, _ = _comparable(answer, check.claimed)
            if value is not None:
                readings.append((occasion, value))
        agreeing = [occasion for occasion, value in readings if check.claimed.holds(value)]
        if not agreeing:
            continue
        numbered = is_numeric_dtype(values)
        shown = "; ".join(f"{column} {_plain(float(occasion)) if numbered else occasion} gives "
                          f"{_plain(value)}" for occasion, value in readings)
        return replace(
            check, reason="missing_context",
            message=f"Each {unit} is measured at several {column} values, and the claim names none: over all "
                    f"of them the data give {_plain(check.compared)}, and by {column}: {shown}. "
                    f"{check.message}",
            clarification=f"At which {column} is the {plan.aggregation} taken?",
        )
    return check


# Fewest subjects a repeated design has, and the share of its rows that may repeat a subject and occasion.
SUBJECTS_MIN, REPEATS_MAX = 5, 0.05


def _measured_unit(frame: Any, occasion: str, used: set) -> str | None:
    """The column of subjects measured at each value of ``occasion``: Subject in Orthodont, whose children are
    each measured at 8, 10, 12 and 14, so that a subject and an age pick one row."""
    for column in frame.columns:
        if column == occasion or column in used:
            continue
        subjects = frame[column].nunique()
        if not SUBJECTS_MIN <= subjects < len(frame):
            continue
        pairs = frame[[column, occasion]].dropna()
        if pairs.duplicated().mean() > REPEATS_MAX or pairs.groupby(column)[occasion].nunique().median() < 2:
            continue
        return str(column)
    return None


# Places a name may also mean when the same column holds them: Washington, the state or the capital.
PLACE_ALIASES = {
    "washington": ("district of columbia", "washington, d.c.", "washington dc", "washington d.c."),
}
# Name words of a column of the larger places a name sits in: Lincoln County, Colorado.
PARENT_PLACE_WORDS = frozenset({"state", "country", "province", "region", "territory", "nation", "county",
                                "st", "prov"})
# Most places a shared name may stand for before it is read as a label, not a place.
NAMESAKES_MAX = 10


def _checked_namesakes(check: ClaimCheck, analysis: Any) -> ClaimCheck:
    """A failed check on a name several places share, made again for each of them.

    "Lincoln County has about 520 wind turbines": t_county = Lincoln County is a county in four states, and
    together they have 1,975. When the rows of one of them (t_state = CO, 521) agree, the claim does not say
    which is meant, and neither verdict stands. A column that splits the name's rows, and holds fewer
    values than the name's own column over the table, is taken as where each one is.
    """
    from pandas.api.types import is_numeric_dtype  # noqa: PLC0415

    from .nlq import execute_plan  # noqa: PLC0415

    if check.reason != "failed_check" or check.answer is None or check.claimed is None:
        return check
    plan, frame = check.answer.plan, analysis.frame
    named = [item for item in plan.filters if len(item.values) == 1 and item.column in frame.columns]
    for item in named:
        others = PLACE_ALIASES.get(str(item.values[0]).lower(), ())
        held = [str(value) for value in frame[item.column].dropna().unique() if str(value).lower() in others]
        for value in held:
            # "Washington's per capita income" in a table with a row for the District of Columbia as well.
            swapped = replace(plan, filters=tuple(
                ValueFilter(other.column, (value,)) if other is item else other for other in plan.filters))
            try:
                answer = execute_plan(swapped, frame, analysis.roles)
            except (KeyError, ValueError):
                continue
            number, _ = _comparable(answer, check.claimed)
            if number is None or not check.claimed.holds(number):
                continue
            alternatives = (
                Alternative(f"{item.column} = {item.values[0]}", check.question or "", check.compared, False,
                            check.answer.calculation),
                Alternative(f"{item.column} = {value}", check.question or "", number, True,
                            answer.calculation),
            )
            return replace(
                check, reason="ambiguous_intent", alternatives=alternatives,
                message=f"{item.values[0]} may also mean {value}, which {item.column} holds too: {value} "
                        f"gives {_plain(number)}, which agrees, and {item.values[0]} "
                        f"{_plain(check.compared)}. {check.message}",
                clarification=f"Is it {item.values[0]} or {value}?",
            )
        rows = frame[frame[item.column].astype(str) == str(item.values[0])]
        whole = frame[item.column].nunique()
        for column in frame.columns:
            if column == item.column or is_numeric_dtype(frame[column]) or column in {
                    other.column for other in plan.filters}:
                continue
            places = rows[column].dropna().astype(str).unique()
            if not 2 <= len(places) <= NAMESAKES_MAX or frame[column].nunique() >= whole or not (
                PARENT_PLACE_WORDS & set(normalized_name(str(column)).split())
            ):
                continue
            readings = []
            for place in sorted(places):
                split = replace(plan, filters=(*plan.filters, ValueFilter(str(column), (place,))))
                try:
                    answer = execute_plan(split, frame, analysis.roles)
                except (KeyError, ValueError):
                    continue
                value, _ = _comparable(answer, check.claimed)
                if value is not None:
                    readings.append((place, value, answer))
            agreeing = [reading for reading in readings if check.claimed.holds(reading[1])]
            if not agreeing:
                continue
            alternatives = (
                Alternative(f"{item.column} = {item.values[0]} (all)", check.question or "", check.compared,
                            False, check.answer.calculation),
                *(Alternative(f"{column} = {place}", check.question or "", value, check.claimed.holds(value),
                              answer.calculation) for place, value, answer in readings),
            )
            shown = "; ".join(f"{column} = {place} gives {_plain(value)}" for place, value, _ in readings)
            fitting = _and_names([str(reading[0]) for reading in agreeing])
            return replace(
                check, reason="ambiguous_intent", alternatives=alternatives,
                message=f"{item.values[0]} is in {len(places)} {column} values, and the claim holds for "
                        f"{fitting} only: {shown}. Together they give {_plain(check.compared)}. "
                        f"{check.message}",
                clarification=f"Which {item.values[0]} is meant?",
            )
    return check


def _checked_clock(check: ClaimCheck, analysis: Any) -> ClaimCheck:
    """A verdict on dates in UTC, made again in every time zone from UTC-12 to UTC+14.

    A claim most likely means its days, weekdays, hours and years in local
    time, which the table does not give: 101 incidents started on 25 March
    2026 in UTC and 110 in Calgary's time. When some zone turns the verdict,
    the claim gets missing_context with the range the zones give.
    """
    import pandas as pd  # noqa: PLC0415

    from .nlq import execute_plan, utc_dates  # noqa: PLC0415

    decided = check.verified or check.reason == "failed_check"
    if not decided or check.answer is None or check.claimed is None:
        return check
    plan, frame = check.answer.plan, analysis.frame
    clock = utc_dates(plan, frame, analysis.roles)
    if clock is None:
        return check
    values = []
    for hours in range(-12, 15):
        shifted = frame.assign(**{clock: frame[clock] + pd.Timedelta(hours=hours)})
        try:
            answer = execute_plan(plan, shifted, analysis.roles)
        except (KeyError, ValueError):
            return check
        value, _ = _comparable(answer, check.claimed)
        if value is None:
            return check
        if plan.intent == "growth" and _says_fall(check.claim, check.claimed):
            value = -value
        values.append(value)
    if all(check.claimed.holds(value) == check.verified for value in values):
        held = f"{clock} is in UTC; the verdict holds in every time zone from UTC-12 to UTC+14."
        return replace(check, assumptions=(*check.assumptions, held))
    return replace(
        check, status="unverified", reason="missing_context",
        message=f"{clock} is in UTC, and the claim most likely means local time, which the table does not "
                f"give: in time zones from UTC-12 to UTC+14 the data give {_plain(min(values))} to "
                f"{_plain(max(values))}, and the verdict turns within that range. {check.message}",
        clarification="What time zone are the claim's dates and hours in?",
    )


# Words that say a total is over every year: "in all", "altogether", "over the period".
ALL_YEARS = re.compile(
    r"\b(?:in\s+all|in\s+total|altogether|all\s+told|combined|over\s+the\s+(?:years|period|decade)|"
    r"(?:since|from|between)\s+(?:19|20)\d{2})\b",
    re.IGNORECASE,
)


def _checked_unnamed_year(check: ClaimCheck, analysis: Any) -> ClaimCheck:
    """A total over every year that fails where one year's total fits, when the claim names no year.

    "California recorded about 5,000 traffic deaths" in a table of 1982 to
    1988: the seven years give 35,315 and 1984 alone 5,020. The claim does not
    say which year it means, so the total over all of them does not prove it wrong.
    """
    if check.reason != "failed_check" or check.answer is None or check.claimed is None:
        return check
    if check.claimed.percent:
        # A share is not a total, and a year's count is no share of anything.
        return check
    plan, date = check.answer.plan, analysis.roles.date
    total = plan.aggregation in ("sum", "count") and plan.intent in ("aggregate", "count")
    dated = plan.year or plan.month or any(
        isinstance(item, Condition) and item.column == date for item in plan.conditions
    )
    if not total or dated or plan.dimension or re.search(r"\b(?:19|20)\d{2}\b", check.claim):
        return check
    if ALL_YEARS.search(check.claim):
        return check
    yearly = _yearly(analysis, plan)
    if yearly is None or len(yearly[0]) < 2:
        return check
    by_year, how = yearly
    fits = [f"{year} ({_plain(float(value))})" for year, value in by_year.items()
            if check.claimed.holds(float(value))]
    if not fits:
        return check
    return replace(
        check, status="unverified", reason="missing_context",
        message=f"The claim names no year, and the table has {len(by_year)}, {by_year.index[0]} to "
                f"{by_year.index[-1]}: {how} over all of them is {_plain(check.compared)}, and the claim's "
                f"number fits {list_names(fits)}. {check.message}",
        clarification="Which year does the claim mean?",
    )


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
    # Rows with no date are in no year: 4,061 agencies yet to start NIBRS are no year of their own.
    dated = years.notna()
    rows, years = rows[dated], years[dated]
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


# "its all-time high", "a record low": the value claimed is the highest or lowest the table holds.
RECORD_EXTREME = re.compile(r"\b(?:all[- ]time|record)\s+(?P<side>high|low|peak)s?\b", re.IGNORECASE)


def _checked_extreme(check: ClaimCheck, analysis: Any, extreme: re.Match) -> ClaimCheck:
    """A verified value that the claim calls its all-time high or low, checked against the highest or lowest
    value of the column over every row the claim's other conditions keep."""
    from .nlq import _apply_filters  # noqa: PLC0415
    from .numeric import to_numbers  # noqa: PLC0415

    answer = check.answer
    assert answer is not None and answer.plan.measure is not None
    date = analysis.roles.date
    plan = replace(answer.plan, year=None, month=None,
                   conditions=tuple(item for item in answer.plan.conditions if item.column != date))
    working, _ = _apply_filters(analysis.frame, plan, analysis.roles)
    values = to_numbers(working[answer.plan.measure]).dropna()
    if values.empty or check.claimed is None:
        return check
    high = extreme.group("side").lower() != "low"
    best = float(values.max() if high else values.min())
    word = "highest" if high else "lowest"
    if check.claimed.holds(best):
        return replace(check, assumptions=(*check.assumptions, f"Checked that {_plain(best)} is the {word} "
                                           f"{answer.plan.measure} over the {len(values):,} rows."))
    return ClaimCheck(
        check.claim, "unverified", "failed_check",
        f"The data give {_plain(check.compared)} for {answer.calculation}, as claimed, but the claim calls "
        f"it the {extreme.group(0)}, and the {word} {answer.plan.measure} over the {len(values):,} rows is "
        f"{_plain(best)}.",
        claimed=check.claimed, question=check.question, answer=answer, compared=best, agreement="mismatch",
        assumptions=check.assumptions, source=check.source,
    )


# A rank closing the claim: ", more than any other state", ", the most of any Senate candidate".
CLOSING_RANK = re.compile(
    r",\s*(?:(?:more|fewer|less)\s+than\s+(?:in\s+)?any\s+other|the\s+(?:most|highest|largest|biggest|fewest|"
    r"lowest|smallest|least)\s+(?:of|among|in)\s+(?:any|all(?:\s+the)?))\s+[^,.;]+?\s*$",
    re.IGNORECASE,
)


def _without_closing_rank(analysis: Any, asked: str, choices: dict[str, str]) -> str:
    """The question without a closing rank whose words only name the groups.

    "Pennsylvania has 1,477 agencies, the most of any state" would ask for each state, which "Pennsylvania"
    cannot also filter, and "more than any other state" names nothing with "other". The rank is checked
    after, from the claim's own words (``_checked_rank_tail``). A rank with a word for a value ("the most of
    any Senate candidate") stays, as that word keeps the rows the rank is over.
    """
    from .folder import singular  # noqa: PLC0415

    found = CLOSING_RANK.search(asked)
    if found is None:
        return asked
    words = [word for word in _words(found.group(0).lower()) if word not in RANK_TAIL_GRAMMAR]
    columns = {singular(word) for column in analysis.frame.columns
               for word in _words(normalized_name(str(column)))}
    if words and all(singular(word) in columns or word in RANK_SCOPE_WORDS
                     or _names_rows([word], analysis.roles.row_noun) for word in words):
        return asked[:found.start()].rstrip(" ,")
    return asked


# A closing rank against the rest: "the most of any Senate candidate", "more than any other state".
RANK_TAIL = re.compile(
    r"\b(?P<word>most|highest|largest|biggest|greatest|lowest|smallest|fewest|least|more|fewer|less)\s+"
    r"(?:[a-z]+\s+){0,3}?(?:of|among|than)\s+(?:in\s+)?(?:any|all)\s+(?:other\s+|the\s+)?(?P<scope>[^,.;]+)",
    re.IGNORECASE,
)
LOW_RANK_WORDS = frozenset({"lowest", "smallest", "fewest", "least", "fewer", "less"})
# The words of a closing rank that only say it ranks: ", more than any other", ", the most of any".
RANK_TAIL_GRAMMAR = frozenset({"more", "fewer", "less", "than", "in", "any", "other", "the", "most",
                               "highest", "largest", "biggest", "fewest", "lowest", "smallest", "least", "of",
                               "among", "all"})
# Words of a closing rank that say what is ranked without naming groups: "candidate", "one", "name".
RANK_SCOPE_WORDS = frozenset({"candidate", "candidates", "one", "ones", "name", "names", "group", "groups",
                              "kind", "kinds", "type", "types"})


def _checked_rank_tail(check: ClaimCheck, analysis: Any, text: str) -> ClaimCheck | None:
    """A verified value for one group, checked against every group the closing words keep, or None.

    "Sherrod Brown raised about $96.5 million, the most of any Senate candidate": the reading keeps
    Cand_Name = BROWN, SHERROD and Cand_Office = S, which "Senate" names. The total is made for every
    Cand_Name with Cand_Office = S, and the claim holds only if Brown's is the highest. None when the
    reading keeps no one group apart from what the closing words name.
    """
    from .nlq import _apply_filters  # noqa: PLC0415

    found = RANK_TAIL.search(text)
    answer = check.answer
    if found is None or answer is None:
        return None
    plan = answer.plan
    counted = plan.measure is None and plan.aggregation == "count"
    if not counted and not (plan.measure and plan.aggregation in ("sum", "mean", "median", "max", "min")):
        return None
    scope = found.group("scope").lower()
    read = [match for item in check.assumptions
            if (match := re.fullmatch(r'Read "(.+?)" as (.+?) = .+\.', item))]
    named = {match.group(2) for match in read if match.group(1).lower() in scope}
    groups = [item for item in plan.filters
              if len(item.values) == 1 and item.column not in named
              and analysis.frame[item.column].nunique(dropna=True) > 1]
    if len(groups) != 1 or plan.dimension or plan.year or plan.month:
        return None
    group = groups[0]
    from .folder import singular  # noqa: PLC0415

    # Each word of the closing words must name the groups ("state"), the rows, or a value read ("Senate"):
    # a word left unread could narrow the groups the rank is over.
    known = {word for match in read for word in _words(match.group(1).lower())}
    group_words = {singular(word) for word in _words(normalized_name(str(group.column)))}
    for word in _words(re.sub(r"\b(?:other|the|in|of)\b", " ", scope)):
        if word not in known and singular(word) not in group_words and not _names_rows(
                [word], analysis.roles.row_noun) and word not in RANK_SCOPE_WORDS:
            return None
    others = replace(plan, filters=tuple(item for item in plan.filters if item is not group))
    rows, _ = _apply_filters(analysis.frame, others, analysis.roles)
    column, value = group.column, str(group.values[0])
    labels = rows[column].astype(str)
    if counted:
        by_group, how = rows.groupby(labels).size(), "row count"
    else:
        by_group = rows.groupby(labels)[plan.measure].agg(plan.aggregation).dropna()
        how = f"{plan.aggregation}({plan.measure})"
    if value not in by_group.index or len(by_group) < 2:
        return None
    low = found.group("word").lower() in LOW_RANK_WORDS
    leader = by_group.idxmin() if low else by_group.idxmax()
    side = "lowest" if low else "highest"
    kept = "".join(f" with {item.column} = {', '.join(map(str, item.values))}" for item in others.filters)
    if by_group[leader] == by_group[value]:
        held = f"{how} is {side} for {value} of the {len(by_group):,} {column} values{kept}."
        return replace(check, assumptions=(*check.assumptions, held))
    return replace(
        check, status="unverified", reason="failed_check", agreement="mismatch",
        message=f"The value agrees ({check.message}), but the {side} {how} of the {column} values{kept} is "
                f"{leader}'s, at {_plain(float(by_group[leader]))}.",
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


# "Queensland's roads were the deadliest in Australia in 2022, with 297 deaths": the group with most deaths.
DEADLIEST_LEAD = re.compile(
    r"^\s*(?P<group>[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,3})(?:'s\s+[a-z]+)?\s+(?:was|were)\s+the\s+deadliest"
    r"(?:\s+[a-z]+)?(?:\s+in\s+[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})?(?P<when>\s+(?:in|during)\s+\d{4})?"
    r",\s*with\s+(?P<value>[\d,.]+)\s+(?P<noun>deaths|fatalities)\b(?P<rest>.*)$"
)
# ", more than New South Wales" after a peak: one of the groups it is above, which the peak check covers.
ABOVE_ONE = re.compile(
    r",\s*(?:more|higher|greater)\s+than\s+[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3}(?=\s*[.!]?\s*$)"
)


# "More agencies began reporting in 2021 than in any other year: 2,664": a peak over the years.
MORE_IN_YEAR = re.compile(
    r"^\s*more\s+(?P<what>.+?)\s+in\s+(?P<year>\d{4})\s+than\s+in\s+any\s+other\s+year\s*[:,]\s*"
    r"(?P<value>.+?)\s*[.!]?\s*$",
    re.IGNORECASE,
)


# "Hakeem Jeffries raised more than any other House candidate in the 2024 cycle, almost $23 million": the rank
# said first, with no noun between "more" and "than", and the value after it.
RANKED_LEAD = re.compile(
    r"^\s*(?P<subject>[A-Z][^,]*?)\s+(?P<verb>[a-z]+)\s+more\s+than\s+any\s+other\s+(?P<kind>[^,]+?),\s*"
    r"(?P<value>[^,]+?)\s*[.!]?\s*$"
)


def _ranked_lead(text: str) -> str:
    """"G raised more than any other D, V" as "G raised V, the most of any D"."""
    found = RANKED_LEAD.match(text)
    if found is None or not NUMBER_PATTERN.search(found.group("value")) or found.group("verb") in (
            "had", "has", "have", "was", "were", "is", "are"):
        return text
    return (f"{found.group('subject')} {found.group('verb')} {found.group('value')}, the most of any "
            f"{found.group('kind')}.")


def _as_most(text: str) -> str:
    """A lead said with "the deadliest", or naming a group it beats, as "G had the most X in Y, with V"."""
    over_years = MORE_IN_YEAR.match(text)
    if over_years is not None:
        return (f"The number of {over_years.group('what')} peaked in {over_years.group('year')} at "
                f"{over_years.group('value')}.")
    found = DEADLIEST_LEAD.match(text)
    if found is not None:
        text = (f"{found.group('group')} had the most {found.group('noun')}{found.group('when') or ''}, "
                f"with {found.group('value')}{found.group('rest')}")
    beaten = ABOVE_ONE.sub("", text)
    return beaten if beaten != text and MOST_LEAD.match(beaten) else text


def _most_as_peak(text: str) -> str:
    """"G had more X than any other D, with V" as "X in G peaked at V", which is checked as a peak."""
    text = _as_most(text)
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
    # "Queensland had the most deaths in 2022": the group leads, so the peak is over the groups that year.
    lead = MOST_LEAD.match(_as_most(text))
    led = bool(groups and lead and lead.group("most"))
    counted = plan is not None and plan.measure is None and plan.aggregation == "count"
    one_year = plan is not None and plan.year is not None and plan.month is None
    in_year = one_year and not over_groups and bool(
        peak and re.search(rf"\b{plan.year}\b", rewritten[peak.end():]) or against
    )
    yearly = one_year and analysis.roles.date is not None and (
        (counted and not (groups and (over_groups or led))) or (combined and in_year)
    )
    if (plan is not None and plan.year is not None and groups and not yearly and combined and not over_groups
            and not led):
        # "Rain at Oxford in 2012 peaked at 984 mm": over the years, or over the stations that year?
        return replace(
            check, status="unverified", reason="missing_context",
            message=f"The claim names a peak, and ADA could not tell whether it is over the years or over "
                    f"{groups[0].column}: {check.message}",
            clarification=f"Is it the highest of the years, or of the {groups[0].column} values in "
                          f"{plan.year}?",
        )
    if plan is None or not (len(groups) == 1 or yearly) or (not yearly and not (combined or counted)):
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
        column, named = group.column, str(group.values[0])
        if counted:
            by_group, how = rows.groupby(rows[column].astype(str)).size(), "row count"
        else:
            by_group = rows.groupby(rows[column].astype(str))[plan.measure].agg(plan.aggregation).dropna()
            how = f"{plan.aggregation}({plan.measure})"
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


def _share_of_two(
    analysis: Any, asked: str, answer: Any, number: ClaimedNumber, choices: dict[str, str]
) -> tuple[QueryAnswer, str] | None:
    """A percentage claim whose words name two columns of counts, asked as a share: "COVID-19 was involved in
    one in ten deaths" is COVID-19 Deaths over Total deaths. None unless the plain question was refused for
    naming two."""
    from .api import Refusal  # noqa: PLC0415

    if not number.percent or not isinstance(answer, Refusal) or "can use one measure" not in answer.message:
        return None
    question = _share_question(asked)
    shared = analysis.ask(question, decisions=choices)
    if isinstance(shared, Refusal) or "share_percent" not in dict(shared.figures):
        return None
    return shared, question


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


def _checked_base(check: ClaimCheck, count: str, noun: str, frame: Any = None, factor: float = 1.0,
                  text: str = "") -> ClaimCheck:
    """A claim that names its base ("Of the 4,526 applicants", "the 93 cars"), with the base checked too.

    When a column is named by the noun ("the 48 states" and a column state),
    the base is its number of distinct values in the rows the answer's
    filters keep. Otherwise it is the share's denominator (times ``factor``
    when a unit column gives it in thousands), or, for any other answer, the
    rows in the answer's scope.
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
    base = None if base is None else base * factor
    of, are = "the share is of", "the share is of"
    if base is None and answer is not None and answer.scope is not None:
        base, of, are = float(answer.scope.rows_in_scope), "the answer's scope holds", "rows in scope"
    # "the 61,000 species records" for 61,119 rows: a base written in round thousands is rounded.
    zeros = len(count.replace(",", "")) - len(count.replace(",", "").rstrip("0"))
    rounding = 0.5 * 10 ** min(zeros, 3) if stated >= 1000 else 0.5
    if base is not None and abs(base - stated) < rounding:
        rounded = ", rounded" if abs(base - stated) >= 0.5 else ""
        note = f"The claim's {named} are the {_plain(base)} {are}{rounded}."
        if _describes_base(check, text, named):
            # "the 149,200 immigrants who arrived in Ireland": the base's count is the share's, so the words
            # that only describe it pick no other rows.
            described = (f"{_quoted(list(check.words_left_out))} only describe the {named}, which are the "
                         f"{_plain(base)} {are}")
            number, computed = check.claimed, _plain(check.compared)
            if check.agreement == "match":
                return replace(
                    check, status="verified", reason=None, clarification=None,
                    message=f"{computed} from {answer.calculation} lies in {number.interval()}, which "
                    f"{number.text} stands for; {described}.",
                    assumptions=(note, *check.assumptions),
                )
            return replace(
                check, reason="failed_check", clarification=None,
                message=f"The data give {computed} for {answer.calculation}, outside {number.interval()}, "
                f"which {number.text} stands for; {described}.",
                assumptions=(note, *check.assumptions),
            )
        return replace(check, assumptions=(note, *check.assumptions))
    found = f"{of} {_plain(base)}" if base is not None else "the answer gives no base to compare"
    return replace(
        check, status="unverified", reason="missing_context",
        message=f"The claim's base is {named}, but {found}; {check.message}",
        clarification=f"Which rows are the {named}?",
    )


def _describes_base(check: ClaimCheck, text: str, named: str) -> bool:
    """Whether a share agreed but for words left out, and every such word sits in the clause after its base:
    "arrived" in "of the 149,200 immigrants who arrived in Ireland"."""
    if check.reason != "missing_context" or check.agreement == "not_compared" or not check.words_left_out:
        return False
    if check.claimed is None or check.compared is None or not check.claimed.percent:
        return False
    at = text.find(named)
    clause = re.split(r"[,;.!?]", text[at + len(named):])[0] if at >= 0 else ""
    return all(re.search(rf"\b{re.escape(word)}\b", clause, re.IGNORECASE) for word in check.words_left_out)


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
            if number.percent and "share_percent" not in dict(other.figures):
                # A percentage is never compared with the count the other option gives.
                continue
            computed, _ = _comparable(other, number)
            if computed is not None and number.holds(computed * factor):
                return decision, option, other, computed * factor
    return None


def _name_words(column: object) -> list[str]:
    """The words of a column's name, less words for counting: "covid", "deaths" for "COVID-19 Deaths"."""
    words = re.findall(r"[a-z]+", normalized_name(str(column)).lower())
    return [word for word in words if word not in COUNTING_WORDS] or words


def _head_sibling(
    analysis: Any, asked: str, number: ClaimedNumber, answer: QueryAnswer, choices: dict[str, str],
    factor: float,
) -> tuple[str, str, QueryAnswer, float, bool] | None:
    """Another column of numbers the claim's word names as well as the one read: "deaths" names Total deaths
    and COVID-19 Deaths when the claim says neither "total" nor "COVID-19". Returns the column, the
    question asked of it, its answer, the number it gives in the claim's unit and whether the claim holds."""
    from pandas.api.types import is_bool_dtype, is_numeric_dtype  # noqa: PLC0415

    from .api import Refusal  # noqa: PLC0415

    frame = analysis.frame
    measure = answer.plan.weight or answer.plan.measure
    if measure is None or measure not in frame.columns:
        return None
    said = set(re.findall(r"[a-z]+", asked.lower()))
    named = _name_words(measure)
    # A name with no words ("2021" in a wide table of years) has no head word to share.
    if not named or named[-1] not in said or set(named[:-1]) & said:
        return None
    head = named[-1]
    for column in frame.columns:
        series = frame[column]
        if column == measure or not is_numeric_dtype(series) or is_bool_dtype(series):
            continue
        words = _name_words(column)
        if not words or words[-1] != head or set(words[:-1]) & said:
            continue
        question = re.sub(rf"\b{re.escape(head)}\b", str(column), asked, count=1, flags=re.IGNORECASE)
        other = analysis.ask(question, decisions=choices)
        if number.percent and not isinstance(other, Refusal) and "share_percent" not in dict(other.figures):
            # A percentage is a share of the other column too.
            question = _share_question(question)
            other = analysis.ask(question, decisions=choices)
        if isinstance(other, Refusal) or (other.plan.weight or other.plan.measure) != column:
            continue
        if number.percent and "share_percent" not in dict(other.figures):
            continue
        value, _ = _comparable(other, number)
        if value is not None:
            return str(column), question, other, value * factor, number.holds(value * factor)
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


def _counted_after(text: str, number: ClaimedNumber) -> str | None:
    """The plural noun, with up to two words before it, right after the claim's number: "nursing homes" in
    "23 nursing homes in Kansas City"."""
    if number.percent or number.text not in text:
        return None
    after = text[text.index(number.text) + len(number.text):]
    found = re.match(r"\s+((?:[A-Za-z-]+\s+){0,2}?[A-Za-z-]{3,}s)\b", after)
    if found is None or found.group(1).split()[-1].lower() in VERB_ENDINGS_S:
        return None
    return found.group(1)


def _counted_total(analysis: Any, answer: QueryAnswer, asked: str, number: ClaimedNumber, text: str,
                   choices: Mapping[str, str]) -> tuple[QueryAnswer, str, str | None]:
    """The total of a column named for the things a claim counts, in place of its average over several rows.

    "Sudan had about 1.1 million refugees in 2022": refugees counts them on each row, so over the 100 rows the
    count is their total. Returns the answer, its question and the note saying so, or the answer as it was.
    """
    from .api import Refusal  # noqa: PLC0415

    phrase = _counted_after(text, number)
    measure = answer.plan.measure
    used = answer.scope.rows_used if answer.scope is not None else None
    if not phrase or not measure or answer.plan.aggregation not in ("mean", "median") or (
            AVERAGING.search(text)):
        return answer, asked, None
    if not (_words_in_name(phrase, str(measure)) and _words_in_name(normalized_name(str(measure)), phrase)):
        return answer, asked, None
    if not used or used <= 1:
        # One row's value is its count.
        return answer, asked, None
    total = analysis.ask(f"total {asked}", decisions=choices)
    if isinstance(total, Refusal) or total.plan.aggregation != "sum" or total.plan.measure != measure:
        return answer, asked, None
    note = f'Read "{number.text} {phrase}" as a total of {measure}, since it counts them.'
    return total, f"total {asked}", note


def _words_in_name(phrase: str, column: str) -> bool:
    """Whether every word of a phrase, or its singular, is a word of a column's name."""
    from .folder import singular  # noqa: PLC0415

    words = set(normalized_name(column).split())
    return all(word in words or singular(word) in words for word in _words(phrase))


def _valued_columns(frame: Any, word: str) -> list[str]:
    """Text columns of labels (not free text) with a value holding ``word`` or its singular as a whole word:
    Category for "plant" in "Vascular Plant"."""
    from .folder import singular  # noqa: PLC0415

    pattern = rf"\b(?:{re.escape(word.lower())}|{re.escape(singular(word.lower()))})\b"
    found = []
    for column in frame.select_dtypes(exclude=["number", "bool", "datetime"]).columns:
        values = frame[column].dropna().astype(str).drop_duplicates()
        if 1 < len(values) <= LABEL_VALUES_MAX and values.str.lower().str.contains(pattern).any():
            found.append(str(column))
    return found


# Words ending in s that are verbs, not the plural noun a modifier comes before: "3 tickets was".
VERB_ENDINGS_S = frozenset({"was", "is", "has", "does", "goes", "makes", "gives", "takes", "comes", "says"})
# Share of a column's values that must differ for a filter on it to pick one thing by its name.
NAME_SHARE = 0.9
# Most distinct values a column of labels holds; more are names or free text.
LABEL_VALUES_MAX = 200


def _in_any_value(frame: Any, word: str) -> bool:
    """Whether ``word`` (or its singular) is a word of a column's name or of any value of a text column."""
    from .folder import singular  # noqa: PLC0415

    if _names_anything(frame, word):
        return True
    pattern = rf"\b(?:{re.escape(word.lower())}|{re.escape(singular(word.lower()))})\b"
    for column in frame.select_dtypes(exclude=["number", "bool", "datetime"]).columns:
        values = frame[column].dropna().astype(str).drop_duplicates()
        if len(values) <= 10_000 and values.str.lower().str.contains(pattern, regex=True).any():
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
    # "died" names Accidental_Drug_Related_Deaths.csv: a word for dying names a table of deaths, and "babies
    # born" a table of births.
    dying = OUTCOME_WORDS["died"] if words & OUTCOME_WORDS["died"] else frozenset()
    # The births are to women, so "women" and "mothers" name the table's rows too.
    born = BIRTH_WORDS | {"mother", "mothers", "woman", "women"} if words & BIRTH_WORDS else frozenset()
    # "railways" in dutch-rail-disruptions: a word of the name of four letters or more may start a longer one.
    # "list" in "on the FBI's list" is the table itself.
    return bool(parts) and all(
        part in words or singular(part) in words or f"{part}s" in words or part in dying or part in born
        or any(len(word) >= 4 and part.startswith(word) for word in words) or part in TABLE_NOUNS
        for part in parts
    )


# Nouns for a table as a whole: "on the FBI's list", "in the register".
TABLE_NOUNS = frozenset({
    # "lists" is left out: "the park lists 3,200 species" is a verb.
    "list", "register", "registry", "database", "roster", "directory", "catalogue", "catalog",
    "inventory",
})


# Words for being born, which name a table of births.
BIRTH_WORDS = frozenset({"born", "birth", "births", "baby", "babies", "newborn", "newborns"})


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
    a category of its own, not "Men" written another way. In a column of
    places, the value with more after a comma is another place ("China, Hong
    Kong SAR" beside "China"). Returns (column, value, the other ways) for the
    first such value.
    """
    from .aggregates import _is_place_column  # noqa: PLC0415

    picked = [item for item in (*answer.plan.filters, *answer.plan.counted) if isinstance(item, ValueFilter)]
    for item in picked:
        if item.column not in frame.columns:
            continue
        places = _is_place_column(item.column)
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
                and not (places and other.lower().startswith(f"{lowered},"))
            ]
            if others:
                # The ways that start with the value come first: "Norovirus Genogroup II" before
                # "Adenovirus;Norovirus Genogroup I".
                return str(item.column), value, sorted(others, key=lambda other: (
                    not other.lower().startswith(lowered), other))
    return None


def _holds_counts(analysis: Any) -> bool:
    """Whether some column could hold a count of what the table is named for: a numeric column with more
    than one value that is not the date, an ID or a code. Colombia's road deaths have none (CODIGO DANE is a
    code and CANTIDAD is 1 on every row), so each row is one death."""
    roles, frame = analysis.roles, analysis.frame
    return any(column not in (roles.date, roles.identifier) and column not in roles.factors
               and frame[column].nunique() > 1 for column in roles.numeric if column in frame.columns)


def _values_written(plan: Any, text: str) -> bool:
    """Whether the claim writes out in full each value the reading filters on: "Antioquia" for DEPARTAMENTO =
    ANTIOQUIA, but not "duty" for cause_of_death = Duty related illness."""
    values = [str(value) for item in (*plan.filters, *plan.counted) if isinstance(item, ValueFilter)
              for value in item.values]
    return all(re.search(rf"(?<!\w){re.escape(value)}(?!\w)", text, re.IGNORECASE) for value in values)


# Nouns for deaths, which name the same rows.
DEATH_NOUNS = frozenset({"death", "deaths", "fatality", "fatalities"})


def _names_rows(phrases: Sequence[str], row_noun: str | None) -> bool:
    """Whether any of the phrases holds the file's own noun for its rows ("characters" in marvel.csv)."""
    from .folder import singular  # noqa: PLC0415

    if not row_noun:
        return False
    forms = {row_noun.lower(), singular(row_noun.lower())}
    if forms & DEATH_NOUNS:
        # "deaths" names the rows of ardd_fatalities.csv, as "fatalities" does.
        forms |= DEATH_NOUNS
    words = {form for phrase in phrases for word in _words(phrase)
             for form in (word.lower(), singular(word.lower()))}
    return bool(forms & words)


def _and_names(names: Sequence[str]) -> str:
    """"a", "a and b", or "a, b and c"."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


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
    currency = _currency_named(analysis.frame, number)
    if currency and not re.search(rf"\b{re.escape(currency)}s?\b", asked, re.IGNORECASE):
        # "the euro gained about 10 yen a year", "averaged about $1.18": the currency is a column of rates.
        asked = f"{asked} {currency}"
    if len(noun) >= 2 and not re.search(rf"\b{re.escape(number.unit or '')}\b", asked, re.IGNORECASE) and any(
            _initials(number.unit or "", str(column)) for column in analysis.frame.columns):
        # "more than 400 hit points": words after the number whose initials name a column (hp_number) say what
        # it measures.
        asked = f"{asked} {number.unit}"
    unsaid = len(noun) == 1 and not re.search(rf"\b{re.escape(noun[0])}\b", asked, re.IGNORECASE)
    if unsaid and not _claim_unit(number) and _kind_value(noun[0].lower(), analysis.frame):
        # "565 boys were named Jack": the boys are Sex = Boy, so the noun picks rows as well as naming them.
        asked = f"{asked} {noun[0]}"
    from .nlq import DEMONYMS  # noqa: PLC0415

    people = noun[0].lower() if noun else ""
    places = DEMONYMS.get(people) or (DEMONYMS.get(people[:-1]) if people.endswith("s") else None) or ()
    if places and not re.search(rf"\b{re.escape(noun[0])}\b", asked, re.IGNORECASE) and any(
            _in_any_value(analysis.frame, place) for place in places):
        # "25 million Texans held a license": the people of a place the table holds pick its rows.
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
    if isinstance(answer, Refusal):
        from .folder import singular  # noqa: PLC0415

        # "babies" in canada-births/13100416.csv: a word that names the table's rows, and nothing in them,
        # picks no rows. A word before the rows' own noun ("The average Lego set") is read with it below.
        # "sea level" in battery-sea-level names MSL: only the words the refusal says name nothing go.
        refused = {word for phrase in _unknown_phrases(answer.message) for word in _words(phrase)}
        row_noun = analysis.roles.row_noun
        nouns = {row_noun, singular(row_noun)} if row_noun else set()
        naming = [word for word in dict.fromkeys(_words(asked))
                  if _names_table(word, table_words) and not _in_any_value(analysis.frame, word)
                  and (not refused or word in refused)
                  and not any(re.search(rf"\b{re.escape(word)}\s+{re.escape(noun)}\b", asked, re.IGNORECASE)
                              for noun in nouns)]
        shorter = re.sub(r"\s+", " ", re.sub(rf"\b(?:{'|'.join(naming)})\b", " ", asked, flags=re.IGNORECASE)
                         ).strip() if naming else asked
        retried = analysis.ask(shorter, decisions=choices) if naming else answer
        if naming and (not isinstance(retried, Refusal) or not any(
                re.search(rf"\b{word}\b", retried.message, re.IGNORECASE) for word in naming)):
            answer, asked, left = retried, shorter, (*left, *naming)
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
        valued = VALUE_IN_COLUMNS.search(answer.message)
        if not named and valued:
            # '"broken-down" is a value of cause_en and statistical_cause_en': each column before the value.
            phrase = valued.group("phrase")
            held = [column for column in analysis.frame.columns
                    if str(column) in re.split(r",\s*|\s+and\s+", valued.group("columns"))]
            at = rf"(?<!\w){re.escape(phrase)}(?!\w)"
            named = [(column, re.sub(at, f"{column} {phrase}", asked, count=1, flags=re.IGNORECASE))
                     for column in held]
        by_unit = not named
        for column, rewritten in named or _unit_columns(analysis, asked, number):
            if not rewritten:
                continue
            other = analysis.ask(rewritten, decisions=choices)
            unknown = _unknown_phrases(other.message) if isinstance(other, Refusal) else []
            named_table = [phrase for phrase in unknown if _names_table(phrase, table_words)]
            if named_table:
                # "railways" in dutch-rail-disruptions names the table, so leaving it out picks no rows.
                for phrase in named_table:
                    at = rf"(?<!\w){re.escape(phrase)}(?!\w)"
                    rewritten = re.sub(at, " ", rewritten, flags=re.IGNORECASE)
                rewritten = re.sub(r"\s+", " ", rewritten).strip()
                other = analysis.ask(rewritten, decisions=choices)
            if isinstance(other, Refusal):
                continue
            if other.plan.count_column == column and not re.search(
                rf"{re.escape(number.text)}\s+{re.escape(str(column))}", rewritten
            ):
                # "2,000 people died on Australia's roads" counts people: "roads" fitting Road User does not
                # make the claim a count of road users.
                continue
            # "1.1 million refugees" is a total in each reading, as in the one checked.
            other, rewritten, _ = _counted_total(analysis, other, rewritten, number, text, choices)
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
        elif (readings and not by_unit and all(reading[3] is not None for reading in readings)
              and len({number.holds(reading[3]) for reading in readings}) == 1):
            # "broken down train" in cause_en and statistical_cause_en: when every column the words fit gives
            # the same verdict, the claim stands or falls whichever is meant.
            shown = "; ".join(f"{column} gives {_plain(computed)}" for column, _, _, computed in readings)
            column, asked, answer, _ = readings[0]
            fitted = _and_names([item[0] for item in readings])
            assumptions.append(f"The claim's words fit {fitted}, and each gives the same verdict ({shown}); "
                               f"the check reads {column}.")
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
            retried, shorter, found = None, asked, unknown
            if not unknown and (shared := _share_of_two(analysis, asked, answer, number, choices)):
                retried, shorter = shared
            # "babies born": leaving out "born" may show that "babies" names nothing either.
            for _ in range(3):
                if not found or any(KEPT_WORDS & set(_words(phrase)) for phrase in found):
                    break
                for phrase in found:
                    shorter = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", " ", shorter, flags=re.IGNORECASE)
                shorter = re.sub(r"\s+", " ", shorter).strip(" .,;:!?")
                retried = analysis.ask(shorter, decisions=choices) if shorter else None
                shared = _share_of_two(analysis, shorter, retried, number, choices) if retried else None
                if shared:
                    retried, shorter = shared
                found = _unknown_phrases(retried.message) if isinstance(retried, Refusal) else []
                found = [phrase for phrase in found if phrase not in unknown]
                unknown = [*unknown, *found]
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
            if unknown:
                assumptions.append(
                    f"{LEFT_OUT_NOTE} {_quoted(unknown)}, which name no column or value in the table."
                )
            answer, asked = retried, shorter
    # A change in "the number of girls" adds up Count, the column that says how many each row holds: the
    # plan keeps it as its weight, read from the counted noun rather than picked by default.
    unnamed = [
        column for column in (answer.plan.measure, answer.plan.partner)
        if column and not _named(asked, column) and column != answer.plan.weight
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
    row_noun = (analysis.roles.row_noun or "").lower()
    # The noun before the change ("Fox rescues nearly doubled"), or the head of the words the question, or the
    # claim, starts with ("stops in Seattle fell", "notices in San Francisco in 2020", "Deaths rose" in
    # australian-road-deaths, whose name took "Deaths" out of the question).
    heads = []
    for words in (asked, text):
        verb = CHANGE_WORDS.search(words) if changed else None
        lead = words[:verb.start()] if verb else words
        heads += [match.group(1) for match in (re.search(r"([A-Za-z]+)\s*$", lead) if verb else None,
                                               LEADING_HEAD.match(lead)) if match]
    subject = next((head for head in heads if _names_rows([head], row_noun)), None)
    level = unnamed == [answer.plan.measure] and answer.plan.intent == "aggregate" and not changed
    if subject and not averaging and row_noun and (changed or level):
        # In animal_rescues.csv the file's noun for its rows, as the subject, counts them.
        rows_counted = analysis.ask(f"number of {asked}", decisions=choices)
        if not isinstance(rows_counted, Refusal) and rows_counted.plan.intent == (
            "growth" if changed else "count"
        ) and not rows_counted.plan.measure:
            answer, asked, unnamed = rows_counted, f"number of {asked}", []
            reading = "the change as one in the number of" if changed else "the number as a count of"
            assumptions.append(f"Read {reading} {subject}, the file's rows.")
    indicator = next((item for item in answer.plan.filters
                      if set(normalized_name(str(item.column)).split()) & INDICATOR_WORDS), None)
    if indicator is None and unnamed == [answer.plan.measure] and cube_measure(analysis.frame) == unnamed[0]:
        # In a cross-tabulated table whose one column of counts is VALUE, a value the claim names ("Net
        # migration" in Inward or Outward Flow) says what VALUE holds; a total level ADA chose does not.
        totals = {(level.column, level.label) for level in total_levels(analysis.frame)}
        indicator = next((item for item in answer.plan.filters if len(item.values) == 1
                          and (item.column, str(item.values[0])) not in totals), None)
    if unnamed == [answer.plan.measure] and indicator and (
            normalized_name(str(unnamed[0])) in VALUE_NAMES or _only_number(analysis, unnamed[0])):
        # "Adult smoking in England" in a long table: Indicator Name = Smoking Prevalence says what Value
        # holds on the rows it keeps. So does type = theft_vehicle_motorcycle for crimes, the one column of
        # numbers in Malaysia's crime counts, in "Motorcycle thefts fell".
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
    answer, asked, note = _counted_total(analysis, answer, asked, number, text, choices)
    if note:
        assumptions.append(note)
    counted_phrase = _counted_after(text, number)
    # A column named for more than the things counted ("Number of administrators who have left the nursing
    # home" for "nursing homes") holds something else about them.
    measured = str(answer.plan.measure) if answer.plan.measure else ""
    counted_measure = bool(counted_phrase and measured) and not averaged and (
        answer.plan.aggregation in ("mean", "median") and _words_in_name(str(counted_phrase), measured)
        and not _words_in_name(normalized_name(measured), str(counted_phrase)))
    if counted_measure:
        # "There are 23 nursing homes in Kansas City": the homes are counted, and averaging "Number of
        # administrators who have left the nursing home" over them counts nothing.
        return ClaimCheck(
            text, "unverified", "missing_context",
            f'"{number.text} {counted_phrase}" counts {counted_phrase}, but ADA read them as '
            f"{answer.plan.measure} and took its {answer.plan.aggregation} ({answer.calculation}), which "
            "counts nothing.",
            claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
            clarification=f"Which column or rows are the {counted_phrase}?",
        )
    written = NUMBER_PATTERN.search(number.text)
    if written and written.group("currency") and answer.plan.aggregation == "count" and not (
        answer.plan.measure or answer.plan.count_column or answer.plan.weight
    ):
        from pandas.api.types import is_numeric_dtype  # noqa: PLC0415

        from .metrics import is_additive_name  # noqa: PLC0415

        frame = analysis.frame
        money = [str(column) for column in frame.columns
                 if is_numeric_dtype(frame[column]) and is_additive_name(str(column))
                 and _money_column(str(column), frame[column])]
        figures = [str(column) for column in frame.columns
                   if normalized_name(str(column)) in VALUE_NAMES and is_numeric_dtype(frame[column])]
        total = analysis.ask(f"total {money[0]} {asked}", decisions=choices) if len(money) == 1 else None
        figure = analysis.ask(f"{figures[0]} {asked}", decisions=choices) if (
            not money and len(figures) == 1) else None
        unit = _row_unit(analysis, figure) if figure is not None and not isinstance(figure, Refusal) else None
        if total is not None and not isinstance(total, Refusal) and total.plan.aggregation == "sum" and (
            total.plan.measure == money[0]
        ):
            # "more than $2 million in parking fines": an amount of money is the total of the table's one
            # column of amounts, not a count of its rows.
            assumptions.append(f"Read the claim as the total of {money[0]}, the only column of amounts of "
                               "money, as the number is an amount of money.")
            answer, asked = total, f"total {money[0]} {asked}"
        elif unit and UNITS[unit[0]][0] == "money" and figure.scope is not None and (
            figure.scope.rows_used == 1
        ):
            # "fish and chips cost under $8" where the one row's UNITS is Dollars: the amount is its value.
            assumptions.append(f"Read the claim as {figures[0]} of the one row it picks, which {unit[1]} "
                               f"gives in {unit[0]}, as the number is an amount of money.")
            answer, asked = figure, f"{figures[0]} {asked}"
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
    date_column = analysis.roles.date
    parts = [*answer.plan.filters, *answer.plan.counted, *answer.plan.whole,
             *(item for item in answer.plan.conditions if item.column != date_column)]
    if number.percent and not parts and dict(answer.figures).get("share_percent", 0.0) >= 100 - 1e-9:
        # "Home teams won 52% of Bundesliga matches between 1963 and 2009" read as distinct HomeTeam values in
        # those years over all of them: a share of every row, kept by the years alone, says nothing of the
        # claim.
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"The claim gives a percentage, but ADA's reading ({answer.calculation}) keeps every row, by the "
            "claim's years alone, so its share says nothing of the claim.",
            claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
            clarification="A percentage of which rows, meeting what?",
        )
    if number.percent and (answer.plan.intent == "count" or answer.plan.aggregation == "sum") and not (
        {"share_percent", "change_percent"} & set(dict(answer.figures))
    ):
        # A percentage compared with a plain count fails a true claim ("half of the squirrels" with 1,434).
        if not re.match(r"share\b", asked, re.IGNORECASE):
            # "good on 75% of the days": the days that were good, as a share of the days.
            retried = _check(analysis, text, _share_question(asked), number, source, choices)
            if retried.answer is not None and "share_percent" in dict(retried.answer.figures):
                return retried
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"The claim gives a percentage, but ADA read the words as a count ({answer.calculation}), not as "
            "a share of a base it could name.",
            claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
            clarification="A percentage of which rows?",
        )
    computed, notes = _comparable(answer, number)
    recorded = None if number.percent or computed is None else _recorded_scale(analysis, answer)
    if recorded:
        notes.append(f"{recorded[2]} gives {answer.plan.measure or answer.plan.weight} in {recorded[1]}, so "
                     f"{_plain(computed)} is {_plain(computed * recorded[0])}.")
        computed *= recorded[0]
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
    said_slope = _time_slope(worded[0] if worded else text)
    if said_slope and answer.plan.intent not in ("trend", "correlation") and not slope:
        # "rose by about 1 a year" over a single year: an average is no change each year.
        return ClaimCheck(
            text, "unverified", "unsupported_analysis",
            f"The claim states a change each {said_slope[2]}, and ADA found no slope over time for it: "
            f"{answer.answer}",
            claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
            clarification="Which column changes, and over which years?",
        )
    timed = said_slope if answer.plan.intent == "trend" else None
    scale = 1.0
    if timed:
        # "grew by about 500 million pounds a year": the slope of the yearly values, by least squares and by
        # Theil-Sen; when only one of them agrees, the claim does not say which it means.
        _, _, unit, falling = timed
        found = dict(answer.figures)
        keys = (("least squares", "least_squares_per_year"), ("Theil-Sen", "slope_per_year"))
        # "29 cents a year" of prices in dollars: both slopes are in the claim's unit before they are
        # compared.
        unit_change = _unit_factor(analysis, answer, number, measure)
        scale = unit_change[0] if unit_change else 1.0
        fits = [(name, found[key] / TIME_SLOPE_UNITS[unit] * (-1 if falling else 1) * scale)
                for name, key in keys if key in found]
        in_unit = [f"Converted {measure} from {unit_change[1]}, {unit_change[2]}, to {_claim_unit(number)}."
                   ] if unit_change else []
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
        reversed_sign = "; the claim says it falls, so the slope's sign is reversed." if falling else "."
        notes = [*in_unit, f"Compared with the least-squares slope of the yearly values, {_plain(computed)} "
                 f"per {unit}{same}{reversed_sign}"]
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
    where = "the unit in its name"
    if claimed_unit and not recorded_unit and (row_unit := _row_unit(analysis, answer)):
        recorded_unit, where = row_unit[0], f"the unit {row_unit[1]} gives for these rows"
    key = _key_filter(analysis.frame, answer)
    if key and answer.plan.aggregation == "count" and not measure and not answer.plan.count_column and (
            not answer.plan.weight):
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
    written = NUMBER_PATTERN.search(number.text)
    if written and written.group("currency") and answer.plan.aggregation == "count" and not measure:
        # "values its trees at almost £1 billion": an amount of money is never a count of things.
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"The claim gives an amount of money, but ADA read it as a count ({answer.calculation}).",
            claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
            clarification="Which column holds the amount?",
        )
    said_noun = (number.unit or "").split()[:1]
    rows = answer.plan.aggregation == "count" and not measure and not answer.plan.count_column and (
        not answer.plan.weight)
    if rows and said_noun and (described := _described_as(analysis.frame, said_noun[0])):
        # "Germany had about 700 cars" where every row reads "Passenger cars - per thousand inhabitants": the
        # cars are what each row's figure counts, not the rows.
        from pandas.api.types import is_numeric_dtype  # noqa: PLC0415

        frame = analysis.frame
        figures = [str(column) for column in frame.columns
                   if normalized_name(str(column)) in VALUE_NAMES and is_numeric_dtype(frame[column])]
        if len(figures) == 1 and figures[0].lower() not in asked.lower():
            check = _check(analysis, text, f"{figures[0]} {asked}", number, source, choices)
            note = f'Read "{said_noun[0]}" as {figures[0]}, the figure every row gives for "{described}".'
            scope = check.answer.scope if check.answer is not None else None
            summed = check.answer is not None and check.answer.plan.aggregation == "sum"
            rate = re.search(r"\bper\b", described, re.IGNORECASE)
            if rate and summed and scope and (scope.rows_used or 0) > 1:
                # A rate "per thousand inhabitants" does not add up over rows.
                return replace(check, status="unverified", reason="missing_context",
                               clarification="Which row is meant?", assumptions=(note, *check.assumptions),
                               message=f"{figures[0]} is a rate ({described}), and the claim's rows hold "
                                       f"{scope.rows_used} of them, which do not add up.")
            return replace(check, assumptions=(note, *check.assumptions))
        return ClaimCheck(
            text, "unverified", "missing_context",
            f'The claim counts {said_noun[0]}, which the table\'s figures are of ("{described}"), but ADA '
            f"read it as a count of rows ({answer.calculation}).",
            claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
            clarification="Which column holds the number?",
        )
    if claimed_unit and answer.plan.aggregation == "count" and not measure and (
        step := _row_step(analysis) or _row_step(analysis, answer.plan)
    ) and UNITS[claimed_unit] == UNITS[step]:
        # "negative prices for more than 500 hours" in a table of one reading an hour: each row is an hour.
        # In a table of several stations, each of one station's rows is a day.
        picked = [f"{item.column} = {list_names([str(value) for value in item.values])}"
                  for item in answer.plan.filters if isinstance(item, ValueFilter)]
        among = f" for {', '.join(picked)}" if picked and not _row_step(analysis) else ""
        assumptions.append(f"Read the {number.unit} as rows, since each row of the table{among} is one "
                           f"{step} of {analysis.roles.date}.")
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
            f'The claim gives the number in {number.unit}, but {measure} is recorded in '
            f'{recorded_unit.replace("°c", "°C").replace("°f", "°F")}, '
            f"which measures {UNITS[recorded_unit][0]}, not {UNITS[claimed_unit][0]}.",
            claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
            clarification=f"Is the number about {measure}, or about another column?",
        )
    factor = scale
    if claimed_unit and recorded_unit and computed is not None and claimed_unit != recorded_unit and (
        not timed
    ):
        factor = UNITS[recorded_unit][1] / UNITS[claimed_unit][1]
        converted = computed * factor
        assumptions.append(
            f"Converted {measure} from {recorded_unit}, {where}, to {claimed_unit}: "
            f"{_plain(computed)} {recorded_unit} is {_plain(converted)} {claimed_unit}."
        )
        computed = converted
    elif number.unit and not recorded_unit:
        said = number.unit.split()[0]
        # The number's text ends with its unit ("2.6 million unemployed people"), so the words after the
        # number are read from where its own figure ends.
        figure = re.match(r"(?:[^\d]*?)[\d.,]+(?:\s*(?:million|billion|thousand|bn|m|k)\b)?", number.text)
        end = text.find(number.text) + (figure.end() if figure else len(number.text))
        after = text[end:] if number.text in text else ""
        following = re.match(rf"\s*{re.escape(said)}\s+([A-Za-z]+)", after)
        modifier = following is not None and not claimed_unit and (
            following.group(1).lower() == (analysis.roles.row_noun or "").lower()
            or (following.group(1).lower().endswith("s") and following.group(1).lower() not in VERB_ENDINGS_S)
            or following.group(1).lower() in IRREGULAR_PLURALS
        )
        filtered = {item.column for item in answer.plan.filters}
        unused = [column for column in _valued_columns(analysis.frame, said) if column not in filtered] if (
            modifier) else []
        if unused:
            # "about 3,200 plant species": "plant" picks Category = Vascular Plant or Non-vascular Plant,
            # which a unit would leave out.
            return ClaimCheck(
                text, "unverified", "missing_context",
                f'The claim\'s "{said}" names values of {unused[0]}, which ADA\'s reading '
                f"({answer.calculation}) does not use, so the number may count only some of its rows.",
                claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
                clarification=f"Which {unused[0]} values are the {said} {following.group(1)}?",
            )
        if not number.scale_word:
            assumptions.append(f'The unit "{number.unit}" comes from the claim; the check compares numbers '
                               "only.")
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
    # A value whose unit column says Dollars for these rows is money too.
    row_unit = _row_unit(analysis, answer)
    money = measure_values is not None and (_money_column(str(measure), measure_values) or bool(
        row_unit and UNITS[row_unit[0]][0] == "money"))
    spent = [phrase for phrase in left_out
             if phrase.lower() in VERB_NOUNS or phrase.lower() in PRICE_WORDS] if money else []
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
    # "more than half of the disruptions on the Dutch railways": a share of the rows' own noun is of the rows.
    # "fifty-one disruptions" in disruptions-2025.csv counts the rows by their own noun.
    counted_noun = re.match(r"\s*how\s+many\s+([A-Za-z]+)", asked)
    rows_named = _names_rows([counted_noun.group(1)], analysis.roles.row_noun) if counted_noun else (
        number.percent and _names_rows(re.findall(r"[A-Za-z]+", asked), analysis.roles.row_noun))
    frame = analysis.frame
    # "Athens is twinned with 49 cities" in cities.csv: name = Athens picks the city itself (a column whose
    # values are nearly all different), so the cities after the number are others, not its rows.
    subject = any(item.column in frame.columns and frame[item.column].nunique() >= NAME_SHARE * frame[
        item.column].notna().sum() for item in plan.filters)
    if not rows_named and analysis.roles.row_noun and number.text in text and not subject:
        # "more than 1,500 law enforcement agencies" in agencies.csv: the rows' own noun right after the
        # number is what it counts.
        after = text[text.index(number.text) + len(number.text):]
        rows_named = _names_rows(re.findall(r"[A-Za-z]+", after)[:4], analysis.roles.row_noun)
    if (named and counts_rows and agreement == "mismatch" and not _names_rows(named, analysis.roles.row_noun)
            and not rows_named and (_holds_counts(analysis) or not _values_written(plan, text))):
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
    timed = {phrase.lower() for phrase in left_out} & TIME_UNIT_WORDS
    # "8,000 vehicles an hour" over rows of an hour each; over rows of a minute an hour adds sixty of them.
    stepped = not timed or timed <= {_row_step(analysis) or "", f"{_row_step(analysis)}s"}
    if agreement == "mismatch" and left_out and stepped and _beyond_every_row(
            answer, number, computed, measure, asked, analysis.frame):
        # "topped 8,000 vehicles an hour" against a highest of 7,280 over every row: words that name nothing
        # can only narrow the rows, and over fewer rows the highest is no higher.
        words = " and ".join(f'"{phrase}"' for phrase in left_out)
        side = {"max": "highest", "min": "lowest", "sum": "total"}[answer.plan.aggregation]
        return ClaimCheck(
            text, "unverified", "failed_check",
            f"The data give {_plain(computed)}{rows} for {described}, outside {number.interval()}. Without "
            f"{words}, which name nothing in the table, this is over every row the rest of the claim keeps; "
            f"the words left out could only keep fewer, and over fewer rows the {side} is never beyond it.",
            claimed=number, question=asked, answer=answer, compared=computed, agreement="mismatch",
            assumptions=tuple(dict.fromkeys(assumptions)), warnings=tuple(dict.fromkeys(warnings)),
            checks=tuple(checks), source=source,
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
    # "Net migration" in Inward or Outward Flow says what VALUE holds on the rows it keeps.
    valued = any(item.startswith(f"Read {measure} as the value of ") for item in assumptions)
    # A column of counts read from the noun counted ("the number of girls") is not a guess either.
    tallied = measure == answer.plan.weight
    if agreement == "mismatch" and measure and not (valued or tallied) and not _names_head(
            " ".join([asked, *named]), measure, strict=True):
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
    sibling = _head_sibling(analysis, asked, number, answer, choices, factor)
    if sibling and sibling[4] != (agreement == "match"):
        # "Women accounted for 47.4% of the deaths": of Total deaths 47.44%, of COVID-19 Deaths 45.07%; "the
        # deaths" names both.
        column, question, other, value, agrees = sibling
        measure_read = str(answer.plan.weight or answer.plan.measure)
        alternatives = (
            Alternative(measure_read, asked, computed, agreement == "match", described),
            Alternative(column, question, value, agrees, other.calculation),
        )
        return ClaimCheck(
            text, "unverified", "ambiguous_intent",
            f"The claim's words name {measure_read} and {column} alike: the data give {_plain(computed)} "
            f"with {measure_read} and {_plain(value)} with {column}, and only one is in {number.interval()}.",
            claimed=number, question=asked, answer=answer, compared=computed, agreement=agreement,
            assumptions=tuple(dict.fromkeys(assumptions)), warnings=tuple(dict.fromkeys(warnings)),
            checks=tuple(checks), alternatives=alternatives, source=source,
            clarification=f"Is the number about {measure_read} or {column}?",
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
