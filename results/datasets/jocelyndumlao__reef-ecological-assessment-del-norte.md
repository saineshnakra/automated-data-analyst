# Reef Ecological Assessment: Del Norte

[jocelyndumlao/reef-ecological-assessment-del-norte](https://www.kaggle.com/datasets/jocelyndumlao/reef-ecological-assessment-del-norte) on Kaggle · environmental · 2,431 rows × 30 columns · licence: CC0: Public Domain · round 2

A fish census from August 2019 on two reef systems in the Gulf of Mexico: 2,431 rows, each one species' count and size class on one 60 m2 transect (15 reefs, 88 transects), with taxonomy, trophic group, length-weight coefficients, derived biomass and protection status (no-take zone, marine protected area, open access).

**First run:** partly useful. The numbers check out and Ask handles simple rankings, but the headline is an effort-driven share, the main comparison (protection status, per transect) is missing, and the differences scan leads with coordinates that differ between reefs.

**Now:** partly useful. The arithmetic is right and the differences card now leads with a real contrast (piscivores are the largest fish), but the headline is an effort-driven share and the protection-status comparison per transect, the point of the survey, appears only if the user asks for it in exactly the right words.

## What ADA finds now

This file is a 2019 fish census on two Gulf of Mexico reef systems, one row per species per transect. The analysis now sets map coordinates and fixed species traits aside and finds sensible contrasts, such as piscivores being much larger than plankton feeders, and it counts species per reef correctly. Its headline, that Alacranes holds 67% of all fish counted, mostly reflects that more transects were surveyed there. It does not compare the no-take zone, protected area and open-access sites per transect in its summary, although asked 'per transect' it correctly finds the no-take zone has about twice as many fish as open access and four times as many as the protected area.

Headline: “Alacranes has the largest share of quantity: 67.2% (11.7K).”

## What the first review found, and where it stands

| Issue | Severity | Now |
|---|---|---|
| The headline is a share of a summed count, which mostly reflects survey effort. | high | not fixed |
| Wrong unit of analysis: groups compared on species records instead of transects. | high | partly fixed |
| Coordinates and site attributes were compared as readings. | high | fixed |
| Market-share framing and a 'top three' sentence with only two groups. | medium | not fixed |
| A defined minimum tied to one category (herbivores at trophic level 2) was called an instrument floor. | medium | fixed |
| Species-level constants were analysed per record, and biomass, an exact function of other columns, was not recognised. | medium | partly fixed |
| Constant columns were kept without a note. | medium | fixed |
| When two groupings split rows identically, one was dropped without mention. | low | fixed |
| An integer reef code was treated as a quantity. | low | partly fixed |
| Counting distinct values within each group was not supported. | low | fixed |

## Still open

- The headline and the concentration card report Alacranes' 67% share of all fish counted, which mostly reflects that it had 54 of the 88 transects; the concentration card even speaks of the 'top three' of two regions, and its advice to 'recompute the overall figures without Alacranes' would leave a single region.
- The brief never compares protection status per transect, the main question for this survey; the no-take zone averages 324 fish per transect against 168 in open access and 82 in the protected area, which only a carefully worded question reveals, while 'does quantity differ by protection_status' answers at record level that open access is highest.
- Transect numbers, which restart at 1 on each reef, and reef codes are still used as ordered groupings with trend tests ('transect explains 1%', 'quantity rises as id_reef increases'), and 'how many transects per region' answers 6 for each region instead of 54 and 34.
- Biomass is not recognised as calculated from count, size and the species coefficients, so its correlation with size is reported as a lead.

## Reproduce

Download the dataset from Kaggle and run `ada analyze "Alacranes Reef and Bajos del Norte reef ecological assesment/fish_data.csv"`. The data is not stored in this repository.
