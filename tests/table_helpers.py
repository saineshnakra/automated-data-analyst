"""Published tables for the question and claim tests."""

from __future__ import annotations

import pandas as pd


def migration() -> pd.DataFrame:
    """A CSO-style table of migration in thousands, every count under each level and again under its total.

    By hand, 2024: immigrants by age band 20, 15, 50, 10 and 5 (100 in all), emigrants 5, 10, 20, 4 and 1
    (40); 2023: immigrants 10, 10, 30, 8 and 2 (60), emigrants 4, 8, 16, 2 and 2 (32). Men are 60% of each.
    """
    bands = ["0 - 14 years", "15 - 24 years", "25 - 44 years", "45 - 64 years", "65 years and over"]
    counts = {
        (2024, "Immigrants: All origins"): [20.0, 15.0, 50.0, 10.0, 5.0],
        (2024, "Emigrants: All destinations"): [5.0, 10.0, 20.0, 4.0, 1.0],
        (2023, "Immigrants: All origins"): [10.0, 10.0, 30.0, 8.0, 2.0],
        (2023, "Emigrants: All destinations"): [4.0, 8.0, 16.0, 2.0, 2.0],
    }
    rows = []
    for (year, flow), values in counts.items():
        for age, value in [("All ages", sum(values)), *zip(bands, values, strict=True)]:
            for sex, part in (("Both sexes", 1.0), ("Male", 0.6), ("Female", 0.4)):
                rows.append({"Year": year, "Age Group": age, "Sex": sex, "Inward or Outward Flow": flow,
                             "UNIT": "Thousand", "VALUE": round(value * part, 1)})
    return pd.DataFrame(rows)



def births() -> pd.DataFrame:
    """A StatCan-style table of births: counts beside percentages, a nation over its provinces, and labels
    that all start "Age of mother, ".

    By hand, 2025: Canada 100 births, 12 to mothers aged 15 to 19, 50 aged 20 to 34, 30 aged 35 to 39 and 8
    aged 40 to 44; Alberta 40 and Nunavut 60, Nunavut's 12 all to mothers aged 15 to 19. 2024 halves each.
    """
    ages = ["Age of mother, all ages", "Age of mother, 15 to 19 years", "Age of mother, 20 to 34 years",
            "Age of mother, 35 to 39 years", "Age of mother, 40 to 44 years"]
    places = {"Alberta, place of residence of mother": [40, 0, 26, 10, 4],
              "Nunavut, place of residence of mother": [60, 12, 24, 20, 4]}
    rows = []
    for year, part in ((2025, 1.0), (2024, 0.5)):
        canada = [sum(column) for column in zip(*places.values(), strict=True)]
        for geo, counts in {"Canada, place of residence of mother": canada, **places}.items():
            for age, count in zip(ages, counts, strict=True):
                number = count * part
                share = number / (counts[0] * part) * 100
                rows.append({"REF_DATE": year, "GEO": geo, "Age of mother": age,
                             "Characteristics": "Number of live births", "UOM": "Number",
                             "COORDINATE": f"{len(rows) % 3 + 1}.{len(rows) % 5 + 1}.1", "VALUE": number})
                rows.append({"REF_DATE": year, "GEO": geo, "Age of mother": age,
                             "Characteristics": "Percentage of live births", "UOM": "Percentage",
                             "COORDINATE": f"{len(rows) % 3 + 1}.{len(rows) % 5 + 1}.2",
                             "VALUE": round(share, 1)})
    return pd.DataFrame(rows)


def registry() -> pd.DataFrame:
    """A PHS-style cancer registry: sites that overlap (Colon within Colorectal cancer) under "All cancer
    types", which leaves out non-melanoma skin cancer, each count under Male, Females and their total All
    (flagged "d" in SexQF), and counts by age band in columns beside the count for all ages.

    By hand, 2024 is 1.2 times 2020: All cancer types 390 in 2020, 468 in 2024 (240 men, 228 women); lung
    104 (50 men, 54 women) in 2020, so 60 and 65 in 2024; breast 120 women and 1 man in 2024; prostate 96
    men; cervix 24 women, with no row for All. Each count for all ages splits into 20% under 50, 25% aged
    75 to 84 and 10% aged 85 and over, rounded down, and the rest aged 50 to 74.
    """
    sites = {
        ("C00-C97, excluding C44", "All cancer types"): (200, 190),
        ("C33-C34", "Trachea, bronchus and lung"): (50, 54),
        ("C50", "Breast"): (1, 100),
        ("C18-C20", "Colorectal cancer"): (40, 30),
        ("C18", "Colon"): (25, 20),
        ("C43", "Malignant melanoma of the skin"): (15, 15),
        ("C44", "Non-melanoma skin cancer"): (60, 50),
        ("C61", "Prostate"): (80, None),
        ("C53", "Cervix uteri"): (None, 20),
        ("C70-C72, D32-D33", "All brain and CNS tumours (malignant and non-malignant)"): (10, 12),
    }
    rows = []
    for year in range(2020, 2025):
        factor = 1 + 0.05 * (year - 2020)
        for (code, site), (men, women) in sites.items():
            counts = {"Male": men, "Females": women}
            counts = {sex: round(count * factor) for sex, count in counts.items() if count is not None}
            if len(counts) == 2 or site == "Prostate":
                counts = {"All": sum(counts.values()), **counts}
            for sex, count in counts.items():
                under, old, oldest = int(count * 0.2), int(count * 0.25), int(count * 0.1)
                rows.append({
                    "Country": "S92000003", "CancerSiteICD10Code": code, "CancerSite": site, "Sex": sex,
                    "SexQF": "d" if sex == "All" else None, "Year": year, "IncidencesAgeUnder50": under,
                    "IncidencesAge50To74": count - under - old - oldest, "IncidencesAge75To84": old,
                    "IncidencesAge85AndOver": oldest, "IncidencesAllAges": count, "EASR": count / 3.7,
                })
    return pd.DataFrame(rows)


def deaths() -> pd.DataFrame:
    """A NISRA-style table of deaths by single year of age, 0 to 70, and sex, each count again under AGE
    "ALL" and SEX "ALL", with the age and the sex each written twice (a code and a label).

    By hand: in 2023 women aged a died 1 + a % 5 times and men 2 + a % 3 times; in 2024 one more woman of
    each age died, so 71 more deaths in all.
    """
    rows = []
    for year in (2023, 2024):
        for sex, label in (("1", "Females"), ("2", "Males"), ("ALL", "All persons")):
            counts = []
            for age in range(71):
                women, men = 1 + age % 5 + (year - 2023), 2 + age % 3
                counts.append(women if sex == "1" else men if sex == "2" else women + men)
            labelled = [*((str(age), count) for age, count in enumerate(counts)), ("ALL", sum(counts))]
            for age, count in labelled:
                rows.append({"STATISTIC": "DTHCOUNT", "Statistic Label": "Deaths", "Year": year, "AGE": age,
                             "Single year of age": "All ages" if age == "ALL" else age, "SEX": sex,
                             "Sex Label": label, "UNIT": "Number", "VALUE": count})
    return pd.DataFrame(rows)


def air_quality() -> pd.DataFrame:
    """EPA-style days of air quality by county, the kinds of day kept in columns that add up to Days with AQI.

    By hand: 610 good days of 1,810 (33.7%); Maricopa 32 of 365 (8.77%).
    """
    return pd.DataFrame({
        "State": ["Arizona", "Arizona", "Alabama", "California", "California"],
        "County": ["Maricopa", "Pima", "Baldwin", "Los Angeles", "San Bernardino"],
        "Year": [2025] * 5,
        "Days with AQI": [365, 360, 355, 365, 365],
        "Good Days": [32, 250, 268, 40, 20],
        "Moderate Days": [280, 100, 87, 227, 235],
        "Unhealthy Days": [53, 10, 0, 98, 110],
        "Max AQI": [180, 120, 87, 200, 220],
    })


def commuting() -> pd.DataFrame:
    """Census 2021-style methods of travel to work by area, each method a column under one heading.

    By hand: the City of London's 3,602 of 5,352 work mainly at or from home (67.3%); Cardiff, the one area
    in Wales (its code starts with W), has 90,000 of 170,000 driving.
    """
    heading = "Method of travel to workplace: "
    return pd.DataFrame({
        "date": [2021] * 5,
        "geography": ["City of London", "Newham", "Cambridge", "Hartlepool", "Cardiff"],
        "geography code": ["E09000001", "E09000025", "E07000008", "E06000001", "W06000015"],
        f"{heading}Total: All usual residents aged 16 years and over in employment the week before the "
        "census": [5352, 163446, 70596, 37750, 170000],
        f"{heading}Work mainly at or from home": [3602, 30000, 25000, 6570, 40000],
        f"{heading}Underground, metro, light rail, tram": [500, 38371, 100, 33, 0],
        f"{heading}Bicycle": [200, 5000, 11836, 561, 8000],
        f"{heading}Driving a car or van": [124, 40000, 15000, 22448, 90000],
        f"{heading}Other method of travel to work": [926, 50075, 18660, 8138, 32000],
    })


def covid_deaths() -> pd.DataFrame:
    """CDC-style deaths by sex and single year of age over one period, all deaths beside those with COVID-19.

    Ages run "0-05 Months", "06-11 Months", "01 Year" to "84 Years" and "85 Years and over". For a row of
    age a (both month rows are 0, the oldest 85), Total deaths is 200 + 10 (a mod 3) for men and
    180 + 10 (a mod 3) for women, and COVID-19 Deaths is 10 + (a mod 5) for men and 5 + (a mod 5) for women.
    """
    ages = [("0-05 Months", 0), ("06-11 Months", 0), ("01 Year", 1),
            *((f"{age:02d} Years", age) for age in range(2, 85)), ("85 Years and over", 85)]
    rows = []
    for sex, extra in (("Male", 1), ("Female", 0)):
        for label, age in ages:
            rows.append({"Data as of": "06/28/2023", "Start Date": "01/01/2020", "End Date": "06/24/2023",
                         "Sex": sex, "Age Years": label, "Total deaths": 180 + 20 * extra + 10 * (age % 3),
                         "COVID-19 Deaths": 5 + 5 * extra + age % 5})
    frame = pd.DataFrame(rows)
    for column in ("Data as of", "Start Date", "End Date"):
        frame[column] = pd.to_datetime(frame[column], format="%m/%d/%Y")
    return frame


def migration_estimates() -> pd.DataFrame:
    """Monthly migration estimates for 2022 and 2023 by direction, sex and age, with TOTAL levels."""
    rows = []
    months = pd.date_range("2022-01-01", periods=24, freq="MS")
    for month in months:
        arrivals = 5000 if month.year == 2022 else 10000
        departures = 4000 if month.year == 2022 else 3000
        for direction, total in (("Arrivals", arrivals), ("Departures", departures),
                                 ("Net", arrivals - departures)):
            for sex, share in (("TOTAL", 1.0), ("Female", 0.5), ("Male", 0.5)):
                for age, part in (("TOTAL", 1.0), ("0-14 years", 0.25), ("15+ years", 0.75)):
                    rows.append({"year_month": month, "month_of_release": month + pd.DateOffset(months=2),
                                 "direction": direction, "sex": sex, "age": age,
                                 "estimate": int(total * share * part), "standard_error": 10})
    return pd.DataFrame(rows)


def road_deaths() -> pd.DataFrame:
    """One row per person killed on Australia's roads in 2021 and 2022, with Year, Month and a Time of day.

    Queensland has the most deaths in 2022 (6) but more in 2021 (8); every state but the three largest has one
    death a year.
    """
    counts = {2021: {"Qld": 8, "NSW": 7, "Vic": 3}, 2022: {"Qld": 6, "NSW": 5, "Vic": 4}}
    regions = {"Qld": "Queensland - Outback", "NSW": "Sydney - City", "Vic": "Melbourne - Inner",
               "SA": "Adelaide - Central", "WA": "Perth - Inner", "Tas": "Hobart", "NT": "Darwin",
               "ACT": "Australian Capital Territory"}
    rows = []
    for year, largest in counts.items():
        for state, region in regions.items():
            for index in range(largest.get(state, 1)):
                rows.append({
                    "Crash ID": len(rows) + 1, "State": state, "Month": index % 12 + 1, "Year": year,
                    "Time": f"{index:02d}:30", "Road User": ("Pedestrian", "Driver", "Passenger")[index % 3],
                    "Age": 70 if state == "ACT" else 20 + 5 * index, "SA4 Name 2021": region,
                })
    return pd.DataFrame(rows)


def food_prices() -> pd.DataFrame:
    """Monthly prices from September 2016 to September 2026 in Stats NZ's layout: Period written as 2016.09, a
    Data_value whose UNITS says Dollars or Index, and Group names that say the series are for New Zealand.

    A dozen eggs rises 2 cents a month from $3.00; a loaf rises 1 cent a month from $1.00 to $2.20.
    """
    months = pd.period_range("2016-09", "2026-09", freq="M")
    rows = []
    for index, month in enumerate(months):
        period = float(f"{month.year}.{month.month:02d}")
        for title, group, units, value in (
            ("Eggs, dozen", "Food Price Index Selected Monthly Weighted Average Prices for New Zealand",
             "Dollars", 3.0 + 0.02 * index),
            ("Bread - white sliced loaf, 600g",
             "Food Price Index Selected Monthly Weighted Average Prices for New Zealand", "Dollars",
             1.0 + 0.01 * index),
            ("Overseas accommodation prepaid in New Zealand", "CPI Level 3 Classes for New Zealand", "Index",
             1000.0 + index),
            ("Rent", "CPI Monthly Rents (National)", "Index", 900.0 + index),
        ):
            rows.append({"Period": period, "Data_value": round(value, 2), "UNITS": units, "Group": group,
                         "Series_title_1": title})
    return pd.DataFrame(rows)


def parking_tickets() -> pd.DataFrame:
    """Parking tickets written in 2015: 3 before 10 am and 7 from 10 am on, $796 in all."""
    times = ["08:15", "09:05", "09:59", "10:00", "10:30", "11:45", "13:00", "14:20", "16:00", "17:59"]
    days = [f"2015-03-{day:02d} {time}" for day, time in zip(range(2, 12), times, strict=True)]
    amounts = [55.0, 55.0, 82.0, 82.0, 110.0, 55.0, 55.0, 82.0, 165.0, 55.0]
    return pd.DataFrame({"IssueDate": pd.to_datetime(days), "Offence": ["FAIL TO PAY"] * 10,
                         "OffenceAmount": amounts})


def baby_names() -> pd.DataFrame:
    """Babies by year of birth, sex, ethnicity and first name, with every 2014 row given twice, as NYC's file
    does. Girls named Sophia: 500 in 2014 (1,000 counting the repeats), 400 in 2015 and 300 in 2016."""
    counts = {
        ("FEMALE", "Sophia"): {2014: (300, 200), 2015: (250, 150), 2016: (180, 120)},
        ("MALE", "Liam"): {2014: (400, 100), 2015: (380, 120), 2016: (350, 150)},
        ("FEMALE", "Emma"): {2014: (220, 80), 2015: (210, 90), 2016: (200, 100)},
    }
    rows = []
    for (sex, name), years in counts.items():
        for year, pair in years.items():
            for ethnicity, count in zip(("HISPANIC", "WHITE NON HISPANIC"), pair, strict=True):
                row = {"Year of Birth": year, "Gender": sex, "Ethnicity": ethnicity,
                       "Child's First Name": name, "Count": count, "Rank": 1 if count >= 200 else 2}
                rows.extend([row, dict(row)] if year == 2014 else [row])
    return pd.DataFrame(rows)


def eviction_notices() -> pd.DataFrame:
    """Eviction notices by File Date, as San Francisco's Rent Board gives them: 6 in 2018, 5 in 2019, 2 in
    2020 and 3 in 2021; non-payment is cited once in 2020 and three times in 2021. Constraints Date is on one
    row and data_as_of is the day of the extract."""
    days = (["2018-03-01"] * 6 + ["2019-04-02"] * 5 + ["2020-05-03"] * 2 + ["2021-06-04"] * 3)
    unpaid = [False] * 11 + [True, False] + [True] * 3
    return pd.DataFrame({
        "Eviction ID": [f"M{index:03d}" for index in range(16)],
        "City": ["San Francisco"] * 16,
        "File Date": pd.to_datetime(days),
        "Non Payment": unpaid,
        "Owner Move In": [True, False] * 8,
        "Constraints Date": pd.to_datetime([None] * 12 + ["2020-07-01"] + [None] * 3),
        "data_as_of": pd.to_datetime(["2025-01-01"] * 16),
    })



def deaths_by_cause_wide() -> pd.DataFrame:
    """Deaths by area and cause with a column for each year, "2019 Deaths" to "2024 Deaths", as Statistics
    Finland gives them. Areas nest (WHOLE COUNTRY is the mainland and Åland; the mainland is two counties;
    MA2 ÅLAND repeats HVA91 Åland), and so do causes (00-54 Total is the three single causes; 04-22
    Neoplasms is the two cancers). The whole country has 1,080 deaths in 2022, the most of any year, and
    864 in 2019."""
    leaves = {"HVA01 East Uusimaa wellbeing services county": 3, "HVA05 Southwest Finland wellbeing services "
              "county": 5, "HVA91 Åland": 1}
    causes = {"04 Malignant neoplasm of lung": 2, "05 Malignant neoplasm of breast": 1, "27 Ischaemic heart "
              "diseases": 3}
    years = range(2019, 2025)
    per_year = {2019: 16, 2020: 17, 2021: 18, 2022: 20, 2023: 19, 2024: 18}

    def deaths(areas: list[str], kinds: list[str], year: int) -> int:
        return sum(leaves[area] * causes[kind] * per_year[year] for area in areas for kind in kinds)

    areas = {
        "WHOLE COUNTRY": list(leaves), "MA1 MAINLAND FINLAND": list(leaves)[:2],
        **{area: [area] for area in list(leaves)[:2]}, "MA2 ÅLAND": ["HVA91 Åland"],
        "HVA91 Åland": ["HVA91 Åland"],
    }
    kinds = {"00-54 Total": list(causes), "04-22 Neoplasms": list(causes)[:2],
             **{kind: [kind] for kind in causes}}
    rows = [{"Wellbeing services county": area, "Underlying cause of death": kind,
             **{f"{year} Deaths": deaths(parts, members, year) for year in years}}
            for area, parts in areas.items() for kind, members in kinds.items()]
    return pd.DataFrame(rows)


def crimes_by_type() -> pd.DataFrame:
    """Crimes by state, category, type and year, as Malaysia's police give them: the state Malaysia is the
    country's total, and type "all" is each category's. Rapes in Malaysia: 1,600 in 2016 rising by 50 a
    year to 1,950 in 2023, the most of any year; motorcycle thefts fall from 30,000 in 2016 to 10,400 in
    2023, 65.3% fewer."""
    rows = []
    for year in range(2016, 2024):
        step = year - 2016
        counts = {
            "Selangor": {"assault": {"rape": 400 + 10 * step, "murder": 50},
                         "property": {"theft_vehicle_motorcycle": 10_000 - 1_000 * step,
                                      "theft_other": 5_000}},
            "Johor": {"assault": {"rape": 1_200 + 40 * step, "murder": 40},
                      "property": {"theft_vehicle_motorcycle": 20_000 - 1_800 * step, "theft_other": 4_000}},
        }
        for state, categories in counts.items():
            for category, types in categories.items():
                for kind, count in {"all": sum(types.values()), **types}.items():
                    rows.append({"state": state, "category": category, "type": kind,
                                 "date": f"{year}-01-01", "crimes": count})
        for category in ("assault", "property"):
            for kind in {"all", *counts["Selangor"][category]}:
                total = sum(row["crimes"] for row in rows
                            if row["date"] == f"{year}-01-01" and row["category"] == category
                            and row["type"] == kind and row["state"] != "Malaysia")
                rows.append({"state": "Malaysia", "category": category, "type": kind, "date": f"{year}-01-01",
                             "crimes": total})
    return pd.DataFrame(rows)


def road_deaths_colombia() -> pd.DataFrame:
    """Road deaths in Colombia, one row each as the Policía Nacional gives them: a municipality code (CODIGO
    DANE), CANTIDAD of 1 on every row, and a town called Colombia among the municipalities. 2024: 6 deaths in
    Antioquia, 4 in Cauca and 1 in Colombia (Huila), 11 in all; 2025: 7 in Antioquia, 5 in Cauca and one in
    each of 30 towns of Valle del Cauca."""
    places = {"Medellín": ("ANTIOQUIA", 5001000), "Popayán": ("CAUCA", 19001000),
              "Colombia": ("HUILA", 41206000)}
    towns = {f"Pueblo {number}": ("VALLE DEL CAUCA", 76001000 + number) for number in range(1, 31)}
    places.update(towns)
    deaths = {2024: {"Medellín": 6, "Popayán": 4, "Colombia": 1},
              2025: {"Medellín": 7, "Popayán": 5, **dict.fromkeys(towns, 1)}}
    rows = [{"DEPARTAMENTO": places[town][0], "MUNICIPIO": town, "CODIGO DANE": places[town][1],
             "FECHA HECHO": f"{day + 1:02d}/03/{year}", "CANTIDAD": 1}
            for year, towns in deaths.items() for town, count in towns.items() for day in range(count)]
    return pd.DataFrame(rows)


def cinema_tickets() -> pd.DataFrame:
    """Cinema tickets as Statistics Denmark gives them (BIO5): a row for every film category, type of film,
    nationality and year (TID), each at every level and again at its totals, with the count in INDHOLD.

    "Premiere films that year" are some of "Screened films (all)"; each "X films, all audiences" adds up
    the adult and children's films of its kind, and "Feature films and documentaries, all audiences" both
    kinds. Screened films, 2025: Denmark 3,674, USA 4,596 and the rest 1,584 thousand tickets, 9,854 in
    all (Danish 37.28%, American 46.64%); 2024: 2,230, 5,770 and 1,500, 9,500 in all.
    """
    leaves = {
        2025: {"Denmark": (2000, 1500, 150, 24), "USA": (2400, 2100, 80, 16),
               "Rest of the World": (900, 600, 70, 14)},
        2024: {"Denmark": (1200, 900, 120, 10), "USA": (3200, 2500, 60, 10),
               "Rest of the World": (1000, 400, 90, 10)},
    }
    rows = []
    for year, nations in leaves.items():
        totals = [sum(counts[index] for counts in nations.values()) for index in range(4)]
        for nation, (feature_adult, feature_young, documentary_adult, documentary_young) in {
                **nations, "All countries": tuple(totals)}.items():
            types = {
                "Feature films and documentaries, all audiences":
                    feature_adult + feature_young + documentary_adult + documentary_young,
                "Feature films, all audiences": feature_adult + feature_young,
                "Feature films, adult audience": feature_adult,
                "Feature films, children / youth / family audience": feature_young,
                "Documentary films, all audiences": documentary_adult + documentary_young,
                "Documentary films, adult audience": documentary_adult,
                "Documentary films, children / youth / family audience": documentary_young,
            }
            for kind, count in types.items():
                for category, share in (("Screened films (all)", 1), ("Premiere films that year", 3)):
                    rows.append({"BIO": "Tickets sold (1,000)", "FILMKAT": category, "TYPMAL": kind,
                                 "NATION2": nation, "TID": year, "INDHOLD": count // share})
    return pd.DataFrame(rows)
