import pandas as pd


def summarize_trend(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> str:
    data = frame[[x_col, y_col]].dropna()

    if data.empty:
        return "No data is available for this trend chart."

    first_value = float(data[y_col].iloc[0])
    last_value = float(data[y_col].iloc[-1])

    if first_value == 0:
        change_text = "with no percentage change available"
    else:
        change = ((last_value - first_value) / abs(first_value)) * 100
        change_text = f"({change:+.1f}% from the first value)"

    if last_value > first_value:
        direction = "increased"
    elif last_value < first_value:
        direction = "decreased"
    else:
        direction = "was unchanged"

    return (
        f"{y_col} {direction} from {first_value:,.2f} "
        f"to {last_value:,.2f} {change_text}."
    )


def summarize_categories(
    frame: pd.DataFrame,
    category_col: str,
    value_col: str,
) -> str:
    data = frame[[category_col, value_col]].dropna()

    if data.empty:
        return "No data is available for this category chart."

    grouped = (
        data.groupby(category_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
    )

    if grouped.empty:
        return "No category values are available."

    highest = grouped.iloc[0]
    lowest = grouped.iloc[-1]

    return (
        f"{category_col} is highest for "
        f"{highest[category_col]} ({highest[value_col]:,.2f}) "
        f"and lowest for {lowest[category_col]} "
        f"({lowest[value_col]:,.2f})."
    )


def summarize_distribution(
    frame: pd.DataFrame,
    value_col: str,
) -> str:
    data = pd.to_numeric(frame[value_col], errors="coerce").dropna()

    if data.empty:
        return "No numeric data is available for this distribution chart."

    return (
        f"{value_col} has {len(data):,} values, "
        f"with a median of {data.median():,.2f}, "
        f"ranging from {data.min():,.2f} to {data.max():,.2f}."
    )


def summarize_relationship(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> str:
    data = frame[[x_col, y_col]].copy()
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna()

    if len(data) < 2:
        return "Not enough numeric data is available to summarize this relationship."

    correlation = data[x_col].corr(data[y_col])

    if pd.isna(correlation):
        return f"No measurable relationship is available between {x_col} and {y_col}."

    if correlation > 0.3:
        direction = "positive"
    elif correlation < -0.3:
        direction = "negative"
    else:
        direction = "weak"

    return (
        f"The relationship between {x_col} and {y_col} is "
        f"{direction}, with a correlation of {correlation:.2f}."
    )


def summarize_heatmap(
    frame: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
) -> str:
    data = frame[[row_col, col_col, value_col]].dropna()

    if data.empty:
        return "No data is available for this heatmap."

    highest = data.loc[data[value_col].idxmax()]
    lowest = data.loc[data[value_col].idxmin()]

    return (
        f"The highest value is {highest[value_col]:,.2f} "
        f"for {row_col}={highest[row_col]} and {col_col}={highest[col_col]}. "
        f"The lowest value is {lowest[value_col]:,.2f} "
        f"for {row_col}={lowest[row_col]} and {col_col}={lowest[col_col]}."
    )


def summarize_movement(
    frame: pd.DataFrame,
    category_col: str,
    value_col: str,
) -> str:
    data = frame[[category_col, value_col]].dropna()

    if data.empty:
        return "No movement data is available."

    positive = data[data[value_col] > 0]
    negative = data[data[value_col] < 0]

    parts = []

    if not positive.empty:
        item = positive.loc[positive[value_col].idxmax()]
        parts.append(
            f"largest increase was {item[category_col]} "
            f"({item[value_col]:+,.2f})"
        )

    if not negative.empty:
        item = negative.loc[negative[value_col].idxmin()]
        parts.append(
            f"largest decrease was {item[category_col]} "
            f"({item[value_col]:+,.2f})"
        )

    net_change = data[value_col].sum()
    parts.append(f"net change was {net_change:+,.2f}")

    return "The " + ", ".join(parts) + "."
