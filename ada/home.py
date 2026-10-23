"""The web app's pages for researchers: the home page, and the example study built on the server.

The home page says what ADA makes from a study and how. The example page writes the
example study (``ada demo``), runs ``ada build`` and ``ada verify`` on it, and shows
every asset they made, so a visitor sees the research side without installing anything.
"""

from __future__ import annotations

import io
import tempfile
import time
import zipfile
from dataclasses import dataclass
from html import escape
from pathlib import Path

import streamlit as st

from . import __version__
from .ui import render_footer, render_section_heading

REPOSITORY = "https://github.com/saineshnakra/automated-data-analyst"
DOCS = f"{REPOSITORY}/blob/main/docs/research.md"
RESULTS = f"{REPOSITORY}/blob/main/results/README.md"
INSTALL = "pip install automated-data-analyst"
#: The example study's figure, shipped with the repository for the README.
FIGURE = Path(__file__).resolve().parent.parent / "assets" / "readme" / "study-figure.png"

STEPS = (
    (
        "01 · DESCRIBE",
        "ada init data.csv",
        "Say what the study is",
        "ADA reads the file, guesses each column's role and unit, and asks what a row is, which column "
        "identifies a subject or site, the design, and which analyses were planned. The answers go into "
        "study.yml, a plain file you can read and edit.",
    ),
    (
        "02 · BUILD",
        "ada build",
        "Get the paper's assets",
        "The data audit, data dictionary, processing record, tables, figures, methods draft and a "
        "reproduction package, written to paper-assets/.",
    ),
    (
        "03 · VERIFY",
        "ada verify",
        "Check it rebuilds",
        "ADA rebuilds everything from study.yml and the data alone in a scratch folder and compares "
        "every number. It exits with 1 if anything differs, so it can run in CI.",
    ),
)

ASSETS = (
    ("audit/", "Data audit",
     "What was read, values outside their range, duplicates, and whether the data fit the stated design."),
    ("dictionary/", "Data dictionary",
     "Each column's meaning, unit and type, with its allowed and observed values."),
    ("processing/", "Processing record",
     "Every step from the original file to the analysed rows, with rows before and after, and the "
     "excluded rows themselves."),
    ("tables/", "Tables",
     "Descriptive tables and the results of the planned analyses: estimates, intervals, test statistics "
     "and adjusted p-values, in Word, LaTeX and CSV."),
    ("figures/", "Figures",
     "Every observation drawn, sized for PLOS, Nature, Cell, Science or eLife, each with its source data "
     "and a caption."),
    ("methods/", "Methods draft",
     "Written from what was run: the design, each exclusion and its reason, each test, the software "
     "versions and references. A results draft too."),
    ("supplement/", "Supplement",
     "Exploratory analyses, each model's output and checks, collected in one Word file."),
    ("reproduce/", "Reproduction package",
     "study.yml, pinned package versions, input checksums and the steps another lab follows to rebuild."),
    ("repository/", "Repository files",
     "README, CITATION.cff, Zenodo metadata, and data and code availability drafts."),
)

PRINCIPLES = (
    "Only the analyses written in study.yml are run, and each is labelled planned or exploratory. "
    "ADA never searches for a significant result.",
    "p-values are adjusted within the families you name, by Holm's method unless you choose another.",
    "No value is dropped on its own. An exclusion is a rule in study.yml with its reason, and the rows "
    "it removed are saved.",
    "A design ADA can't fit, such as a crossover trial or nested random effects, is refused with the "
    "reason, not approximated.",
    "Your own pandas tables and Matplotlib figures join the assets through a custom script.",
)

DESIGNS = (
    ("Independent", "Welch's t-test or ANOVA", "Linear regression"),
    ("Paired", "Paired t-test", "Mixed model"),
    ("Repeated measures", "Mixed model, random intercept per subject", "Mixed model"),
    ("Clustered", "Mixed model, random intercept per cluster", "Mixed model"),
)

# From the example study: drop the exclusion of mouse M07 and build again. Shortened to fit.
CHANGES_EXAMPLE = """## Build 2

Since the previous build:
- study.yml changed in processing.
- Step 3 was removed: Exclude rows where
  Mouse is one of M07.
- Rows used: 134 → 143.

Assets that changed:
- results changed (Table 3):
  rows used 133 → 142;
  weight-gain.estimate 0.5598 → 0.589;
  weight-gain.ci low 0.518 → 0.5449; …
- weight-curves changed (Figure 1):
  n 133 → 142; …"""

VERIFY_EXAMPLE = """$ ada verify
Inputs
  ok    data/mouse-weights.csv matches
        study.yml and the build.
Out of date
  ok    All 17 assets were built from the
        current study.yml.
Edited since the build
  ok    None of the 50 files recorded in
        provenance.json has changed.
Rebuild in a scratch folder
  ok    Rebuilt from study.yml and the data
        alone, and every file matches: 12 CSV
        files number by number, …

Everything matches: 17 assets and their 50 files."""


def _cards(rows, card_class: str) -> str:
    return "".join(
        f'<article class="{card_class}"><div class="asset-folder">{escape(folder)}</div>'
        f"<h3>{escape(title)}</h3><p>{escape(text)}</p></article>"
        for folder, title, text in rows
    )


def render_home(example_page, first_look_page, *, build: str = "") -> None:
    """The home page: what ADA makes from a study, how, and where to try it."""
    st.markdown(
        """
        <section class="hero">
          <div class="hero-copy">
            <div class="eyebrow">For researchers</div>
            <h1>Your paper's tables and figures,<br><span class="grad">from your data, traceable to it.</span></h1>
            <p>Describe your study once in a plain file. ADA writes the data audit, the tables in Word, LaTeX and CSV, the figures with their source data and captions, and a methods draft, then checks that a rebuild gives the same numbers. It is open source and runs on your own computer, so the data stays with you.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    actions = st.columns([0.27, 0.2, 0.53], vertical_alignment="center")
    if actions[0].button("See an example study built here", type="primary", width="stretch", key="home_example"):
        st.switch_page(example_page)
    if actions[1].button("Look at a table now", width="stretch", key="home_first_look"):
        st.switch_page(first_look_page)
    actions[2].markdown(
        f'<div class="install-chip"><span class="prompt">$</span>{INSTALL}</div>', unsafe_allow_html=True
    )

    render_section_heading(
        "How it works",
        "Three commands, in the folder with your data",
        "ADA asks what only you know about the study, then makes everything else from the data.",
    )
    st.markdown(
        '<section class="step-grid">'
        + "".join(
            f'<article class="step-card"><div class="how-number">{escape(number)}</div>'
            f'<code class="step-command">{escape(command)}</code><h3>{escape(title)}</h3>'
            f"<p>{escape(text)}</p></article>"
            for number, command, title, text in STEPS
        )
        + "</section>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        "What ada build makes",
        "Everything the paper needs, in one folder",
        "Each asset records where its numbers came from. Rebuilding an unchanged study gives the same "
        "bytes, so git diff stays quiet.",
    )
    st.markdown(f'<section class="asset-grid">{_cards(ASSETS, "asset-card")}</section>', unsafe_allow_html=True)

    figure_column, text_column = st.columns([0.56, 0.44], gap="large", vertical_alignment="center")
    if FIGURE.is_file():
        figure_column.image(str(FIGURE), width="stretch")
    text_column.markdown(
        """
        <div class="figure-note">
          <div class="section-kicker">A figure from the example study</div>
          <h3>Figure 1. Body weight over week by Diet</h3>
          <p>Thin lines join the measurements of each mouse; 15 mice (7 Chow, 8 High fat). Markers and bars show the mean and 95% confidence interval at each week for each diet. Analysis weight-gain (linear mixed model over Week, random intercept per subject) was planned before the data were seen.</p>
          <ul>
            <li>Every animal is drawn, not only the means.</li>
            <li>The caption is written from what was drawn: the n, the rows left out, the test behind it.</li>
            <li>The plotted numbers are saved beside it in weight-curves.source.csv.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_heading(
        "Statistics",
        "The design picks the test, not the data",
        "You state the design and the planned analyses before the build, and ADA fits what the design calls for.",
    )
    principles_column, table_column = st.columns([0.5, 0.5], gap="large")
    principles_column.markdown(
        '<ul class="principles">' + "".join(f"<li>{escape(line)}</li>" for line in PRINCIPLES) + "</ul>",
        unsafe_allow_html=True,
    )
    table_column.markdown(
        '<table class="design-table"><thead><tr><th>Design</th><th>Comparing groups</th>'
        "<th>A numeric predictor</th></tr></thead><tbody>"
        + "".join(
            f"<tr><td>{escape(design)}</td><td>{escape(groups)}</td><td>{escape(numeric)}</td></tr>"
            for design, groups, numeric in DESIGNS
        )
        + "</tbody></table>"
        '<p class="table-note">Binary outcomes get the risk difference, the odds ratio, and a χ² or '
        "Fisher's exact test, or logistic regression. Change over time is a mixed model with time × group.</p>",
        unsafe_allow_html=True,
    )

    render_section_heading(
        "Traceability",
        "Every number can be traced, and checked",
        "provenance.json records, for each asset, the input file's checksum, the rows used and the exclusion "
        "steps behind them, the code that made it and every package version.",
    )
    changes_column, verify_column = st.columns(2, gap="large")
    with changes_column:
        st.markdown('<div class="section-label">CHANGES.md after you change a rule</div>', unsafe_allow_html=True)
        st.code(CHANGES_EXAMPLE, language="markdown")
    with verify_column:
        st.markdown('<div class="section-label">ada verify</div>', unsafe_allow_html=True)
        st.code(VERIFY_EXAMPLE, language="text")

    st.markdown(
        f"""
        <section class="info-grid">
          <article class="info-card"><h3>Restricted data</h3><p><code>ada preview</code> lists what would be shared and flags withheld columns or individual rows in any file; <code>ada share</code> copies only what may leave. With demo data, others run the reproduction on a synthetic table with the same columns.</p></article>
          <article class="info-card"><h3>With an AI assistant</h3><p><code>ada skill</code> installs a skill that tells an AI coding assistant to ask you for the design and the planned analyses, write study.yml, build and verify, and pass on a refusal rather than work around it.</p></article>
          <article class="info-card"><h3>Tested on real data</h3><p>The first look was run on 60 research datasets, from blood tests to soil chemistry, and every finding was checked against SciPy and statsmodels. <a href="{RESULTS}" target="_blank">The results ↗</a></p></article>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_section_heading("Try it", "See it work before you install it", "")
    example_column, look_column, install_column = st.columns(3, gap="medium")
    with example_column:
        st.markdown(
            '<div class="try-card"><h3>The example study</h3><p>Sixteen mice on two diets, weighed weekly. '
            "ADA builds its 17 assets on this server in a few seconds: tables, figures, methods draft, "
            "and a zip to download.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Open the example study", type="primary", width="stretch", key="try_example"):
            st.switch_page(example_page)
    with look_column:
        st.markdown(
            '<div class="try-card"><h3>A first look at any table</h3><p>Drop in a CSV or Excel file of '
            "measurements and see what is wrong with it, which groups differ, what relates to what and "
            "what changed over time. Nothing to install.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Look at a table", width="stretch", key="try_first_look"):
            st.switch_page(first_look_page)
    with install_column:
        st.markdown(
            '<div class="try-card"><h3>On your own study</h3><p>Run these in the folder that holds the '
            f'data. <a href="{DOCS}" target="_blank">The study.yml format and every file written ↗</a></p></div>',
            unsafe_allow_html=True,
        )
        st.code(f"{INSTALL}\nada init data.csv\nada build\nada verify", language="bash")

    render_footer(build=build)


@dataclass
class Example:
    """The example study after ``ada build`` and ``ada verify`` on this server."""

    folder: Path
    built: object
    seconds: float
    verified: str
    ok: bool
    archive: bytes


@st.cache_resource(show_spinner=False)
def build_example(version: str) -> Example:
    """Write the example study to a scratch folder, build and verify it, and zip the result.

    Cached per ADA version, so it runs once per server process.
    """
    from .research import build  # noqa: PLC0415 - Matplotlib and python-docx load only when asked for
    from .research.example import write_example  # noqa: PLC0415
    from .research.verify import verify  # noqa: PLC0415

    root = Path(tempfile.mkdtemp(prefix="ada-example-"))
    folder = root / "ada-demo"
    write_example(folder)
    started = time.perf_counter()
    built = build(folder / "study.yml")
    seconds = time.perf_counter() - started
    report = verify(folder / "study.yml")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(root).as_posix())
    verified = report.text().replace(str(folder), "ada-demo")
    return Example(folder, built, seconds, verified, report.ok, archive.getvalue())


def _read(folder: Path, name: str) -> str:
    path = folder / "paper-assets" / name
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def render_example_study(first_look_page, *, build: str = "") -> None:
    """The example study, built on this server, with every asset it made."""
    st.set_page_config(page_title="ADA | Example study")
    st.markdown(
        """
        <section class="hero compact">
          <div class="hero-copy">
            <div class="eyebrow">Example study</div>
            <h1>Sixteen mice, two diets, weighed weekly.</h1>
            <p>Simulated data, described in a study.yml with authors, units, two exclusions with their reasons, a planned and an exploratory analysis and a custom table. Below is everything <code>ada build</code> made from it on this server, and what <code>ada verify</code> said about the rebuild. <code>ada demo</code> writes the same study on your computer.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    with st.spinner("Building the example study…"):
        example = build_example(__version__)
    built = example.built
    analyses = list(built.results.values())
    fitted = sum(1 for found in analyses if found.supported)
    summary = st.columns(4)
    summary[0].metric("Assets built", f"{len(built.assets)}")
    summary[1].metric("Rows analysed", f"{len(built.data):,} of {len(built.processed.original):,}")
    summary[2].metric("Analyses fitted", f"{fitted} of {len(analyses)}")
    summary[3].metric("Build time", f"{example.seconds:.1f} s")
    actions = st.columns([0.3, 0.7], vertical_alignment="center")
    actions[0].download_button(
        "Download the study and its assets (.zip)",
        data=example.archive,
        file_name="ada-example-study.zip",
        mime="application/zip",
        width="stretch",
    )
    actions[1].markdown(
        f'<div class="install-chip"><span class="prompt">$</span>{INSTALL} &amp;&amp; ada demo</div>',
        unsafe_allow_html=True,
    )

    figures, tables, methods, audit, processing, study, verify = st.tabs(
        ["Figures", "Tables", "Methods draft", "Data audit", "Processing record", "study.yml", "Verify"]
    )
    with figures:
        for asset in built.assets:
            if asset.kind != "figure":
                continue
            png = next((path for path in asset.files if path.suffix == ".png"), None)
            if png is not None and png.is_file():
                st.image(str(png), width=720)
            st.caption(asset.caption)
            st.caption(f"Files: {', '.join(path.name for path in asset.files)}")
    with tables:
        for table in built.tables:
            if not table.number.startswith("Table"):
                continue
            st.markdown(f"**{table.caption}**")
            st.dataframe(table.display, hide_index=True, width="stretch")
            for note in table.notes:
                st.caption(note)
    with methods:
        st.markdown(_read(example.folder, "methods/methods.md"))
        st.divider()
        st.markdown(_read(example.folder, "methods/results.md"))
    with audit:
        st.markdown(_read(example.folder, "audit/data-audit.md"))
    with processing:
        st.markdown(_read(example.folder, "processing/processing-record.md"))
    with study:
        st.code((example.folder / "study.yml").read_text(encoding="utf-8"), language="yaml")
        for script in sorted((example.folder / "analysis").glob("*.py")):
            st.markdown(f"**{script.relative_to(example.folder).as_posix()}**, listed under `custom:`")
            st.code(script.read_text(encoding="utf-8"), language="python")
    with verify:
        (st.success if example.ok else st.error)(
            "Every file matches a rebuild." if example.ok else "The rebuild did not match."
        )
        st.code(example.verified, language="text", wrap_lines=True)

    render_section_heading(
        "Your own study",
        "Run the same three commands on your data",
        "ADA runs on your computer, so the data stays with you.",
    )
    own, look = st.columns([0.55, 0.45], gap="large")
    own.code(f"{INSTALL}\nada init data.csv     # asks about the study; writes study.yml\nada build\nada verify",
             language="bash")
    with look:
        st.markdown(
            '<div class="try-card"><h3>Or look at a table first</h3><p>See what is in a file before you '
            "describe the study: data checks, group differences, relationships and trends, in the "
            "browser.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Look at a table", width="stretch", key="example_first_look"):
            st.switch_page(first_look_page)
    render_footer(build=build)
