"""Publication figures: journal presets, style, export, source data and captions.

The figure types themselves are in ``plots``. This module sets the size, fonts
and resolution from the journal preset and ``output`` in ``study.yml``, saves
each figure in every requested format with editable text, writes the data
behind it as CSV, and records its caption and provenance.

Text stays editable: PDF and EPS embed TrueType fonts (type 42) and SVG keeps
text as text. File metadata is fixed so a rebuild gives the same files.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .provenance import Asset

MM_PER_INCH = 25.4

#: Colour-blind-safe categorical colours (Okabe and Ito, 2008), in a fixed order
#: that keeps neighbouring colours apart under the common colour-vision deficiencies.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9")
#: Ink for estimates, error bars, text and axes.
INK = "#1a1a1a"
MUTED = "#6b6b6b"
#: Fallbacks when the preset's font is not installed; Liberation Sans has Arial's metrics.
FALLBACK_FONTS = ("Liberation Sans", "DejaVu Sans")


@dataclass(frozen=True)
class Journal:
    name: str
    #: Named widths in millimetres; "single" and "double" are always present.
    widths_mm: dict[str, float]
    max_height_mm: float
    dpi: int
    fonts: tuple[str, ...]
    font_size: float
    #: Formats the journal asks for, recommended when ``output.figures`` is not set.
    formats: tuple[str, ...]
    guide: str = ""


JOURNALS: dict[str, Journal] = {
    "default": Journal(
        name="default",
        widths_mm={"single": 85.0, "double": 170.0},
        max_height_mm=225.0,
        dpi=300,
        fonts=("Arial", "Helvetica"),
        font_size=8.0,
        formats=("pdf", "svg", "png"),
    ),
}


@dataclass(frozen=True)
class Style:
    """The resolved figure settings for one study."""

    journal: Journal
    width_mm: float
    max_height_mm: float
    dpi: int
    fonts: tuple[str, ...]
    font_size: float
    formats: tuple[str, ...]
    notes: tuple[str, ...] = field(default=())


def resolve_style(study) -> Style:
    """Combine the journal preset with any size, font or resolution set under ``output``."""
    output = study.output
    journal = JOURNALS.get(output.journal.lower())
    notes = []
    if journal is None:
        notes.append(
            f"Unknown journal preset {output.journal!r}; used the default. Presets: "
            + ", ".join(sorted(JOURNALS))
            + "."
        )
        journal = JOURNALS["default"]
    fonts = tuple(dict.fromkeys(((output.font,) if output.font else ()) + journal.fonts + FALLBACK_FONTS))
    installed = _installed_fonts()
    chosen = next((font for font in fonts if font in installed), "DejaVu Sans")
    wanted = output.font or journal.fonts[0]
    if chosen != wanted:
        notes.append(
            f"{wanted} is not installed here, so figures use {chosen}. SVG files keep text as text and "
            f"show it in {wanted} wherever it is installed."
        )
    return Style(
        journal=journal,
        width_mm=float(output.width_mm or journal.widths_mm["single"]),
        max_height_mm=float(output.height_mm or journal.max_height_mm),
        dpi=int(output.dpi or journal.dpi),
        fonts=(chosen, *[font for font in fonts if font != chosen]),
        font_size=float(output.font_size or journal.font_size),
        formats=output.figures,
        notes=tuple(notes),
    )


def _installed_fonts() -> set[str]:
    from matplotlib import font_manager  # noqa: PLC0415

    return {entry.name for entry in font_manager.fontManager.ttflist}


@contextmanager
def styled(style: Style):
    """Matplotlib settings for publication figures, active inside the ``with`` block."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    size = style.font_size
    settings = {
        "font.family": "sans-serif",
        "font.sans-serif": list(style.fonts),
        "font.size": size,
        "axes.titlesize": size,
        "axes.labelsize": size,
        "xtick.labelsize": size - 1,
        "ytick.labelsize": size - 1,
        "legend.fontsize": size - 1,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=list(PALETTE)),
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": style.dpi,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "ada",
        "figure.constrained_layout.use": True,
    }
    with plt.rc_context(settings):
        yield plt


def figure_size(style: Style, width: str | float = "single", aspect: float = 0.75) -> tuple[float, float]:
    """Width and height in inches: a named journal width (or millimetres) and height = width × aspect."""
    width_mm = style.journal.widths_mm.get(width, style.width_mm) if isinstance(width, str) else float(width)
    if width == "single":
        width_mm = style.width_mm
    height_mm = min(width_mm * aspect, style.max_height_mm)
    return width_mm / MM_PER_INCH, height_mm / MM_PER_INCH


def _metadata(kind: str) -> dict | None:
    """Fixed metadata, so the same figure gives the same file at every build."""
    return {
        "pdf": {"CreationDate": None, "ModDate": None, "Creator": "ADA", "Producer": "matplotlib"},
        "svg": {"Date": None, "Creator": "ADA"},
        "png": {"Software": "ADA"},
        "eps": {"Creator": "ADA"},
    }.get(kind)


def save(
    result,
    figure,
    *,
    ident: str,
    title: str,
    caption: str,
    source: pd.DataFrame | None,
    depends: tuple[str, ...],
    code: str,
    calculation: str = "",
    section: str = "main",
    references: tuple[str, ...] = (),
    rows: dict | None = None,
    numbers: dict | None = None,
) -> Asset:
    """Save ``figure`` in each format, write its source data and record it as an asset."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    style = getattr(result, "style", None) or resolve_style(result.study)
    folder = result.folder("figures")
    existing = sum(1 for asset in result.assets if asset.kind == "figure" and asset.section == section)
    number = f"Figure {existing + 1}" if section == "main" else f"Figure S{existing + 1}"
    files: list[Path] = []
    for kind in style.formats:
        path = folder / f"{ident}.{kind}"
        options = {"format": kind, "dpi": style.dpi, "metadata": _metadata(kind)}
        if kind == "tiff":
            options["pil_kwargs"] = {"compression": "tiff_lzw"}
        figure.savefig(path, **options)
        files.append(path)
    plt.close(figure)
    if source is not None:
        source_path = folder / f"{ident}.source.csv"
        source.to_csv(source_path, index=False, lineterminator="\n")
        files.append(source_path)
    full_caption = f"{number}. {title}" + (f" {caption}" if caption else "")
    asset = Asset(
        id=ident,
        kind="figure",
        title=title,
        files=files,
        depends=depends,
        calculation=calculation,
        references=references,
        code=code,
        rows=rows or result.rows(),
        numbers=numbers or {},
        number=number,
        caption=full_caption,
        section=section,
        row_level=source is not None,
    )
    result.assets.append(asset)
    return asset


def write_captions(result) -> Path | None:
    """All figure captions in one file, in figure order, ready to paste into a manuscript."""
    figures = [asset for asset in result.assets if asset.kind == "figure"]
    if not figures:
        return None
    path = result.folder("figures") / "captions.md"
    lines = ["# Figure captions", ""]
    for asset in figures:
        lines += [asset.caption, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(result) -> None:
    """Draw each figure in ``study.yml`` (or the default set) and write the captions file."""
    from . import plots  # noqa: PLC0415

    result.style = resolve_style(result.study)
    result.warnings.extend(result.style.notes)
    with styled(result.style):
        plots.draw_all(result)
