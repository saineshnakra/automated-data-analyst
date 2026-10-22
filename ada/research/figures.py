"""Publication figures: journal presets, style, export, source data and captions.

The figure types themselves are in ``plots``. This module sets the size, fonts
and resolution from the journal preset and ``output`` in ``study.yml``, saves
each figure in every requested format with editable text, writes the data
behind it as CSV, and records its caption and provenance.

Text stays editable: PDF and EPS embed TrueType fonts (type 42) and SVG keeps
text as text. File metadata is fixed so a rebuild gives the same files. TIFF is
written as journals ask for it (RGB, no alpha channel, LZW), and because EPS
cannot hold transparency, a study that asks for EPS gets opaque tints in place
of see-through colours in every format.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from io import BytesIO
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
    #: The smallest text the journal allows; tick labels and legends never go below it.
    min_font_size: float = 6.0
    #: Panel labels: "upper" (A, B, C) or "lower" (a, b, c), in bold at this size (None: font size + 1).
    panel_labels: str = "upper"
    panel_label_size: float | None = None


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
    # PLOS gives a minimum and maximum width and the width of its text column, not column widths:
    # "single" is the text column and "double" the maximum. TIFF needs LZW, RGB and no alpha channel.
    "plos": Journal(
        name="plos",
        widths_mm={"single": 132.0, "double": 190.5, "minimum": 66.8},
        max_height_mm=222.3,
        dpi=300,
        fonts=("Arial", "Times", "Symbol"),
        font_size=8.0,
        min_font_size=8.0,
        formats=("tiff", "eps"),
        guide="https://journals.plos.org/plosone/s/figures",
    ),
    # Nature: 5 to 7 pt text, panel labels 8 pt bold lower case, vector files preferred. Its pages
    # give 300 dpi for photographs and ask for 450 dpi in the export checklist; the higher is used.
    "nature": Journal(
        name="nature",
        widths_mm={"single": 89.0, "one-and-a-half": 120.0, "one-and-a-half-wide": 136.0, "double": 183.0},
        max_height_mm=170.0,
        dpi=450,
        fonts=("Arial", "Helvetica"),
        font_size=7.0,
        min_font_size=5.0,
        formats=("pdf", "eps"),
        guide="https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/",
        panel_labels="lower",
        panel_label_size=8.0,
    ),
    # Cell Press, two-column journals: 6 to 8 pt Arial, panel labels in capitals. The guide only asks
    # that a figure fit on one 8.5 × 11 inch page, so the height is a conservative 225 mm. It asks
    # 300 dpi for colour images, 500 dpi for images combined with text and 1000 dpi for black-and-white
    # line art; graphs with coloured points are combination art.
    "cell": Journal(
        name="cell",
        widths_mm={"single": 85.0, "one-and-a-half": 114.0, "double": 174.0},
        max_height_mm=225.0,
        dpi=500,
        fonts=("Arial", "Helvetica"),
        font_size=8.0,
        min_font_size=6.0,
        formats=("pdf", "tiff"),
        guide="https://www.cell.com/information-for-authors/figure-guidelines",
    ),
    # The Science family: 8 pt Myriad or another sans serif (6 pt at least), panel labels 9 pt bold
    # capitals. Widths between one and two columns stop at 14.4 cm, and 19.9 cm is the deepest figure
    # that leaves room for the legend. Raster files need 300 dpi (1200 for black-and-white line art)
    # and are reduced to 600 dpi above that, so 600 is used.
    "science": Journal(
        name="science",
        widths_mm={"single": 90.0, "intermediate": 144.0, "double": 184.0},
        max_height_mm=199.0,
        dpi=600,
        fonts=("Myriad Pro", "Helvetica", "Arial"),
        font_size=8.0,
        min_font_size=6.0,
        formats=("pdf", "tiff"),
        guide="https://www.science.org/cms/asset/76887697-528f-4a22-afea-cee904504e76/"
        "sciadv_guide_to_preparing_figures_2026.pdf",
        panel_label_size=9.0,
    ),
    # eLife asks for at least 300 dpi, a width of at least 10 cm (20 cm for full page width) and a
    # figure that fits one page with its legend. It gives no font, font size or height, so these
    # are conservative: Arial at 8 pt with nothing below 7 pt, and 200 mm high.
    "elife": Journal(
        name="elife",
        widths_mm={"single": 100.0, "double": 200.0},
        max_height_mm=200.0,
        dpi=300,
        fonts=("Arial", "Helvetica"),
        font_size=8.0,
        min_font_size=7.0,
        formats=("pdf", "tiff"),
        guide="https://elife-rp.msubmit.net/html/elife-rp_author_instructions.html",
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
    small = small_font_size(style)
    settings = {
        "font.family": "sans-serif",
        "font.sans-serif": list(style.fonts),
        "font.size": size,
        "axes.titlesize": size,
        "axes.labelsize": size,
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "legend.fontsize": small,
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


def opaque(style: Style) -> bool:
    """EPS has no transparency, so partly transparent marks are drawn as the opaque tint they would show."""
    return "eps" in style.formats


def tint(colour: str, alpha: float, style: Style) -> tuple[float, float, float, float]:
    """``colour`` at ``alpha`` over white: see-through, or the same tint drawn opaque for EPS."""
    from matplotlib.colors import to_rgb  # noqa: PLC0415

    red, green, blue = to_rgb(colour)
    if opaque(style):
        return (*(alpha * value + 1 - alpha for value in (red, green, blue)), 1.0)
    return red, green, blue, alpha


def small_font_size(style: Style) -> float:
    """Tick labels and legends: a point smaller than the text, but never below the journal's minimum."""
    return min(style.font_size, max(style.font_size - 1, style.journal.min_font_size))


def panel_label(style: Style, index: int) -> tuple[str, float]:
    """The label of the ``index``-th panel (A, B, … or a, b, …) and its size in points."""
    letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[index % 26]
    letter = letter.lower() if style.journal.panel_labels == "lower" else letter
    return letter, style.journal.panel_label_size or style.font_size + 1


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


def _save_tiff(figure, path: Path, dpi: int) -> None:
    """TIFF as journals ask for it: 8-bit RGB on white with no alpha channel, LZW-compressed, dpi recorded."""
    from PIL import Image  # noqa: PLC0415

    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, metadata=_metadata("png"))
    buffer.seek(0)
    with Image.open(buffer) as image:
        rgba = image.convert("RGBA")
        flat = Image.new("RGB", rgba.size, "white")
        flat.paste(rgba, mask=rgba.getchannel("A"))
        flat.save(path, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))


@contextmanager
def _fixed_creation_date(active: bool):
    """EPS files carry a creation date; Matplotlib takes it from SOURCE_DATE_EPOCH when that is set."""
    if not active:
        yield
        return
    before = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = before


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
        if kind == "tiff":
            _save_tiff(figure, path, style.dpi)
        else:
            with _fixed_creation_date(kind == "eps"):
                figure.savefig(path, format=kind, dpi=style.dpi, metadata=_metadata(kind))
        files.append(path)
    plt.close(figure)
    if source is not None:
        source_path = folder / f"{ident}.source.csv"
        source.to_csv(source_path, index=False, lineterminator="\n")
        files.append(source_path)
    heading = title if title.endswith((".", "?", "!")) else f"{title}."
    full_caption = f"{number}. {heading}" + (f" {caption}" if caption else "")
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
