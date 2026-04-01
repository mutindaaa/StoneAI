"""
generate_report.py — Combines analytics PNG outputs into a one-page A4 landscape PDF match report.

Usage:
    python scripts/generate_report.py --match_id 3788741 --title "Turkey vs Italy — Euro 2020"
"""

import argparse
import csv
import logging
import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm, mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

# ── Layout constants (all in points, 1pt = 1/72 inch) ─────────────────────────
PAGE_W, PAGE_H = landscape(A4)           # ~841 × 595 pt
MARGIN        = 1.0 * cm
HEADER_H      = 1.6 * cm
FOOTER_H      = 1.0 * cm
ROW_GAP       = 0.4 * cm

LOGO_FONT     = "Helvetica-Bold"
LOGO_SIZE     = 16
TITLE_FONT    = "Helvetica-Bold"
TITLE_SIZE    = 13
META_FONT     = "Helvetica"
META_SIZE     = 9
FOOTER_FONT   = "Helvetica"
FOOTER_SIZE   = 8

HEADER_BG     = colors.HexColor("#1a1a2e")   # dark navy
FOOTER_BG     = colors.HexColor("#16213e")
HEADER_FG     = colors.white
ACCENT        = colors.HexColor("#e94560")    # red accent

ROW1_SPLIT    = 0.60    # left column gets 60 % of usable width in row 1
ROW2_SPLIT    = 0.50    # 50/50 in row 2


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_image(canvas: Canvas, path: Path, x: float, y: float, w: float, h: float) -> bool:
    """Draw an image scaled to fit (w × h) box, centred. Returns False if missing."""
    if not path.exists():
        log.warning("Missing image, skipping: %s", path)
        return False

    img = ImageReader(str(path))
    img_w, img_h = img.getSize()
    scale = min(w / img_w, h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale
    # Centre inside the allocated box
    offset_x = (w - draw_w) / 2
    offset_y = (h - draw_h) / 2
    canvas.drawImage(img, x + offset_x, y + offset_y, width=draw_w, height=draw_h,
                     preserveAspectRatio=True, mask="auto")
    return True


def _read_stats(csv_path: Path) -> dict:
    """
    Parse xt_top_players.csv and return a dict with:
        top_player, top_xt, shots, goals, teams (list[str])
    """
    stats = {"top_player": "N/A", "top_xt": 0.0, "shots": "N/A", "goals": "N/A", "teams": []}
    if not csv_path.exists():
        log.warning("Stats CSV not found: %s", csv_path)
        return stats

    rows: list[dict] = []
    teams: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
            teams.add(row.get("team_id", ""))

    if rows:
        top = rows[0]   # already sorted by total_xt descending
        stats["top_player"] = top.get("player_id", "N/A")
        try:
            stats["top_xt"] = float(top.get("total_xt", 0))
        except ValueError:
            pass
    stats["teams"] = sorted(teams)
    return stats


def _detect_team_images(match_dir: Path, teams: list[str]) -> tuple[Optional[Path], Optional[Path]]:
    """
    Return (home_pass_network, away_pass_network) paths based on teams found in CSV,
    falling back to alphabetical order when team list is incomplete.
    """
    pngs = sorted(match_dir.glob("pass_network_*.png"))
    if len(pngs) == 0:
        return None, None
    if len(pngs) == 1:
        return pngs[0], None

    # Use team ordering from CSV if we have exactly two teams
    if len(teams) == 2:
        home, away = teams[0], teams[1]
        home_path = match_dir / f"pass_network_{home}.png"
        away_path = match_dir / f"pass_network_{away}.png"
        return home_path, away_path

    return pngs[0], pngs[1]


# ── Drawing sub-routines ───────────────────────────────────────────────────────

def _draw_header(c: Canvas, title: str, generated_date: str) -> None:
    """Full-width header bar with logo, title, and date."""
    x0 = MARGIN
    y0 = PAGE_H - MARGIN - HEADER_H
    bar_w = PAGE_W - 2 * MARGIN

    # Background
    c.setFillColor(HEADER_BG)
    c.rect(x0, y0, bar_w, HEADER_H, fill=1, stroke=0)

    # Accent left stripe
    c.setFillColor(ACCENT)
    c.rect(x0, y0, 4, HEADER_H, fill=1, stroke=0)

    c.setFillColor(HEADER_FG)

    # Logo text
    c.setFont(LOGO_FONT, LOGO_SIZE)
    c.drawString(x0 + 12, y0 + (HEADER_H - LOGO_SIZE) / 2 + 2, "STONE AI")

    # Title (centred)
    c.setFont(TITLE_FONT, TITLE_SIZE)
    title_w = c.stringWidth(title, TITLE_FONT, TITLE_SIZE)
    c.drawString(x0 + bar_w / 2 - title_w / 2, y0 + (HEADER_H - TITLE_SIZE) / 2 + 2, title)

    # Date (right aligned)
    c.setFont(META_FONT, META_SIZE)
    date_label = f"Generated: {generated_date}"
    date_w = c.stringWidth(date_label, META_FONT, META_SIZE)
    c.drawString(x0 + bar_w - date_w - 8, y0 + (HEADER_H - META_SIZE) / 2 + 2, date_label)


def _draw_footer(c: Canvas, stats: dict, shots: str, goals: str) -> None:
    """Full-width footer with key stats."""
    x0 = MARGIN
    y0 = MARGIN
    bar_w = PAGE_W - 2 * MARGIN

    c.setFillColor(FOOTER_BG)
    c.rect(x0, y0, bar_w, FOOTER_H, fill=1, stroke=0)

    top_xt_str = f"{stats['top_xt']:.3f}" if isinstance(stats["top_xt"], float) else stats["top_xt"]
    text = (
        f"Shots: {shots}  |  Goals: {goals}  |  "
        f"Top xT: {stats['top_player']} ({top_xt_str})  |  Generated by Stone AI"
    )
    c.setFillColor(HEADER_FG)
    c.setFont(FOOTER_FONT, FOOTER_SIZE)
    text_w = c.stringWidth(text, FOOTER_FONT, FOOTER_SIZE)
    c.drawString(x0 + bar_w / 2 - text_w / 2, y0 + (FOOTER_H - FOOTER_SIZE) / 2 + 1, text)


def _draw_row_labels(c: Canvas, left_label: str, right_label: str,
                     x_left: float, x_right: float, y: float, w_left: float, w_right: float) -> None:
    """Small grey section labels above image rows."""
    label_h = 0.4 * cm
    c.setFillColor(colors.HexColor("#dddddd"))
    c.rect(x_left, y, w_left, label_h, fill=1, stroke=0)
    c.rect(x_right, y, w_right, label_h, fill=1, stroke=0)

    c.setFillColor(colors.HexColor("#333333"))
    c.setFont(LOGO_FONT, 7)
    c.drawString(x_left + 4, y + 3, left_label.upper())
    c.drawString(x_right + 4, y + 3, right_label.upper())

    return label_h


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_report(
    match_id: str,
    analytics_dir: Path,
    output_path: Path,
    title: str,
    shots: str = "—",
    goals: str = "—",
) -> Path:
    """
    Build a one-page A4 landscape PDF match report.

    Parameters
    ----------
    match_id:       StatsBomb match identifier
    analytics_dir:  Root directory containing per-match subdirectories
    output_path:    Destination PDF path
    title:          Report title shown in header
    shots:          Shots count string for footer (override via CLI if known)
    goals:          Goals count string for footer

    Returns
    -------
    Path to the written PDF file.
    """
    match_dir = analytics_dir / match_id
    csv_path  = match_dir / "xt_top_players.csv"
    stats     = _read_stats(csv_path)

    home_png, away_png = _detect_team_images(match_dir, stats["teams"])
    shot_png  = match_dir / "shot_map.png"
    xt_png    = match_dir / "xt_top_players.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = Canvas(str(output_path), pagesize=landscape(A4))

    generated_date = date.today().isoformat()

    # ── Header ──────────────────────────────────────────────────────────────
    _draw_header(c, title, generated_date)

    # ── Usable vertical space between header and footer ──────────────────────
    content_top    = PAGE_H - MARGIN - HEADER_H - ROW_GAP
    content_bottom = MARGIN + FOOTER_H + ROW_GAP
    content_h      = content_top - content_bottom
    row_h          = (content_h - ROW_GAP) / 2          # two equal rows
    label_h        = 0.4 * cm

    usable_w = PAGE_W - 2 * MARGIN

    # ── Row 1: shot_map (60 %) | xt_top_players (40 %) ──────────────────────
    r1_y       = content_bottom + ROW_GAP + row_h        # top of row 1
    r1_img_y   = r1_y - row_h
    r1_w_left  = usable_w * ROW1_SPLIT - ROW_GAP / 2
    r1_w_right = usable_w * (1 - ROW1_SPLIT) - ROW_GAP / 2
    r1_x_left  = MARGIN
    r1_x_right = MARGIN + r1_w_left + ROW_GAP

    img_h_row1 = row_h - label_h - 2

    _draw_row_labels(c, "Shot Map", "xT — Top Players",
                     r1_x_left, r1_x_right, r1_y - label_h, r1_w_left, r1_w_right)

    _safe_image(c, shot_png,  r1_x_left,  r1_img_y, r1_w_left,  img_h_row1)
    _safe_image(c, xt_png,    r1_x_right, r1_img_y, r1_w_right, img_h_row1)

    # ── Row 2: pass_network home (50 %) | pass_network away (50 %) ──────────
    r2_y       = content_bottom + row_h                  # top of row 2
    r2_img_y   = content_bottom
    r2_w       = usable_w / 2 - ROW_GAP / 2
    r2_x_left  = MARGIN
    r2_x_right = MARGIN + r2_w + ROW_GAP

    img_h_row2 = row_h - label_h - 2

    home_label = home_png.stem.replace("pass_network_", "") + " — Pass Network" if home_png else "Pass Network (Home)"
    away_label = away_png.stem.replace("pass_network_", "") + " — Pass Network" if away_png else "Pass Network (Away)"

    _draw_row_labels(c, home_label, away_label,
                     r2_x_left, r2_x_right, r2_y - label_h, r2_w, r2_w)

    if home_png:
        _safe_image(c, home_png, r2_x_left,  r2_img_y, r2_w, img_h_row2)
    if away_png:
        _safe_image(c, away_png, r2_x_right, r2_img_y, r2_w, img_h_row2)

    # ── Footer ───────────────────────────────────────────────────────────────
    _draw_footer(c, stats, shots, goals)

    c.save()
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a one-page PDF match report from analytics PNGs."
    )
    parser.add_argument("--match_id",      required=True,  help="StatsBomb match ID, e.g. 3788741")
    parser.add_argument("--analytics_dir", default="output_videos/analytics/",
                        help="Root analytics output directory (default: output_videos/analytics/)")
    parser.add_argument("--output",        default=None,
                        help="Destination PDF path (default: output_videos/reports/{match_id}_report.pdf)")
    parser.add_argument("--title",         default="Match Report", help="Report title in header")
    parser.add_argument("--shots",         default="—",  help="Shots count for footer")
    parser.add_argument("--goals",         default="—",  help="Goals count for footer")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    analytics_dir = Path(args.analytics_dir)
    output_path   = Path(args.output) if args.output else \
                    Path("output_videos/reports") / f"{args.match_id}_report.pdf"

    pdf_path = generate_report(
        match_id=args.match_id,
        analytics_dir=analytics_dir,
        output_path=output_path,
        title=args.title,
        shots=args.shots,
        goals=args.goals,
    )

    size_kb = pdf_path.stat().st_size / 1024
    print(f"\nReport written to: {pdf_path}")
    print(f"File size:         {size_kb:.1f} KB")
