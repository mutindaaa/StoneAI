"""
Analytics Configuration Constants

All numeric thresholds, pitch dimensions, visual style parameters, and grid
sizes used by the analytics package are defined here.  Import from this module
rather than embedding inline literals in other files.
"""

# ---------------------------------------------------------------------------
# Pitch dimensions (metres — SPADL / FIFA standard)
# ---------------------------------------------------------------------------
PITCH_LENGTH_M: float = 105.0
PITCH_WIDTH_M: float = 68.0

# StatsBomb coordinate space (used by kloppy normalization)
STATSBOMB_LENGTH: float = 120.0
STATSBOMB_WIDTH: float = 80.0

# ---------------------------------------------------------------------------
# xT grid
# ---------------------------------------------------------------------------
XT_GRID_COLS: int = 16  # horizontal zones (own-goal → opp-goal)
XT_GRID_ROWS: int = 12  # vertical zones   (left touchline → right)
XT_ITERATIONS: int = 50  # Karun Singh value-iteration max iterations
XT_CONVERGENCE_EPS: float = 1e-6  # stop when max delta drops below this
XT_ROUND_DECIMALS: int = 5  # rounding precision for stored xT values

# ---------------------------------------------------------------------------
# Shot map
# ---------------------------------------------------------------------------
SHOT_MAP_FIGSIZE: tuple[int, int] = (10, 8)
SHOT_MAP_XG_SCALE: float = 1200.0  # multiply xG → scatter point area
SHOT_MAP_XG_BASE: float = 100.0  # minimum point area when xG is zero/missing
SHOT_MAP_COLOR_GOAL: str = "#00d4aa"
SHOT_MAP_COLOR_NO_GOAL: str = "#ff6b6b"
SHOT_MAP_BG_COLOR: str = "#1a1a2e"

# ---------------------------------------------------------------------------
# Pass network
# ---------------------------------------------------------------------------
PASS_NET_FIGSIZE: tuple[int, int] = (14, 9)
PASS_NET_PAD: int = 15  # pitch padding on all sides (mplsoccer units)
PASS_NET_LINE_SCALE: float = 0.4  # pass count → edge line-width multiplier
PASS_NET_LINE_MAX: float = 6.0  # maximum edge line width
PASS_NET_NODE_SCALE: float = 800.0  # pass count → node area multiplier
PASS_NET_NODE_BASE: float = 200.0  # minimum node area
PASS_NET_NODE_COLOR: str = "#f5c07a"
PASS_NET_MIN_PASSES: int = 2  # minimum pass pair count to draw an edge

# ---------------------------------------------------------------------------
# Player heatmap
# ---------------------------------------------------------------------------
HEATMAP_KDE_LEVELS: int = 100
HEATMAP_KDE_THRESH: float = 0.01

# ---------------------------------------------------------------------------
# xT bar chart
# ---------------------------------------------------------------------------
BAR_CHART_HEIGHT_PER_PLAYER: float = 0.45  # inches of figure height per row
BAR_CHART_MIN_HEIGHT: float = 4.0  # minimum total figure height (inches)
BAR_CHART_LABEL_OFFSET: float = 0.001  # x-offset for bar value labels

# ---------------------------------------------------------------------------
# Video bridge
# ---------------------------------------------------------------------------
DEFAULT_FRAME_W: int = 1280
DEFAULT_FRAME_H: int = 720

# Resolution heuristic breakpoints used in infer_frame_dimensions
RES_4K_THRESHOLD: int = 1400  # max_x > this → treat as 4K / 1080p source
RES_4K_W: int = 1920
RES_4K_H: int = 1080
RES_HD_THRESHOLD: int = 1000  # max_x > this → treat as 720p source
RES_HD_W: int = 1280
RES_HD_H: int = 720

# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR: str = "output_videos/analytics"
TOP_PLAYERS_N: int = 15  # rows in xT ranking tables
