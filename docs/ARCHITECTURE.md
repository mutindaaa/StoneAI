# Multi-Sport Scouting Platform - Architecture

## Vision
Automated scouting platform that analyzes match footage across multiple sports, providing coaches, scouts, and academies with actionable player performance data.

## Design Principles

1. **Sport-Agnostic Core**: Common detection and tracking engine
2. **Modular Sports**: Easy to add new sports via plugins
3. **User Input Required**: Teams/coaches provide player rosters for accuracy
4. **Database-Driven**: Store and compare performance across matches
5. **API-First**: Enable integrations and mobile apps

---

## System Architecture

### Layer 1: Video Processing Core (Sport-Agnostic)

**Responsibilities:**
- Video ingestion and preprocessing
- Object detection (players, ball, referees)
- Multi-object tracking across frames
- Motion analysis (speed, distance, acceleration)
- Team color assignment

**Technologies:**
- YOLOv8 for object detection
- ByteTrack for multi-object tracking
- OpenCV for video processing
- PyTorch for deep learning

**Key Modules:**
```
core/
├── video_processor.py       # Video I/O, chunking, preprocessing
├── object_detector.py        # YOLO-based detection
├── tracker.py                # ByteTrack wrapper
├── motion_analyzer.py        # Speed, distance, acceleration
└── team_assigner.py          # Color-based team assignment
```

---

### Layer 2: Sport-Specific Modules

Each sport implements a standard interface but provides custom logic.

**Standard Interface:**
```python
class SportAnalyzer(ABC):
    @abstractmethod
    def calibrate_field(self, frame) -> FieldCalibration

    @abstractmethod
    def calculate_metrics(self, tracks, match_config) -> PlayerMetrics

    @abstractmethod
    def detect_events(self, tracks, frames) -> List[Event]

    @abstractmethod
    def generate_visualization(self, frames, tracks, metrics) -> List[Frame]
```

**Soccer Module:**
```python
class SoccerAnalyzer(SportAnalyzer):
    field_dimensions = (105, 68)  # meters

    events = ["goal", "shot", "pass", "tackle", "corner", "free_kick"]

    metrics = [
        "distance_covered",
        "top_speed",
        "sprints",
        "possession_time",
        "successful_passes",
        "pass_accuracy",
        "shots_on_target",
        "tackles_won",
        "heat_map"
    ]
```

**Basketball Module (Future):**
```python
class BasketballAnalyzer(SportAnalyzer):
    field_dimensions = (28.65, 15.24)  # meters

    events = ["shot", "rebound", "assist", "steal", "block"]

    metrics = [
        "distance_covered",
        "top_speed",
        "shot_attempts",
        "shot_percentage",
        "rebounds",
        "assists",
        "court_position_map"
    ]
```

**Directory Structure:**
```
sports/
├── __init__.py
├── base.py                  # SportAnalyzer abstract class
├── soccer/
│   ├── __init__.py
│   ├── analyzer.py          # SoccerAnalyzer
│   ├── field_calibrator.py  # Perspective transformation
│   ├── events.py            # Goal, pass, shot detection
│   └── metrics.py           # Soccer-specific calculations
└── basketball/              # Future expansion
    ├── __init__.py
    ├── analyzer.py
    ├── court_calibrator.py
    ├── events.py
    └── metrics.py
```

---

### Layer 3: User Input & Configuration

**Match Configuration File:**
```json
{
  "match_id": "match_2026_02_16_001",
  "sport": "soccer",
  "date": "2026-02-16",
  "competition": "Premier League",
  "venue": "Emirates Stadium",

  "team_home": {
    "id": "arsenal",
    "name": "Arsenal FC",
    "color_primary": "red",
    "color_secondary": "white",
    "players": [
      {
        "jersey_number": 7,
        "name": "Bukayo Saka",
        "position": "RW",
        "player_id": "saka_b_001"
      },
      {
        "jersey_number": 9,
        "name": "Gabriel Jesus",
        "position": "ST",
        "player_id": "jesus_g_001"
      }
    ]
  },

  "team_away": {
    "id": "chelsea",
    "name": "Chelsea FC",
    "color_primary": "blue",
    "color_secondary": "white",
    "players": [...]
  },

  "video_path": "input_videos/arsenal_vs_chelsea.mp4"
}
```

**Why User Input?**
- ✅ Eliminates unreliable OCR jersey detection
- ✅ Enables player identity persistence across camera cuts
- ✅ Allows personalized player reports
- ✅ Builds historical player database
- ✅ Matches industry standard (StatsBomb, Wyscout require roster input)

---

### Layer 4: Database Schema

**PostgreSQL Schema:**

```sql
-- Players
CREATE TABLE players (
    player_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    team_id VARCHAR(50),
    position VARCHAR(20),
    jersey_number INT
);

-- Teams
CREATE TABLE teams (
    team_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    sport VARCHAR(20),
    color_primary VARCHAR(20),
    color_secondary VARCHAR(20)
);

-- Matches
CREATE TABLE matches (
    match_id VARCHAR(50) PRIMARY KEY,
    sport VARCHAR(20),
    date DATE,
    competition VARCHAR(100),
    venue VARCHAR(100),
    team_home_id VARCHAR(50) REFERENCES teams(team_id),
    team_away_id VARCHAR(50) REFERENCES teams(team_id),
    video_path VARCHAR(255),
    processed BOOLEAN DEFAULT FALSE
);

-- Player Match Performance
CREATE TABLE player_match_stats (
    id SERIAL PRIMARY KEY,
    match_id VARCHAR(50) REFERENCES matches(match_id),
    player_id VARCHAR(50) REFERENCES players(player_id),

    -- Universal metrics (all sports)
    minutes_played FLOAT,
    distance_covered_m FLOAT,
    top_speed_kmh FLOAT,
    avg_speed_kmh FLOAT,
    sprints_count INT,

    -- Sport-specific metrics (JSON)
    sport_metrics JSONB,

    -- Timestamp
    created_at TIMESTAMP DEFAULT NOW()
);

-- Example sport_metrics for soccer:
{
  "possession_time_sec": 420,
  "passes_attempted": 45,
  "passes_completed": 38,
  "pass_accuracy": 0.844,
  "shots": 3,
  "shots_on_target": 2,
  "tackles_won": 7,
  "tackles_attempted": 9
}

-- Events
CREATE TABLE match_events (
    id SERIAL PRIMARY KEY,
    match_id VARCHAR(50) REFERENCES matches(match_id),
    player_id VARCHAR(50) REFERENCES players(player_id),
    event_type VARCHAR(50),  -- "goal", "shot", "pass", etc.
    timestamp_sec FLOAT,
    frame_number INT,
    location_x FLOAT,
    location_y FLOAT,
    metadata JSONB
);
```

---

### Layer 5: API & Web Dashboard

**REST API Endpoints:**

```python
# Match processing
POST   /api/matches/upload          # Upload video + config
GET    /api/matches/{match_id}      # Get match details
POST   /api/matches/{match_id}/process  # Start analysis
GET    /api/matches/{match_id}/status   # Processing status

# Player analytics
GET    /api/players/{player_id}               # Player profile
GET    /api/players/{player_id}/matches       # Match history
GET    /api/players/{player_id}/stats         # Aggregate stats
GET    /api/players/{player_id}/compare/{player_id2}  # Compare

# Team analytics
GET    /api/teams/{team_id}           # Team profile
GET    /api/teams/{team_id}/players   # Roster
GET    /api/teams/{team_id}/stats     # Team aggregate stats

# Reports
GET    /api/reports/player/{player_id}/pdf    # Generate PDF
GET    /api/reports/match/{match_id}/pdf      # Match report
```

**Web Dashboard (Streamlit Prototype → React Production):**

```
Dashboard Features:
├── Upload Match
│   ├── Video upload
│   ├── Match config (JSON or form)
│   └── Processing queue
│
├── Match Analysis
│   ├── Video playback with annotations
│   ├── Real-time stats overlay
│   ├── Event timeline
│   └── Heatmaps
│
├── Player Profiles
│   ├── Career statistics
│   ├── Match history
│   ├── Performance trends
│   └── Comparison tools
│
└── Reports
    ├── PDF generation
    ├── Excel exports
    └── Share links
```

---

## Performance Optimizations

### Current Issues → Solutions

**Issue 1: Laggy Icons/Tracking**

**Root Causes:**
1. High-resolution video (1080p+) processing
2. OpenCV drawing operations on every frame
3. Not using GPU acceleration properly

**Solutions:**
```python
# 1. Downscale for processing, upscale for display
def preprocess_video(video_path, target_height=720):
    """Process at 720p for speed, maintain quality"""
    cap = cv2.VideoCapture(video_path)
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_factor = target_height / original_h
    # Process at 720p, store scale factor for coordinate mapping

# 2. Pre-render icon templates (alpha blending)
class IconRenderer:
    def __init__(self):
        self.player_ellipse_cache = {}
        self.ball_triangle_cache = {}

    def draw_player_optimized(self, frame, bbox, color, track_id):
        # Use cv2.addWeighted instead of drawing primitives
        # 3-5x faster for complex overlays

# 3. Batch GPU operations
def process_frames_batch(frames, model, batch_size=8):
    """Process frames in batches on GPU"""
    results = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        batch_results = model(batch)  # Single GPU call
        results.extend(batch_results)
    return results
```

**Issue 2: Incorrect Speed/Distance**

**Root Cause:**
Field calibration failing or inaccurate perspective transformation

**Solution:**
```python
# Manual field calibration with user verification
class FieldCalibratorInteractive:
    def calibrate_with_user_input(self, frame):
        """
        1. Auto-detect field keypoints
        2. Show to user for verification/correction
        3. Allow manual adjustment via GUI
        4. Save calibration for this camera angle
        """
        keypoints = self.auto_detect_keypoints(frame)

        # Show frame with detected keypoints
        verified_keypoints = self.show_calibration_ui(frame, keypoints)

        # Calculate transformation matrix
        transform = self.calculate_perspective_transform(verified_keypoints)

        return transform
```

**Issue 3: Video Quality Degradation**

**Root Cause:**
Codec/compression issues when saving output

**Solution:**
```python
def save_video_high_quality(frames, output_path, fps, codec='h264'):
    """
    Use H.264 codec with high bitrate for quality
    """
    height, width = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
        isColor=True
    )

    # Set high bitrate (prevent compression artifacts)
    # Alternative: use FFmpeg subprocess for better quality control
```

---

## Implementation Roadmap

### Phase 1: Fix Soccer Platform (2 weeks)

**Week 1: Performance & Accuracy**
- [ ] GPU optimization verification
- [ ] Video downscaling for processing
- [ ] Optimized icon rendering (pre-rendered templates)
- [ ] Interactive field calibration tool
- [ ] High-quality video output (H.264)

**Week 2: User Input System**
- [ ] Match configuration JSON schema
- [ ] Config file loader
- [ ] Player-tracking ID mapping (jersey # + team color → player identity)
- [ ] Simple web form for match config (Streamlit)

### Phase 2: Database & API (1 month)

**Week 3-4: Database**
- [ ] PostgreSQL setup
- [ ] Schema implementation
- [ ] Match data ingestion pipeline
- [ ] Player stats aggregation

**Week 5-6: API & Dashboard**
- [ ] FastAPI REST endpoints
- [ ] Streamlit dashboard v1
- [ ] Player profile pages
- [ ] Match visualization

### Phase 3: Production Features (1-2 months)

- [ ] PDF report generation
- [ ] Heat maps
- [ ] Player comparison tools
- [ ] Multi-match aggregation
- [ ] Authentication & user accounts

### Phase 4: Multi-Sport Expansion (3-6 months)

- [ ] Abstract base classes (SportAnalyzer)
- [ ] Basketball module
- [ ] Basketball YOLO model training
- [ ] Basketball-specific metrics
- [ ] Test with basketball footage

---

## Technology Stack

**Backend:**
- Python 3.10+
- FastAPI (REST API)
- PostgreSQL (database)
- Redis (caching, job queue)
- Celery (async task processing)

**ML/CV:**
- PyTorch (GPU acceleration)
- YOLOv8 (object detection)
- OpenCV (video processing)
- NumPy, Pandas (data processing)

**Frontend:**
- Streamlit (MVP dashboard)
- React + TypeScript (production)
- Tailwind CSS (styling)
- Chart.js / Plotly (visualizations)

**Infrastructure:**
- Docker (containerization)
- AWS S3 (video storage)
- AWS EC2/Lambda (compute)
- Nginx (reverse proxy)

---

## Revenue Model

**Tiered Subscription:**

| Tier | Price/Month | Features |
|------|-------------|----------|
| Individual | $19 | 5 matches/month, basic stats |
| Team | $99 | Unlimited matches, full roster, PDF reports |
| Academy | $299 | Multi-team, advanced analytics, API access |
| Enterprise | Custom | White-label, custom integrations |

**Target Customers:**
1. Youth soccer academies (primary)
2. High school/college teams
3. Semi-professional clubs
4. Individual players/parents
5. Professional scouts (premium tier)

---

## Competitive Analysis

**Competitors:**
- Hudl (market leader, $100M+ revenue)
- Wyscout (professional scouting)
- StatsBomb (data analytics)
- Veo (automated camera + analysis)

**Our Differentiation:**
1. **Multi-sport** (competitors are single-sport)
2. **Affordable** (Hudl is $500-2000/year, we start at $19/month)
3. **No special camera required** (use any video)
4. **AI-powered** (automated analysis, minimal manual work)

---

## Next Steps

Ready to start implementation?

1. **Immediate**: Fix performance issues with current soccer platform
2. **This week**: Implement match configuration system
3. **Next week**: Set up database and start storing match data
4. **Month 2**: Build API and dashboard

Let's begin! 🚀
