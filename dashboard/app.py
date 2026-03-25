"""
Sports Analytics Dashboard

Streamlit UI that talks to the FastAPI backend at API_URL.

Run with:
  streamlit run dashboard/app.py

Set the API URL via env var (default: http://localhost:8000):
  API_URL=http://my-server:8000 streamlit run dashboard/app.py
"""

import io
import json
import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sports Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("⚽ Sports Analytics")
page = st.sidebar.radio(
    "Navigation",
    ["New Analysis", "Job Queue", "Results Viewer", "Analytics"],
    index=0,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STEP_EMOJIS = {
    "Queued": "🕐",
    "Loading video": "📂",
    "Object detection & tracking": "🔍",
    "Camera movement estimation": "🎥",
    "Field calibration": "📐",
    "Team assignment (SigLIP)": "🧠",
    "Jersey number detection": "🔢",
    "Speed & distance calculation": "⚡",
    "Ball possession assignment": "⚽",
    "Rendering output video": "🎬",
    "Complete": "✅",
}


def _emoji(step: str) -> str:
    return STEP_EMOJIS.get(step, "⚙️")


def _api(method: str, path: str, **kwargs):
    try:
        resp = getattr(requests, method)(f"{API_URL}{path}", **kwargs)
        resp.raise_for_status()
        return resp
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_URL}. Is the server running?")
        st.stop()
    except requests.exceptions.HTTPError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
        return None


def _status_badge(status: str) -> str:
    colors = {
        "queued": "🟡",
        "processing": "🔵",
        "done": "🟢",
        "failed": "🔴",
    }
    return f"{colors.get(status, '⚪')} {status.upper()}"


# ---------------------------------------------------------------------------
# Page: New Analysis
# ---------------------------------------------------------------------------


def _default_config() -> dict:
    return {
        "sport": "soccer",
        "competition": "My Match",
        "team_home": {
            "id": "team_home",
            "name": "Home Team",
            "color_primary": "red",
            "color_secondary": "white",
            "players": [],
        },
        "team_away": {
            "id": "team_away",
            "name": "Away Team",
            "color_primary": "blue",
            "color_secondary": "white",
            "players": [],
        },
        "processing_options": {
            "use_gpu": True,
            "downscale_for_processing": True,
            "target_processing_height": 720,
            "enable_jersey_ocr": True,
            "jersey_ocr_sample_interval": 30,
            "enable_radar": True,
            "pitch_model_path": "models/pitch_detection.pt",
        },
    }


def page_new_analysis():
    st.header("New Analysis")
    st.markdown("Upload a match video and configure team details to start processing.")

    col_upload, col_config = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("1 · Upload Video")
        video_file = st.file_uploader(
            "Match video",
            type=["mp4", "avi", "mkv", "mov"],
            help="Short clips (<5 min) run entirely in RAM. Longer videos use chunked processing.",
        )
        if video_file:
            st.video(video_file)

    with col_config:
        st.subheader("2 · Match Config")

        # Quick-fill mode: paste full JSON or use form
        config_mode = st.radio(
            "Config mode", ["Form", "Paste JSON"], horizontal=True
        )

        if config_mode == "Paste JSON":
            raw = st.text_area(
                "Paste config JSON",
                value=json.dumps(_default_config(), indent=2),
                height=400,
            )
            try:
                config_data = json.loads(raw)
                st.success("JSON is valid")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                config_data = None
        else:
            cfg = _default_config()

            cfg["competition"] = st.text_input("Competition", value=cfg["competition"])
            cfg["sport"] = st.selectbox("Sport", ["soccer"], index=0)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Home Team**")
                cfg["team_home"]["name"] = st.text_input("Home team name", value="Home Team")
                cfg["team_home"]["color_primary"] = st.selectbox(
                    "Home primary color",
                    ["red", "blue", "white", "black", "green", "yellow", "orange", "purple"],
                    index=0,
                )
            with c2:
                st.markdown("**Away Team**")
                cfg["team_away"]["name"] = st.text_input("Away team name", value="Away Team")
                cfg["team_away"]["color_primary"] = st.selectbox(
                    "Away primary color",
                    ["blue", "red", "white", "black", "green", "yellow", "orange", "purple"],
                    index=0,
                )

            with st.expander("Processing options"):
                cfg["processing_options"]["use_gpu"] = st.checkbox(
                    "Use GPU (CUDA)", value=True
                )
                cfg["processing_options"]["downscale_for_processing"] = st.checkbox(
                    "Downscale to 720p for processing", value=True
                )
                cfg["processing_options"]["enable_jersey_ocr"] = st.checkbox(
                    "Jersey number OCR", value=True
                )
                cfg["processing_options"]["enable_radar"] = st.checkbox(
                    "Radar bird's-eye overlay", value=True
                )

            config_data = cfg

    st.divider()

    if st.button("🚀 Start Analysis", type="primary", disabled=(video_file is None)):
        if config_data is None:
            st.error("Fix config JSON errors before submitting.")
            return

        with st.spinner("Uploading video..."):
            resp = _api(
                "post",
                "/jobs",
                files={"video": (video_file.name, video_file.getvalue(), "video/mp4")},
                data={"config": json.dumps(config_data)},
                timeout=120,
            )

        if resp and resp.ok:
            job = resp.json()
            st.success(f"Job started! ID: `{job['job_id']}`")
            st.session_state["active_job_id"] = job["job_id"]
            st.info("Switch to **Job Queue** in the sidebar to monitor progress.")
        else:
            st.error("Failed to start job. Check the API server logs.")


# ---------------------------------------------------------------------------
# Page: Job Queue
# ---------------------------------------------------------------------------


def page_job_queue():
    st.header("Job Queue")

    auto_refresh = st.checkbox("Auto-refresh every 5 s", value=True)

    resp = _api("get", "/jobs")
    if not resp:
        return

    jobs = resp.json()

    if not jobs:
        st.info("No jobs yet. Go to **New Analysis** to upload a video.")
        return

    # Active job progress widget
    active_id = st.session_state.get("active_job_id")
    if active_id:
        active = next((j for j in jobs if j["job_id"] == active_id), None)
        if active and active["status"] in ("queued", "processing"):
            st.subheader(f"Active job · `{active_id[:8]}…`")
            emoji = _emoji(active["step"])
            st.progress(
                active["progress"],
                text=f"{emoji} {active['step']} ({active['progress']:.0%})",
            )
            if auto_refresh:
                time.sleep(5)
                st.rerun()

    # Jobs table
    st.subheader("All jobs")
    df = pd.DataFrame(
        [
            {
                "ID": j["job_id"][:8] + "…",
                "Full ID": j["job_id"],
                "Status": _status_badge(j["status"]),
                "Progress": f"{j['progress']:.0%}",
                "Step": j["step"],
                "File": j["video_filename"],
                "Created": j["created_at"][:19].replace("T", " "),
                "Finished": (j["finished_at"] or "")[:19].replace("T", " "),
            }
            for j in jobs
        ]
    )

    selected = st.dataframe(
        df[["ID", "Status", "Progress", "Step", "File", "Created", "Finished"]],
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    rows = selected.selection.get("rows", [])
    if rows:
        job_id = df.iloc[rows[0]]["Full ID"]
        st.session_state["results_job_id"] = job_id
        if st.button("📊 View Results for selected job"):
            st.session_state["active_page"] = "Results Viewer"
            st.rerun()

    # Manual refresh
    if st.button("🔄 Refresh now"):
        st.rerun()


# ---------------------------------------------------------------------------
# Page: Results Viewer
# ---------------------------------------------------------------------------


def page_results_viewer():
    st.header("Results Viewer")

    # Job selector
    resp = _api("get", "/jobs")
    if not resp:
        return
    jobs = resp.json()
    done_jobs = [j for j in jobs if j["status"] == "done"]

    if not done_jobs:
        st.info("No completed jobs yet.")
        return

    preselect = st.session_state.get("results_job_id", done_jobs[0]["job_id"])
    options = [j["job_id"] for j in done_jobs]
    labels = [f"{j['job_id'][:8]}… — {j['video_filename']}" for j in done_jobs]
    label_map = dict(zip(options, labels))

    selected_id = st.selectbox(
        "Select job",
        options=options,
        format_func=lambda x: label_map[x],
        index=options.index(preselect) if preselect in options else 0,
    )

    st.divider()

    tab_video, tab_metrics, tab_events = st.tabs(["🎬 Video", "📊 Metrics", "⚡ Events"])

    # --- Video tab ---
    with tab_video:
        st.subheader("Annotated Output Video")
        video_url = f"{API_URL}/jobs/{selected_id}/video"
        st.video(video_url)
        st.markdown(f"[⬇ Download MP4]({video_url})")

    # --- Metrics tab ---
    with tab_metrics:
        st.subheader("Player Metrics")
        resp = _api("get", f"/jobs/{selected_id}/metrics")
        if resp:
            metrics_raw = resp.json()
            # Support both new format {_summary, players} and legacy flat format
            summary = metrics_raw.get("_summary", {})
            players_dict = metrics_raw.get("players", metrics_raw)
            unit_system = summary.get("unit_system", "metric")
            use_imperial = unit_system == "imperial"
            dist_label = "Distance (yd)" if use_imperial else "Distance (m)"
            speed_label = "Top Speed (mph)" if use_imperial else "Top Speed (km/h)"
            avg_spd_label = "Avg Speed (mph)" if use_imperial else "Avg Speed (km/h)"

            # Team summary cards
            if summary.get("teams"):
                st.markdown("#### Team Summary")
                team_cols = st.columns(len(summary["teams"]))
                for col, (tid, tt) in zip(team_cols, summary["teams"].items()):
                    with col:
                        d_u = tt.get("distance_unit", "m")
                        s_u = tt.get("speed_unit", "km/h")
                        st.metric(tid, f"{tt['total_distance']:.0f} {d_u}")
                        st.caption(f"Max: {tt['max_speed']} {s_u}  |  "
                                   f"Passes: {tt['total_passes']}  |  "
                                   f"Poss: {tt['total_possession_sec']:.0f}s")

            rows = []
            for pid, m in players_dict.items():
                if not isinstance(m, dict):
                    continue
                row = {
                    "Player": m.get("player_name", pid),
                    "Team": m.get("team_id", "—"),
                    "Minutes": round(m.get("minutes_played", 0), 1),
                    dist_label: m.get("distance_covered", 0),
                    speed_label: m.get("top_speed", 0),
                    avg_spd_label: m.get("avg_speed", 0),
                    "Sprints": m.get("sprints_count", 0),
                }
                sport_m = m.get("sport_metrics", {})
                row["Passes"] = sport_m.get("passes_attempted", 0)
                row["Shots"] = sport_m.get("shots", 0)
                row["Poss(s)"] = sport_m.get("possession_time_sec", 0)
                rows.append(row)

            df = pd.DataFrame(rows).sort_values(dist_label, ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Quick bar charts
            st.markdown(f"#### {dist_label} per player (top 20)")
            st.bar_chart(df.head(20).set_index("Player")[dist_label])

            st.markdown(f"#### {speed_label} per player (top 20)")
            st.bar_chart(df.head(20).set_index("Player")[speed_label])

            # Download
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇ Download metrics CSV",
                data=csv,
                file_name=f"{selected_id[:8]}_metrics.csv",
                mime="text/csv",
            )

    # --- Events tab ---
    with tab_events:
        st.subheader("Match Events")
        resp = _api("get", f"/jobs/{selected_id}/events")
        if resp:
            events = resp.json()
            if not events:
                st.info("No events detected.")
            else:
                df_ev = pd.DataFrame(events)

                # Summary
                c1, c2, c3 = st.columns(3)
                passes = df_ev[df_ev["event_type"] == "pass"]
                shots = df_ev[df_ev["event_type"] == "shot"]
                poss = df_ev[df_ev["event_type"] == "possession_change"]
                c1.metric("Passes", len(passes))
                c2.metric("Shots", len(shots))
                c3.metric("Possession Changes", len(poss))

                # Filter
                event_types = ["All"] + sorted(df_ev["event_type"].unique().tolist())
                selected_type = st.selectbox("Filter by type", event_types)
                if selected_type != "All":
                    df_ev = df_ev[df_ev["event_type"] == selected_type]

                # Display
                cols_to_show = [
                    c
                    for c in [
                        "timestamp_sec",
                        "event_type",
                        "player_id",
                        "team_id",
                        "location_x",
                        "location_y",
                    ]
                    if c in df_ev.columns
                ]
                st.dataframe(df_ev[cols_to_show], use_container_width=True, hide_index=True)

                # Timeline
                st.markdown("#### Event timeline")
                timeline = df_ev.copy()
                timeline["count"] = 1
                if "timestamp_sec" in timeline.columns:
                    timeline["minute"] = (timeline["timestamp_sec"] // 60).astype(int)
                    pivot = (
                        timeline.groupby(["minute", "event_type"])["count"]
                        .sum()
                        .unstack(fill_value=0)
                    )
                    st.bar_chart(pivot)

                st.download_button(
                    "⬇ Download events JSON",
                    data=json.dumps(events, indent=2),
                    file_name=f"{selected_id[:8]}_events.json",
                    mime="application/json",
                )


# ---------------------------------------------------------------------------
# Page: Analytics
# ---------------------------------------------------------------------------


def page_analytics():
    st.header("Analytics")
    st.markdown(
        "Explore StatsBomb open data or run analytics on a completed video job. "
        "Requires: `kloppy`, `statsbombpy`, `socceraction`, `mplsoccer`."
    )

    tab_sb, tab_video = st.tabs(["StatsBomb Open Data", "Video Match Analytics"])

    # ------------------------------------------------------------------
    # Tab A: StatsBomb open data explorer
    # ------------------------------------------------------------------
    with tab_sb:
        st.subheader("StatsBomb Open Data Explorer")

        # Lazy import — fail gracefully if not installed
        try:
            from analytics.statsbomb_loader import list_open_competitions, list_open_matches, load_match
            from analytics.spadl_pipeline import statsbomb_to_spadl, compute_xt, top_players_by_xt
            from analytics.visualizer import shot_map, pass_network, xt_bar_chart
            libs_ok = True
        except ImportError as e:
            st.error(f"Analytics libraries not installed: {e}\n\nRun: `pip install kloppy statsbombpy socceraction mplsoccer`")
            libs_ok = False

        if libs_ok:
            with st.spinner("Loading competition list..."):
                try:
                    comps_df = list_open_competitions()
                    comps_df = comps_df.sort_values("competition_name")
                except Exception as e:
                    st.error(f"Could not load competitions: {e}")
                    comps_df = None

            if comps_df is not None and not comps_df.empty:
                comp_options = comps_df.apply(
                    lambda r: f"{r['competition_name']} — {r['season_name']}", axis=1
                ).tolist()
                selected_comp = st.selectbox("Competition / Season", comp_options)
                comp_idx = comp_options.index(selected_comp)
                comp_row = comps_df.iloc[comp_idx]

                with st.spinner("Loading matches..."):
                    try:
                        matches_df = list_open_matches(
                            int(comp_row["competition_id"]),
                            int(comp_row["season_id"]),
                        )
                    except Exception as e:
                        st.error(f"Could not load matches: {e}")
                        matches_df = None

                if matches_df is not None and not matches_df.empty:
                    match_labels = matches_df.apply(
                        lambda r: f"{r.get('home_team','?')} vs {r.get('away_team','?')} ({r.get('match_date','')})",
                        axis=1,
                    ).tolist()
                    selected_match_label = st.selectbox("Match", match_labels)
                    match_idx = match_labels.index(selected_match_label)
                    match_id = int(matches_df.iloc[match_idx]["match_id"])
                    st.caption(f"Match ID: `{match_id}`")

                    viz_choice = st.multiselect(
                        "Visualizations",
                        ["Shot Map", "Pass Network (Home)", "Pass Network (Away)", "xT Top Players"],
                        default=["Shot Map", "xT Top Players"],
                    )

                    if st.button("Run Analytics", type="primary"):
                        with st.spinner(f"Loading match {match_id}..."):
                            try:
                                result = load_match(match_id)
                                events_df = result["events"]
                                teams = events_df["team"].unique().tolist() if "team" in events_df.columns else ["home", "away"]
                                home_team = teams[0] if teams else "home"
                                away_team = teams[1] if len(teams) > 1 else "away"
                                st.success(f"Loaded {len(events_df)} events — {home_team} vs {away_team}")
                            except Exception as e:
                                st.error(f"Failed to load match: {e}")
                                events_df = None

                        if events_df is not None:
                            # xT
                            actions_df = None
                            if any(v in viz_choice for v in ["xT Top Players"]):
                                with st.spinner("Computing xT..."):
                                    try:
                                        actions_df = statsbomb_to_spadl(events_df, home_team_id=home_team)
                                        actions_df = compute_xt(actions_df)
                                        xt_df = top_players_by_xt(actions_df, n=15)
                                    except Exception as e:
                                        st.warning(f"xT computation failed: {e}")
                                        xt_df = None

                            cols = st.columns(2)
                            col_idx = 0

                            if "Shot Map" in viz_choice:
                                with cols[col_idx % 2]:
                                    st.markdown("**Shot Map**")
                                    fig, _ = shot_map(events_df, title=f"Shot Map")
                                    st.pyplot(fig)
                                col_idx += 1

                            if "Pass Network (Home)" in viz_choice:
                                with cols[col_idx % 2]:
                                    st.markdown(f"**Pass Network — {home_team}**")
                                    fig, _ = pass_network(events_df, team=home_team)
                                    st.pyplot(fig)
                                col_idx += 1

                            if "Pass Network (Away)" in viz_choice:
                                with cols[col_idx % 2]:
                                    st.markdown(f"**Pass Network — {away_team}**")
                                    fig, _ = pass_network(events_df, team=away_team)
                                    st.pyplot(fig)
                                col_idx += 1

                            if "xT Top Players" in viz_choice and xt_df is not None and not xt_df.empty:
                                with cols[col_idx % 2]:
                                    st.markdown("**Top Players by Expected Threat (xT)**")
                                    fig, _ = xt_bar_chart(xt_df)
                                    st.pyplot(fig)
                                st.dataframe(xt_df, use_container_width=True, hide_index=True)
                                csv = xt_df.to_csv(index=False)
                                st.download_button(
                                    "Download xT CSV",
                                    data=csv,
                                    file_name=f"match_{match_id}_xt.csv",
                                    mime="text/csv",
                                )

    # ------------------------------------------------------------------
    # Tab B: Video match analytics
    # ------------------------------------------------------------------
    with tab_video:
        st.subheader("Video Match Analytics")
        st.markdown(
            "Runs xT and shot map on events detected by the video pipeline. "
            "Select a completed job below."
        )

        try:
            from analytics.video_bridge import load_video_events, infer_frame_dimensions
            from analytics.spadl_pipeline import compute_xt, top_players_by_xt
            from analytics.visualizer import shot_map, xt_bar_chart
            video_libs_ok = True
        except ImportError as e:
            st.error(f"Analytics libraries not installed: {e}")
            video_libs_ok = False

        if video_libs_ok:
            resp = _api("get", "/jobs")
            if resp:
                jobs = resp.json()
                done_jobs = [j for j in jobs if j["status"] == "done"]

                if not done_jobs:
                    st.info("No completed jobs yet. Run the video pipeline first.")
                else:
                    options = [j["job_id"] for j in done_jobs]
                    labels = [f"{j['job_id'][:8]}… — {j['video_filename']}" for j in done_jobs]
                    label_map = dict(zip(options, labels))

                    selected_id = st.selectbox(
                        "Select job",
                        options=options,
                        format_func=lambda x: label_map[x],
                    )

                    if st.button("Run Video Analytics", type="primary"):
                        # Fetch events from API
                        ev_resp = _api("get", f"/jobs/{selected_id}/events")
                        if ev_resp:
                            events_raw = ev_resp.json()
                            if not events_raw:
                                st.warning("No events found for this job.")
                            else:
                                import tempfile, json as _json
                                with tempfile.NamedTemporaryFile(
                                    mode="w", suffix="_events.json", delete=False
                                ) as tmp:
                                    _json.dump(events_raw, tmp)
                                    tmp_path = tmp.name

                                with st.spinner("Computing analytics..."):
                                    frame_w, frame_h = infer_frame_dimensions(tmp_path)
                                    actions_df = load_video_events(tmp_path, frame_w, frame_h)

                                if actions_df.empty:
                                    st.warning("No SPADL-mappable events found.")
                                else:
                                    actions_df = compute_xt(actions_df)
                                    xt_df = top_players_by_xt(actions_df, n=15)

                                    st.success(f"Processed {len(actions_df)} actions")

                                    col1, col2 = st.columns(2)

                                    # Shot map
                                    shots = actions_df[actions_df["type_name"] == "shot"].rename(
                                        columns={"start_x": "coordinates_x", "start_y": "coordinates_y",
                                                 "type_name": "event_type", "result_name": "result"}
                                    )
                                    with col1:
                                        st.markdown("**Shot Map**")
                                        fig, _ = shot_map(shots, pitch_type="custom",
                                                          title="Shot Map (video)")
                                        st.pyplot(fig)

                                    # xT chart
                                    if not xt_df.empty:
                                        with col2:
                                            st.markdown("**Top Players by xT**")
                                            fig, _ = xt_bar_chart(xt_df)
                                            st.pyplot(fig)

                                        st.dataframe(xt_df, use_container_width=True, hide_index=True)

                                    # Actions table
                                    with st.expander("All SPADL actions"):
                                        display_cols = [c for c in [
                                            "time_seconds", "type_name", "result_name",
                                            "player_id", "team_id", "start_x", "start_y",
                                            "xt_value",
                                        ] if c in actions_df.columns]
                                        st.dataframe(
                                            actions_df[display_cols],
                                            use_container_width=True, hide_index=True,
                                        )

                                    os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "New Analysis":
    page_new_analysis()
elif page == "Job Queue":
    page_job_queue()
elif page == "Results Viewer":
    page_results_viewer()
elif page == "Analytics":
    page_analytics()
