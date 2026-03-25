"""
Video Download and Clipping Tool

Downloads YouTube videos and clips them for analytics testing.
Uses OpenCV for clipping (no FFmpeg needed).

Usage:
    # Download a single video
    python download_and_clip.py --url "https://youtu.be/ktFPvKw5rZM" --download --output input_videos/chicago_vs_scwave.mp4

    # Download an entire playlist into a folder
    python download_and_clip.py --download-playlist "https://youtube.com/playlist?list=PLIVc4N0HjStpJ5SDDk-6cWfFUQmoE6ylD" --output-dir input_videos/chicago_u19

    # List all videos in a playlist (no download)
    python download_and_clip.py --list-playlist "https://youtube.com/playlist?list=PLIVc4N0HjStpJ5SDDk-6cWfFUQmoE6ylD"

    # Create generic clips (5-min test, first half, second half)
    python download_and_clip.py --input input_videos/match.mp4 --create-clips --clip-prefix chicago --pre-match-offset 300

    # Check video info
    python download_and_clip.py --info input_videos/match.mp4
"""

import subprocess
import sys
import os
import cv2
import argparse
from pathlib import Path


def list_playlist(url: str) -> list:
    """
    Print all videos in a YouTube playlist without downloading.
    Returns list of dicts with id, title, duration_string.
    """
    print(f"\nFetching playlist info: {url}\n")
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp",
         "--flat-playlist",
         "--print", "%(id)s\t%(title)s\t%(duration_string)s",
         url],
        capture_output=True, text=True
    )
    videos = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            vid_id = parts[0].strip()
            title  = parts[1].strip()
            dur    = parts[2].strip() if len(parts) > 2 else "?"
            videos.append({"id": vid_id, "title": title, "duration": dur})

    print(f"{'#':<4} {'Duration':<10} {'Title'}")
    print("-" * 80)
    for i, v in enumerate(videos, 1):
        print(f"{i:<4} {v['duration']:<10} {v['title']}")
    print(f"\nTotal: {len(videos)} videos\n")
    return videos


def download_playlist(url: str, output_dir: str, quality: str = "best[height<=1080]",
                      skip_existing: bool = True):
    """
    Download all videos in a YouTube playlist into output_dir.

    Files are named: YYYYMMDD_<sanitised_title>.mp4
    Skips already-downloaded files by default.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_template = str(Path(output_dir) / "%(upload_date)s_%(title)s.%(ext)s")

    print(f"\nDownloading playlist → {output_dir}")
    print(f"Quality: {quality}\n")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        url,
        "-f", quality,
        "-o", output_template,
        "--no-playlist-reverse",   # chronological order
        "--progress",
        "--merge-output-format", "mp4",
    ]
    if skip_existing:
        cmd.append("--no-overwrites")

    result = subprocess.run(cmd)
    if result.returncode == 0:
        downloaded = sorted(Path(output_dir).glob("*.mp4"))
        print(f"\nPlaylist download complete: {len(downloaded)} files in {output_dir}")
        for f in downloaded:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}  ({size_mb:.0f} MB)")
        return [str(f) for f in downloaded]
    else:
        print("\nSome downloads may have failed. Re-run to resume — yt-dlp will skip existing files.")
        return []


def download_video(url: str, output_path: str, quality: str = "best[height<=1080]"):
    """Download video from YouTube using yt-dlp."""
    print(f"\nDownloading video from YouTube...")
    print(f"  URL: {url}")
    print(f"  Quality: {quality}")
    print(f"  Output: {output_path}\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "yt_dlp",
        url,
        "-f", quality,
        "-o", output_path,
        "--no-playlist",
        "--progress"
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"\nDone: Download complete: {output_path} ({size_mb:.0f} MB)")
        return output_path
    else:
        print("\nIf download failed, try manually:")
        print(f'  python -m yt_dlp "{url}" -f "{quality}" -o "{output_path}"')
        raise RuntimeError("Download failed")


def clip_video_opencv(input_path: str, output_path: str, start_sec: float, end_sec: float):
    """Clip a video using OpenCV (no FFmpeg needed)."""
    print(f"\nClipping video...")
    print(f"  Input:    {input_path}")
    print(f"  Output:   {output_path}")
    print(f"  From:     {_sec_to_hms(start_sec)}")
    print(f"  To:       {_sec_to_hms(end_sec)}")
    print(f"  Duration: {end_sec - start_sec:.0f}s")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), total_frames)
    frames_to_write = end_frame - start_frame

    print(f"  Video:    {width}x{height} @ {fps:.1f} FPS")
    print(f"  Frames:   {start_frame} to {end_frame} ({frames_to_write} frames)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_written = 0
    report_every = max(1, frames_to_write // 10)

    for i in range(frames_to_write):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_written += 1
        if frames_written % report_every == 0:
            pct = int(frames_written / frames_to_write * 100)
            print(f"  Progress: {pct}% ({frames_written}/{frames_to_write})", end='\r')

    cap.release()
    out.release()

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nDone: Clip saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


def get_video_info(video_path: str) -> dict:
    """Get video duration and properties using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()

    return {
        'fps': fps, 'width': width, 'height': height,
        'total_frames': total_frames, 'duration_sec': duration_sec,
        'duration_hms': _sec_to_hms(duration_sec)
    }


def _sec_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _hms_to_sec(hms: str) -> float:
    try:
        return float(hms)
    except ValueError:
        parts = hms.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        raise ValueError(f"Invalid time format: {hms}. Use HH:MM:SS or seconds.")


def create_generic_clips(input_path: str, output_prefix: str = "clip", pre_match_offset: int = 0):
    """
    Create standard clips from any match video.

    Args:
        input_path:        Path to downloaded video.
        output_prefix:     Filename prefix for clips (e.g. 'veo_match').
        pre_match_offset:  Seconds of pre-match content before kickoff (default 0).
    """
    info = get_video_info(input_path)
    print(f"\nVideo: {info['width']}x{info['height']} @ {info['fps']:.1f} FPS")
    print(f"Duration: {info['duration_hms']} ({info['duration_sec']:.0f}s)")
    if pre_match_offset:
        print(f"Pre-match offset: {_sec_to_hms(pre_match_offset)} ({pre_match_offset}s)")

    ko = pre_match_offset  # kickoff offset

    clips = []

    # Always create: first 5 min, first half, second half
    clips.append({
        "name": f"input_videos/{output_prefix}_test_5min.mp4",
        "start": ko, "end": ko + 300,
        "desc": "Quick test — first 5 min of match",
    })
    if info['duration_sec'] > ko + 45 * 60:
        clips.append({
            "name": f"input_videos/{output_prefix}_first_half.mp4",
            "start": ko, "end": ko + 45 * 60,
            "desc": "First half (45 min)",
        })
    if info['duration_sec'] > ko + 50 * 60:
        clips.append({
            "name": f"input_videos/{output_prefix}_second_half.mp4",
            "start": ko + 47 * 60, "end": min(info['duration_sec'], ko + 97 * 60),
            "desc": "Second half (~45 min, with 2 min HT buffer)",
        })

    print(f"\nCreating {len(clips)} clips...")
    print("=" * 60)
    created = []
    for i, clip in enumerate(clips):
        if clip["start"] >= info["duration_sec"]:
            print(f"\nSkipping (beyond video): {clip['desc']}")
            continue
        end = min(clip["end"], info["duration_sec"])
        print(f"\n[{i+1}/{len(clips)}] {clip['desc']}")
        try:
            clip_video_opencv(input_path, clip["name"], clip["start"], end)
            created.append(clip["name"])
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n{'='*60}")
    print(f"Created {len(created)} clips!\n")
    print("Run analytics on the quick 5-min test first:")
    if created:
        print(f"  python main_v3.py --config match_configs/your_config.json --video {created[0]}")
    return created


def create_spurs_chelsea_clips(input_path: str):
    """
    Create strategically timed clips from the Spurs vs Chelsea Dec 2024 match.

    Match events:
    - 6':  Solanke goal (1-0 Spurs)
    - 13': Kulusevski goal (2-0 Spurs)
    - 65': Sancho goal (2-1)
    - 74': Palmer penalty (2-2)
    - 79': Palmer penalty (2-3)
    - 83': Fernandez goal (2-4)
    Final: Spurs 3-4 Chelsea

    Note: NBC Sports video may have pre-match content before kickoff.
    If clips seem off, adjust pre_match_offset below.
    """
    info = get_video_info(input_path)
    print(f"\nVideo: {info['width']}x{info['height']} @ {info['fps']:.1f} FPS")
    print(f"Duration: {info['duration_hms']} ({info['duration_sec']:.0f}s)")
    print(f"\nNote: If your clips start at wrong times, the video may have")
    print(f"pre-match content. Edit 'pre_match_offset' in download_and_clip.py\n")

    # NBC Sports pre-match broadcast is ~5:30 before kickoff
    # Detected via YOLO sampling: players first appear consistently at 5m30s
    pre_match_offset = 330  # 5 min 30 sec of pre-match content

    clips = [
        {
            "name": "input_videos/test_01_kickoff_5min.mp4",
            "start": pre_match_offset + 0,
            "end":   pre_match_offset + 300,
            "desc": "Kickoff first 5 min of actual match"
        },
        {
            "name": "input_videos/test_02_solanke_goal.mp4",
            "start": pre_match_offset + (5 * 60),
            "end":   pre_match_offset + (8 * 60),
            "desc": "Solanke goal at match 6' (video 11:30)"
        },
        {
            "name": "input_videos/test_03_kulusevski_goal.mp4",
            "start": pre_match_offset + (11 * 60),
            "end":   pre_match_offset + (15 * 60),
            "desc": "Kulusevski goal at match 13' (video 17:00)"
        },
        {
            "name": "input_videos/test_04_first_20min.mp4",
            "start": pre_match_offset + 0,
            "end":   pre_match_offset + (20 * 60),
            "desc": "Full first 20 min of match (both Spurs goals)"
        },
        {
            "name": "input_videos/test_05_chelsea_comeback.mp4",
            "start": pre_match_offset + (63 * 60),
            "end":   pre_match_offset + (85 * 60),
            "desc": "Chelsea comeback: Sancho(65'), Palmer x2(74',79'), Fernandez(83')"
        },
    ]

    print(f"Creating {len(clips)} clips...")
    print("=" * 60)

    created = []
    for clip in clips:
        if clip["start"] >= info["duration_sec"]:
            print(f"\nSkipping (beyond video): {clip['desc']}")
            continue

        end = min(clip["end"], info["duration_sec"])
        print(f"\n[{clips.index(clip)+1}/{len(clips)}] {clip['desc']}")

        try:
            clip_video_opencv(input_path, clip["name"], clip["start"], end)
            created.append(clip["name"])
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n{'='*60}")
    print(f"Created {len(created)} clips successfully!\n")
    print("Next steps - run analytics on each clip:")
    print()
    for clip_path in created:
        print(f"  python main_v3.py --config match_configs/tottenham_vs_chelsea_dec2024.json --video {clip_path}")

    print(f"\nQuickest test (5 min clip):")
    if created:
        print(f"  python main_v3.py --config match_configs/tottenham_vs_chelsea_dec2024.json --video {created[0]}")

    return created


def main():
    parser = argparse.ArgumentParser(
        description="Video Download and Clipping Tool for Sports Analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Chicago FC United U19 playlist workflow:

  # List all 35 games (no download):
  python download_and_clip.py --list-playlist "https://youtube.com/playlist?list=PLIVc4N0HjStpJ5SDDk-6cWfFUQmoE6ylD"

  # Download a single game:
  python download_and_clip.py --url "https://youtu.be/ktFPvKw5rZM" --download --output input_videos/chicago_vs_scwave.mp4

  # Download the whole playlist (resumable — re-run to continue):
  python download_and_clip.py --download-playlist "https://youtube.com/playlist?list=PLIVc4N0HjStpJ5SDDk-6cWfFUQmoE6ylD" --output-dir input_videos/chicago_u19

  # Create 5-min test clip + both halves from a downloaded game:
  python download_and_clip.py --input input_videos/chicago_vs_scwave.mp4 --create-clips --clip-prefix scwave --pre-match-offset 5:00

  # Run analytics:
  python main_v3.py --config match_configs/chicago_fc_united_u19.json --video input_videos/scwave_test_5min.mp4
        """
    )

    parser.add_argument("--url", type=str, help="YouTube URL to download")
    parser.add_argument("--download", action="store_true", help="Download single video from --url")
    parser.add_argument("--download-playlist", type=str, metavar="PLAYLIST_URL",
                        help="Download all videos in a YouTube playlist")
    parser.add_argument("--list-playlist", type=str, metavar="PLAYLIST_URL",
                        help="List all videos in a playlist without downloading")
    parser.add_argument("--output-dir", type=str, default="input_videos/playlist",
                        help="Folder for --download-playlist output (default: input_videos/playlist)")
    parser.add_argument("--input", type=str, help="Input video path")
    parser.add_argument("--output", type=str,
                        default="input_videos/downloaded.mp4",
                        help="Output path for downloaded single video or custom clip")
    parser.add_argument("--start", type=str, help="Clip start (HH:MM:SS or seconds)")
    parser.add_argument("--end", type=str, help="Clip end (HH:MM:SS or seconds)")
    parser.add_argument("--create-test-clips", action="store_true",
                        help="Create strategic test clips for Spurs vs Chelsea match")
    parser.add_argument("--create-clips", action="store_true",
                        help="Create generic clips (5-min test, first half, second half)")
    parser.add_argument("--clip-prefix", type=str, default="match",
                        help="Filename prefix for --create-clips (default: match)")
    parser.add_argument("--pre-match-offset", type=str, default="0",
                        help="Pre-match content before kickoff (HH:MM:SS or seconds, default: 0)")
    parser.add_argument("--info", type=str, metavar="VIDEO_PATH",
                        help="Show info for a local video file")
    parser.add_argument("--quality", type=str, default="best[height<=1080]",
                        help="yt-dlp quality (default: best up to 1080p)")

    args = parser.parse_args()

    # List playlist (no download)
    if args.list_playlist:
        list_playlist(args.list_playlist)
        return

    # Download entire playlist
    if args.download_playlist:
        download_playlist(args.download_playlist, args.output_dir, args.quality)
        return

    # Show local video info
    if args.info:
        info = get_video_info(args.info)
        size_mb = Path(args.info).stat().st_size / (1024 * 1024)
        print(f"\nVideo: {args.info}")
        print(f"  Resolution: {info['width']}x{info['height']}")
        print(f"  FPS:        {info['fps']:.2f}")
        print(f"  Duration:   {info['duration_hms']} ({info['duration_sec']:.0f}s)")
        print(f"  Frames:     {info['total_frames']:,}")
        print(f"  File size:  {size_mb:.0f} MB")
        return

    # Download single video
    downloaded_path = None
    if args.download:
        if not args.url:
            print("Error: --url required for --download")
            sys.exit(1)
        downloaded_path = download_video(args.url, args.output, args.quality)
    else:
        downloaded_path = args.input

    # Custom clip
    if args.start and args.end:
        if not downloaded_path:
            print("Error: Need --input or --download to clip")
            sys.exit(1)
        start_sec = _hms_to_sec(args.start)
        end_sec = _hms_to_sec(args.end)
        clip_out = args.output if not args.download else f"input_videos/clip_{int(start_sec)}_{int(end_sec)}.mp4"
        clip_video_opencv(downloaded_path, clip_out, start_sec, end_sec)

    # Strategic test clips (Spurs vs Chelsea)
    elif args.create_test_clips:
        if not downloaded_path:
            print("Error: Need --input or --download to create clips")
            sys.exit(1)
        create_spurs_chelsea_clips(downloaded_path)

    # Generic clips (any match)
    elif args.create_clips:
        if not downloaded_path:
            print("Error: Need --input or --download to create clips")
            sys.exit(1)
        offset_sec = int(_hms_to_sec(args.pre_match_offset))
        create_generic_clips(downloaded_path, output_prefix=args.clip_prefix,
                             pre_match_offset=offset_sec)

    elif not any([args.info, args.download, args.download_playlist,
                  args.list_playlist, args.start,
                  args.create_test_clips, args.create_clips]):
        parser.print_help()


if __name__ == "__main__":
    main()
