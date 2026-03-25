import cv2
print(cv2.__version__)

def read_video(video_path):
    """
    Read entire video into memory (use for short videos only!)

    For long videos, use read_video_chunks() instead
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def get_video_properties(video_path):
    """Get video properties without loading frames"""
    cap = cv2.VideoCapture(video_path)
    props = {
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration_sec': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    cap.release()
    return props

def read_video_chunks(video_path, chunk_size=100):
    """
    Generator that yields video frames in chunks

    Args:
        video_path: Path to video file
        chunk_size: Number of frames per chunk

    Yields:
        (chunk_frames, chunk_start_index) tuples
    """
    cap = cv2.VideoCapture(video_path)
    chunk_index = 0

    while True:
        chunk_frames = []
        chunk_start = chunk_index * chunk_size

        for i in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            chunk_frames.append(frame)

        if len(chunk_frames) == 0:
            break

        yield chunk_frames, chunk_start

        chunk_index += 1

    cap.release()

def save_video(output_video_frames, output_video_path, fps=24):
    """Save frames to video file"""
    if len(output_video_frames) == 0:
        raise ValueError("No frames to save")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for .mp4 output
    height, width = output_video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in output_video_frames:
        out.write(frame)

    out.release()
    print(f"Saved {len(output_video_frames)} frames to {output_video_path}")

def save_video_streaming(output_video_path, fps, width, height):
    """
    Create a video writer for streaming output (chunked processing)

    Returns a VideoWriter object that can be written to incrementally
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))