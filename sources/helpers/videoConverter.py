"""
iPhone H.265 (10-bit/HDR) -> H.264 8-bit SDR
ffmpeg conversion for OpenCV compatibility on Windows
"""
import subprocess
from pathlib import Path

def convert(input_video):
    input_video = Path(input_video).resolve()
    output_video = input_video.with_suffix(".opencv.mp4")

    print("Input exists:", input_video.exists())
    print("Running ffmpeg...")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_video)
    ], check=True)

    return output_video