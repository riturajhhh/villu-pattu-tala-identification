"""
utils/youtube_downloader.py
===========================
Helper utility to download audio from YouTube links using yt-dlp.
Requires ffmpeg installed on the host system.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

sys_path = str(Path(__file__).parent.parent)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from utils.logger import get_logger

logger = get_logger("villu_pattu.youtube")


def download_youtube_audio(
    url: str,
    output_dir: str | Path,
    file_id: str,
) -> Optional[Path]:
    """Download audio from a YouTube URL and extract to WAV.

    Parameters
    ----------
    url:
        The YouTube video URL.
    output_dir:
        Directory where the WAV file will be saved.
    file_id:
        Unique name/ID to use for the downloaded file.

    Returns
    -------
    Path to the extracted WAV file or None if download failed.
    """
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp is not installed. Please install it using pip.")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Template format path for download
    outtmpl_path = str(output_dir / f"{file_id}.%(ext)s")

    base_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    # Try strategies in order: without cookies first, then with browser cookies
    strategies = [
        ("no cookies", {}),
        ("edge cookies", {"cookiesfrombrowser": ("edge", )}),
    ]

    for strategy_name, extra_opts in strategies:
        opts = {**base_opts, **extra_opts}
        try:
            logger.info(f"Downloading YouTube audio ({strategy_name}): {url}")
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                expected_wav = output_dir / f"{file_id}.wav"

                if expected_wav.exists():
                    logger.info(f"Successfully downloaded and extracted audio: {expected_wav}")
                    return expected_wav
                else:
                    logger.error("Audio downloaded but extracted WAV file not found.")
                    return None

        except Exception as exc:
            logger.warning(f"Strategy '{strategy_name}' failed for {url}: {exc}")
            continue

    logger.error(f"All download strategies failed for {url}")
    return None
