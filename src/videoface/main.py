#!/usr/bin/python3

import typer
from typing import List
from .detect import *
from .downloadVideo import *
import logging
from urllib.parse import unquote

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = typer.Typer()


def clean_url(url: str) -> str:
    """Clean URL by removing escape characters and unquoting"""
    return unquote(url.replace('\\', ''))


@app.command()
def process(
        videos: List[str] = typer.Argument(
            ...,
            help="YouTube video URLs to process",
            callback=lambda urls: [clean_url(url) for url in urls]
        )
):
    """Process each video for face detection."""
    if not videos:
        logging.error("No video URLs provided")
        raise typer.BadParameter("Please provide at least one video URL")

    logging.info(f"Processing {len(videos)} videos")

    # Print cleaned URLs for verification
    for url in videos:
        logging.info(f"Processing URL: {url}")

    # Download videos
    for video in videos:
        try:
            logging.info(f"Downloading video: {video}")
            DownloadVideo(video)
        except Exception as e:
            logging.error(f"Failed to download video {video}: {str(e)}")
            continue

    # Process downloaded videos
    video_dir = "../videos"
    if not os.path.exists(video_dir):
        logging.error(f"Video directory not found: {video_dir}")
        return

    video_paths = os.listdir(video_dir)
    if not video_paths:
        logging.warning("No videos found in directory after download")
        return

    for video_name in video_paths:
        print(video_name)
        video_path = os.path.join(video_dir, video_name)
        logging.info(f"Processing video: {video_name}")

        try:
            frames_output_dir, frames_with_boxes_dir, faces_dir = setup(video_path)
            videoFrames(video_path, frames_output_dir)
            detectFaces(frames_output_dir, frames_with_boxes_dir, faces_dir)
            cleanUp(frames_output_dir, frames_with_boxes_dir)
        except Exception as e:
            logging.error(f"Error processing {video_name}: {str(e)}")


if __name__ == "__main__":
    app()