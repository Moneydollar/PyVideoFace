from yt_dlp import YoutubeDL
import os

def DownloadVideo(video_url):
    output_directory = '../videos'

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Define yt-dlp options
    ydl_opts = {
        'quiet': False,
        'verbose': True,
        'noplaylist': True,
        'outtmpl': os.path.join(output_directory, '%(title)s.%(ext)s'),
        # Format options to get video without audio
        'format': 'bestvideo[ext=mp4]',  # Get best video quality in mp4 format
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',  # Ensure final format is mp4
        }],
        'no_warnings': True,
    }

    # Download the video using yt-dlp
    with YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])
        except Exception as e:
            print(f"Error downloading video: {e}")
        else:
            print("Downloaded video")

