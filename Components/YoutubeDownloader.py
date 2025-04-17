import os
from pytubefix import YouTube
import ffmpeg
import subprocess


def get_video_size(stream):
    return stream.filesize / (1024 * 1024)


def download_youtube_video(url):
    try:
        yt = YouTube(url, "WEB")

        video_streams = yt.streams.filter(type="video").order_by("resolution").desc()
        audio_stream = yt.streams.filter(only_audio=True).first()

        print("Available video streams:")
        for i, stream in enumerate(video_streams):
            size = get_video_size(stream)
            stream_type = "Progressive" if stream.is_progressive else "Adaptive"
            print(
                f"{i}. Resolution: {stream.resolution}, Size: {size:.2f} MB, Type: {stream_type}"
            )

        choice = int(input("Enter the number of the video stream to download: "))
        selected_stream = video_streams[choice]
        
        # DEBUG: Print selected stream info
        print(f"DEBUG: Selected Stream Info:")
        print(f"  Resolution: {getattr(selected_stream, 'resolution', 'N/A')}")
        print(f"  FPS: {getattr(selected_stream, 'fps', 'N/A')}")
        print(f"  Mime Type: {getattr(selected_stream, 'mime_type', 'N/A')}")
        print(f"  Is Progressive: {getattr(selected_stream, 'is_progressive', 'N/A')}")
        # Try accessing width/height directly if available
        selected_width = getattr(selected_stream, 'width', None)
        selected_height = getattr(selected_stream, 'height', None)
        print(f"  Reported Width: {selected_width}")
        print(f"  Reported Height: {selected_height}")

        if not os.path.exists("videos"):
            os.makedirs("videos")

        print(f"Downloading video: {yt.title}")
        video_file = selected_stream.download(
            output_path="videos", filename_prefix="video_"
        )

        if not selected_stream.is_progressive:
            print("Downloading audio...")
            audio_file = audio_stream.download(
                output_path="videos", filename_prefix="audio_"
            )

            print("Merging video and audio...")
            merged_output_file = os.path.join("videos", f"{yt.title}_merged.mp4")
            output_file = merged_output_file
            
            # Prepare ffmpeg inputs
            in_video = ffmpeg.input(video_file)
            in_audio = ffmpeg.input(audio_file)
            
            # Prepare ffmpeg output arguments
            output_args = {
                "vcodec": "libx264",
                "acodec": "aac",
                "strict": "experimental",
                "crf": "23", # Reasonable quality
                "preset": "medium" # Encoding speed
            }
            
            # Add size option ONLY if width and height were found
            if selected_width and selected_height:
                print(f"  Setting merge output size to: {selected_width}x{selected_height}")
                output_args['s'] = f'{selected_width}x{selected_height}'
            else:
                print("  Warning: Could not determine resolution from selected stream, merge might use default size.")

            # Create the output stream specification
            merged_stream = ffmpeg.output(
                in_video,         # Map video from video file
                in_audio,         # Map audio from audio file
                merged_output_file,
                **output_args     # Pass arguments dictionary
            )
            
            # Run the merge command
            print(f"  Running merge command: {ffmpeg.compile(merged_stream)}")
            ffmpeg.run(merged_stream, overwrite_output=True)

            os.remove(video_file)
            os.remove(audio_file)
        else:
            output_file = video_file

        print(f"Downloaded: {yt.title} to 'videos' folder")
        print(f"File path: {output_file}")
        return output_file

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(
            "Please make sure you have the latest version of pytube and ffmpeg-python installed."
        )
        print("You can update them by running:")
        print("pip install --upgrade pytube ffmpeg-python")
        print(
            "Also, ensure that ffmpeg is installed on your system and available in your PATH."
        )
        return None


if __name__ == "__main__":
    youtube_url = input("Enter YouTube video URL: ")
    # Just call the download function for direct testing
    downloaded_file = download_youtube_video(youtube_url)
    
    if downloaded_file:
        print(f"\nDownload finished. File available at: {downloaded_file}")
    else:
        print("\nDownload failed.")
