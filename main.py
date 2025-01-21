from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import extractAudio, crop_video
from Components.Transcription import transcribeAudio
from Components.LanguageTasks import GetHighlights
from Components.FaceCrop import crop_to_vertical, combine_videos
from Components.Database import VideoDatabase
import os


def process_video(url: str = None, local_path: str = None):
    db = VideoDatabase()
    video_path = None
    video_id = None

    if not url and not local_path:
        print("Error: Must provide either URL or local path")
        return None

    if url:
        print(f"Processing YouTube URL: {url}")
        # Check if we already have this video
        cached_data = db.get_cached_processing(youtube_url=url)
        if cached_data:
            print("Found cached video from URL!")
            video_path = cached_data["video"][2]  # local_path from video table
            video_id = cached_data["video"][0]  # id from video table
        else:
            # Download new video
            video_path = download_youtube_video(url)
            if not video_path:
                print("Failed to download video")
                return None
            video_path = video_path.replace(".webm", ".mp4")
            print(f"Downloaded video to: {video_path}")

    else:
        print(f"Processing local file: {local_path}")
        if not os.path.exists(local_path):
            print("Error: Local file does not exist")
            return None

        # Check if we already have this local file
        cached_data = db.get_cached_processing(local_path=local_path)
        if cached_data:
            print("Found cached local video!")
            video_path = local_path
            video_id = cached_data["video"][0]
        else:
            video_path = local_path

    # Ensure we have a valid video path
    if not video_path:
        print("No valid video path obtained")
        return None

    # Audio Processing
    audio_path = None
    if cached_data and cached_data["video"][3]:  # audio_path from video table
        print("Using cached audio file")
        audio_path = cached_data["video"][3]
        if not os.path.exists(audio_path):
            print("Cached audio file not found, extracting again")
            audio_path = None

    if not audio_path:
        print("Extracting audio from video")
        audio_path = extractAudio(video_path)
        if not audio_path:
            print("Failed to extract audio")
            return None

    if not video_id:
        video_id = db.add_video(url, video_path, audio_path)

    transcriptions = None
    if cached_data and cached_data.get("transcription"):
        print("Using cached transcription")
        transcriptions = cached_data["transcription"]

    if not transcriptions:
        print("Generating new transcription")
        transcriptions = transcribeAudio(audio_path)
        if transcriptions:
            db.add_transcription(video_id, transcriptions)
        else:
            print("Transcription failed")
            return None

    # Format transcription text
    TransText = ""
    for text, start, end in transcriptions:
        start_time = float(start)
        end_time = float(end)
        TransText += f"[{start_time:.2f}] Speaker: {text.strip()} [{end_time:.2f}]\n"

    print("\nFirst 200 characters of transcription:")
    print(TransText[:200] + "...")

    # Highlight Processing
    try:
        # NOTE: Highlight caching disabled for debugging
        # if cached_data and cached_data.get("highlights"):
        #     print("Using cached highlights")
        #     highlight = cached_data["highlights"][0]
        #     start, stop = highlight[0], highlight[1]
        # else:
        print("Generating new highlights")
        highlights = GetHighlights(TransText)
        if not highlights or len(highlights) == 0:
            print("No valid highlights found")
            return None

        start, stop = highlights[0]
        if not isinstance(start, (int, float)) or not isinstance(stop, (int, float)):
            print(f"Invalid timestamp types: start={type(start)}, stop={type(stop)}")
            return None

        print(f"Processing highlight: Start: {start:.2f}, End: {stop:.2f}")

        # Video Processing
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_base = f"processed_{base_name}"
        temp_output = f"{output_base}_temp.mp4"
        cropped = f"{output_base}_cropped.mp4"
        final_output = f"{output_base}_final.mp4"

        # Process video segments
        print("Cropping video...")
        crop_video(video_path, temp_output, start, stop)
        print("Creating vertical crop...")
        crop_to_vertical(temp_output, cropped)
        print("Combining videos...")
        combine_videos(temp_output, cropped, final_output)

        # Cleanup
        if os.path.exists(temp_output):
            os.remove(temp_output)
        if os.path.exists(cropped):
            os.remove(cropped)

        # Save to database
        db.add_highlight(video_id, start, stop, final_output)
        return final_output

    except Exception as e:
        print(f"Error in highlight processing: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\nVideo Processing Options:")
    print("1. Process YouTube URL")
    print("2. Process Local File")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        url = input("Enter YouTube URL: ")
        output = process_video(url=url)
    elif choice == "2":
        local_file = input("Enter path to local video file: ")
        output = process_video(local_path=local_file)
    else:
        print("Invalid choice")
        output = None

    if output:
        print(f"\nSuccess! Output saved to: {output}")
    else:
        print("\nProcessing failed!")
