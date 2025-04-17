from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip
import subprocess
import math
import tempfile
import os
import shlex
import json # For ffprobe output

# Import caption functions from the new module
from .Captions import burn_captions, animate_captions

def extractAudio(video_path):
    try:
        video_clip = VideoFileClip(video_path)
        audio_path = "audio.wav"
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()
        print(f"Extracted audio to: {audio_path}")
        return audio_path
    except Exception as e:
        print(f"An error occurred while extracting audio: {e}")
        return None


def crop_video(input_file, output_file, start_time, end_time, original_width, original_height):
    """Extracts a video segment using FFmpeg, ensuring original resolution."""
    try:
        duration = end_time - start_time
        if duration <= 0:
            print("Error: End time must be after start time for cropping.")
            return False
            
        ffmpeg_command = [
            'ffmpeg',
            '-ss', str(start_time), 
            '-i', input_file,
            '-t', str(duration),
            '-map', '0:v:0',
            '-map', '0:a:0',
            '-c:v', 'libx264',  
            '-preset', 'medium',
            '-crf', '23',
            # Explicitly set output size to match original
            '-s', f'{original_width}x{original_height}', 
            '-c:a', 'aac',      
            '-b:a', '128k',
            '-sn',
            '-y',
            output_file
        ]
        
        print("Running FFmpeg command for segment extraction (crop_video):")
        cmd_string = ' '.join([str(arg) for arg in ffmpeg_command])
        print(f"Command: {cmd_string}")

        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"Successfully extracted segment to: {output_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg during segment extraction: {e}")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"An error occurred during segment extraction: {e}")
        return False

# Function to format time in SRT format
def format_time(seconds):
    milliseconds = int((seconds - math.floor(seconds)) * 1000)
    seconds = int(math.floor(seconds))
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# Function to format time in ASS format (hours:mm:ss.cc)
def format_time_ass(seconds):
    centiseconds = int((seconds - math.floor(seconds)) * 100)
    seconds = int(math.floor(seconds))
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    return f"{hours:d}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

def crop_bottom_video(input_path, output_path, crop_percentage_bottom):
    """Crops a percentage from the bottom of the video using FFmpeg."""
    try:
        if not 0 < crop_percentage_bottom < 1:
            print("Error: Crop percentage must be between 0 and 1 (exclusive).")
            return False
            
        height_multiplier = 1.0 - crop_percentage_bottom
        
        ffmpeg_command = [
            'ffmpeg',
            '-i', input_path,
            # vf filter: keep original width (iw), calculate new height based on percentage, ensure it's even, crop from top-left (0,0)
            '-vf', f'crop=iw:floor(ih*{height_multiplier}/2)*2:0:0',
            '-c:v', 'libx264', # Re-encode video
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'copy',   # Copy existing audio stream
            '-y',           # Overwrite output file
            output_path
        ]

        print("Running FFmpeg command to crop bottom of video:")
        cmd_string = ' '.join([str(arg) for arg in ffmpeg_command])
        print(f"Command: {cmd_string}")

        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"Successfully cropped bottom off video to: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg during bottom crop: {e}")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")
        # Clean up potentially incomplete output file
        if os.path.exists(output_path):
             try: os.remove(output_path) 
             except: pass
        return False
    except Exception as e:
        print(f"An unexpected error occurred during bottom cropping: {e}")
        # Clean up potentially incomplete output file
        if os.path.exists(output_path):
             try: os.remove(output_path) 
             except: pass
        return False

def get_video_dimensions(video_path):
    """Gets the width and height of a video file using ffprobe."""
    try:
        print(f"Checking dimensions for: {video_path}")
        if not os.path.exists(video_path):
            print("  Error: File not found.")
            return None, None
            
        ffprobe_command = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0', # Select the first video stream
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(ffprobe_command, check=True, capture_output=True, text=True)
        output_json = json.loads(result.stdout)
        
        if output_json and 'streams' in output_json and len(output_json['streams']) > 0:
            width = output_json['streams'][0].get('width')
            height = output_json['streams'][0].get('height')
            if width is not None and height is not None:
                 print(f"  Dimensions found: {width}x{height}")
                 return int(width), int(height)
            else:
                 print("  Error: Could not find width/height in ffprobe stream data.")
                 return None, None
        else:
            print("  Error: No video streams found by ffprobe or invalid JSON output.")
            print(f"  ffprobe output: {output_json}")
            return None, None
            
    except subprocess.CalledProcessError as e:
        print(f"  Error running ffprobe: {e}")
        print(f"  ffprobe stderr: {e.stderr}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"  Error decoding ffprobe JSON output: {e}")
        if 'result' in locals(): print(f"  Raw ffprobe output: {result.stdout}")
        return None, None
    except Exception as e:
        print(f"  An unexpected error occurred getting dimensions: {e}")
        return None, None

# Example usage:
# if __name__ == "__main__":
#    # ... (old example usage)

