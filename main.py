from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import extractAudio, crop_video, burn_captions, crop_bottom_video, animate_captions, get_video_dimensions
from Components.Transcription import transcribeAudio, transcribe_segment_word_level
from Components.LanguageTasks import GetHighlights
from Components.FaceCrop import crop_to_vertical_average_face
from Components.Database import VideoDatabase
import os
import traceback

# --- Configuration Flags --- 
# Set to True to use two words-level animated captions (slower but nicer)
# Set to False to use the default, faster ASS subtitle burning
USE_ANIMATED_CAPTIONS = True

# Define the output directory for final shorts
SHORTS_DIR = "shorts"

# Define the crop percentage for the bottom of the video (useful when there are integrated captions in the original video)
CROP_PERCENTAGE_BOTTOM = 0

def process_video(url: str = None, local_path: str = None):
    db = VideoDatabase()
    video_path = None
    video_id = None
    cached_data = None
    is_from_cache = False # Flag to track if using cached video path

    # Ensure the final shorts directory exists
    os.makedirs(SHORTS_DIR, exist_ok=True)

    if not url and not local_path:
        print("Error: Must provide either URL or local path")
        return None

    if url:
        print(f"Processing YouTube URL: {url}")
        cached_data = db.get_cached_processing(youtube_url=url)
        if cached_data:
            print("Found cached video from URL!")
            video_path = cached_data["video"][2]
            video_id = cached_data["video"][0]
            if not os.path.exists(video_path):
                print(f"Cached video path not found: {video_path}. Re-downloading.")
                video_path = None
                cached_data = None
                video_id = None
        else:
                is_from_cache = True # Mark as using cache only if file exists
        
        if not video_path:
            video_path = download_youtube_video(url)
            if not video_path:
                print("Failed to download video")
                return None
            if not video_path.lower().endswith('.mp4'):
                 base, _ = os.path.splitext(video_path)
                 new_path = base + ".mp4"
                 try:
                     os.rename(video_path, new_path)
                     video_path = new_path
                     print(f"Renamed downloaded file to: {video_path}")
                 except OSError as e:
                     print(f"Error renaming file to mp4: {e}. Trying conversion.")
                     pass 

    else:
        print(f"Processing local file: {local_path}")
        if not os.path.exists(local_path):
            print("Error: Local file does not exist")
            return None
        video_path = local_path
        cached_data = db.get_cached_processing(local_path=local_path)
        if cached_data:
            print("Found cached local video!")
            video_id = cached_data["video"][0]
            is_from_cache = True # Mark as cached if DB entry exists for local path

    if not video_path or not os.path.exists(video_path):
        print("No valid video path obtained or file does not exist.")
        return None

    # --- CHECK DIMENSIONS: Initial --- 
    print("\n--- Checking Initial Video Dimensions ---")
    initial_width, initial_height = get_video_dimensions(video_path)
    if initial_width is None or initial_height is None:
        print("Error: Could not determine initial video dimensions. Aborting.")
        return None
    if initial_width < 100 or initial_height < 100: # Basic sanity check
        print(f"Warning: Initial video dimensions ({initial_width}x{initial_height}) seem very small.")
    print("--- Initial Check Done ---")

    # --- Audio Processing (Run this *before* any cropping that might drop audio) --- 
    audio_path = None
    if cached_data and cached_data["video"][3]:
        print("Using cached audio file reference")
        audio_path = cached_data["video"][3]
        if not os.path.exists(audio_path):
            print("Cached audio file not found, extracting again")
            audio_path = None

    if not audio_path:
        print(f"Extracting audio from video: {video_path}")
        audio_path = extractAudio(video_path)
        if not audio_path:
            print("Failed to extract audio")
            return None
        if video_id:
             db.update_video_audio_path(video_id, audio_path)
        else:
             video_id = db.add_video(url, video_path, audio_path)

    if not video_id:
         video_entry = db.get_video(youtube_url=url, local_path=video_path)
         if video_entry:
             video_id = video_entry[0]
         else:
             print("Error: Video ID could not be determined after processing.")
             return None 

    transcriptions = None
    if cached_data and cached_data.get("transcription"):
        print("Using cached segment transcription")
        transcriptions = cached_data["transcription"]

    if not transcriptions:
        print("Generating new segment transcription")
        transcriptions = transcribeAudio(audio_path)
        if transcriptions:
            db.add_transcription(video_id, transcriptions)
        else:
            print("Segment-level transcription failed. Cannot proceed.")
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
        print("Generating new highlights")
        highlights = GetHighlights(TransText)
        if not highlights or len(highlights) == 0:
            print("No valid highlights found")
            return None

        # --- Loop through highlights and process each --- 
        final_output_paths = [] # Store paths of successfully created shorts
        temp_segment = None
        cropped_vertical_temp = None # Intermediate vertical crop before bottom crop
        cropped_vertical_final = None # After bottom crop
        segment_audio_path = None
        transcription_result = None
        
        for i, highlight_data in enumerate(highlights):
            # Reset temp file paths for this loop iteration
            temp_segment = None
            cropped_vertical_temp = None
            cropped_vertical_final = None
            segment_audio_path = None
            transcription_result = None
            
            try:
                # Extract start and end times from the enriched highlight data
                start = float(highlight_data["start"])
                stop = float(highlight_data["end"])
                
                print(f"\n--- Processing Highlight {i+1}/{len(highlights)}: Start={start:.2f}s, End={stop:.2f}s ---")
                if "caption_with_hashtags" in highlight_data:
                    print(f"Caption: {highlight_data['caption_with_hashtags']}")
                
                # --- Define File Paths --- 
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_base = f"{base_name}_highlight_{i+1}"
                temp_segment = os.path.join("videos", f"{output_base}_temp_segment.mp4")
                cropped_vertical_temp = os.path.join("videos", f"{output_base}_vertical_temp.mp4")
                cropped_vertical_final = os.path.join("videos", f"{output_base}_vertical_final.mp4")
                final_output_with_captions = os.path.join(SHORTS_DIR, f"{output_base}_final.mp4")
                segment_audio_path = None # Reset here before checking USE_ANIMATED_CAPTIONS
                if USE_ANIMATED_CAPTIONS:
                    segment_audio_path = os.path.join("videos", f"{output_base}_temp_audio.wav")
                # --- End Define File Paths ---
                
                # --- Processing Steps --- 
                # 1. Extract Segment (Video + Audio, Original Aspect Ratio)
                print(f"1. Extracting segment...")
                extract_success = crop_video(video_path, temp_segment, start, stop, initial_width, initial_height)
                if not extract_success:
                    print(f"Failed step 1 for highlight {i+1}. Skipping.")
                    if os.path.exists(temp_segment): 
                        try: os.remove(temp_segment) 
                        except Exception as clean_e: print(f"Warning: Could not remove temp segment file: {clean_e}")
                    continue
                
                # --- CHECK DIMENSIONS: Segment --- 
                print("\n--- Checking Segment Video Dimensions ---")
                segment_width, segment_height = get_video_dimensions(temp_segment)
                if segment_width is None or segment_height is None:
                    print("Error: Could not determine segment video dimensions. Skipping highlight.")
                    # Cleanup?
                    if os.path.exists(temp_segment): 
                        try: 
                            os.remove(temp_segment) 
                        except Exception as clean_e: 
                            print(f"Warning: Could not remove temp segment file: {clean_e}")
                    continue
                if segment_width != initial_width or segment_height != initial_height:
                     print(f"Warning: Segment dimensions ({segment_width}x{segment_height}) differ from initial ({initial_width}x{initial_height}).")
                     # Decide whether to continue or abort? For now, continue.
                print("--- Segment Check Done ---")
                # --- End Check ---
                
                # 2. Create Vertical Crop (Based on Average Face Position)
                print(f"2. Creating average face centered vertical crop...")
                vert_crop_path = crop_to_vertical_average_face(temp_segment, cropped_vertical_temp)
                if not vert_crop_path:
                    print(f"Failed step 2 (average face crop) for highlight {i+1}. Skipping.")
                    if os.path.exists(temp_segment): 
                        try: 
                            os.remove(temp_segment) 
                        except Exception as clean_e:
                             print(f"Warning: Could not remove temp segment file: {clean_e}")
                    if os.path.exists(cropped_vertical_temp): 
                        try: 
                            os.remove(cropped_vertical_temp) 
                        except Exception as clean_e:
                             print(f"Warning: Could not remove temp vertical crop file: {clean_e}")
                    continue
                cropped_vertical_temp = vert_crop_path # Use returned path

                # 3. Crop Bottom Off Vertical Video (Temporary Fix)
                if CROP_PERCENTAGE_BOTTOM > 0:
                    print(f"3. Applying bottom crop to vertical video...")
                 
                    bottom_crop_success = crop_bottom_video(cropped_vertical_temp, cropped_vertical_final, CROP_PERCENTAGE_BOTTOM)
                    if not bottom_crop_success:
                        print(f"Failed step 3 for highlight {i+1}. Skipping.")
                        # Clean up previous steps
                        if os.path.exists(temp_segment): 

                            try: os.remove(temp_segment) 
                            except: pass
                        if os.path.exists(cropped_vertical_temp):

                            try: os.remove(cropped_vertical_temp) 
                            except: pass
                        if os.path.exists(cropped_vertical_final): 

                            try: os.remove(cropped_vertical_final) 
                            except: pass
                        continue
                else:
                    print("No bottom crop applied")
                    cropped_vertical_final = cropped_vertical_temp


                # 4. Choose Captioning Method
                captioning_success = False
                if USE_ANIMATED_CAPTIONS:
                    print("Attempting Word-Level Animated Captions...")
                    print(f"Extracting audio for segment {i+1}...")
                    segment_audio_extracted_path = extractAudio(temp_segment) 
                    if not segment_audio_extracted_path:
                        print("Failed to extract audio from segment. Cannot perform word transcription. Skipping highlight.")
                        if os.path.exists(temp_segment): 
                            try: os.remove(temp_segment)
                            except Exception as clean_e: print(f"Warning: Could not remove temp segment file: {clean_e}")
                        if os.path.exists(cropped_vertical_temp): 
                            try: os.remove(cropped_vertical_temp)
                            except Exception as clean_e: print(f"Warning: Could not remove cropped vertical file: {clean_e}")
                        if os.path.exists(cropped_vertical_final): 
                            try: os.remove(cropped_vertical_final)
                            except Exception as clean_e: print(f"Warning: Could not remove cropped vertical file: {clean_e}")
                        continue # Skip to next highlight
                    else:
                        segment_audio_path = segment_audio_extracted_path 
                        
                    transcription_result = transcribe_segment_word_level(segment_audio_path)
                    if transcription_result:
                        captioning_success = animate_captions(cropped_vertical_final, temp_segment, transcription_result, final_output_with_captions)
                    else:
                        print("Word-level transcription failed for segment. Skipping animation.")
                        captioning_success = False
                else:
                    print("Using Standard ASS Caption Burning...")
                    captioning_success = burn_captions(cropped_vertical_final, temp_segment, transcriptions, start, stop, final_output_with_captions)
                
                # 5. Handle Captioning Result
                if not captioning_success:
                    print(f"Caption generation failed for highlight {i+1}. Skipping.")
                    # Cleanup intermediate files 
                    if os.path.exists(temp_segment): 
                        try: os.remove(temp_segment)
                        except Exception as clean_e: 
                            print(f"Warning: Could not remove temp segment file: {clean_e}")
                    if os.path.exists(cropped_vertical_temp): # Use correct variable name
                        try: os.remove(cropped_vertical_temp)
                        except Exception as clean_e: 
                            print(f"Warning: Could not remove temp vertical file: {clean_e}")
                    if os.path.exists(cropped_vertical_final): # Use correct variable name
                        try: os.remove(cropped_vertical_final)
                        except Exception as clean_e: 
                            print(f"Warning: Could not remove final vertical file: {clean_e}")
                    if segment_audio_path and os.path.exists(segment_audio_path): 
                        try: os.remove(segment_audio_path)
                        except Exception as clean_e: 
                            print(f"Warning: Could not remove segment audio file: {clean_e}")
                    continue

                # --- Success for this highlight --- 
                print(f"Successfully processed highlight {i+1}.")
                final_output_paths.append(final_output_with_captions)
                print(f"Saving highlight {i+1} info to database: {final_output_with_captions}")
                
                # Get the enriched data from the highlight
                segment_text = highlight_data.get('segment_text', '')
                caption = highlight_data.get('caption_with_hashtags', '')
                
                # Save to database with enriched data
                db.add_highlight(
                    video_id, 
                    start, 
                    stop, 
                    final_output_with_captions,
                    segment_text=segment_text,
                    caption_with_hashtags=caption
                )

                # --- Cleanup Intermediate Files --- 
                print("Cleaning up intermediate files for this highlight...")
                if os.path.exists(temp_segment): 
                    try: os.remove(temp_segment)
                    except Exception as clean_e: 
                        print(f"Warning: Could not remove temp segment file: {clean_e}")
                if os.path.exists(cropped_vertical_temp): # Use correct variable name 
                    try: os.remove(cropped_vertical_temp)
                    except Exception as clean_e: 
                        print(f"Warning: Could not remove temp vertical file: {clean_e}")
                if os.path.exists(cropped_vertical_final): # Use correct variable name
                    try: os.remove(cropped_vertical_final)
                    except Exception as clean_e: 
                        print(f"Warning: Could not remove final vertical file: {clean_e}") 
                if segment_audio_path and os.path.exists(segment_audio_path): 
                    try: os.remove(segment_audio_path)
                    except Exception as clean_e: 
                        print(f"Warning: Could not remove segment audio file: {clean_e}")
            
            except Exception as e_inner:
                print(f"\n--- Error processing highlight {i+1} --- ")
                traceback.print_exc()
                print(f"Continuing to next highlight if available.")
                # Attempt cleanup even on inner loop error
                if temp_segment and os.path.exists(temp_segment): 
                    try: os.remove(temp_segment)
                    except Exception as clean_e: 
                        print(f"Warning: Could not remove temp segment file: {clean_e}")
                if cropped_vertical_temp and os.path.exists(cropped_vertical_temp): # Use correct variable name
                    try: os.remove(cropped_vertical_temp)
                    except Exception as clean_e: 
                        print(f"Warning: Could not remove temp vertical file: {clean_e}")
                if cropped_vertical_final and os.path.exists(cropped_vertical_final): # Use correct variable name
                    try: os.remove(cropped_vertical_final)
                    except Exception as clean_e: 
                        print(f"Warning: Could not remove final vertical file: {clean_e}")
                if segment_audio_path and os.path.exists(segment_audio_path): 
                    try: os.remove(segment_audio_path)
                    except Exception as clean_e: 
                        print(f"Warning: Could not remove segment audio file: {clean_e}")
        # --- End of loop for processing highlights --- 

        if not final_output_paths:
             print("\nProcessing finished, but no highlight segments were successfully converted.")
             return None
        else:
             print(f"\nProcessing finished. Generated {len(final_output_paths)} shorts in '{SHORTS_DIR}' directory.")
             # Return the list of paths or just the first one?
             # Returning list is more informative.
             return final_output_paths 

    except Exception as e:
        print(f"Error in overall highlight processing: {str(e)}")
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
        # If output is a list (multiple shorts generated)
        if isinstance(output, list):
            print(f"\nSuccess! Output saved to:")
            for path in output:
                print(f"- {path}")
        else: # Should not happen with current logic, but handle just in case
            print(f"\nSuccess! Output saved to: {output}")
    else:
        print("\nProcessing failed or no shorts generated!")
