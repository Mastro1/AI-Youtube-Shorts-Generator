import cv2
import numpy as np
from moviepy.editor import *
# Note: detect_faces_and_speakers and Frames are no longer used by crop_to_vertical_static
# from Components.Speaker import detect_faces_and_speakers, Frames
global Fps

def crop_to_vertical_static(input_video_path, output_video_path):
    """Crops the video to a 9:16 aspect ratio using a static centered crop."""
    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None # Return None on failure

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_height == 0 or fps == 0:
        print("Error: Video properties (height/fps) are invalid.")
        cap.release()
        return None

    # Calculate target 9:16 width based on original height
    vertical_height = original_height
    vertical_width = int(vertical_height * 9 / 16)

    # Ensure the calculated width is even (required by some codecs)
    if vertical_width % 2 != 0:
        vertical_width -= 1 

    print(f"Original Dims: {original_width}x{original_height} @ {fps:.2f}fps")
    print(f"Target Vertical Dims: {vertical_width}x{vertical_height}")

    if original_width < vertical_width or vertical_width <= 0:
        print("Error: Original video width is less than the calculated vertical width or width is invalid.")
        cap.release()
        return None

    # Calculate static horizontal crop start/end points (centered)
    x_start = (original_width - vertical_width) // 2
    x_end = x_start + vertical_width
    print(f"Static Crop Range (Horizontal): {x_start} to {x_end}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for .mp4 output
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (vertical_width, vertical_height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        cap.release()
        return None
        
    # Set global Fps (if needed elsewhere, otherwise consider removing global)
    global Fps
    Fps = fps

    processed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Apply the static crop
        cropped_frame = frame[:, x_start:x_end]

        # Basic check in case cropping resulted in unexpected shape
        if cropped_frame.shape[1] != vertical_width or cropped_frame.shape[0] != vertical_height:
             print(f"Warning: Cropped frame shape {cropped_frame.shape} doesn't match target {vertical_width}x{vertical_height}. Adjusting...")
             # Attempt to resize, though this indicates an issue upstream or with calculations
             cropped_frame = cv2.resize(cropped_frame, (vertical_width, vertical_height))

        out.write(cropped_frame)
        processed_frames += 1

    print(f"Processed {processed_frames}/{total_frames} frames.")
    cap.release()
    out.release()
    print(f"Static vertical cropping complete. Video saved to: {output_video_path}")
    return output_video_path # Return path on success


def crop_to_vertical(input_video_path, output_video_path):
    # detect_faces_and_speakers(input_video_path, "DecOut.mp4")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vertical_height = int(original_height)
    vertical_width = int(vertical_height * 9 / 16)
    print(vertical_height, vertical_width)


    if original_width < vertical_width:
        print("Error: Original video width is less than the desired vertical width.")
        return

    x_start = (original_width - vertical_width) // 2
    x_end = x_start + vertical_width
    print(f"start and end - {x_start} , {x_end}")
    print(x_end-x_start)
    half_width = vertical_width // 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (vertical_width, vertical_height))
    global Fps
    Fps = fps
    print(fps)
    count = 0
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) >-1:
            if len(faces) == 0:
                (x, y, w, h) = Frames[count]

            # (x, y, w, h) = faces[0]  
            try:
                #check if face 1 is active
                (X, Y, W, H) = Frames[count]
            except Exception as e:
                print(e)
                (X, Y, W, H) = Frames[count][0]
                print(Frames[count][0])
            
            for f in faces:
                x1, y1, w1, h1 = f
                center = x1+ w1//2
                if center > X and center < X+W:
                    x = x1
                    y = y1
                    w = w1
                    h = h1
                    break

            # print(faces[0])
            centerX = x+(w//2)
            print(centerX)
            print(x_start - (centerX - half_width))
            if count == 0 or (x_start - (centerX - half_width)) <1 :
                ## IF dif from prev fram is low then no movement is done
                pass #use prev vals
            else:
                x_start = centerX - half_width
                x_end = centerX + half_width


                if int(cropped_frame.shape[1]) != x_end- x_start:
                    if x_end < original_width:
                        x_end += int(cropped_frame.shape[1]) - (x_end-x_start)
                        if x_end > original_width:
                            x_start -= int(cropped_frame.shape[1]) - (x_end-x_start)
                    else:
                        x_start -= int(cropped_frame.shape[1]) - (x_end-x_start)
                        if x_start < 0:
                            x_end += int(cropped_frame.shape[1]) - (x_end-x_start)
                    print("Frame size inconsistant")
                    print(x_end- x_start)

        count += 1
        cropped_frame = frame[:, x_start:x_end]
        if cropped_frame.shape[1] == 0:
            x_start = (original_width - vertical_width) // 2
            x_end = x_start + vertical_width
            cropped_frame = frame[:, x_start:x_end]
        
        print(cropped_frame.shape)

        out.write(cropped_frame)

    cap.release()
    out.release()
    print("Cropping complete. The video has been saved to", output_video_path, count)


# --- New Function: Average Face Centered Crop --- 

def crop_to_vertical_average_face(input_video_path, output_video_path, sample_interval_seconds=0.5):
    """Crops video to 9:16 based on the average horizontal face position sampled periodically."""
    print("Starting average face centered vertical crop...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return None

    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_height <= 0 or fps <= 0:
        print("Error: Invalid video properties (height or fps <= 0).")
        cap.release()
        return None
        
    print(f"Input: {original_width}x{original_height} @ {fps:.2f}fps")

    # Calculate target 9:16 width
    vertical_height = original_height
    vertical_width = int(vertical_height * 9 / 16)
    if vertical_width % 2 != 0: vertical_width -= 1

    if original_width < vertical_width or vertical_width <= 0:
        print("Error: Original width too small for vertical crop.")
        cap.release()
        return None

    print(f"Target Vertical Dims: {vertical_width}x{vertical_height}")

    # --- First Pass: Sample face positions --- 
    face_centers_x = []
    frames_to_skip = int(fps * sample_interval_seconds)
    if frames_to_skip < 1: frames_to_skip = 1 # Sample at least every frame if interval is too small
    
    frame_count = 0
    print(f"Sampling face position every {frames_to_skip} frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frames_to_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Adjust detection parameters if needed (e.g., scaleFactor, minNeighbors)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            if len(faces) > 0:
                # Assume the largest face is the main one if multiple are detected
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]
                centerX = x + w / 2
                face_centers_x.append(centerX)
                # Optional: Draw box on sample frame for debugging
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # cv2.imshow('Sample', frame); cv2.waitKey(1)
                
        frame_count += 1
        
    # --- Calculate Average Position --- 
    average_face_center_x = None
    if face_centers_x:
        average_face_center_x = np.mean(face_centers_x)
        print(f"Found {len(face_centers_x)} face samples. Average center X: {average_face_center_x:.2f}")
    else:
        print("Warning: No faces detected during sampling. Falling back to frame center.")
        average_face_center_x = original_width / 2

    # --- Calculate Static Crop Box --- 
    half_vertical_width = vertical_width // 2
    x_start = int(average_face_center_x - half_vertical_width)
    x_end = x_start + vertical_width

    # Clamp crop box to frame boundaries
    x_start = max(0, x_start)
    x_end = min(original_width, x_end)

    # Adjust x_start if clamping x_end changed the width
    if x_end - x_start != vertical_width:
         x_start = x_end - vertical_width
         x_start = max(0, x_start) # Re-clamp x_start just in case
    
    # Final check if width calculation is still correct after clamping
    if x_end - x_start != vertical_width:
        print(f"Error: Could not calculate valid crop window ({x_start}-{x_end}) for width {vertical_width}. Check logic.")
        cap.release()
        return None
        
    print(f"Calculated Static Crop Box: X = {x_start} to {x_end}")

    # --- Second Pass: Apply crop and write video --- 
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind video capture
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (vertical_width, vertical_height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        cap.release()
        return None

    print("Applying static crop and writing output video...")
    written_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cropped_frame = frame[:, x_start:x_end]
        
        # Sanity check shape before writing (optional but good)
        if cropped_frame.shape[1] != vertical_width or cropped_frame.shape[0] != vertical_height:
            print(f"Warning: Frame {written_frames} cropped shape {cropped_frame.shape} != target {vertical_width}x{vertical_height}. Resizing.")
            cropped_frame = cv2.resize(cropped_frame, (vertical_width, vertical_height))
            
        out.write(cropped_frame)
        written_frames += 1

    print(f"Finished writing {written_frames} frames.")
    cap.release()
    out.release()
    print(f"Average face centered vertical crop complete. Saved to: {output_video_path}")
    return output_video_path





