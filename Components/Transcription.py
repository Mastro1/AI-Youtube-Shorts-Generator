from faster_whisper import WhisperModel
import whisper_timestamped as whisper
import torch
import os


def transcribeAudio(audio_path):
    try:
        print("Transcribing audio (segment-level)...")

        # Optimize CPU settings for i5
        cpu_threads = 10  # Use most of available threads
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)

        model = WhisperModel(
            "base.en",
            device="cpu",
            compute_type="int8",  # Use int8 quantization for faster CPU processing
            cpu_threads=cpu_threads,
            num_workers=2,
        )

        print("Faster-Whisper model loaded")

        segments, info = model.transcribe(
            audio=audio_path,
            beam_size=3,
            language="en",
            max_new_tokens=128,
            condition_on_previous_text=False,
        )

        segments = list(segments)
        extracted_texts = [
            [segment.text, float(segment.start), float(segment.end)] for segment in segments
        ]
        return extracted_texts

    except Exception as e:
        print("Segment-level Transcription Error:", e)
        return []


def transcribe_segment_word_level(audio_path):
    """Generates structured word-level timestamps for a given audio segment."""
    try:
        print(f"Loading whisper-timestamped model for word timings...")
        model = whisper.load_model("base.en", device="cpu") # Or choose another model size
        
        print(f"Transcribing audio for word timestamps: {audio_path}")
        # Keep alignment_heads=True for potentially better timing if needed
        result = whisper.transcribe(model, audio_path, language="en", vad=True, detect_disfluencies=True) 
        
        # Return the raw result dictionary which contains segments and words
        if result and result.get("segments"):
             print(f"Generated word timestamps (structured). Found {len(result['segments'])} segments.")
             return result # Return the whole dictionary
        else:
             print("Warning: whisper-timestamped did not return valid segments.")
             return None

    except FileNotFoundError:
        print(f"Error: Audio file not found for word transcription: {audio_path}")
        return None
    except Exception as e:
        print(f"Word-level Transcription Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    audio_path = "audio.wav"
    
    print("--- Testing Segment Transcription ---")
    segment_transcriptions = transcribeAudio(audio_path)
    if segment_transcriptions:
        TransText = ""
        for text, start, end in segment_transcriptions:
            TransText += f"{start:.2f} - {end:.2f}: {text}\n"
        print(TransText)
    else:
        print("Segment transcription failed.")
        
    print("\n--- Testing Word Transcription ---")
    word_transcription_result = transcribe_segment_word_level(audio_path) 
    if word_transcription_result:
        print("Transcription Result Keys:", word_transcription_result.keys())
        print("Number of segments:", len(word_transcription_result.get('segments', [])))
        if word_transcription_result.get('segments'):
             print("First segment words:")
             first_segment_words = word_transcription_result['segments'][0].get('words', [])
             for i, word_info in enumerate(first_segment_words[:10]):
                  print(f"  {word_info['start']:.2f} - {word_info['end']:.2f}: {word_info['text']}")
    else:
        print("Word transcription failed.")

