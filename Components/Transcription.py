from faster_whisper import WhisperModel
import torch
import os


def transcribeAudio(audio_path):
    try:
        print("Transcribing audio...")

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

        print("Model loaded")

        segments, info = model.transcribe(
            audio=audio_path,
            beam_size=3,
            language="en",
            max_new_tokens=128,
            condition_on_previous_text=False,
        )

        segments = list(segments)
        extracted_texts = [
            [segment.text, segment.start, segment.end] for segment in segments
        ]
        return extracted_texts

    except Exception as e:
        print("Transcription Error:", e)
        return []


if __name__ == "__main__":
    audio_path = "audio.wav"
    transcriptions = transcribeAudio(audio_path)
    print("Done")
    TransText = ""

    for text, start, end in transcriptions:
        TransText += f"{start} - {end}: {text}"
    print(TransText)

