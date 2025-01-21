import google.generativeai as genai
from typing import TypedDict, List
import json
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "Google API key not found. Make sure it is defined in the .env file."
    )

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")


class Message(TypedDict):
    role: str
    content: str


class HighlightData(TypedDict):
    start: float
    end: float


def validate_highlight(highlight: HighlightData) -> bool:
    """Validate a single highlight segment."""
    try:
        if not all(key in highlight for key in ["start", "end"]):
            return False

        start = float(highlight["start"])
        end = float(highlight["end"])

        # Check for valid duration (60 seconds with 0.1s tolerance)
        if abs((end - start) - 60.0) > 0.1:
            print("")

        # Check for valid ordering
        if start >= end:
            return False

        return True
    except (ValueError, TypeError):
        return False


def validate_highlights(highlights: List[HighlightData]) -> bool:
    """Validate all highlights and check for overlaps."""
    if not highlights:
        print("No Highlights Passed")
        return False

    # Validate each individual highlight
    if not all(validate_highlight(h) for h in highlights):
        print("Validation Error")
        return False

    # Check for overlapping segments
    sorted_highlights = sorted(highlights, key=lambda x: float(x["start"]))
    for i in range(len(sorted_highlights) - 1):
        if float(sorted_highlights[i]["end"]) > float(
            sorted_highlights[i + 1]["start"]
        ):
            return False

    return True


def extract_highlights(
    transcription: str, max_attempts: int = 3
) -> List[HighlightData]:
    """Extract highlights with retry logic."""
    system_prompt = """
    Analyze the provided transcription and select multiple NON-OVERLAPPING segments that would make engaging longer-form videos. 
    Return ONLY a JSON array. Do not include any explanations, text, or formatting outside JSON.

    CRITICAL REQUIREMENTS:
    1. Time Duration Requirements:
       - EXACTLY 60 seconds between start and end for each segment
       - No overlap between segments
       - Return as many valid 60-second segments as possible
    
    2. Selection Criteria:
       - Choose engaging or impactful continuous segments
       - Must include complete thoughts/sentences
       - Select moments with clear context (avoid starting mid-conversation)
       - Prefer segments with natural breaks or topic transitions
    
    3. Timestamp Accuracy:
       - Use EXACT timestamps from the transcription
       - Do not make up or modify timestamps
       - Start and end times must correspond to actual transcript markers
    
    4. Format Requirements:
       Return ONLY a JSON array of objects in this exact format:
       [{
           "start": <exact_start_timestamp>,
           "end": <exact_end_timestamp>
       }]

    Important VALIDATION:
    - Verify each end_time - start_time is EXACTLY 60 seconds
    - Ensure timestamps match actual transcript markers
    - Verify segments don't overlap

    Return ONLY the JSON. No explanations or additional text.
    """

    for attempt in range(max_attempts):
        try:
            prompt = f"{system_prompt}\n\nTranscription:\n{transcription}"
            response = model.generate_content(prompt)

            # Extract JSON from response
            response_text = response.text
            json_string = response_text.strip("`json\n").strip()
            # print(json_string)
            highlights = json.loads(json_string)
            # print(type(highlights))

            # Validate the highlights
            if validate_highlights(highlights):
                return highlights

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            continue

    return []


def GetHighlights(transcription: str) -> list[tuple[float, float]]:
    """Main function to get multiple 60-second highlights from transcription."""
    try:
        # Clean and validate the input transcription
        if not transcription or not transcription.strip():
            print("Empty transcription")
            return []

        # Prepare initial state with cleaned transcription
        initial_state = {
            "messages": [{"role": "user", "content": transcription.strip()}],
            "highlights": [],
            "error": None,
        }

        # Extract timestamps and text
        highlights = extract_highlights(transcription)

        # Validate the highlights
        if not highlights:
            print("No highlights extracted")
            return []

        # Convert to list of tuples and validate each pair
        result = []
        for h in highlights:
            try:
                start = float(h["start"])
                end = float(h["end"])
                if start >= 0 and end > start:
                    result.append((start, end))
            except (ValueError, KeyError, TypeError) as e:
                print(f"Error processing highlight: {str(e)}")
                continue

        if not result:
            print("No valid highlights found")
            return []

        return result

    except Exception as e:
        print(f"Error in GetHighlights: {str(e)}")
        import traceback

        traceback.print_exc()
        return []


if __name__ == "__main__":
    example_transcription = """
    [0.0] Speaker 1: Welcome to our discussion about artificial intelligence.
    [15.5] Speaker 1: Today we'll explore the fascinating world of machine learning.
    [30.2] Speaker 2: One of the most exciting applications is in video processing.
    [45.8] Speaker 1: Let's look at how AI can automatically generate video highlights.
    [60.0] Speaker 2: This technology is revolutionizing content creation.
    [75.5] Speaker 1: We're seeing it used in social media, entertainment, and education.
    [90.2] Speaker 2: The ability to automatically process and understand video content is remarkable.
    [105.8] Speaker 1: It's changing how we create and consume digital content.
    [120.0] Speaker 2: Let's dive into some specific examples.
    """

    segments = GetHighlights(example_transcription)
    if segments:
        for i, (start, end) in enumerate(segments, 1):
            print(f"Segment {i}: Start: {start}, End: {end}")
    else:
        print("No valid segments found.")
