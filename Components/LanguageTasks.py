from google import genai
from typing import TypedDict, List, Optional
import json
import os
import re # Import regex for parsing transcription
from dotenv import load_dotenv
from google.genai import types

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "Google API key not found. Make sure it is defined in the .env file."
    )

client = genai.Client(
        api_key=GOOGLE_API_KEY,
    )
# Consider using a more capable model if generating descriptions needs more nuance
# model = genai.GenerativeModel("gemini-1.5-flash") # Example alternative
model = "gemini-2.0-flash"


class Message(TypedDict):
    role: str
    content: str


class HighlightSegment(TypedDict):
    start: float
    end: float

# New type for the enriched highlight data
class EnrichedHighlightData(TypedDict):
    start: float
    end: float
    caption_with_hashtags: str
    segment_text: str # Store the text used for generation


def validate_highlight(highlight: HighlightSegment) -> bool:
    """Validate a single highlight segment's time duration and format."""
    try:
        if not all(key in highlight for key in ["start", "end"]):
            print(f"Validation Fail: Missing 'start' or 'end' key in {highlight}")
            return False

        start = float(highlight["start"])
        end = float(highlight["end"])
        duration = end - start

        # Check for valid duration (between ~29 and ~61 seconds)
        min_duration = 30.0 - 1.0 # Increased tolerance
        max_duration = 60.0 + 1.0 # Increased tolerance

        if not (min_duration <= duration <= max_duration):
            print(f"Validation Fail: Duration {duration:.2f}s out of range [~29s, ~61s] for {highlight}")
            return False

        # Check for valid ordering
        if start >= end:
            print(f"Validation Fail: Start time {start} >= end time {end} for {highlight}")
            return False

        return True
    except (ValueError, TypeError) as e:
        print(f"Validation Fail: Invalid type or value in {highlight} - {e}")
        return False


def validate_highlights(highlights: List[HighlightSegment]) -> bool:
    """Validate all highlights and check for overlaps."""
    if not highlights:
        print("Validation: No highlights provided.")
        return False

    # Validate each individual highlight (already checks duration)
    if not all(validate_highlight(h) for h in highlights):
        # Specific errors printed within validate_highlight
        print("Validation: One or more highlights failed individual checks.")
        return False

    # Check for overlapping segments
    sorted_highlights = sorted(highlights, key=lambda x: float(x["start"]))
    for i in range(len(sorted_highlights) - 1):
        if float(sorted_highlights[i]["end"]) > float(
            sorted_highlights[i + 1]["start"]
        ):
            print(f"Validation Fail: Overlap detected between {sorted_highlights[i]} and {sorted_highlights[i+1]}")
            return False

    return True


def extract_highlights(
    transcription: str, max_attempts: int = 3
) -> List[HighlightSegment]:
    """Extracts highlight time segments from transcription, validates, checks overlaps, with retry logic."""
    # System instruction based on Google AI Studio code
    system_instruction_text = """
Act as a social media content creator. Extract as many non-overlapping segments as possible from the provided transcript that would be suitable and engaging as short social media video clips. Prioritize identifying a variety of valid segments.
Return ONLY a JSON array of objects. Each object should have the keys "start" and "end" containing the exact timestamps from the beginning and end of the segment. Do not include any explanations, text, or formatting outside of the JSON.

Selection Criteria:
◦ Select key points, explanations, questions, conclusions, or generally engaging parts of the conversation.
◦ Try to include complete thoughts or sentences.
◦ Prefer moments with relatively clear context, but prioritize meeting the time constraints if necessary.
◦ Segments with natural breaks or transitions are preferred, but not required.

Duration Requirements:
◦ Ensure that the duration of each segment (end - start) is STRICTLY BETWEEN 30 and 60 seconds (inclusive).
◦ Select non-overlapping segments only.
◦ Find and return between 10 and 20 valid segments that meet this duration requirement — no more.

Timestamp Accuracy:
◦ Use the EXACT timestamps from the transcript.
◦ Do not make up or alter timestamps.

Example JSON output (with sample timestamps):
[
  {
    "start": "8.96",
    "end": "42.20"
  },
  {
    "start": "115.08",
    "end": "156.12"
  },
  {
    "start": "1381.68",
    "end": "1427.40"
  }
]

• Your main goal is to identify multiple segments whose durations are strictly BETWEEN 30 and 60 seconds.
• Verify that timestamps match actual transcript markers or logical boundaries.
• Ensure that segments do not overlap.
    """

    # Define generation config based on AI Studio code
    generation_config = types.GenerateContentConfig(
        temperature=0.2,
        system_instruction=[types.Part.from_text(text=system_instruction_text)]
        # response_mime_type="text/plain" # Not typically set here for generate_content
    )

    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt + 1} to generate and validate highlight time segments...")
        try:
            # Structure the prompt and system instruction for generate_content
            user_prompt_text = f"Transcription:\\n{transcription}"
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt_text)])]
            # Use the global model but with the new config
            response = client.models.generate_content(model=model,contents=contents,
                                              config=generation_config)

            # Basic safety check for response content
            if not response or not response.text:
                 print(f"Attempt {attempt + 1} failed: Empty response from LLM.")
                 continue

            # Extract JSON from response
            response_text = response.text
            # Handle potential markdown code blocks
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
            if match:
                json_string = match.group(1).strip()
            else:
                 # Assume the whole text is JSON if no markdown block found
                 json_string = response_text.strip()

            raw_highlights = json.loads(json_string)

            if not isinstance(raw_highlights, list):
                 print(f"Attempt {attempt + 1} failed: LLM response was not a JSON list.")
                 print(f"Raw LLM Response: {response.text}")
                 continue

            # Filter the highlights: Keep only those passing individual validation
            valid_highlights = [h for h in raw_highlights if validate_highlight(h)]

            if not valid_highlights:
                print("No highlights passed individual duration/format validation in this attempt.")
                continue # Try next attempt

            # Check for overlaps ONLY among the valid duration highlights
            # Sort again just to be sure, although validate_highlights also sorts internally for its check
            sorted_highlights = sorted(valid_highlights, key=lambda x: float(x["start"]))
            overlaps_found = False
            for i in range(len(sorted_highlights) - 1):
                if float(sorted_highlights[i]["end"]) > float(sorted_highlights[i + 1]["start"]):
                    print(f"Overlap detected between {sorted_highlights[i]} and {sorted_highlights[i+1]}")
                    overlaps_found = True
                    break # No need to check further overlaps in this attempt

            if overlaps_found:
                print("Overlap check failed for this attempt.")
                continue # Try next attempt

            # If we reach here, we have a non-empty list of valid, non-overlapping highlights
            print(f"Attempt {attempt + 1} successful. Found {len(sorted_highlights)} valid highlight time segments.")
            return sorted_highlights # Return the validated and sorted list

        except json.JSONDecodeError:
             print(f"Attempt {attempt + 1} failed: Invalid JSON response from LLM.")
             if 'response_text' in locals(): print(f"Raw LLM Response: {response_text}")
             continue
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with unexpected error: {str(e)}")
            if 'response_text' in locals(): print(f"Raw LLM Response on Error: {response_text}")
            continue

    print("Max attempts reached for extracting time segments, returning empty list.")
    return []


# --- New Functions ---

def extract_text_for_segment(transcription: str, start_time: float, end_time: float) -> str:
    """Extracts speaker text from transcription within a given time range."""
    segment_text = []
    # Regex to capture timestamp and text, robust to formats like:
    # [0.00] Speaker: Text [8.96]
    # [8.96] Text
    # [12.32] Speaker: Text
    # It captures the start time and the main text content.
    line_pattern = re.compile(r"^\s*\[\s*(\d+\.\d+)\s*\]\s*(.*?)(?:\s*\[\d+\.\d+\s*\])?$")

    lines = transcription.strip().splitlines() # Use splitlines for robustness
    for i, line in enumerate(lines):
        match = line_pattern.match(line)
        if match:
            try:
                timestamp = float(match.group(1))
                text_content = match.group(2).strip()

                # Remove speaker prefix like "Speaker X:" if present
                text_content = re.sub(r"^[Ss]peaker\s*\d*:\s*", "", text_content).strip()

                # Include lines starting within the time range
                if timestamp < end_time and timestamp >= start_time:
                    if text_content: # Avoid adding empty lines
                         segment_text.append(text_content)
            except (ValueError, IndexError):
                # Ignore lines that don't match the expected format
                continue
        # else: Line doesn't match pattern, ignore

    return "\n".join(segment_text)


def generate_description_and_hashtags(segment_text: str, max_attempts: int = 3) -> Optional[str]:
    """Generates a description with appended hashtags for a text segment using LLM."""
    if not segment_text or not segment_text.strip():
        print("Skipping description generation: Empty segment text provided.")
        return None

    system_prompt = """
    You are provided with the text content of a short video clip (typically 30-60 seconds).
    Your task is to generate a single string containing:
    1. A short, concise, and engaging description (max 1-2 sentences) summarizing the clip.
    2. Followed by a space and 3-5 relevant hashtags appended directly to the description. Hashtags should be lowercase, start with #, and contain no spaces.

    Instructions:
    1.  Read the provided text carefully to understand the main topic and key points.
    2.  Combine the description and hashtags into a single string.
    3.  Return ONLY a JSON object in the following exact format. Do not include any explanations, markdown formatting, or text outside the JSON structure:
        {
            "caption_with_hashtags": "Your engaging description here. #hashtag1 #hashtag2 #hashtag3"
        }

    Example Input Text:
    "One of the most exciting applications is in video processing. Let's look at how AI can automatically generate video highlights. This technology is revolutionizing content creation."

    Example Output JSON:
    {
        "caption_with_hashtags": "Discover how AI automatically creates video highlights, changing content creation forever! #ai #videohighlights #contentcreation #machinelearning"
    }

    Return ONLY the JSON object.
    """

    # Define generation config based on AI Studio code
    generation_config = types.GenerateContentConfig(
        temperature=1,
        system_instruction=[types.Part.from_text(text=system_prompt)]
    )

    for attempt in range(max_attempts):
        try:
            # Structure the prompt and system instruction for generate_content
            user_prompt_text = f"Segment Text:\n{segment_text}"
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt_text)])]
            
            # Use the global model with the new config
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generation_config
            )

            if not response or not response.text:
                print(f"Attempt {attempt + 1} failed: Empty response from LLM for description.")
                continue

            # Extract JSON from response
            response_text = response.text
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
            if match:
                json_string = match.group(1).strip()
            else:
                json_string = response_text.strip()

            data = json.loads(json_string)

            # Validate the structure and types
            if not isinstance(data, dict) or \
               "caption_with_hashtags" not in data or \
               not isinstance(data["caption_with_hashtags"], str):
                print(f"Attempt {attempt + 1} failed: Invalid structure or types in JSON response.")
                print(f"Raw LLM Response: {response_text}")
                continue

            return data["caption_with_hashtags"].strip()

        except json.JSONDecodeError:
            print(f"Attempt {attempt + 1} failed: Invalid JSON response from LLM for description.")
            if 'response_text' in locals(): print(f"Raw LLM Response: {response_text}")
            continue
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with unexpected error during description generation: {str(e)}")
            if 'response_text' in locals(): print(f"Raw LLM Response on Error: {response_text}")
            continue

    print("Max attempts reached for generating description/hashtags, returning None.")
    return None


# --- Updated Main Function ---

def GetHighlights(transcription: str) -> List[EnrichedHighlightData]:
    """
    Main function to get multiple highlight segments from transcription,
    each enriched with an LLM-generated description and hashtags.
    """
    enriched_highlights = []
    try:
        # Clean and validate the input transcription
        if not transcription or not transcription.strip():
            print("Error: Empty transcription provided.")
            return []

        # 1. Extract highlight time segments
        highlight_segments = extract_highlights(transcription.strip())

        if not highlight_segments:
            print("No valid highlight time segments were extracted.")
            return []

        print(f"\nProceeding to generate descriptions for {len(highlight_segments)} segments...")

        # 2. For each segment, extract text and generate description/hashtags
        for segment in highlight_segments:
            # Convert string timestamps to floats first
            try:
                start_time = float(segment["start"])
                end_time = float(segment["end"])
            except ValueError:
                print(f"Warning: Could not convert timestamps to float for segment {segment}. Skipping.")
                continue

            # 2a. Extract text for this segment
            segment_text = extract_text_for_segment(transcription, start_time, end_time)

            if not segment_text.strip():
                 print("Warning: No text extracted for this segment. Skipping description generation.")
                 # Optionally skip this segment entirely or add placeholder description
                 continue # Skip this segment

            # 2b. Generate description and hashtags for the segment text
            caption_string = generate_description_and_hashtags(segment_text)

            if caption_string:
                # 2c. Combine time segment with description data
                enriched_data: EnrichedHighlightData = {
                    "start": start_time,
                    "end": end_time,
                    "segment_text": segment_text, # Store the original text
                    "caption_with_hashtags": caption_string
                }
                enriched_highlights.append(enriched_data)
            else:
                print(f"Warning: Failed to generate description for segment {start_time:.2f}-{end_time:.2f}. Skipping.")
                # Optionally add segment with placeholder/error description instead of skipping

        if not enriched_highlights:
            print("No highlights could be successfully enriched with descriptions.")
            return []

        print(f"\nSuccessfully enriched {len(enriched_highlights)} highlights.")
        return enriched_highlights

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
    [60.0] Speaker 2: This technology is revolutionizing content creation, making it faster and easier.
    [75.5] Speaker 1: We're seeing it used widely in social media, entertainment, and even education platforms to deliver personalized content.
    [90.2] Speaker 2: The ability for AI to not just cut clips but understand context and find truly engaging moments is key.
    [105.8] Speaker 1: It's changing how we create and consume digital content daily. Think about personalized news feeds.
    [120.0] Speaker 2: Let's dive into some specific examples of tools available now.
    [135.0] Speaker 1: Good idea. Tool number one uses advanced natural language processing...
    """

    final_highlights = GetHighlights(example_transcription)
    if final_highlights:
        print("\n--- Final Enriched Highlights ---")
        for i, highlight in enumerate(final_highlights, 1):
            print(f"Highlight {i}:")
            print(f"  Time: {highlight['start']:.2f}s - {highlight['end']:.2f}s")
            print(f"  Text: {highlight['segment_text'][:100]}...") # Print snippet of text
            print(f"  Caption: {highlight['caption_with_hashtags']}")
            print("-" * 10)
    else:
        print("\nNo valid enriched highlights found.")
