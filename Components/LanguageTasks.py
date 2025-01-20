import google.generativeai as genai
from typing import Annotated, Any, Dict, TypedDict, List
from langgraph.graph import StateGraph
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


# Type definitions
class Message(TypedDict):
    role: str
    content: str


class HighlightData(TypedDict):
    start: float
    content: str
    end: float


class GraphState(TypedDict):
    messages: List[Message]
    highlights: list[HighlightData]
    error: str | None


def extract_highlights(state: GraphState) -> GraphState:
    """Extract highlights from transcription using Gemini."""
    try:
        messages = state["messages"]
        last_message = messages[-1]["content"]

        system_prompt = """
        Analyze the provided transcription and select ONE continuous segment that would make an engaging short-form video. 
        
        CRITICAL REQUIREMENTS:
        1. Important Time Duration:
           - Minimum: 30 seconds between start and end
           - Maximum: 60 seconds between start and end
           - Target: Aim for 45 seconds when possible
        
        2. Selection Criteria:
           - Choose the MOST engaging or impactful continuous segment
           - Must include complete thoughts/sentences
           - Select moments with clear context (avoid starting mid-conversation)
           - Prefer segments with a clear hook or interesting opening
        
        3. Timestamp Accuracy:
           - Use EXACT timestamps from the transcription
           - Do not make up or modify timestamps
           - Start and end times must correspond to actual transcript markers
        
        4. Format Requirements:
           Return ONLY a JSON array with ONE object in this exact format:
           [{
               "start": <exact_start_timestamp>,
               "content": "complete segment content",
               "end": <exact_end_timestamp>
           }]

        Important VALIDATION:
        - Verify end_time - start_time is between 30 and 60 seconds
        - Ensure timestamps match actual transcript markers
        - Confirm the content is a continuous, complete segment

        Return ONLY the JSON. No explanations or additional text.
        """
        prompt = f"{system_prompt}\n\nTranscription:\n{last_message}"
        response = model.generate_content(prompt)

        # Extract JSON from response
        response_text = response.text
        json_string = response_text.strip("`json\n").strip()
        highlights = json.loads(json_string)

        return {**state, "highlights": highlights, "error": None}
    except Exception as e:
        return {**state, "error": str(e)}


def validate_highlights(state: GraphState) -> str:
    """Determine next step based on validation."""
    if state.get("error") or not state.get("highlights"):
        return "extract"

    highlights = state.get("highlights", [])
    if not highlights or len(highlights) != 1:
        return "extract"

    highlight = highlights[0]
    if not all(key in highlight for key in ["start", "content", "end"]):
        return "extract"

    try:
        start = float(highlight["start"])
        end = float(highlight["end"])
        if start >= end:
            return "extract"
    except ValueError:
        return "extract"

    return "end"


def create_highlight_graph() -> StateGraph:
    """Create the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("extract", extract_highlights)
    workflow.add_node("end", lambda x: x)

    # Add conditional edges
    workflow.add_conditional_edges(
        "extract", validate_highlights, {"extract": "extract", "end": "end"}
    )

    # Set entry point
    workflow.set_entry_point("extract")

    return workflow.compile()


def GetHighlight(transcription: str) -> tuple[float, float]:
    """Main function to get highlights from transcription."""
    try:
        # Initialize the graph
        workflow = create_highlight_graph()

        # Prepare initial state
        initial_state: GraphState = {
            "messages": [{"role": "user", "content": transcription}],
            "highlights": [],
            "error": None,
        }

        # Run the graph
        final_state = workflow.invoke(initial_state)

        highlights = final_state.get("highlights", [])
        if highlights and len(highlights) > 0:
            highlight = highlights[0]
            return float(highlight["start"]), float(highlight["end"])

        return 0, 0
    except Exception as e:
        print(f"Error in get_highlight: {e}")
        return 0, 0


if __name__ == "__main__":
    example_transcription = """
    [0.0] Speaker 1: Welcome to our discussion about artificial intelligence.
    [15.5] Speaker 1: Today we'll explore the fascinating world of machine learning.
    [30.2] Speaker 2: One of the most exciting applications is in video processing.
    [45.8] Speaker 1: Let's look at how AI can automatically generate video highlights.
    [60.0] Speaker 2: This technology is revolutionizing content creation.
    """
    start, end = GetHighlight(example_transcription)
    print(f"Start: {start}, End: {end}")
