# AI-Youtube-Shorts-Generator

An AI-powered tool that automatically generates engaging short-form videos from longer YouTube content.
Forked from [SamurAIGPT/AI-Youtube-Shorts-Generator](https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator)

## Features
- **Video Download**: Given a YouTube URL, the tool downloads the video.
- **Transcription**: Uses Whisper to transcribe the video.
- **Highlight Extraction**: Utilizes OpenAI's Gemini-Pro to identify the most engaging parts of the video.
- **Speaker Detection**: Detects speakers in the video.
- **Vertical Cropping**: Crops the highlighted sections vertically, making them perfect for shorts.
- **Caching System**: 
  - Stores processed video data in SQLite database
  - Caches transcriptions to avoid reprocessing
  - Saves highlight timestamps for quick retrieval
  - Improves processing speed for previously analyzed videos

## Installation
### Prerequisites
- Python 3.7 or higher
- FFmpeg
- OpenCV
- LangGraph
- SQLite3

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator.git
   cd AI-Youtube-Shorts-Generator
   ```

2. Create a virtual environment:
   ```bash
   python3.10 -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate # On Windows: venv\Scripts\activate
   ```

4. Install the python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration
1. Set up the environment variables.
   Create a `.env` file in the project root directory and add your API key from Google AI Studio (it's free):
   ```bash
   GOOGLE_API_KEY=your_key_here
   ```

## Usage
1. Ensure your `.env` file is correctly set up with your API key.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Enter either:
   - A YouTube URL to process a new video
   - A local file path to process a video from your system

The tool will:
- Check if the video has been processed before
- Use cached data if available
- Only perform necessary processing steps for new videos
- Store results for future use

## Database Structure
The caching system uses SQLite with three main tables:
- `videos`: Stores video metadata and file paths
- `transcriptions`: Stores video transcription data
- `highlights`: Stores extracted highlight segments
- If you face any issues or missing files with that try to remove the .db file

## Known Issues
- Face detection and vertical cropping may to be fixed

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.
