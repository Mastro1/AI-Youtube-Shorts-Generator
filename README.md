# AI-Youtube-Shorts-Generator-Gemini

An AI-powered tool that automatically generates engaging short-form videos from longer YouTube content, optimized for platforms like YouTube Shorts, Instagram Reels, and TikTok and for static videos with a 1 person speaking.

## Key Features

- **Smart Video Download**: 
  - Downloads videos from YouTube URLs with quality selection
  - Supports both progressive and adaptive streams
  - Automatically merges video and audio for best quality
  - Handles local video files as input

- **Advanced Transcription**:
  - Uses `faster-whisper` (base.en model) for efficient transcription
  - Provides both segment-level and word-level timestamps
  - CPU-optimized processing with int8 quantization
  - Multi-threaded performance for faster processing

- **AI-Powered Highlight Detection**:
  - Leverages Google's Gemini-2.0-flash model for content analysis
  - Identifies the most engaging segments from transcriptions
  - Generates relevant hashtags and captions
  - Smart content selection based on engagement potential

- **Intelligent Video Processing**:
  - Multiple vertical cropping strategies:
    - Static centered crop
    - Face-detection based dynamic cropping
    - Average face position based cropping
  - Maintains optimal 9:16 aspect ratio for shorts
  - Automatic bottom margin cropping for better framing
  - Supports both static and animated captions

- **Robust Caching System**:
  - SQLite database for efficient data management
  - Caches processed videos, audio, and transcriptions
  - Prevents redundant processing of previously handled content
  - Easy cache management and cleanup

## Prerequisites

- Python 3.10 or higher
- FFmpeg (latest version recommended)
- CUDA-compatible GPU (optional, for faster processing)
- 4GB+ RAM recommended

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-Youtube-Shorts-Generator.git
   cd AI-Youtube-Shorts-Generator
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY=your_google_ai_studio_key_here
   ```

## Usage

1. Start the tool:
   ```bash
   python main.py
   ```

2. Input either:
   - A YouTube URL
   - A path to a local video file

3. Select video quality when prompted (for YouTube downloads)

4. The tool will process your video through several stages:
   - Download/import video
   - Extract and transcribe audio
   - Identify engaging segments
   - Create vertical crops
   - Add captions
   - Generate final shorts

5. Find your processed shorts in the `shorts` directory

## Configuration Options

- `USE_ANIMATED_CAPTIONS`: Toggle between static and animated captions (in main.py) (reccomended)
- `SHORTS_DIR`: Customize output directory for processed videos
- CPU thread optimization in `Components/Transcription.py`

## Project Structure

```
AI-Youtube-Shorts-Generator/
├── Components/
│   ├── Captions.py       # Caption generation and rendering
│   ├── Database.py       # SQLite database management
│   ├── Edit.py          # Video editing and processing
│   ├── FaceCrop.py      # Vertical cropping algorithms
│   ├── LanguageTasks.py # AI content analysis
│   ├── Speaker.py       # Speaker detection (experimental)
│   ├── Transcription.py # Audio transcription
│   └── YoutubeDownloader.py # Video download handling
├── main.py              # Main execution script
├── requirements.txt     # Python dependencies
└── .env                # Environment variables
```

## Database Schema

The SQLite database (`video_processing.db`) contains three main tables:

1. **videos**:
   - id (PRIMARY KEY)
   - youtube_url
   - local_path
   - audio_path
   - created_at

2. **transcriptions**:
   - id (PRIMARY KEY)
   - video_id (FOREIGN KEY)
   - transcription_data
   - created_at

3. **highlights**:
   - id (PRIMARY KEY)
   - video_id (FOREIGN KEY)
   - start_time
   - end_time
   - output_path
   - segment_text
   - caption_with_hashtags
   - created_at

## Known Issues & Limitations

1. **Face Detection**:
   - The face-based cropping can be inconsistent with multiple faces
   - May need manual adjustment for optimal framing in some cases

2. **Speaker Detection**:
   - Current implementation uses basic voice activity detection
   - Full speaker diarization not yet implemented

3. **Resource Usage**:
   - Processing long videos can be memory-intensive
   - GPU acceleration limited to specific components

## Troubleshooting

1. If facing cache-related issues:
   - Delete `video_processing.db` to clear the cache
   - Remove temporary files in the `videos` directory

2. For video processing errors:
   - Ensure FFmpeg is properly installed and accessible
   - Check available disk space for temporary files
   - Verify input video format compatibility

3. For AI-related issues:
   - Confirm Google API key is valid and has sufficient quota
   - Check internet connectivity for API calls

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- SQL integration made by [YassineKADER](https://github.com/YassineKADER/AI-Youtube-Shorts-Generator-)
- Original project by [SamurAIGPT](https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator)
- Uses Google's Gemini AI for content analysis
- Powered by faster-whisper for transcription
