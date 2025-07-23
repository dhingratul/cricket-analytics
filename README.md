# Cricket Analytics

This project provides a tool to analyze cricket game videos, rating the quality of shots and providing improvement feedback using Google Gemini 2.5 Pro.

## Features
- Takes an input video of a cricket game.
- Analyzes each frame (at a configurable interval) to rate the quality of the cricket shot.
- Provides a score out of 10 and a one-line improvement suggestion (if not a perfect shot).
- Overlays the feedback directly onto the video.
- Uses Google Gemini 2.5 Pro for AI-powered analysis.

## Requirements
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for package management

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd cricket-analytics
   ```
2. Install dependencies using uv:
   ```bash
   uv venv
   uv pip install -e .
   uv sync
   ```
3. Create a `.env` file in the project root with your Gemini API key:
   ```env
   GOOGLE_API_KEY=your_actual_gemini_api_key_here
   ```

## Usage
1. Prepare your cricket video (e.g., `data/sample.mp4`).
2. Run the analysis script:
   ```bash
   .venv/bin/python src/cricket.py <input_video> <output_video> [--interval N]
   ```
   - `<input_video>`: Path to the input cricket video file.
   - `<output_video>`: Path to save the annotated output video.
   - `--interval N`: (Optional) Analyze every Nth frame (default: 30).

Example:
```bash
.venv/bin/python src/cricket.py data/sample.mp4 data/output.mp4 --interval 30
```

## How it Works
- The script extracts frames from the input video at the specified interval.
- Each frame is sent to Gemini 2.5 Pro with a prompt to rate the shot and provide feedback.
- The score and feedback are overlayed on the video frame.
- The annotated video is saved to the specified output path.

## Notes
- Make sure your `.env` file contains a valid `GOOGLE_API_KEY` for Gemini API access.
- The script uses OpenCV for video processing and overlays, and Pillow for image conversion.
- The analysis is only as good as the Gemini model's understanding of cricket shots from the video frames.

## License
MIT
