import cv2
import numpy as np
import google.generativeai as genai
import openai
import os
from PIL import Image
from dotenv import load_dotenv
import io
import base64

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")


def analyze_video_with_openai(video_path):
    """
    Analyze cricket video with OpenAI GPT-4o vision model using key frames.
    Returns comprehensive feedback on shot quality.
    """
    if not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found in environment. Please set it in your .env file."
        )

    client = openai.OpenAI(api_key=openai_api_key)

    # Extract key frames from video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract frames at regular intervals (max 10 frames for API efficiency)
    frame_indices = [int(i * total_frames / 10) for i in range(10)]
    frames_base64 = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert frame to base64
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")
            frames_base64.append(img_base64)

    cap.release()

    prompt = """Analyze this cricket shot sequence from these key frames and provide comprehensive feedback on the batting technique and shot quality. Please evaluate:

1. Overall shot quality (score out of 10)
2. Timing and footwork progression
3. Bat swing and follow-through
4. Balance and stance throughout
5. Shot selection and execution
6. Key areas for improvement

Provide a concise but thorough analysis focusing on technical aspects of the batting technique across the entire sequence."""

    # Create content with all frames
    content = [{"type": "text", "text": prompt}]
    for i, frame_b64 in enumerate(frames_base64):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{frame_b64}"},
            }
        )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert cricket batting coach. Analyze the sequence of frames showing a cricket shot and provide detailed technical feedback.",
                },
                {"role": "user", "content": content},
            ],
            max_tokens=1500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error analyzing video with OpenAI: {e}")
        return f"Analysis failed: {str(e)}"


def analyze_video_with_gemini(video_path):
    """
    Analyze entire cricket video with Google Gemini.
    Returns comprehensive feedback on shot quality.
    """
    if not gemini_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not found in environment. Please set it in your .env file."
        )

    genai.configure(api_key=gemini_api_key)

    try:
        # Upload video file to Gemini
        video_file = genai.upload_file(path=video_path)

        # Wait for processing
        while video_file.state.name == "PROCESSING":
            print("Processing video...")
            import time

            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError("Video processing failed")

        model = genai.GenerativeModel("gemini-2.5-pro")

        prompt = """Analyze this cricket video and provide comprehensive feedback on the batting technique and shot quality. Please evaluate:

1. Overall shot quality (score out of 10)
2. Timing and footwork
3. Bat swing and follow-through
4. Balance and stance
5. Shot selection and execution
6. Key areas for improvement

Provide a concise but thorough analysis focusing on technical aspects of the batting."""

        response = model.generate_content([video_file, prompt])

        # Clean up uploaded file
        genai.delete_file(video_file.name)

        return response.text
    except Exception as e:
        print(f"Error analyzing video with Gemini: {e}")
        return f"Analysis failed: {str(e)}"


def process_video(input_path, output_path, provider="openai"):
    """
    Process cricket video by sending entire video to AI for analysis.
    Creates a copy of the original video and provides comprehensive feedback.
    """
    print(f"Analyzing cricket video: {input_path}")
    print(f"Using AI provider: {provider}")
    print("Sending entire video for analysis...")

    # Copy original video to output (no modifications needed)
    import shutil

    shutil.copy2(input_path, output_path)
    print(f"Original video copied to: {output_path}")

    # Analyze entire video with selected AI provider
    print("Starting video analysis...")
    try:
        if provider == "gemini":
            feedback = analyze_video_with_gemini(input_path)
        else:
            feedback = analyze_video_with_openai(input_path)

        # Display results
        print(f"\n{'='*60}")
        print("CRICKET SHOT ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(feedback)
        print(f"{'='*60}")
        print(f"Original video saved to: {output_path}")

        return feedback

    except Exception as e:
        error_msg = f"Video analysis failed: {str(e)}"
        print(f"\n{'='*60}")
        print("CRICKET SHOT ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(error_msg)
        print(f"{'='*60}")
        return error_msg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze entire cricket video for comprehensive shot quality feedback."
    )
    parser.add_argument("input_video", help="Path to input cricket video")
    parser.add_argument("output_video", help="Path to save video copy")
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default="openai",
        help="AI provider to use (default: openai)",
    )
    args = parser.parse_args()
    process_video(args.input_video, args.output_video, args.provider)
