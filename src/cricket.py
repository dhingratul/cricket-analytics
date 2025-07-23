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
openai_api_key = os.getenv('OPENAI_API_KEY')
gemini_api_key = os.getenv('GOOGLE_API_KEY')

def analyze_frame_with_openai(frame):
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment. Please set it in your .env file.")
    openai.api_key = openai_api_key
    # Convert OpenCV frame (BGR) to RGB and encode as PNG in memory
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    prompt = "Rate the quality of shot in this frame, and provide a score out of 10. Provide 1 line feedback on how to improve the shot if not 10."
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a cricket video analyst."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]}
        ],
        max_tokens=256
    )
    return response.choices[0].message.content.strip()

def analyze_frame_with_gemini(frame):
    if not gemini_api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in environment. Please set it in your .env file.")
    genai.configure(api_key=gemini_api_key)
    # Convert OpenCV frame (BGR) to PIL Image (RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    # Use Gemini API to analyze the frame
    model = genai.GenerativeModel('gemini-2.5-pro')
    prompt = "Rate the quality of shot in this frame, and provide a score out of 10. Provide 1 line feedback on how to improve if not 10."
    response = model.generate_content([
        prompt,
        pil_img
    ])
    return response.text

def process_video(input_path, output_path, frame_interval=30, provider='openai'):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare to write output video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    feedback_cache = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            print(f"Analyzing frame {frame_idx}/{total_frames}")
            if provider == 'gemini':
                feedback = analyze_frame_with_gemini(frame)
            else:
                feedback = analyze_frame_with_openai(frame)
            feedback_cache[frame_idx] = feedback
        else:
            feedback = feedback_cache.get(frame_idx - (frame_idx % frame_interval), "")
        # Overlay feedback
        if feedback:
            y0, dy = 30, 30
            for i, line in enumerate(feedback.split('\n')):
                y = y0 + i*dy
                cv2.putText(frame, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze cricket video for shot quality and overlay feedback.")
    parser.add_argument('input_video', help='Path to input cricket video')
    parser.add_argument('output_video', help='Path to save annotated video')
    parser.add_argument('--interval', type=int, default=30, help='Frame interval for analysis (default: 30)')
    parser.add_argument('--provider', choices=['openai', 'gemini'], default='openai', help='AI provider to use (default: openai)')
    args = parser.parse_args()
    process_video(args.input_video, args.output_video, args.interval, args.provider)
