import streamlit as st
import os
import tempfile
from pathlib import Path
from cricket import analyze_video_with_openai, analyze_video_with_gemini

def main():
    st.set_page_config(
        page_title="Cricket Shot Analysis",
        page_icon="🏏",
        layout="wide"
    )
    
    st.title("🏏 Cricket Shot Analysis")
    st.markdown("Select a cricket video and get AI-powered batting feedback!")
    
    # Create two columns - video selection on left, feedback on right
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📹 Video Selection")
        
        # Get list of video files from data directory
        data_dir = Path("data")
        if not data_dir.exists():
            st.error("Data directory not found! Please make sure 'data/' folder exists.")
            return
            
        video_files = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.avi")) + list(data_dir.glob("*.mov"))
        
        if not video_files:
            st.warning("No video files found in data/ directory!")
            return
        
        # Video selection dropdown
        selected_video = st.selectbox(
            "Choose a video:",
            options=video_files,
            format_func=lambda x: x.name
        )
        
        # AI Provider selection
        provider = st.selectbox(
            "Select AI Provider:",
            options=["openai", "gemini"],
            index=0
        )
        
        # Display selected video
        if selected_video:
            st.subheader("Selected Video:")
            st.video(str(selected_video))
        
        # Analysis button
        analyze_button = st.button("🔍 Analyze Shot", type="primary")
    
    with col2:
        st.header("📊 Analysis Results")
        
        if analyze_button and selected_video:
            with st.spinner(f"Analyzing video with {provider.upper()}... This may take a moment."):
                try:
                    # Create temporary output file
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                        temp_output = tmp_file.name
                    
                    # Run analysis
                    if provider == 'gemini':
                        feedback = analyze_video_with_gemini(str(selected_video))
                    else:
                        feedback = analyze_video_with_openai(str(selected_video))
                    
                    # Display results in a nice format
                    st.success("✅ Analysis Complete!")
                    
                    # Create an attractive results container
                    with st.container():
                        st.subheader("🎯 Shot Analysis Report")
                        
                        # Display feedback in a styled text area
                        st.markdown("### 📋 Technical Feedback:")
                        st.text_area(
                            "feedback_display",
                            value=feedback,
                            height=400,
                            disabled=True,
                            label_visibility="collapsed"
                        )
                        
                        # Add some styling
                        st.markdown("---")
                        st.caption(f"Analysis powered by {provider.upper()} • Video: {selected_video.name}")
                    
                    # Clean up temp file
                    if os.path.exists(temp_output):
                        os.unlink(temp_output)
                        
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("💡 Make sure your API keys are set in your .env file!")
        
        elif not analyze_button:
            # Show placeholder when no analysis is running
            st.info("👆 Select a video and click 'Analyze Shot' to get started!")
            
            # Show some tips
            with st.expander("💡 Tips for Best Results"):
                st.markdown("""
                - **Video Quality**: Higher resolution videos provide better analysis
                - **Shot Visibility**: Ensure the batsman and ball are clearly visible
                - **OpenAI**: Analyzes key frames from the video sequence
                - **Gemini**: Analyzes the complete video for more context
                - **API Keys**: Make sure your `.env` file contains valid API keys
                """)
    
    # Add footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🏏 Cricket Analytics Tool • Built with Streamlit"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 