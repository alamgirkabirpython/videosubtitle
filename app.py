import streamlit as st
from transformers import pipeline
from pydub import AudioSegment
import torch
import os
import time
import yt_dlp
import tempfile

# Set Streamlit page config
st.set_page_config(page_title="Audio/Video-to-Text Transcription", layout="centered", initial_sidebar_state="auto")

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the Whisper model pipeline
@st.cache_resource
def load_model():
    return pipeline(
        "automatic-speech-recognition",
        "openai/whisper-large-v2",
        chunk_length_s=30,
        stride_length_s=3,
        return_timestamps=True,
        device=device,
    )

pipe = load_model()

# Function to download audio using yt-dlp
def download_audio_youtube(video_url, output_path="temp_audio.mp4"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return output_path
    except yt_dlp.utils.DownloadError as e:
        st.error(f"Error downloading audio from YouTube: {e}")
        return None

# Function to format transcription into SRT format
def format_transcription_to_srt(transcription):
    srt_output = ""
    counter = 1

    for chunk in transcription.get('chunks', []):
        text = chunk["text"]
        timestamps = chunk.get("timestamp", None)

        if timestamps:
            start_time, end_time = timestamps
            # Convert timestamps to SRT format
            start_time_srt = time.strftime('%H:%M:%S', time.gmtime(start_time)) + f",{int((start_time % 1) * 1000):03}"
            end_time_srt = time.strftime('%H:%M:%S', time.gmtime(end_time)) + f",{int((end_time % 1) * 1000):03}"

            srt_output += f"{counter}\n{start_time_srt} --> {end_time_srt}\n{text.strip()}\n\n"
            counter += 1

    return srt_output.strip()

# Function to convert audio to WAV format using pydub
def convert_to_wav(input_file, output_file="temp_audio.wav"):
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")
    return output_file

# Main app function
def main():
    st.markdown("<h1 style='color: #00bfff;'>Audio/Video-to-Text Transcription App</h1>", unsafe_allow_html=True)

    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["Audio File", "YouTube Video"])

    # Tab for audio file upload
    with tab1:
        st.subheader("Transcribe Audio File")
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
        st.audio(uploaded_file)

        if uploaded_file:
            if st.button("Transcribe Audio"):
                with st.spinner("Processing..."):
                    start_time = time.time()

                    # Use tempfile for temporary file storage
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_file_path = temp_file.name

                    # Convert to WAV using pydub
                    wav_file = convert_to_wav(temp_file_path)

                    # Perform transcription
                    transcription = pipe(wav_file)
                    st.write(transcription)  # Debugging: Check the transcription output
                    srt_transcription = format_transcription_to_srt(transcription)

                    st.success("Transcription completed!")
                    st.subheader("Subtitles (SRT Format)")
                    st.text_area("SRT Output", value=srt_transcription, height=400)

                    # Download options
                    st.download_button("Download Subtitles (SRT File)", srt_transcription, file_name="transcription.srt")

                    end_time = time.time()
                    st.write(f"Time taken: {round(end_time - start_time, 2)} seconds")

                    # Clean up
                    os.remove(temp_file_path)
                    os.remove(wav_file)

    # Tab for YouTube video transcription
    with tab2:
        st.subheader("Transcribe YouTube Video")
        video_url = st.text_input("Enter YouTube video link:")

        if video_url:
            if st.button("Transcribe Video"):
                with st.spinner("Processing..."):
                    try:
                        # Download audio
                        st.info("Downloading audio...")
                        audio_file = download_audio_youtube(video_url)

                        if audio_file:
                            # Convert to WAV using pydub
                            st.info("Extracting audio...")
                            wav_file = convert_to_wav(audio_file)

                            # Perform transcription
                            st.info("Transcribing audio...")
                            transcription = pipe(wav_file)
                            st.write(transcription)  # Debugging: Check the transcription output
                            srt_transcription = format_transcription_to_srt(transcription)

                            st.success("Transcription completed!")
                            st.subheader("Subtitles (SRT Format)")
                            st.text_area("SRT Output", value=srt_transcription, height=400)

                            # Download options
                            st.download_button("Download Subtitles (SRT File)", srt_transcription, file_name="transcription.srt")

                            # Clean up
                            os.remove(audio_file)
                            os.remove(wav_file)

                    except Exception as e:
                        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
