import streamlit as st
import speech_recognition as sr
import pandas as pd
from pathlib import Path
from jiwer import wer

st.set_page_config(page_title="STT Report Tool", layout="wide")

# 1. Setup Session State (This saves results even if you switch files)
if 'history' not in st.session_state:
    st.session_state.history = []

st.title(" LibriSpeech Accuracy & Reporting Tool")

# HARDCODED PATH: Keep this pointing to your audio files
data_path = Path(r"C:\Users\HP 640 G2\Downloads\PROJECT\141231")



if not data_path.exists():
    st.error(f"Audio path not found! Check: {data_path}")
    st.stop()

audio_files = [f.name for f in data_path.glob("*.flac")]

# Sidebar for History and Downloads
with st.sidebar:
    st.header(" Test History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df[["File", "Accuracy"]])
        
        # Download Button
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=" Download Full Report (CSV)",
            data=csv,
            file_name="stt_accuracy_report.csv",
            mime="text/csv",
        )
        
        if st.button(" Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("No tests run yet.")

# Main UI
selected_file = st.selectbox("Select Audio File:", audio_files)

# Show audio player


if st.button(" Transcribe & Save Result"):
    # --- THE FIX: Define file_path right here ---
 file_path = str(data_path/selected_file)
    
with st.spinner("Processing..."):
        try:
            # A. Transcribe
            recognizer = sr.Recognizer()
            with sr.AudioFile(file_path) as source:
                audio = recognizer.record(source)
                predicted_text = recognizer.recognize_google(audio).lower()

            # B. Get Actual Text
            actual_text = ""
            text_files = list(data_path.glob("*.txt"))
            if text_files:
                # Open the first txt file found in the folder
                with open(text_files[0], "r") as f:
                    for line in f:
                        if selected_file.replace(".flac", "") in line:
                            # LibriSpeech format: "ID TEXT" -> take parts[1]
                            parts = line.strip().split(" ", 1)
                            if len(parts) > 1:
                                actual_text = parts[1].lower()
                            break

            # C. Calculate Metrics
            if not actual_text:
                st.warning("Ground truth text not found in .txt file.")
                actual_text = "not found"
                accuracy = 0.0
            else:
                error = wer(actual_text, predicted_text)
                accuracy = max(0, (1 - error) * 100)

            # D. SAVE TO HISTORY
            new_result = {
                "File": selected_file,
                "Accuracy": f"{accuracy:.2f}%",
                "Actual Text": actual_text,
                "Predicted Text": predicted_text
            }
            st.session_state.history.append(new_result)

            # E. Display Current Result
            st.success(f"Result Saved! Current Accuracy: {accuracy:.2f}%")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("**Ground Truth**")

                st.write(actual_text)
            with col2:
                st.warning("**Prediction**")
                st.write(predicted_text)
            
            # Refresh to show in sidebar history
            st.rerun()
                
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.caption("LibriSpeech STT Benchmarking Project")
