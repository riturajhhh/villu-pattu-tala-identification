"""
frontend/app.py
================
Streamlit web application for the Villu Pattu Tala Identification System.
Provides a modern glassmorphic dark interface to upload files, predict Talas,
render signal waveforms, beats, and spec plots, and view prediction histories.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.components.ui_helpers import (
    card_metric,
    draw_header,
    inject_custom_css,
    tala_badge,
    render_metronome,
    feedback_form,
)
from frontend.components.visualizations import (
    plot_prediction_distribution,
    render_base64_image,
)

# Backend FastAPI URL
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Villu Pattu Tala Identifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
draw_header()

# Define navigation tabs
tabs = st.tabs(["🎵 Home & Inference", "📜 History Logs", "⚙ Model Overview", "🧠 Explainability & Feedback"])


# ---------------------------------------------------------------------------
# Home & Inference Tab
# ---------------------------------------------------------------------------

with tabs[0]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Upload Villu Pattu Performance Audio")
    
    col_input, col_info = st.columns([2, 1])
    
    audio_file = None
    with col_input:
        input_method = st.radio("Choose Audio Source:", ["📁 Upload File", "🎙️ Record Microphone", "🔗 YouTube Link"], horizontal=True)
        youtube_url = ""
        if input_method == "📁 Upload File":
            audio_file = st.file_uploader(
                "Upload an audio file (WAV, MP3, or FLAC)",
                type=["wav", "mp3", "flac", "ogg"],
                label_visibility="collapsed"
            )
        elif input_method == "🎙️ Record Microphone":
            try:
                audio_file = st.audio_input("Record audio from your microphone")
            except AttributeError:
                st.warning("Live recording widget (st.audio_input) is not supported in this Streamlit version. Please use file upload.")
        else:
            youtube_url = st.text_input("Enter Villu Pattu YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")

        model_selection = st.selectbox(
            "Select Classifier Model Type:",
            options=["auto", "classical", "cnn", "crnn"],
            index=0,
            help="Choose the machine learning or deep learning architecture for classification."
        )

    with col_info:
        st.markdown(
            """
            **Supported Rhythmic cycles (Talas):**
            - **Adi**: 8 beats (Laghu 4 + Drutam 2 + Drutam 2)
            - **Rupaka**: 6 beats (Drutam 2 + Laghu 4)
            - **Misra Chapu**: 7 beats (3 + 4 grouping)
            - **Khanda Chapu**: 5 beats (2 + 3 grouping)
            - **Other**: Alternate or mixed rhythm patterns
            """
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")

    can_analyze = (audio_file is not None) or (input_method == "🔗 YouTube Link" and youtube_url != "")

    if can_analyze:
        if st.button("🚀 Analyze & Identify Tala"):
            with st.spinner("Processing audio, extracting rhythmic features, and classifying..."):
                try:
                    if input_method == "🔗 YouTube Link":
                        # 1. Download YouTube link via backend
                        import urllib.parse
                        encoded_url = urllib.parse.quote(youtube_url, safe="")
                        upload_res = requests.post(f"{API_URL}/upload/youtube?url={encoded_url}")
                        
                        if upload_res.status_code != 200:
                            st.error(f"Failed to process YouTube link: {upload_res.json().get('detail', upload_res.text)}")
                            st.stop()
                            
                        upload_data = upload_res.json()
                        file_id = upload_data["file_id"]
                        display_filename = upload_data["filename"]

                        # 2. Call predict by ID
                        predict_res = requests.post(
                            f"{API_URL}/predict/by-id?file_id={file_id}&model_type={model_selection}"
                        )
                    else:
                        # 1. Post file to /upload route to get unique reference ID
                        files = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                        upload_res = requests.post(f"{API_URL}/upload", files=files)
                        
                        if upload_res.status_code != 200:
                            st.error(f"Upload failed: {upload_res.text}")
                            st.stop()
                        
                        upload_data = upload_res.json()
                        file_id = upload_data["file_id"]
                        display_filename = audio_file.name

                        # 2. Call /predict with same file to get Tala predictions
                        files_predict = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                        predict_res = requests.post(
                            f"{API_URL}/predict?model_type={model_selection}",
                            files=files_predict
                        )

                    if predict_res.status_code != 200:
                        st.error(f"Inference prediction failed: {predict_res.text}")
                        st.stop()
                        
                    pred_data = predict_res.json()

                    # 3. Call /features with file_id to get visualization plots and features
                    feature_res = requests.get(f"{API_URL}/features?file_id={file_id}")
                    if feature_res.status_code != 200:
                        st.error(f"Feature visualization fetch failed: {feature_res.text}")
                        st.stop()
                        
                    feat_data = feature_res.json()

                    # --- Prediction Results Layout ---
                    st.success("Tala Identification completed successfully!")
                    
                    st.markdown("### 🏆 Prediction Summary")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        tala_badge(pred_data["predicted_tala"])
                    with col2:
                        card_metric("Prediction Confidence", f"{pred_data['confidence']:.2%}", "🎯")
                    with col3:
                        card_metric("Detected Tempo (BPM)", f"{pred_data['bpm']:.1f}", "⏱")
                    with col4:
                        card_metric("Pulse Clarity Score", f"{pred_data['pulse_clarity']:.2f}", "⚡")

                    # Draw confidence distribution and details
                    col_dist, col_meta = st.columns([3, 2])
                    with col_dist:
                        plot_prediction_distribution(pred_data["top_predictions"])
                    with col_meta:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.subheader("Signal Characteristics")
                        st.markdown(
                            f"""
                            - **Filename:** `{display_filename}`
                            - **Inference Duration:** `{pred_data['duration']:.2f} seconds`
                            - **Architecture Used:** `{pred_data['model_used'].upper()}`
                            - **Detected Beats:** `{len(pred_data['beat_positions'])}`
                            """
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add metronome animation
                        render_metronome(pred_data['bpm'])

                    # Stash data for explainability tab
                    st.session_state["last_file_id"] = file_id
                    st.session_state["last_tala"] = pred_data["predicted_tala"]
                    st.session_state["last_confidence"] = pred_data["confidence"]
                    if "gradcam" in feat_data["plots"]:
                        st.session_state["last_gradcam"] = feat_data["plots"]["gradcam"]
                    else:
                        st.session_state["last_gradcam"] = None
                        
                    # Also fetch classes for feedback form
                    if "tala_classes" not in st.session_state:
                        try:
                            model_res = requests.get(f"{API_URL}/model-info")
                            st.session_state["tala_classes"] = model_res.json().get("classes", [])
                        except:
                            st.session_state["tala_classes"] = ["Adi", "Rupaka", "Misra_Chapu", "Khanda_Chapu"]

                    # --- Visualization Plots Section ---
                    st.markdown("### 📊 Rhythmic & Spectral Analytics")
                    
                    # Beat tracking & waveform
                    st.markdown("#### Beat Positions & Amplitude Waveform")
                    render_base64_image(
                        feat_data["plots"]["beat_tracking"],
                        caption="Detected musical beats (red dotted lines) overlaid on preprocessed audio waveform."
                    )

                    # Spectrum plots
                    col_spec1, col_spec2 = st.columns(2)
                    with col_spec1:
                        st.markdown("#### Mel Spectrogram (2D Spectral Image)")
                        render_base64_image(
                            feat_data["plots"]["mel_spectrogram"],
                            caption="Log-amplitude Mel filterbank energy across frequencies and frames."
                        )
                    with col_spec2:
                        st.markdown("#### MFCC Heatmap")
                        render_base64_image(
                            feat_data["plots"]["mfcc"],
                            caption="Mel-Frequency Cepstral Coefficients over time capturing vocal/instrumental timbres."
                        )

                    # Onset Envelope
                    st.markdown("#### Onset Strength Envelope")
                    render_base64_image(
                        feat_data["plots"]["onset_envelope"],
                        caption="Signal novelty curve demonstrating beat attack strengths."
                    )

                    # Advanced Plots (Chromagram & Tempogram)
                    col_spec3, col_spec4 = st.columns(2)
                    with col_spec3:
                        if "chromagram" in feat_data["plots"]:
                            st.markdown("#### Chromagram (Pitch Class)")
                            render_base64_image(
                                feat_data["plots"]["chromagram"],
                                caption="Pitch class distribution revealing harmonic/melodic content alongside rhythm."
                            )
                    with col_spec4:
                        if "tempogram" in feat_data["plots"]:
                            st.markdown("#### Fourier Tempogram")
                            render_base64_image(
                                feat_data["plots"]["tempogram"],
                                caption="Tempo variations over time, highlighting cyclic structural alignments."
                            )

                except Exception as exc:
                    st.error(f"Connection error to backend API: {exc}")


# ---------------------------------------------------------------------------
# History Logs Tab
# ---------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Recent Tala Prediction History Logs")
    if st.button("🔄 Refresh History logs"):
        st.rerun()

    try:
        history_res = requests.get(f"{API_URL}/history?limit=50")
        if history_res.status_code == 200:
            hist_data = history_res.json()
            if not hist_data:
                st.info("No prediction logs found in database. Run inference from the Home tab first.")
            else:
                import pandas as pd
                df = pd.DataFrame(hist_data)
                # Reorder columns for readability
                df = df[["timestamp", "filename", "predicted_tala", "confidence", "bpm", "model_used"]]
                
                # Format scores for display
                df["confidence"] = df["confidence"].map(lambda x: f"{x:.2%}")
                df["bpm"] = df["bpm"].map(lambda x: f"{x:.1f}")
                df.columns = ["Timestamp", "Audio File", "Identified Tala", "Confidence", "Tempo (BPM)", "Model Used"]
                
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.error(f"Failed to fetch log history: {history_res.text}")
    except Exception as exc:
        st.error(f"Could not connect to database history API: {exc}")


# ---------------------------------------------------------------------------
# Model Overview Tab
# ---------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Model Architectures & Metrics")
    
    try:
        model_res = requests.get(f"{API_URL}/model-info")
        if model_res.status_code == 200:
            m_info = model_res.json()
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f"**Active Model Type:** `{m_info['model_type'].upper()}`")
                if m_info.get("best_model_name"):
                    st.markdown(f"**Underlying Model Name:** `{m_info['best_model_name']}`")
                st.markdown(f"**Trained Classes:** `{', '.join(m_info['classes'])}`")
            with col_m2:
                st.metric("Model Test Accuracy", f"{m_info['accuracy']:.2%}")
                if m_info.get("n_features"):
                    st.markdown(f"**Feature Count:** `{m_info['n_features']}`")
            st.markdown('</div>', unsafe_allow_html=True)

            # Details about deep learning models
            st.markdown("### Deep Learning Architecture Details")
            st.markdown(
                """
                - **CNN Classifier (2D Mel Spec):** Uses a 3-layer Convolutional Neural Network with Batch Normalisation, ReLU activation, Max Pooling, and Global Average Pooling to classify 128x128 Mel Spectrogram frequency-spatial maps.
                - **CRNN Rhythm Classifier (CNN + LSTM):** Uses 2D Conv layers to capture local spectral characteristics, reshapes the output keeping the time frame dimension, and feeds the sequence into a Bidirectional LSTM layer to capture long-term temporal and rhythmic sequence changes (beats, delays) before final classification.
                """
            )
        else:
            st.error(f"Failed to fetch model info: {model_res.text}")
    except Exception as exc:
        st.error(f"Could not connect to model overview API: {exc}")


# ---------------------------------------------------------------------------
# Explainability & Feedback Tab
# ---------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Explainable AI & Active Learning")
    
    col_exp, col_fb = st.columns([1, 1])
    
    with col_exp:
        st.markdown("### 🔍 Model Explainability (Grad-CAM)")
        st.markdown("For Deep Learning (CNN/CRNN) models, we use Gradient-weighted Class Activation Mapping to highlight which parts of the spectrogram influenced the model's decision.")
        
        if st.session_state.get("last_gradcam"):
            render_base64_image(
                st.session_state["last_gradcam"], 
                caption=f"Grad-CAM overlay for predicted Tala: {st.session_state.get('last_tala')}"
            )
        else:
            st.info("No Grad-CAM explanation available. Run an inference using a Deep Learning model (CNN/CRNN) to generate visual explanations.")
            
    with col_fb:
        if st.session_state.get("last_file_id"):
            feedback_form(
                st.session_state.get("tala_classes", []),
                st.session_state["last_file_id"],
                st.session_state.get("last_tala", ""),
                st.session_state.get("last_confidence", 0.0)
            )
        else:
            st.info("Run an inference first to submit expert feedback.")
            
    st.divider()
    st.markdown("### 📈 Expert Feedback History")
    
    if st.button("🔄 Refresh Feedback Stats"):
        st.rerun()
        
    try:
        fb_res = requests.get(f"{API_URL}/feedback/stats")
        if fb_res.status_code == 200:
            fb_data = fb_res.json()
            
            st.markdown(f"**Total Corrections Submitted:** `{fb_data['total_feedbacks']}`")
            
            if fb_data['total_feedbacks'] > 0:
                recent = fb_data.get('recent_corrections', [])
                if recent:
                    import pandas as pd
                    df_fb = pd.DataFrame(recent)
                    df_fb = df_fb[["timestamp", "original_tala", "corrected_tala", "file_id"]]
                    df_fb.columns = ["Timestamp", "Original Prediction", "Expert Correction", "File ID"]
                    st.dataframe(df_fb, use_container_width=True, hide_index=True)
        else:
            st.error("Failed to load feedback stats.")
    except Exception as exc:
        st.error(f"Could not connect to feedback stats API: {exc}")

