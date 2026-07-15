"""
frontend/components/ui_helpers.py
==================================
CSS injection, custom HTML components, and UI styling utilities
for the Streamlit application.
"""

from __future__ import annotations

import streamlit as st


def inject_custom_css() -> None:
    """Inject glassmorphic premium dark-theme CSS override into the page."""
    custom_css = """
    <style>
        /* Main page adjustments */
        .main {
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Glassmorphism containers */
        div[data-testid="stVerticalBlock"] > div:has(div.glass-card) {
            background: rgba(22, 27, 34, 0.6);
            border: 1px solid rgba(240, 246, 252, 0.1);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(5px);
        }
        
        /* Metric cards */
        .metric-card {
            background: rgba(33, 38, 45, 0.8);
            border: 1px solid rgba(240, 246, 252, 0.15);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #58a6ff;
            margin: 5px 0;
        }
        .metric-label {
            font-size: 13px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Header titles */
        h1, h2, h3 {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        /* Style Streamlit badges/buttons */
        .stButton>button {
            background-color: #238636 !important;
            color: white !important;
            border-radius: 6px !important;
            border: 1px solid rgba(240,246,252,0.1) !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        .stButton>button:hover {
            background-color: #2ea043 !important;
            transform: translateY(-1px);
        }
        
        /* Metronome Animation */
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(88, 166, 255, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(88, 166, 255, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(88, 166, 255, 0); }
        }
        .metronome-dot {
            width: 30px;
            height: 30px;
            background-color: #58a6ff;
            border-radius: 50%;
            margin: 10px auto;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def draw_header() -> None:
    """Render a premium styled project title header."""
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0 30px 0;">
            <h1 style="margin: 0; font-size: 2.5rem; background: linear-gradient(90deg, #58a6ff, #1f6feb); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Villu Pattu Tala Identification System
            </h1>
            <p style="color: #8b949e; font-size: 1.1rem; margin-top: 10px;">
                Rhythmic Cycle & Beat Tracking Analytics Powered by Artificial Intelligence
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_metric(label: str, value: str, icon: str = "") -> None:
    """Render a glassmorphic metric card component."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size: 20px;">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tala_badge(tala_name: str) -> None:
    """Render a styled badge matching the predicted Tala."""
    colors = {
        "Adi": "#238636",
        "Rupaka": "#1f6feb",
        "Misra_Chapu": "#d29922",
        "Khanda_Chapu": "#8957e5",
        "Other": "#8b949e",
    }
    color = colors.get(tala_name, "#8b949e")
    st.markdown(
        f"""
        <div style="text-align: center; margin: 15px 0;">
            <span style="background-color: {color}; color: white; padding: 10px 30px; border-radius: 20px; font-size: 24px; font-weight: bold; box-shadow: 0 4px 15px {color}33;">
                {tala_name}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metronome(bpm: float) -> None:
    """Render a CSS-animated bouncing dot synced to the BPM."""
    if bpm <= 0:
        return
    
    # Calculate animation duration in seconds
    duration_s = 60.0 / bpm
    
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 20px;">
            <p style="color: #8b949e; margin-bottom: 5px;">Visual Metronome</p>
            <div class="metronome-dot" style="animation: pulse {duration_s}s infinite ease-in-out;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feedback_form(tala_classes: list[str], file_id: str, original_tala: str, confidence: float) -> None:
    """Render a form for experts to submit corrections."""
    import requests
    from utils.config_loader import get_config
    try:
        API_URL = get_config().api.base_url
    except Exception:
        API_URL = "http://localhost:8000"
        
    st.markdown("### 📝 Expert Correction")
    st.markdown("If the predicted Tala is incorrect, please select the correct one below to improve future models.")
    
    with st.form(key=f"feedback_form_{file_id}"):
        # Pre-select the original prediction or "Other" if not in list
        default_idx = tala_classes.index(original_tala) if original_tala in tala_classes else 0
        
        correct_tala = st.selectbox(
            "Correct Tala:", 
            tala_classes + ["Other (Specify in notes)"],
            index=default_idx
        )
        notes = st.text_area("Additional Notes (optional)", placeholder="E.g., Difficult tempo change around 1:20")
        
        submit = st.form_submit_button("Submit Correction")
        
        if submit:
            payload = {
                "file_id": file_id,
                "filename": f"expert_feedback_{file_id}",
                "original_tala": original_tala,
                "corrected_tala": correct_tala,
                "confidence": confidence,
                "notes": notes
            }
            try:
                res = requests.post(f"{API_URL}/feedback", json=payload)
                if res.status_code == 200:
                    st.success("✅ Thank you! Your feedback has been recorded for the next training cycle.")
                else:
                    st.error(f"Failed to submit feedback: {res.text}")
            except Exception as e:
                st.error(f"Error connecting to feedback API: {e}")
