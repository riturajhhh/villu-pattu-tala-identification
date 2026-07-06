# AI-Based Tala Identification System for Villu Pattu Folk Music

This is a research-grade, production-ready artificial intelligence system designed to automatically identify the **Tala (rhythmic cycle)** of **Villu Pattu** (a traditional folk music form of Tamil Nadu and Kerala, South India) from audio recordings. The system processes uploaded audio files, detects beat locations, calculates the tempo (BPM), and classifies the recording into its respective rhythmic cycle using both classical Machine Learning models and Deep Learning convolutional architectures.

---

## 🎵 Rhythmic Cycle (Tala) Definitions

The system natively supports and classifies performances into five distinct categories:

| Tala Class | Beats (Aksharas) | Canonical Pattern (Structure) | Musical Description |
|---|---|---|---|
| **Adi** | 8 | `[4, 2, 2]` | 1 Laghu (4 beats) + 2 Drutams (2 beats each). Common in South Indian classical and folk music. |
| **Rupaka** | 6 | `[2, 4]` | 1 Drutam (2 beats) + 1 Laghu (4 beats). |
| **Misra Chapu** | 7 | `[3, 4]` | Fast folk syncopation grouped as 3 + 4 beats. Frequently heard in Villu Pattu. |
| **Khanda Chapu** | 5 | `[2, 3]` | Pentametric syncopation grouped as 2 + 3 beats. |
| **Other** | Varied | Varied | Represents alternative, mixed, or unrecognized folk meters. |

---

## 📁 System Architecture

```
folk music/
├── config/
│   └── config.yaml             # Central configuration (rhythm, audio, directories)
├── dataset/
│   ├── dataset_builder.py      # Combines audio files into flat CSV features and splits
│   ├── synthetic_generator.py  # Synthesises percussive templates to bootstrap pipelines
│   └── augmentation.py         # Waveform augmentations (pitch shift, time stretch, reverb)
├── preprocessing/
│   ├── audio_preprocessor.py   # Signal cleaning (mono, resample, trim, noise reduction)
│   └── preprocess.py           # CLI batch preprocessing utility
├── feature_extraction/
│   ├── time_domain.py          # ZCR, RMS energy, Amplitude Envelope
│   ├── frequency_domain.py     # MFCC, Chroma, Spectral descriptors, CQT, HPSS
│   ├── rhythm_features.py      # Onset strength, Tempograms, Rhythm Histograms, Pulse Clarity
│   └── feature_extractor.py    # Facade extracting tabular feature vectors & Mel images
├── training/
│   ├── train_classical.py      # Trains RF, XGBoost, SVM, KNN & Voting Ensemble
│   ├── train_cnn.py            # Trains PyTorch CNN on Mel Spectrogram maps
│   ├── train_crnn.py           # Trains PyTorch CRNN (CNN + LSTM) on sequences
│   ├── train.py                # Orchestrates training and prints comparisons
│   └── predict.py              # CLI predictor file for audio inference
├── evaluation/
│   ├── evaluate.py             # Generates classification reports and loss curves
│   ├── explainability.py       # SHAP / Feature importances and confidence rankings
│   └── visualizer.py           # Waveform, spec, beat track overlay matplotlib plots
├── api/
│   ├── main.py                 # FastAPI application startup & endpoints
│   ├── database.py             # SQLite prediction history database via SQLAlchemy
│   ├── schemas.py              # Pydantic schemas for endpoint schemas
│   └── routes/                 # Endpoint logic (predict, upload, features, history)
├── frontend/
│   ├── app.py                  # Streamlit dark mode frontend UI dashboard
│   └── components/             # Custom Plotly plots and custom metric widgets
├── Dockerfile                  # Production container recipe
├── docker-compose.yml          # Container configuration for API and frontend services
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation (this file)
```

---

## 🛠 Setup & Installation

### Option 1: Using Conda (Recommended)

1. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate villu_pattu
   ```

### Option 2: Using pip & virtualenv

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Option 3: Using Docker (Fastest deployment)

1. Build and launch all services with docker-compose:
   ```bash
   docker-compose up --build
   ```
   This will spin up:
   - The FastAPI backend at `http://localhost:8000`
   - The Streamlit interface at `http://localhost:8501`

---

## 🚀 Execution Workflow

Since no public database of Villu Pattu audio files exists, the system is designed to bootstrap itself using synthetic rhythmic generation. Follow these steps to set up, train, and run the pipeline:

### Step 1: Bootstrap the Dataset
Generate synthetic percussive audio recordings that encode the rhythm metrics of the specified Talas:
```bash
python -m dataset.synthetic_generator --samples 50 --duration 10
```
This populates the `dataset/` directory with foldered `.wav` samples for each class.

### Step 2: Extract Rhythmic and Spectral Features
Analyze the audio files and create a flat `.csv` dataset along with split folds:
```bash
python -m feature_extraction.extract_features
python -m dataset.dataset_builder
```

### Step 3: Run Model Training Pipeline
Train and compare traditional ML models along with CNN and CRNN architectures:
```bash
python -m training.train --epochs 10
```

### Step 4: Run the API and Web App
Launch the FastAPI server and Streamlit frontend in separate terminals (if not using Docker):

**Terminal 1: FastAPI**
```bash
python api/main.py
```

**Terminal 2: Streamlit**
```bash
streamlit run frontend/app.py
```
Open `http://localhost:8501` in your browser. Upload an audio recording, and explore the predicted Tala, beat tracks, onset envelopes, and confidence distributions!

---

## 🧪 Automated Testing

Verify system integrity by running unit tests:
```bash
pytest tests/ -v
```
These tests cover preprocessing transformations, feature extraction bounds, and API endpoint routing.
