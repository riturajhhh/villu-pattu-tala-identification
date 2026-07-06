"""
training/predict.py
====================
Inference and prediction engine for Tala identification.
Accepts an audio file path, runs preprocessing, feature extraction,
loads the best trained model (classical or deep learning), and returns
the predicted Tala along with confidence scores and audio characteristics.

Usage
-----
    python -m training.predict --audio path/to/sample.wav
    python -m training.predict --audio path/to/sample.wav --model-type cnn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extraction.feature_extractor import FeatureExtractor
from utils.logger import setup_logger

logger = setup_logger("villu_pattu.predict", level="INFO")


class TalaPredictor:
    """Predictor facade for handling Tala inference on audio files."""

    def __init__(self, model_dir: str | Path = "models/saved_models") -> None:
        self.model_dir = Path(model_dir)
        self.extractor = FeatureExtractor()
        
        # Lazy loaded models
        self.classical_pipeline = None
        self.label_encoder = None
        self.cnn_model = None
        self.crnn_model = None
        self.classes = None

    def _load_classical(self) -> bool:
        if self.classical_pipeline is not None:
            return True
        try:
            pipeline_path = self.model_dir / "best_classical_model.pkl"
            le_path = self.model_dir / "label_encoder.pkl"
            if not (pipeline_path.exists() and le_path.exists()):
                return False
            self.classical_pipeline = joblib.load(pipeline_path)
            self.label_encoder = joblib.load(le_path)
            self.classes = self.label_encoder.classes_.tolist()
            return True
        except Exception as exc:
            logger.error(f"Failed to load classical model: {exc}")
            return False

    def _load_cnn(self) -> bool:
        if self.cnn_model is not None:
            return True
        try:
            import torch
            from training.train_cnn import build_cnn_model
            
            meta_path = self.model_dir / "cnn_metadata.json"
            model_path = self.model_dir / "cnn_model.pt"
            if not (meta_path.exists() and model_path.exists()):
                return False
                
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.classes = meta["classes"]
            
            self.cnn_model = build_cnn_model(len(self.classes))
            self.cnn_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            self.cnn_model.eval()
            return True
        except Exception as exc:
            logger.error(f"Failed to load CNN model: {exc}")
            return False

    def _load_crnn(self) -> bool:
        if self.crnn_model is not None:
            return True
        try:
            import torch
            from training.train_crnn import build_crnn_model
            
            meta_path = self.model_dir / "crnn_metadata.json"
            model_path = self.model_dir / "crnn_model.pt"
            if not (meta_path.exists() and model_path.exists()):
                return False
                
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.classes = meta["classes"]
            
            self.crnn_model = build_crnn_model(len(self.classes))
            self.crnn_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            self.crnn_model.eval()
            return True
        except Exception as exc:
            logger.error(f"Failed to load CRNN model: {exc}")
            return False

    def predict(
        self,
        audio_path: str | Path,
        model_type: str = "auto",
    ) -> Optional[Dict[str, Any]]:
        """Predict the Tala for an audio file.

        Parameters
        ----------
        audio_path:
            Path to WAV/MP3/FLAC audio file.
        model_type:
            ``'classical'``, ``'cnn'``, ``'crnn'``, or ``'auto'`` (loads best available).

        Returns
        -------
        dict with prediction results or None.
        """
        import librosa
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None

        # Determine best available model type
        if model_type == "auto":
            if self._load_classical():
                model_type = "classical"
            elif self._load_cnn():
                model_type = "cnn"
            elif self._load_crnn():
                model_type = "crnn"
            else:
                logger.error("No trained models found in saved directory.")
                return None

        logger.info(f"Using {model_type.upper()} model for prediction ...")

        # Load models
        loaded = False
        if model_type == "classical":
            loaded = self._load_classical()
        elif model_type == "cnn":
            loaded = self._load_cnn()
        elif model_type == "crnn":
            loaded = self._load_crnn()

        if not loaded:
            logger.error(f"Failed to load model for type '{model_type}'")
            return None

        from preprocessing.audio_preprocessor import create_preprocessor
        from feature_extraction.rhythm_features import extract_tempo_and_beats

        # 1. Preprocess only ONCE
        preprocessor = create_preprocessor()
        try:
            y, sr = preprocessor.preprocess(audio_path)
        except Exception as exc:
            logger.error(f"Preprocessing failed: {exc}")
            return None

        # 2. Extract rhythm details directly from waveform (always needed for UI)
        rhythm_details = extract_tempo_and_beats(y, sr, hop_length=self.extractor.hop_length)
        bpm = rhythm_details.get("tempo_bpm", 0.0)
        pulse_clarity = rhythm_details.get("pulse_clarity", 0.0)
        beat_positions = rhythm_details.get("beat_positions", [])
        duration = float(len(y) / sr)

        # 3. Model-specific feature extraction
        probabilities = None
        predicted_idx = -1

        if model_type == "classical":
            # Extract tabular features only if using classical ML
            features = self.extractor.extract_from_waveform(y, sr)
            if features is None:
                logger.error("Feature extraction failed.")
                return None
                
            feature_cols = joblib.load(self.model_dir / "feature_columns.pkl")
            X = np.array([features[c] for c in feature_cols]).reshape(1, -1)
            
            # Predict probabilities
            probabilities = self.classical_pipeline.predict_proba(X)[0]
            predicted_idx = int(np.argmax(probabilities))
            
        else:
            # CNN / CRNN take 2D Mel Spectrogram image
            import torch
            # Extract mel image from the already processed waveform
            mel = self.extractor.extract_mel_image_from_waveform(y, sr)
            if mel is None:
                logger.error("Mel spectrogram extraction failed.")
                return None
                
            tensor_x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch & channel dims
            
            model = self.cnn_model if model_type == "cnn" else self.crnn_model
            with torch.no_grad():
                logits = model(tensor_x)
                probs = torch.softmax(logits, dim=1)[0]
                probabilities = probs.cpu().numpy()
                predicted_idx = int(np.argmax(probabilities))

        # Derive result variables from model outputs
        predicted_tala = self.classes[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        # Build sorted top-predictions list
        sorted_indices = np.argsort(probabilities)[::-1]
        top_predictions = [
            {"tala": self.classes[i], "confidence": float(probabilities[i])}
            for i in sorted_indices
        ]

        return {
            "predicted_tala": predicted_tala,
            "confidence": confidence,
            "bpm": bpm,
            "pulse_clarity": pulse_clarity,
            "top_predictions": top_predictions,
            "beat_positions": beat_positions,
            "model_used": model_type,
            "duration": duration,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Tala of a Villu Pattu audio clip")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--model-type", default="auto", choices=["auto", "classical", "cnn", "crnn"])
    parser.add_argument("--models-dir", default="models/saved_models")
    args = parser.parse_args()

    predictor = TalaPredictor(model_dir=args.models_dir)
    res = predictor.predict(args.audio, model_type=args.model_type)

    if res:
        print("\n" + "=" * 45)
        print("VILLU PATTU TALA PREDICTION RESULT")
        print("=" * 45)
        print(f"Predicted Tala  : {res['predicted_tala']} ({res['confidence']:.2%})")
        print(f"Tempo (BPM)     : {res['bpm']:.1f}")
        print(f"Pulse Clarity   : {res['pulse_clarity']:.2f}")
        print(f"Clip Duration   : {res['duration']:.2f} seconds")
        print(f"Model Used      : {res['model_used'].upper()}")
        print("-" * 45)
        print("All Predictions:")
        for pred in res["top_predictions"]:
            print(f"  - {pred['tala']:<15s}: {pred['confidence']:.2%}")
        print("=" * 45)
