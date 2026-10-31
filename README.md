# CoSigning — Sign Language Translator

Two-way translation between American Sign Language (ASL) and speech.

- **Speech → Sign:** speak (or type) a sentence and an avatar plays the sign for each word.
- **Sign → Speech:** sign in front of the webcam (or upload a video). A Transformer model reads hand and body movement (MediaPipe) and visual features (ViT), then turns the signs into an English sentence that is read aloud.

**Live demo:** https://laraomar25.github.io/CoSigning/

## Tech
Python · PyTorch · Vision Transformer (ViT) · MediaPipe · OpenCV · FastAPI · JavaScript · Web Speech API · HTML · CSS

## Run locally
```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --port 8000
```
Then open `index.html` in Chrome or Edge.

## Model
Cross-validation accuracy: about 96% (see `features_5.0/features_5.0/cv_results_5_0.csv`).
