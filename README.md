# ASL Interpreter — Real-Time Sign Language Recognition

An AI-powered American Sign Language (ASL) recognition and translation project designed to make online communication more accessible.

The long-term goal is to develop a real-time sign language interpreter that can integrate with video conferencing platforms such as **Zoom and Microsoft Teams**, converting ASL into written English and synthesized speech.

The project currently focuses on fine-tuning a pretrained video transformer to recognize **50 isolated ASL signs**, establishing a foundation for future continuous sign language translation.

## Project Overview

Sign languages are visual languages that communicate through hand movements, facial expressions, and body positioning. Unlike traditional speech recognition, sign language interpretation requires understanding both spatial and temporal information.

This project explores the application of deep learning and computer vision to recognize ASL signs from video sequences.

### Goals

- Recognize ASL signs using computer vision and deep learning.
- Fine-tune pretrained video recognition models on ASL datasets.
- Develop a real-time recognition pipeline using webcam input.
- Convert recognized signs into text and synthesized speech.
- Explore continuous ASL recognition and sentence-level translation.
- Eventually integrate the interpreter with Zoom and Microsoft Teams.

> **Note:** The current model performs isolated-sign classification, not continuous sign language translation. Recognizing individual signs is an initial step toward the broader goal.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Pretrained model | VideoMAE Base |
| Model library | Hugging Face Transformers |
| Video processing | OpenCV |
| Numerical computing | NumPy |
| Dataset | ASL Citizen |
| Training hardware | NVIDIA RTX 5070 Ti (16 GB VRAM) |
| GPU acceleration | CUDA |
| Mixed precision | BF16 |

**Planned technologies:** FastAPI, React/TypeScript, MediaPipe, text-to-speech, and meeting-platform integration APIs.

## Model Architecture

The initial recognition system uses **VideoMAE (Video Masked Autoencoder)**, a pretrained video transformer capable of learning spatial and temporal representations from video sequences.

Instead of training a video recognition model from scratch, the pretrained VideoMAE encoder is fine-tuned for ASL classification.

### Recognition Pipeline

```text
ASL Video
    |
    v
Video Decoding (OpenCV)
    |
    v
Temporal Frame Sampling (16 Frames)
    |
    v
Image Preprocessing (224 x 224)
    |
    v
Pretrained VideoMAE Encoder
    |
    v
50-Class Classification Head
    |
    v
Predicted ASL Sign
```

The model receives 16 RGB frames per video and predicts one of 50 ASL sign classes.

### Model Configuration

| Parameter | Value |
|---|---|
| Base checkpoint | MCG-NJU/videomae-base |
| Architecture | Vision Transformer |
| Parameters | Approximately 94 million |
| Input frames | 16 |
| Input resolution | 224 × 224 |
| Output classes | 50 |
| Optimizer | AdamW |
| Initial learning rate | 2e-5 |
| Batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Precision | BF16 |
| Initial training epochs | 10 |