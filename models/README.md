# Model Weights

This directory contains the trained model weights and the MediaPipe hand landmarker used for inference. These files are not tracked by git due to their size.

## Files

| File | Size | Description |
|------|------|-------------|
| `static/best_model.pth` | ~2.2 MB | ResidualMLP CNN for static letters (A-I, K-Y) |
| `dynamic/best_model.pth` | ~2.4 MB | Bidirectional LSTM for dynamic letters (J, Z) |
| `dynamic/classes.npy` | <1 KB | Class label mapping for the LSTM model |
| `hand_landmarker.task` | ~7.5 MB | MediaPipe hand landmark detection model |
| `static/label_encoder.pkl` | <1 KB | Label encoder for static model class mapping |

## How to obtain

### Option 1: Download from release

Download the model files from the [latest release](https://github.com/filipp-lotsmanov/sign-language/releases) and place them in this directory following the structure above.

### Option 2: Download MediaPipe model manually

The `hand_landmarker.task` file can be downloaded directly from Google:

```bash
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### Option 3: Train from scratch

See the [`training/`](../training/) directory for training scripts and instructions.
