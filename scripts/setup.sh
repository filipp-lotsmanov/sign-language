#!/usr/bin/env bash
set -e

REPO_URL="https://github.com/filipp-lotsmanov/sign-language"
RELEASE_TAG="v0.1.0"
DOWNLOAD_URL="$REPO_URL/releases/download/$RELEASE_TAG"

REQUIRED_PYTHON="3.12"

echo "=================================="
echo " NGT Sign Language - Setup Script"
echo "=================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    echo "Install Python $REQUIRED_PYTHON+ from https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]; }; then
    echo "Error: Python $REQUIRED_PYTHON+ required, found $PYTHON_VERSION"
    exit 1
fi
echo "[OK] Python $PYTHON_VERSION"

# Check uv
if ! command -v uv &> /dev/null; then
    echo ""
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
echo "[OK] uv installed"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
uv venv
source .venv/bin/activate
echo "[OK] Virtual environment created"

# Install dependencies
echo ""
echo "Installing dependencies..."
uv pip install -e .
echo "[OK] Dependencies installed"

# Download model weights
echo ""
echo "Downloading model weights from release $RELEASE_TAG..."

mkdir -p models/static models/dynamic

download_file() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "  Exists: $dest"
    else
        echo "  Downloading: $dest"
        curl -L --fail --silent --show-error -o "$dest" "$url"
    fi
}

download_file "$DOWNLOAD_URL/static_best_model.pth" "models/static/best_model.pth"
download_file "$DOWNLOAD_URL/static_label_encoder.pkl" "models/static/label_encoder.pkl"
download_file "$DOWNLOAD_URL/dynamic_best_model.pth" "models/dynamic/best_model.pth"
download_file "$DOWNLOAD_URL/dynamic_classes.npy" "models/dynamic/classes.npy"
download_file "$DOWNLOAD_URL/hand_landmarker.task" "models/hand_landmarker.task"

echo "[OK] Model weights downloaded"

# Done
echo ""
echo "=================================="
echo " Setup complete!"
echo "=================================="
echo ""
echo " To run the app:"
echo "   source .venv/bin/activate"
echo "   python main.py"
echo ""
echo " Then open http://localhost:8000"
echo ""
