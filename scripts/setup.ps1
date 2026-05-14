$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/filipp-lotsmanov/sign-language"
$ReleaseTag = "v0.1.0"
$DownloadUrl = "$RepoUrl/releases/download/$ReleaseTag"
$RequiredPython = "3.12"

Write-Host "==================================" -ForegroundColor Cyan
Write-Host " NGT Sign Language - Setup Script" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python (\d+\.\d+)") {
            $pythonCmd = $cmd
            break
        }
    } catch { }
}

if (-not $pythonCmd) {
    Write-Host "Error: Python is not installed." -ForegroundColor Red
    Write-Host "Install Python $RequiredPython+ from https://www.python.org/downloads/"
    exit 1
}

$pythonVersion = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$parts = $pythonVersion.Split(".")
$major = [int]$parts[0]
$minor = [int]$parts[1]

if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 12)) {
    Write-Host "Error: Python $RequiredPython+ required, found $pythonVersion" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Python $pythonVersion" -ForegroundColor Green

# Check uv
$uvInstalled = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvInstalled) {
    Write-Host ""
    Write-Host "uv not found. Installing..." -ForegroundColor Yellow
    irm https://astral.sh/uv/install.ps1 | iex
}
Write-Host "[OK] uv installed" -ForegroundColor Green

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..."
uv venv
.\.venv\Scripts\Activate.ps1
Write-Host "[OK] Virtual environment created" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..."
uv pip install -e .
Write-Host "[OK] Dependencies installed" -ForegroundColor Green

# Download model weights
Write-Host ""
Write-Host "Downloading model weights from release $ReleaseTag..."

New-Item -ItemType Directory -Force -Path "models\static" | Out-Null
New-Item -ItemType Directory -Force -Path "models\dynamic" | Out-Null

function Download-File {
    param([string]$Url, [string]$Dest)
    if (Test-Path $Dest) {
        Write-Host "  Exists: $Dest"
    } else {
        Write-Host "  Downloading: $Dest"
        Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
    }
}

Download-File "$DownloadUrl/static_best_model.pth" "models\static\best_model.pth"
Download-File "$DownloadUrl/static_label_encoder.pkl" "models\static\label_encoder.pkl"
Download-File "$DownloadUrl/dynamic_best_model.pth" "models\dynamic\best_model.pth"
Download-File "$DownloadUrl/dynamic_classes.npy" "models\dynamic\classes.npy"
Download-File "$DownloadUrl/hand_landmarker.task" "models\hand_landmarker.task"

Write-Host "[OK] Model weights downloaded" -ForegroundColor Green

# Done
Write-Host ""
Write-Host "==================================" -ForegroundColor Cyan
Write-Host " Setup complete!" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host " To run the app:"
Write-Host "   .\.venv\Scripts\Activate.ps1"
Write-Host "   python main.py"
Write-Host ""
Write-Host " Then open http://localhost:8000"
Write-Host ""
