#!/bin/bash
# =============================================================================
# Gradient Notebook Setup Script
# =============================================================================
# This script runs BEFORE Jupyter Lab starts, configuring the entire environment.
# Host on GitHub Gist and use in startup command:
#   curl -sSL https://gist.githubusercontent.com/USER/ID/raw/setup.sh | bash && jupyter lab ...
#
# All output is logged to /notebooks/startup.log for debugging
# =============================================================================

set -e  # Exit on any error

# Log file location (persistent storage)
LOG="/notebooks/startup.log"

# Logging function - writes to both log file and stdout
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG"
}

# Start fresh log
echo "==========================================" > "$LOG"
log "Gradient Notebook Setup Starting..."
log "=========================================="

# =============================================================================
# 1. SYSTEM PACKAGES (apt)
# =============================================================================
# - ffmpeg: Audio/video processing
# - libtbb12: Intel Threading Building Blocks (required by fairseq2)
# - libsndfile1: Audio file I/O library
# - libc++1, libc++abi1: LLVM C++ runtime (required by some ML libs)
# =============================================================================
log ""
log "[1/6] Installing system packages..."

apt update -qq >> "$LOG" 2>&1
apt install -y -qq \
    ffmpeg \
    libtbb12 \
    libsndfile1 \
    libc++1 \
    libc++abi1 \
    >> "$LOG" 2>&1

# Refresh shared library cache
ldconfig

log "   OK: System packages installed"

# =============================================================================
# 2. UV - Fast Python Package Installer
# =============================================================================
# uv is 10-100x faster than pip for installing packages
# Official install script from astral.sh
# =============================================================================
log ""
log "[2/6] Installing uv (fast pip replacement)..."

curl -LsSf https://astral.sh/uv/install.sh | sh >> "$LOG" 2>&1

# Add uv to PATH for this session
export PATH="$HOME/.local/bin:$PATH"

log "   OK: uv installed at $HOME/.local/bin/uv"

# =============================================================================
# 3. JUPYTER SERVER PROXY
# =============================================================================
# Allows accessing HTTP servers on internal ports via /proxy/PORT/
# This is how we expose our processing server to the outside world
# =============================================================================
log ""
log "[3/6] Installing jupyter-server-proxy..."

pip install -q jupyter-server-proxy >> "$LOG" 2>&1

log "   OK: jupyter-server-proxy installed"

# =============================================================================
# 4. CACHE DIRECTORY SYMLINKS
# =============================================================================
# Models are cached in ~/.cache but that's not persistent
# Symlink to /notebooks/.cache which IS persistent across restarts
# This saves ~10GB of re-downloading models each time
# =============================================================================
log ""
log "[4/6] Setting up cache symlinks..."

# Create persistent cache directories
mkdir -p /notebooks/.cache/fairseq2
mkdir -p /notebooks/.cache/huggingface
mkdir -p /notebooks/.cache/torch

# Remove existing cache dirs if they exist (not symlinks)
for cache_dir in fairseq2 huggingface torch; do
    home_cache="$HOME/.cache/$cache_dir"
    nb_cache="/notebooks/.cache/$cache_dir"

    if [ -d "$home_cache" ] && [ ! -L "$home_cache" ]; then
        # It's a real directory, move contents and replace with symlink
        cp -r "$home_cache"/* "$nb_cache/" 2>/dev/null || true
        rm -rf "$home_cache"
    fi

    if [ ! -L "$home_cache" ]; then
        mkdir -p "$(dirname "$home_cache")"
        ln -sf "$nb_cache" "$home_cache"
        log "   Linked $home_cache -> $nb_cache"
    fi
done

log "   OK: Cache symlinks configured"

# =============================================================================
# 5. ENVIRONMENT VARIABLES (SYSTEM-WIDE)
# =============================================================================
# Set environment variables globally so ALL processes inherit them:
# - /etc/environment: Read by PAM, applies to all users/processes
# - /etc/profile.d/: Read by login shells
# - Jupyter kernels will inherit these via the environment
# =============================================================================
log ""
log "[5/6] Configuring environment variables (system-wide)..."

# /etc/environment - System-wide, read by PAM (affects ALL processes including Jupyter kernels)
cat >> /etc/environment << 'EOF'
LD_LIBRARY_PATH="/notebooks/.venv/lib"
VIRTUAL_ENV="/notebooks/.venv"
FAIRSEQ2_CACHE_DIR="/notebooks/.cache/fairseq2"
HF_HOME="/notebooks/.cache/huggingface"
TORCH_HOME="/notebooks/.cache/torch"
EOF

# /etc/profile.d/ - For interactive shells (terminals)
cat > /etc/profile.d/gradient-setup.sh << 'EOF'
# Gradient Notebook Environment Setup
export LD_LIBRARY_PATH="/notebooks/.venv/lib:$LD_LIBRARY_PATH"
export VIRTUAL_ENV="/notebooks/.venv"
export PATH="/notebooks/.venv/bin:$HOME/.local/bin:$PATH"
export FAIRSEQ2_CACHE_DIR="/notebooks/.cache/fairseq2"
export HF_HOME="/notebooks/.cache/huggingface"
export TORCH_HOME="/notebooks/.cache/torch"
EOF
chmod +x /etc/profile.d/gradient-setup.sh

# Also add to .bashrc for non-login shells
cat >> ~/.bashrc << 'EOF'

# === Gradient Setup ===
export LD_LIBRARY_PATH="/notebooks/.venv/lib:$LD_LIBRARY_PATH"
export VIRTUAL_ENV="/notebooks/.venv"
export PATH="/notebooks/.venv/bin:$HOME/.local/bin:$PATH"
export FAIRSEQ2_CACHE_DIR="/notebooks/.cache/fairseq2"
export HF_HOME="/notebooks/.cache/huggingface"
export TORCH_HOME="/notebooks/.cache/torch"
EOF

log "   OK: Environment variables configured:"
log "      - /etc/environment (system-wide, all processes)"
log "      - /etc/profile.d/gradient-setup.sh (login shells)"
log "      - ~/.bashrc (interactive shells)"

# =============================================================================
# 6. START ASR SERVER (Background)
# =============================================================================
# Launch the ASR FastAPI server in background using the ASR venv
# Server runs on port 5000, accessible via /proxy/5000/
# =============================================================================
log ""
log "[6/6] Starting ASR server..."

ASR_VENV="/notebooks/asr"
ASR_SERVER="/notebooks/asr_serve.py"
ASR_LOG="/notebooks/asr_server.log"
ASR_PID="/notebooks/asr_server.pid"

if [ -d "$ASR_VENV" ] && [ -f "$ASR_SERVER" ]; then
    # Check if already running
    if [ -f "$ASR_PID" ]; then
        OLD_PID=$(cat "$ASR_PID" 2>/dev/null)
        if kill -0 "$OLD_PID" 2>/dev/null; then
            log "   ASR server already running (PID: $OLD_PID)"
        else
            rm -f "$ASR_PID"
        fi
    fi

    # Start if not running
    if [ ! -f "$ASR_PID" ]; then
        log "   Launching ASR server in background..."

        # Launch with proper environment
        (
            cd /notebooks
            source "$ASR_VENV/bin/activate"
            export LD_LIBRARY_PATH="$ASR_VENV/lib:$LD_LIBRARY_PATH"
            nohup python "$ASR_SERVER" > "$ASR_LOG" 2>&1 &
            echo $! > "$ASR_PID"
        )

        # Wait briefly and check if started
        sleep 2
        if [ -f "$ASR_PID" ] && kill -0 "$(cat "$ASR_PID")" 2>/dev/null; then
            log "   OK: ASR server started (PID: $(cat "$ASR_PID"))"
        else
            log "   WARN: ASR server may not have started - check $ASR_LOG"
        fi
    fi
else
    log "   SKIP: ASR venv or server not found"
    log "      - venv: $ASR_VENV (exists: $([ -d "$ASR_VENV" ] && echo yes || echo no))"
    log "      - server: $ASR_SERVER (exists: $([ -f "$ASR_SERVER" ] && echo yes || echo no))"
fi

# =============================================================================
# DONE!
# =============================================================================
log ""
log "=========================================="
log "Setup Complete!"
log "=========================================="
log ""
log "Installed:"
log "  - System: ffmpeg, libtbb12, libsndfile1, libc++"
log "  - Tools: uv (fast pip)"
log "  - Jupyter: jupyter-server-proxy"
log ""
log "Services:"
log "  - ASR server: port 5000 (via /proxy/5000/)"
log ""
log "Paths:"
log "  - ASR venv: /notebooks/asr"
log "  - cache: /notebooks/.cache"
log "  - uv: ~/.local/bin/uv"
log "  - logs: /notebooks/startup.log, /notebooks/asr_server.log"
log ""
log "Starting Jupyter Lab..."
log "=========================================="
