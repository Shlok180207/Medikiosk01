"""
MediKiosk v5 — Automated Google Colab GPU Setup & Runner
Runs 100% automated on Colab GPU: installs Ollama, pulls models, loads Whisper GPU,
launches FastAPI, and creates a public Cloudflare tunnel.
"""

import subprocess
import time
import sys
import os
import re

def run_cmd(cmd, desc=""):
    if desc:
        print(f"\n⚙️ {desc}...")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"⚠️ Warning: Command exited with code {res.returncode}")

def main():
    print("=" * 65)
    print("🏥 Starting MediKiosk v5 Automated Setup on Colab GPU")
    print("=" * 65)

    # 1. Install zstd & Ollama
    run_cmd("apt-get update -qq && apt-get install -y -qq zstd", "Installing zstd")
    run_cmd("curl -fsSL https://ollama.com/install.sh | sh", "Installing Ollama")

    # 2. Start Ollama daemon in background
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        print("✅ Ollama is already running!")
    except Exception:
        print("Starting Ollama daemon...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)

    # 3. Pull required clinical models
    run_cmd("ollama pull qwen2.5:7b", "Pulling Qwen 2.5 7B (Clinical Triage)")
    run_cmd("ollama pull qwen2.5:3b", "Pulling Qwen 2.5 3B (Fast Fallback)")
    run_cmd("ollama pull moondream", "Pulling Moondream (Vision OCR)")

    # 4. Install Python dependencies
    run_cmd(
        'pip install -q faster-whisper fastapi "uvicorn[standard]" pydantic sqlalchemy python-dotenv python-multipart gTTS ollama PyMuPDF pycloudflared',
        "Installing Python dependencies"
    )

    # 5. Ensure cloudflared binary is installed
    run_cmd(
        "which cloudflared > /dev/null || (wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && dpkg -i cloudflared-linux-amd64.deb)",
        "Ensuring cloudflared binary is installed"
    )

    # 6. Pre-generate TTS audio
    if os.path.exists("pre_generate_tts.py"):
        run_cmd(f"{sys.executable} pre_generate_tts.py", "Pre-generating TTS audio responses")

    # 7. Start FastAPI Backend in background with GPU environment variables
    backend_env = os.environ.copy()
    backend_env["OLLAMA_MODEL"] = "qwen2.5:7b"
    backend_env["FALLBACK_MODEL"] = "qwen2.5:3b"
    backend_env["VISION_MODEL"] = "moondream"
    backend_env["WHISPER_MODEL"] = "large-v3"

    print("\n🏥 Launching MediKiosk FastAPI Backend on GPU...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        env=backend_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    time.sleep(8)

    # 8. Establish Cloudflare Tunnel
    print("\nEstablishing Cloudflare HTTPS tunnel...")
    tunnel_url = None
    try:
        from pycloudflared import try_cloudflare
        tunnel = try_cloudflare(port=8000)
        tunnel_url = tunnel.tunnel.strip() if hasattr(tunnel, "tunnel") else getattr(tunnel, "tunnel_url", str(tunnel)).strip()
    except Exception:
        pass

    if not tunnel_url or "http" not in tunnel_url:
        cf_proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        for _ in range(25):
            time.sleep(1)
            line = cf_proc.stdout.readline() if cf_proc.stdout else ""
            m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if m:
                tunnel_url = m.group(0)
                break

    print("\n" + "=" * 65)
    print("🚀 MEDIKIOSK BACKEND IS LIVE ON GOOGLE COLAB GPU!")
    print("=" * 65)
    print(f"🌐 Public Base URL:     {tunnel_url}")
    print(f"🩺 API Endpoint:        {tunnel_url}/api")
    print(f"📖 Interactive Swagger: {tunnel_url}/docs")
    print("=" * 65)
    print("\n👉 Copy this line into your Antigravity 'frontend/.env' file:")
    print(f"VITE_API_BASE_URL={tunnel_url}/api\n")
    print("=" * 65 + "\n")

    # Stream backend logs in real-time
    try:
        for line in iter(backend_proc.stdout.readline, ""):
            print(line, end="")
    except KeyboardInterrupt:
        print("\nShutting down backend process...")
        backend_proc.terminate()

if __name__ == "__main__":
    main()
