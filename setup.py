#!/usr/bin/env python3
"""
Basketball Stats Tracker — Script de instalación y verificación.
Correlo antes de la primera vez que uses la app.
"""
import subprocess, sys, importlib, os, shutil, platform, urllib.request

# ── Colores ANSI ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
CHECK  = f"{GREEN}✅{RESET}"
CROSS  = f"{RED}❌{RESET}"
WARN   = f"{YELLOW}⚠️{RESET}"

# Habilitar ANSI en Windows
if platform.system() == "Windows":
    os.system("")

print(f"\n{BOLD}🏀 Basketball Stats Tracker v2.0 — Setup{RESET}")
print("=" * 50)

# ══════════════════════════════════════════════════════════════════════════
#  1. DEPENDENCIAS PYTHON
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{CYAN}── Dependencias Python ──{RESET}")

REQUIRED = [
    ("ultralytics", "ultralytics>=8.0"),
    ("fastapi",     "fastapi"),
    ("uvicorn",     "uvicorn[standard]"),
    ("cv2",         "opencv-python"),
    ("numpy",       "numpy"),
    ("multipart",   "python-multipart"),
]

missing = []
for mod, pkg in REQUIRED:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "")
        print(f"  {CHECK} {pkg} {f'({ver})' if ver else ''}")
    except ImportError:
        print(f"  {CROSS} {pkg} — FALTA")
        missing.append(pkg)

if missing:
    print(f"\n{YELLOW}Instalando {len(missing)} paquetes faltantes...{RESET}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    print(f"{CHECK} Instalación completa.")
else:
    print(f"\n{CHECK} Todas las dependencias instaladas.")

# ══════════════════════════════════════════════════════════════════════════
#  2. MODELO YOLO
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{CYAN}── Modelo YOLOv8 ──{RESET}")
try:
    from ultralytics import YOLO
    model_name = "yolov8s.pt"
    print(f"  Descargando/verificando {model_name}...")
    m = YOLO(model_name)
    print(f"  {CHECK} Modelo cargado: {model_name}")
    basket_model = "basketball_shot_yolov8.pt"
    basket_url = "https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker/raw/master/best.pt"
    if not os.path.exists(basket_model):
        print(f"  Descargando modelo especializado basket ({basket_model})...")
        urllib.request.urlretrieve(basket_url, basket_model)
    bm = YOLO(basket_model)
    print(f"  {CHECK} Modelo basket cargado: {basket_model} {bm.names}")
except Exception as e:
    print(f"  {WARN} Error con YOLO: {e}")
    print(f"  El modelo se descargará automáticamente al detectar el primer video.")

# ══════════════════════════════════════════════════════════════════════════
#  3. GPU / CUDA
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{CYAN}── Hardware ──{RESET}")
gpu_available = False
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  {CHECK} GPU detectada: {BOLD}{gpu_name}{RESET} ({vram:.1f} GB VRAM)")
        print(f"  {CHECK} CUDA versión: {torch.version.cuda}")
        gpu_available = True
    else:
        print(f"  {WARN} GPU no disponible — se usará CPU")
        print(f"      Para GPU, instalá CUDA y corré:")
        print(f"      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
except ImportError:
    print(f"  {WARN} PyTorch no instalado — se usará CPU")
    print(f"      YOLO instalará una versión CPU automáticamente.")

# ══════════════════════════════════════════════════════════════════════════
#  4. FFMPEG
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{CYAN}── FFmpeg (audio en video anotado) ──{RESET}")
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        version_line = result.stdout.split("\n")[0] if result.stdout else "versión desconocida"
        print(f"  {CHECK} ffmpeg encontrado: {ffmpeg_path}")
        print(f"      {version_line}")
    except Exception:
        print(f"  {CHECK} ffmpeg encontrado: {ffmpeg_path}")
else:
    print(f"  {WARN} ffmpeg NO encontrado")
    print(f"      El video anotado se generará sin audio.")
    print(f"      Para instalar ffmpeg:")
    if platform.system() == "Windows":
        print(f"      → Descargá de https://ffmpeg.org/download.html")
        print(f"      → O con chocolatey: choco install ffmpeg")
        print(f"      → O con winget: winget install Gyan.FFmpeg")
    else:
        print(f"      → sudo apt install ffmpeg  (Debian/Ubuntu)")
        print(f"      → brew install ffmpeg  (macOS)")

# ══════════════════════════════════════════════════════════════════════════
#  5. ESTIMACIÓN DE VELOCIDAD
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{CYAN}── Velocidad estimada ──{RESET}")
print(f"  Para un video de 10 minutos:")
if gpu_available:
    print(f"  {CHECK} GPU: ~1-3 minutos para detección + análisis")
else:
    print(f"  {WARN} CPU: ~8-15 minutos para detección + análisis")
print(f"      (El video anotado toma tiempo adicional)")

# ══════════════════════════════════════════════════════════════════════════
#  6. DIRECTORIOS
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{CYAN}── Directorios ──{RESET}")
base = os.path.dirname(os.path.abspath(__file__))
for d in ["uploads", "processed", "static"]:
    path = os.path.join(base, d)
    os.makedirs(path, exist_ok=True)
    print(f"  {CHECK} {d}/")

# ══════════════════════════════════════════════════════════════════════════
#  RESUMEN
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 50}")
print(f"{BOLD}🚀 Para arrancar el servidor corré:{RESET}")
print(f"   {CYAN}python server.py{RESET}")
print(f"\n{BOLD}🌐 Luego abrí:{RESET}")
print(f"   {CYAN}http://localhost:8000{RESET}")
print()
