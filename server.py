"""
Basketball Stats Tracker v2.0 - Servidor Local
Usa YOLOv8 para detectar jugadores, pelota y aro en video de pickup basketball.
Incluye motor de estadísticas avanzado con máquina de estados para detección de tiros.
"""

import os, sys, json, uuid, time, math, threading, queue, subprocess, shutil
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Rutas del proyecto ─────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROC_DIR   = BASE_DIR / "processed"
STATIC_DIR = BASE_DIR / "static"
for d in [UPLOAD_DIR, PROC_DIR, STATIC_DIR]:
    d.mkdir(exist_ok=True)

# ── Fix encoding para consola Windows ──────────────────────────────────────
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass  # En algunos entornos reconfigure no está disponible

# ── Modelo YOLO ────────────────────────────────────────────────────────────
# El usuario puede cambiar el modelo desde la UI
MODEL_NAME = 'yolov8s.pt'
BALL_HOOP_MODEL_NAME = 'basketball_shot_yolov8.pt'

# Modelos disponibles con descripción para la UI
AVAILABLE_MODELS = {
    'yolov8n.pt': {
        'name': 'YOLOv8 Nano',
        'quality': '⭐⭐ Básica',
        'speed': '⚡⚡⚡ Muy rápido',
        'size': '~6 MB',
        'description': 'Ideal para videos largos o PCs sin GPU. Menor precisión.',
        'time_10min_cpu': '5-8 min',
        'time_10min_gpu': '~30 seg',
    },
    'yolov8s.pt': {
        'name': 'YOLOv8 Small',
        'quality': '⭐⭐⭐ Buena',
        'speed': '⚡⚡ Rápido',
        'size': '~22 MB',
        'description': 'Balance ideal entre velocidad y precisión. Recomendado.',
        'time_10min_cpu': '8-15 min',
        'time_10min_gpu': '~1 min',
    },
    'yolov8m.pt': {
        'name': 'YOLOv8 Medium',
        'quality': '⭐⭐⭐⭐ Alta',
        'speed': '⚡ Lento',
        'size': '~50 MB',
        'description': 'Máxima precisión. Recomendado solo con GPU dedicada.',
        'time_10min_cpu': '20-35 min',
        'time_10min_gpu': '~2 min',
    },
}

# ── Aplicación FastAPI ─────────────────────────────────────────────────────
app = FastAPI(title="Basketball Tracker v2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/processed", StaticFiles(directory=str(PROC_DIR)), name="processed")

# ── Carga diferida del modelo YOLO ─────────────────────────────────────────
_model = None
_ball_hoop_model = None
_model_lock = threading.Lock()
_ball_hoop_model_lock = threading.Lock()

def get_model():
    """Carga el modelo YOLO de forma diferida y thread-safe."""
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO
            print(f'🧠 Cargando modelo {MODEL_NAME}...')
            _model = YOLO(MODEL_NAME)
            print('✅ Modelo cargado correctamente')
        return _model

def get_ball_hoop_model():
    """Carga el modelo especializado en pelota y aro si esta disponible."""
    global _ball_hoop_model
    model_path = BASE_DIR / BALL_HOOP_MODEL_NAME
    if not model_path.exists():
        return None

    with _ball_hoop_model_lock:
        if _ball_hoop_model is None:
            from ultralytics import YOLO
            print(f'🏀 Cargando modelo basket {BALL_HOOP_MODEL_NAME}...')
            _ball_hoop_model = YOLO(str(model_path))
            print(f'✅ Modelo basket cargado: {_ball_hoop_model.names}')
        return _ball_hoop_model

# ── Almacén de jobs en memoria ─────────────────────────────────────────────
jobs: dict = {}

# ── IDs de clase COCO relevantes ───────────────────────────────────────────
PERSON_CLASS      = 0
SPORTS_BALL_CLASS = 32
BASKETBALL_ALIASES = {'basketball', 'ball'}
HOOP_ALIASES       = {'basketball hoop', 'hoop', 'rim', 'basket', 'net'}

# ── Estados de la máquina de estados de tiros ──────────────────────────────
SHOT_IDLE        = 0   # Sin actividad de tiro
SHOT_APPROACHING = 1   # Pelota moviéndose hacia el aro
SHOT_NEAR_HOOP   = 2   # Pelota en zona del aro


def _create_job(video_path: str, filename: str) -> dict:
    """Crea la estructura de datos inicial para un nuevo job."""
    return {
        'status': 'uploaded',
        'progress': 0,
        'stage': '',
        'detail': '',
        'video_path': video_path,
        'filename': filename,
        'frames_data': None,
        'player_map': {},
        'stats': {},
        'annotated_video': None,
        'detection_result': None,
        'error': None,
        'game_format': '5v5',       # formato del partido: 1v1, 2v2, 3v3, 4v4, 5v5
        'expected_players': 10,     # jugadores esperados en cancha
        '_raw_track_data': {},
        '_ball_positions': [],
        '_hoop_region': {},
        '_manual_calibration': None,
        '_fps': 30,
        '_total_frames': 0,
        '_wh': (1280, 720),
    }


# ══════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════════════

def _check_gpu() -> bool:
    """Verifica si hay GPU CUDA disponible."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _class_ids_by_alias(model, aliases):
    names = getattr(model, 'names', {}) or {}
    return {
        int(cls_id)
        for cls_id, name in names.items()
        if str(name).strip().lower() in aliases
    }


def _detect_basket_objects(model, frame, frame_idx, conf=0.15):
    """Devuelve detecciones del modelo especializado: pelotas y aros."""
    ball_ids = _class_ids_by_alias(model, BASKETBALL_ALIASES)
    hoop_ids = _class_ids_by_alias(model, HOOP_ALIASES)
    balls, hoops = [], []

    results = model.predict(frame, conf=conf, iou=0.5, verbose=False)
    if not results or results[0].boxes is None:
        return balls, hoops

    boxes = results[0].boxes
    for i, raw_cls_id in enumerate(boxes.cls.tolist()):
        cls_id = int(raw_cls_id)
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        det_conf = float(boxes.conf[i])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        item = {
            "frame": frame_idx, "x": cx, "y": cy,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "w": x2 - x1, "h": y2 - y1, "conf": det_conf
        }
        if cls_id in ball_ids:
            balls.append(item)
        elif cls_id in hoop_ids:
            hoops.append(item)

    return balls, hoops


def _estimate_hoop(ball_positions, h, w, hoop_positions=None):
    """
    Estima la posición del aro usando el 10% superior de posiciones de la pelota.
    Usa las posiciones más altas (y más bajo numéricamente) ponderadas por ubicación.
    """
    if hoop_positions:
        best = sorted(hoop_positions, key=lambda p: p.get("conf", 0), reverse=True)
        best = best[:min(50, len(best))]
        cx = float(np.mean([p["x"] for p in best]))
        cy = float(np.mean([p["y"] for p in best]))
        box_sizes = [max(p.get("w", 0), p.get("h", 0)) for p in best]
        radius = float(max(h * 0.035, np.median(box_sizes) * 0.45))
        return {
            "cx": cx, "cy": cy, "radius": radius,
            "source": "basketball_shot_yolov8",
            "detections": len(hoop_positions),
        }

    if not ball_positions:
        # Valor por defecto: zona central-superior del frame
        return {"cx": w * 0.5, "cy": h * 0.2, "radius": h * 0.08, "source": "default"}

    # Tomar el 10% superior de posiciones de la pelota (las de menor y)
    n_top = max(1, len(ball_positions) // 10)
    top_positions = sorted(ball_positions, key=lambda b: b["y"])[:n_top]

    cx = float(np.mean([b["x"] for b in top_positions]))
    cy = float(np.mean([b["y"] for b in top_positions]))
    radius = float(h * 0.08)

    return {"cx": cx, "cy": cy, "radius": radius, "source": "ball_trajectory"}


def _consolidate_tracks(track_data, expected_players, fps, sample_every):
    """
    Consolida tracks fragmentados que probablemente pertenecen a la misma persona.
    Usa un algoritmo de fusión iterativo greedy:
    1. Calcula la "posición promedio" de cada track
    2. Busca pares de tracks con bajo solapamiento temporal
    3. Fusiona el par más compatible (menor solapamiento + menor distancia)
    4. Repite hasta llegar al número esperado de jugadores
    """
    if len(track_data) <= expected_players:
        return track_data

    # Construir dict mutable de tracks
    tracks = {}
    for tid, td in track_data.items():
        frames_set = set(td['frames'])
        bboxes = sorted(td['bbox_history'], key=lambda b: b['frame'])
        # Posición promedio del track
        avg_cx = np.mean([b['cx'] for b in bboxes]) if bboxes else 0
        avg_cy = np.mean([b['cy'] for b in bboxes]) if bboxes else 0
        tracks[tid] = {
            'frames': frames_set,
            'bbox_history': bboxes,
            'thumbnail': td.get('thumbnail'),
            'first_frame': td.get('first_frame', 0),
            'avg_cx': avg_cx,
            'avg_cy': avg_cy,
        }

    def _overlap_pct(a, b):
        """Porcentaje de frames compartidos entre dos tracks."""
        if not a or not b:
            return 0.0
        overlap = len(a & b)
        return overlap / min(len(a), len(b))

    def _merge_score(tid_a, tid_b):
        """
        Score de compatibilidad para fusionar dos tracks.
        Retorna None si son incompatibles, o un score (menor = mejor).
        """
        ta = tracks[tid_a]
        tb = tracks[tid_b]
        overlap = _overlap_pct(ta['frames'], tb['frames'])
        # Si se solapan más del 30%, son personas distintas
        if overlap > 0.30:
            return None
        # Distancia espacial promedio
        spatial_dist = math.hypot(
            ta['avg_cx'] - tb['avg_cx'],
            ta['avg_cy'] - tb['avg_cy']
        )
        # Score: combinar solapamiento y distancia
        # Penalizar solapamiento mucho más que distancia
        score = spatial_dist + overlap * 5000
        return score

    def _do_merge(main_tid, other_tid):
        """Fusiona other_tid en main_tid."""
        main = tracks[main_tid]
        other = tracks[other_tid]
        main['frames'].update(other['frames'])
        main['bbox_history'] = sorted(
            main['bbox_history'] + other['bbox_history'],
            key=lambda b: b['frame']
        )
        # Recalcular posición promedio
        all_bboxes = main['bbox_history']
        main['avg_cx'] = np.mean([b['cx'] for b in all_bboxes])
        main['avg_cy'] = np.mean([b['cy'] for b in all_bboxes])
        # Mantener el mejor thumbnail (del track con más frames originales)
        if not main.get('thumbnail') and other.get('thumbnail'):
            main['thumbnail'] = other['thumbnail']
        del tracks[other_tid]

    # Iteración: seguir fusionando hasta llegar al target
    max_iterations = 500  # safety limit
    iteration = 0
    while len(tracks) > expected_players and iteration < max_iterations:
        iteration += 1
        best_score = None
        best_pair = None

        tids = list(tracks.keys())
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                score = _merge_score(tids[i], tids[j])
                if score is not None and (best_score is None or score < best_score):
                    best_score = score
                    # Fusionar el más pequeño en el más grande
                    if len(tracks[tids[i]]['frames']) >= len(tracks[tids[j]]['frames']):
                        best_pair = (tids[i], tids[j])
                    else:
                        best_pair = (tids[j], tids[i])

        if best_pair is None:
            break  # No hay más pares fusionables

        _do_merge(best_pair[0], best_pair[1])
        print(f'    ⤴️ Fusionado track {best_pair[1]} -> {best_pair[0]} (score: {best_score:.0f}, quedan: {len(tracks)})')

    # Convertir de vuelta al formato esperado
    result = {}
    for tid, td in tracks.items():
        result[tid] = {
            'frames': sorted(td['frames']),
            'bbox_history': td['bbox_history'],
            'thumbnail': td.get('thumbnail'),
            'first_frame': td.get('first_frame', 0),
        }
    return result


def _match_manual_players_to_tracks(manual_players, tracks, w, h):
    """Empareja jugadores marcados en el primer frame con tracks detectados."""
    if not manual_players or not tracks:
        return {}, {}

    matches = {}
    suggestions = {}
    used_tracks = set()

    def _track_score(player, tid, td):
        bbox_hist = sorted(td.get('bbox_history', []), key=lambda b: b.get('frame', 0))
        if not bbox_hist:
            return None

        # Priorizar los primeros bboxes: la calibracion se hizo sobre el frame 0.
        candidates = bbox_hist[:min(8, len(bbox_hist))]
        px = player.get('x', 0) + player.get('w', 0) / 2
        py = player.get('y', 0) + player.get('h', 0) / 2
        pw = max(1, player.get('w', 1))
        ph = max(1, player.get('h', 1))
        diag = math.hypot(w, h)

        best = None
        for b in candidates:
            bw = max(1, b.get('x2', 0) - b.get('x1', 0))
            bh = max(1, b.get('y2', 0) - b.get('y1', 0))
            dist = math.hypot(px - b.get('cx', 0), py - b.get('cy', 0)) / diag
            size_delta = abs(pw - bw) / max(pw, bw) + abs(ph - bh) / max(ph, bh)
            score = dist * 4 + size_delta * 0.35
            if best is None or score < best:
                best = score
        return best

    for player in manual_players:
        best_tid = None
        best_score = None
        for tid, td in tracks.items():
            if tid in used_tracks:
                continue
            score = _track_score(player, tid, td)
            if score is not None and (best_score is None or score < best_score):
                best_score = score
                best_tid = tid

        # Umbral amplio para videos con perspectiva baja y deteccion inicial imperfecta.
        if best_tid is not None and best_score is not None and best_score < 0.8:
            used_tracks.add(best_tid)
            info = {
                "name": player.get("name") or f"Jugador {len(matches) + 1}",
                "team": player.get("team", "A"),
                "color": player.get("color", "#58a6ff"),
            }
            matches[str(best_tid)] = info
            suggestions[str(best_tid)] = {**info, "manual_match_score": best_score}

    return matches, suggestions


def _try_merge_audio(original_path: str, annotated_path: str) -> str:
    """Intenta fusionar el audio del video original al anotado usando ffmpeg."""
    try:
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            print('⚠️  ffmpeg no encontrado, video sin audio')
            return annotated_path
        temp_out = annotated_path.replace('.mp4', '_audio.mp4')
        subprocess.run([
            ffmpeg, '-y',
            '-i', annotated_path,
            '-i', original_path,
            '-c:v', 'copy', '-c:a', 'aac',
            '-map', '0:v:0', '-map', '1:a:0?',
            '-shortest', temp_out
        ], capture_output=True, timeout=120)
        if os.path.exists(temp_out) and os.path.getsize(temp_out) > 0:
            os.replace(temp_out, annotated_path)
            print('🔊 Audio fusionado correctamente')
        return annotated_path
    except Exception as e:
        print(f'⚠️  Error fusionando audio: {e}')
        return annotated_path


def _hex_to_bgr(hex_color: str) -> tuple:
    """Convierte color hexadecimal (#RRGGBB) a tupla BGR para OpenCV."""
    hex_c = hex_color.lstrip("#")
    if len(hex_c) != 6:
        hex_c = "3498db"
    r, g, b = (int(hex_c[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


# ══════════════════════════════════════════════════════════════════════════
#  ENDPOINTS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Sirve la página principal (index.html)."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Recibe el video y lo almacena para procesamiento posterior."""
    job_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{job_id}{ext}"

    # Guardar archivo de video al disco
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    jobs[job_id] = _create_job(str(video_path), file.filename)

    return {"job_id": job_id, "filename": file.filename}


@app.get("/first_frame/{job_id}")
async def first_frame(job_id: str):
    """Devuelve el primer frame del video para calibracion manual."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job no encontrado"}, status_code=404)

    job = jobs[job_id]
    frame_path = PROC_DIR / f"{job_id}_first_frame.jpg"
    if not frame_path.exists():
        cap = cv2.VideoCapture(job["video_path"])
        if not cap.isOpened():
            return JSONResponse({"error": "No se pudo abrir el video"}, status_code=500)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return JSONResponse({"error": "No se pudo leer el primer frame"}, status_code=500)
        cv2.imwrite(str(frame_path), frame)
    return FileResponse(str(frame_path), media_type="image/jpeg")


@app.get("/progress/{job_id}")
async def progress_stream(job_id: str):
    """
    Endpoint SSE (Server-Sent Events) para transmitir progreso en tiempo real.
    El cliente recibe actualizaciones cada 500ms hasta que el job termina.
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job no encontrado"}, status_code=404)

    def _generate():
        """Generador que envía eventos SSE con el progreso del job."""
        # Estados finales que detienen el streaming
        estados_finales = {'done', 'error', 'detected', 'assigned'}
        while True:
            if job_id not in jobs:
                # El job fue eliminado
                payload = json.dumps({
                    'status': 'error',
                    'progress': 0,
                    'stage': 'Job eliminado',
                    'detail': ''
                })
                yield f"data: {payload}\n\n"
                break

            job = jobs[job_id]
            payload = json.dumps({
                'status': job['status'],
                'progress': job['progress'],
                'stage': job['stage'],
                'detail': job['detail']
            })
            yield f"data: {payload}\n\n"

            if job['status'] in estados_finales:
                break

            time.sleep(0.5)

    return StreamingResponse(
        _generate(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        }
    )


@app.post("/detect/{job_id}")
async def detect_objects(job_id: str):
    """
    Lanza la detección YOLO con tracking en segundo plano.
    Devuelve inmediatamente y el progreso se puede seguir vía /progress/{job_id}.
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job no encontrado"}, status_code=404)

    job = jobs[job_id]
    if job['status'] not in ('uploaded', 'calibrated'):
        return JSONResponse(
            {"error": f"Estado inválido: {job['status']}. Se requiere 'uploaded'."},
            status_code=400
        )

    job['status'] = 'detecting'
    job['progress'] = 0
    job['stage'] = 'Iniciando detección...'

    # Lanzar hilo de detección en segundo plano
    threading.Thread(target=_detect_worker, args=(job_id,), daemon=True).start()

    return {'status': 'detecting', 'job_id': job_id}


@app.post("/configure/{job_id}")
async def configure_game(job_id: str, data: dict):
    """
    Configura el formato del partido antes de la detección.
    Recibe: { "game_format": "1v1" } donde el valor puede ser 1v1, 2v2, 3v3, 4v4, 5v5.
    Esto permite filtrar tracks de forma inteligente.
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job no encontrado"}, status_code=404)

    job = jobs[job_id]
    game_format = data.get('game_format', '5v5')
    valid_formats = {'1v1': 2, '2v2': 4, '3v3': 6, '4v4': 8, '5v5': 10}

    if game_format not in valid_formats:
        return JSONResponse(
            {'error': f'Formato inválido. Opciones: {list(valid_formats.keys())}'},
            status_code=400
        )

    job['game_format'] = game_format
    job['expected_players'] = valid_formats[game_format]
    return {'ok': True, 'game_format': game_format, 'expected_players': valid_formats[game_format]}


@app.post("/calibrate/{job_id}")
async def calibrate_court(job_id: str, data: dict):
    """Guarda aro y jugadores marcados manualmente sobre el primer frame."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job no encontrado"}, status_code=404)

    hoop = data.get("hoop") or {}
    players = data.get("players") or []
    frame = data.get("frame") or {}

    if not hoop or "x" not in hoop or "y" not in hoop:
        return JSONResponse({"error": "Marcá la posición del aro antes de calibrar"}, status_code=400)
    if not players:
        return JSONResponse({"error": "Marcá al menos un jugador"}, status_code=400)

    cleaned_players = []
    for i, p in enumerate(players):
        try:
            cleaned_players.append({
                "id": i + 1,
                "name": (p.get("name") or f"Jugador {i + 1}").strip(),
                "team": p.get("team", "A") if p.get("team") in ("A", "B") else "A",
                "color": p.get("color", "#58a6ff"),
                "x": float(p.get("x", 0)),
                "y": float(p.get("y", 0)),
                "w": max(1.0, float(p.get("w", 1))),
                "h": max(1.0, float(p.get("h", 1))),
            })
        except Exception:
            continue

    if not cleaned_players:
        return JSONResponse({"error": "No se pudieron leer los jugadores marcados"}, status_code=400)

    calibration = {
        "hoop": {
            "cx": float(hoop["x"]),
            "cy": float(hoop["y"]),
            "radius": max(8.0, float(hoop.get("radius", 40))),
            "source": "manual_first_frame",
        },
        "players": cleaned_players,
        "frame": {
            "width": int(frame.get("width", 0) or 0),
            "height": int(frame.get("height", 0) or 0),
        },
    }

    job = jobs[job_id]
    job["_manual_calibration"] = calibration
    job["_hoop_region"] = calibration["hoop"]
    if job["status"] == "uploaded":
        job["status"] = "calibrated"
    return {"ok": True, "players": len(cleaned_players), "hoop": calibration["hoop"]}


@app.post("/assign_players/{job_id}")
async def assign_players(job_id: str, data: dict):
    """
    Recibe el mapeo track_id -> { name, team, color } del usuario.
    Ejemplo: { "players": { "3": { "name": "Juan", "team": "A", "color": "#e74c3c" }, ... } }
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job no encontrado"}, status_code=404)

    job = jobs[job_id]
    job["player_map"] = data.get("players", {})
    job["status"] = "assigned"
    return {"ok": True, "players": len(job["player_map"])}


@app.post("/merge_tracks/{job_id}")
async def merge_tracks(job_id: str, data: dict):
    """
    Fusiona dos tracks manualmente (el usuario identifica que son la misma persona).
    Recibe: { "keep": 2, "merge": 147 } donde keep es el track a mantener
    y merge es el track a absorber.
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job no encontrado"}, status_code=404)

    job = jobs[job_id]
    keep_id = data.get('keep')
    merge_id = data.get('merge')

    if keep_id is None or merge_id is None:
        return JSONResponse({'error': 'Se requieren keep y merge'}, status_code=400)

    # Convertir a int para comparar con las claves del track_data
    keep_id = int(keep_id)
    merge_id = int(merge_id)

    raw = job.get('_raw_track_data', {})
    dr = job.get('detection_result', {})

    if keep_id not in raw or merge_id not in raw:
        return JSONResponse({'error': 'Track ID no encontrado'}, status_code=404)

    # Fusionar frames y bbox_history
    keep_track = raw[keep_id]
    merge_track = raw[merge_id]
    merged_frames = sorted(set(keep_track['frames']) | set(merge_track['frames']))
    merged_bboxes = sorted(
        keep_track.get('bbox_history', []) + merge_track.get('bbox_history', []),
        key=lambda b: b['frame']
    )
    keep_track['frames'] = merged_frames
    keep_track['bbox_history'] = merged_bboxes

    # Eliminar el track absorbido
    del raw[merge_id]

    # Actualizar detection_result
    tracks_dict = dr.get('tracks', {})
    if str(keep_id) in tracks_dict:
        tracks_dict[str(keep_id)]['frame_count'] = len(merged_frames)
    if str(merge_id) in tracks_dict:
        del tracks_dict[str(merge_id)]
    dr['track_ids'] = [tid for tid in dr.get('track_ids', []) if tid != merge_id]

    print(f'\U0001f517 Tracks fusionados: {merge_id} -> {keep_id} (job {job_id})')

    return {
        'ok': True,
        'kept': keep_id,
        'merged': merge_id,
        'new_frame_count': len(merged_frames),
        'remaining_tracks': len(raw),
    }


@app.post("/analyze/{job_id}")
async def analyze_game(job_id: str, data: dict = {}):
    """
    Lanza el análisis estadístico completo en segundo plano.
    Requiere que los jugadores ya estén asignados.
    """
    if job_id not in jobs:
        return JSONResponse({"error": "Job no encontrado"}, status_code=404)

    job = jobs[job_id]
    if job["status"] not in ("assigned", "detected"):
        return JSONResponse(
            {"error": "Primero detectá y asigná jugadores"},
            status_code=400
        )

    # Guardar opción de skip video
    job['skip_video'] = data.get('skip_video', False) if isinstance(data, dict) else False

    job['status'] = 'analyzing'
    job['progress'] = 0
    job['stage'] = 'Iniciando análisis...'

    # Lanzar hilo de análisis en segundo plano
    threading.Thread(target=_analyze_worker, args=(job_id,), daemon=True).start()

    return {'status': 'analyzing', 'job_id': job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Devuelve el estado completo del job incluyendo progreso y resultados."""
    if job_id not in jobs:
        return JSONResponse({"error": "No encontrado"}, status_code=404)
    j = jobs[job_id]
    return {
        "status": j["status"],
        "progress": j.get("progress", 0),
        "stage": j.get("stage", ""),
        "detail": j.get("detail", ""),
        "stats": j.get("stats"),
        "annotated_video": j.get("annotated_video"),
        "detection_result": j.get("detection_result"),
        "error": j.get("error"),
        "filename": j.get("filename"),
    }


@app.post("/reset")
async def reset_all():
    """Limpia todos los jobs y archivos temporales."""
    jobs.clear()
    # Limpiar directorio de uploads
    for f in UPLOAD_DIR.iterdir():
        try:
            if f.is_file():
                f.unlink(missing_ok=True)
        except Exception:
            pass
    # Limpiar directorio de procesados
    for f in PROC_DIR.iterdir():
        try:
            if f.is_file():
                f.unlink(missing_ok=True)
        except Exception:
            pass
    return {'ok': True, 'message': 'Todo limpiado'}


@app.get("/health")
async def health():
    """Endpoint de salud del servidor con información del entorno."""
    return {
        'status': 'ok',
        'model': MODEL_NAME,
        'ball_hoop_model': BALL_HOOP_MODEL_NAME if (BASE_DIR / BALL_HOOP_MODEL_NAME).exists() else None,
        'gpu': _check_gpu(),
        'ffmpeg': shutil.which('ffmpeg') is not None,
        'jobs_count': len(jobs),
    }


@app.get("/models")
async def list_models():
    """Devuelve los modelos YOLO disponibles y cuál está activo."""
    return {
        'current': MODEL_NAME,
        'models': AVAILABLE_MODELS,
    }


@app.post("/set_model")
async def set_model(data: dict):
    """Cambia el modelo YOLO activo. Se recarga en la próxima detección."""
    global MODEL_NAME, _model
    model_name = data.get('model', '')
    if model_name not in AVAILABLE_MODELS:
        return JSONResponse(
            {'error': f'Modelo no válido. Opciones: {list(AVAILABLE_MODELS.keys())}'},
            status_code=400
        )
    if model_name != MODEL_NAME:
        MODEL_NAME = model_name
        with _model_lock:
            _model = None  # Forzar recarga en la próxima detección
        print(f'🔄 Modelo cambiado a {MODEL_NAME}')
    return {'ok': True, 'model': MODEL_NAME, 'info': AVAILABLE_MODELS[MODEL_NAME]}


# ══════════════════════════════════════════════════════════════════════════
#  WORKERS DE SEGUNDO PLANO
# ══════════════════════════════════════════════════════════════════════════

def _detect_worker(job_id: str):
    """
    Worker de detección que corre en un hilo separado.
    Ejecuta YOLOv8 con tracking sobre todo el video, actualizando
    el progreso del job en tiempo real.
    """
    job = jobs[job_id]
    try:
        model = get_model()
        ball_hoop_model = get_ball_hoop_model()
        video_path = job["video_path"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Estructuras de tracking
        track_data: dict = {}     # track_id -> datos del track
        ball_positions: list = []
        hoop_positions: list = []

        frame_idx = 0
        # Procesar ~5 fps para balancear velocidad y precisión
        SAMPLE_EVERY = max(1, int(fps / 5))

        if ball_hoop_model is not None:
            job['stage'] = 'Detectando jugadores, pelota y aro...'
        else:
            job['stage'] = 'Detectando jugadores y pelota...'

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % SAMPLE_EVERY == 0:
                results = model.track(
                    frame, persist=True,
                    classes=[PERSON_CLASS] if ball_hoop_model is not None else [PERSON_CLASS, SPORTS_BALL_CLASS],
                    conf=0.3, iou=0.6, verbose=False,
                    tracker='bytetrack.yaml'
                )

                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i, raw_cls_id in enumerate(boxes.cls.tolist()):
                        cls_id = int(raw_cls_id)
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                        conf = float(boxes.conf[i])
                        tid = int(boxes.id[i]) if boxes.id is not None else -1

                        if cls_id == PERSON_CLASS and tid >= 0:
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            if tid not in track_data:
                                track_data[tid] = {
                                    "frames": [],
                                    "thumbnail": None,
                                    "bbox_history": [],
                                    "first_frame": frame_idx,
                                    "_best_conf": 0.0,
                                    "_best_crop_pending": None,
                                }
                            track_data[tid]["frames"].append(frame_idx)
                            track_data[tid]["bbox_history"].append({
                                "frame": frame_idx,
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "cx": cx, "cy": cy, "conf": conf
                            })

                            # Selección de mejor thumbnail: guardar bbox con mayor confianza
                            if conf > track_data[tid]["_best_conf"]:
                                track_data[tid]["_best_conf"] = conf
                                pad = 10
                                crop = frame[
                                    max(0, int(y1) - pad):min(h, int(y2) + pad),
                                    max(0, int(x1) - pad):min(w, int(x2) + pad)
                                ]
                                if crop.size > 0:
                                    track_data[tid]["_best_crop_pending"] = crop.copy()

                        elif cls_id == SPORTS_BALL_CLASS:
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            ball_positions.append({
                                "frame": frame_idx, "x": cx, "y": cy,
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "conf": conf
                            })

                if ball_hoop_model is not None:
                    balls, hoops = _detect_basket_objects(ball_hoop_model, frame, frame_idx)
                    ball_positions.extend(balls)
                    hoop_positions.extend([hdet for hdet in hoops if hdet.get("conf", 0) >= 0.25])

            # Actualizar progreso
            if total_frames > 0:
                job['progress'] = int(frame_idx / total_frames * 100)
                job['detail'] = f'Frame {frame_idx}/{total_frames}'

            frame_idx += 1

        cap.release()

        # ── Guardar thumbnails con la mejor detección de cada track ─────────
        job['stage'] = 'Generando thumbnails...'
        for tid, td in track_data.items():
            best_crop = td.pop("_best_crop_pending", None)
            td.pop("_best_conf", None)
            if best_crop is not None and best_crop.size > 0:
                crop_small = cv2.resize(best_crop, (80, 120))
                thumb_path = PROC_DIR / f"{job_id}_player_{tid}.jpg"
                cv2.imwrite(str(thumb_path), crop_small)
                td["thumbnail"] = f"/processed/{job_id}_player_{tid}.jpg"

        # ── Estimar posición del aro ───────────────────────────────────────
        manual_calibration = job.get("_manual_calibration") or {}
        hoop_region = _estimate_hoop(ball_positions, h, w, hoop_positions)
        if manual_calibration.get("hoop"):
            hoop_region = manual_calibration["hoop"]

        # ── Filtrado inteligente basado en formato del partido ────────────
        expected_players = job.get('expected_players', 10)
        game_format = job.get('game_format', '5v5')

        job['stage'] = 'Consolidando tracks...'

        # Umbral mínimo de frames adaptivo
        video_duration_frames = total_frames / SAMPLE_EVERY
        min_frames_pct = max(5, int(video_duration_frames * 0.03))
        min_frames_time = max(5, int(fps / SAMPLE_EVERY * 2))
        min_frames = max(min_frames_pct, min_frames_time)

        print(f'📊 Formato: {game_format} ({expected_players} jugadores esperados)')
        print(f'📊 Tracks crudos: {len(track_data)}, umbral min_frames: {min_frames}')

        # Paso 1: filtrar tracks muy cortos
        filtered_tracks = {
            tid: td for tid, td in track_data.items()
            if len(td['frames']) >= min_frames
        }
        print(f'📊 Tracks tras filtro básico: {len(filtered_tracks)}')

        # Paso 2: consolidar tracks fragmentados
        consolidated = _consolidate_tracks(filtered_tracks, expected_players, fps, SAMPLE_EVERY)
        print(f'📊 Tracks tras consolidación: {len(consolidated)}')

        # Paso 3: limitar a los tracks más relevantes
        sorted_tracks = sorted(
            consolidated.items(),
            key=lambda x: len(x[1]['frames']),
            reverse=True
        )
        max_tracks = min(len(sorted_tracks), expected_players + 4)
        valid_tracks = dict(sorted_tracks[:max_tracks])

        manual_player_map, manual_suggestions = _match_manual_players_to_tracks(
            manual_calibration.get("players", []), valid_tracks, w, h
        )
        if manual_player_map:
            job["player_map"] = manual_player_map

        print(f'✅ Tracks finales mostrados: {len(valid_tracks)}')

        # ── Construir resultado de detección ──────────────────────────────
        detection_result = {
            "track_ids": list(valid_tracks.keys()),
            "tracks": {
                str(tid): {
                    "thumbnail": td.get("thumbnail"),
                    "frame_count": len(td["frames"]),
                    "first_frame": td["first_frame"],
                    "suggested_player": manual_suggestions.get(str(tid)),
                } for tid, td in valid_tracks.items()
            },
            "ball_detections": len(ball_positions),
            "hoop_detections": len(hoop_positions),
            "total_frames": total_frames,
            "fps": fps,
            "resolution": f"{w}x{h}",
            "hoop_region": hoop_region,
            "ball_hoop_model": BALL_HOOP_MODEL_NAME if ball_hoop_model is not None else None,
            "game_format": game_format,
            "expected_players": expected_players,
            "raw_tracks_found": len(track_data),
            "manual_calibration": bool(manual_calibration),
            "manual_players_matched": len(manual_player_map),
        }

        # ── Guardar resultados en el job ──────────────────────────────────
        job["status"] = "detected"
        job["progress"] = 100
        job["stage"] = "Detección completada"
        job["detail"] = f"{len(valid_tracks)} jugadores (de {len(track_data)} tracks crudos), {len(ball_positions)} pelotas, {len(hoop_positions)} aros"
        job["detection_result"] = detection_result
        job["_raw_track_data"] = valid_tracks
        job["_ball_positions"] = ball_positions
        job["_hoop_region"] = hoop_region
        job["_fps"] = fps
        job["_total_frames"] = total_frames
        job["_wh"] = (w, h)

        print(f"✅ Detección completada para job {job_id}: "
              f"{len(valid_tracks)} tracks, {len(ball_positions)} pelotas")

    except Exception as e:
        import traceback
        job["status"] = "error"
        job["error"] = str(e)
        job["stage"] = "Error en detección"
        job["detail"] = traceback.format_exc()
        print(f"❌ Error en detección job {job_id}: {e}")


def _analyze_worker(job_id: str):
    """
    Worker de análisis que corre en un hilo separado.
    Etapa 1 (0-60%): cálculo de estadísticas
    Etapa 2 (60-95%): generación de video anotado
    Etapa 3 (95-100%): fusión de audio con ffmpeg
    """
    job = jobs[job_id]
    try:
        player_map   = job["player_map"]
        track_data   = job.get("_raw_track_data", {})
        ball_pos     = job.get("_ball_positions", [])
        hoop         = job.get("_hoop_region", {})
        fps          = job.get("_fps", 30)
        total_frames = job.get("_total_frames", 0)
        w, h         = job.get("_wh", (1280, 720))
        SAMPLE_EVERY = max(1, int(fps / 5))

        # ── Etapa 1: Cálculo de estadísticas (0-60%) ──────────────────────
        def progress_stats(pct):
            """Callback de progreso para el cálculo de estadísticas."""
            job['progress'] = int(pct * 0.6)
            job['stage'] = 'Calculando estadísticas...'

        stats = _compute_stats(
            player_map=player_map,
            track_data=track_data,
            ball_positions=ball_pos,
            hoop=hoop,
            fps=fps,
            total_frames=total_frames,
            w=w, h=h,
            sample_every=SAMPLE_EVERY,
            progress_callback=progress_stats,
        )
        job['stats'] = stats

        # ── Etapa 2: Generación de video anotado (60-95%) - OPCIONAL ──────
        annotated_path = None
        skip_video = job.get('skip_video', False)

        if not skip_video:
            def progress_video(pct):
                job['progress'] = 60 + int(pct * 0.35)
                job['stage'] = 'Generando video anotado...'

            annotated_path = _generate_annotated_video(
                job_id, job, player_map, hoop, stats,
                progress_callback=progress_video,
            )

            # ── Etapa 3: Fusión de audio (95-100%) ───────────────────────
            job['progress'] = 95
            job['stage'] = 'Fusionando audio...'
            if annotated_path:
                full_annotated = str(PROC_DIR / annotated_path.lstrip("/processed/"))
                annotated_path_final = _try_merge_audio(job["video_path"], full_annotated)
                annotated_path = f"/processed/{Path(annotated_path_final).name}"
        else:
            job['progress'] = 95
            job['stage'] = 'Video anotado omitido'

        # ── Completado ───────────────────────────────────────────────────
        job["annotated_video"] = annotated_path
        job["status"] = "done"
        job["progress"] = 100
        job["stage"] = "Análisis completado"
        job["detail"] = f"{len(player_map)} jugadores analizados"

        print(f"✅ Análisis completado para job {job_id}")

    except Exception as e:
        import traceback
        job["status"] = "error"
        job["error"] = str(e)
        job["stage"] = "Error en análisis"
        job["detail"] = traceback.format_exc()
        print(f"❌ Error en análisis job {job_id}: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  MOTOR DE ESTADÍSTICAS AVANZADO
# ══════════════════════════════════════════════════════════════════════════

def _compute_stats(player_map, track_data, ball_positions, hoop, fps,
                   total_frames, w, h, sample_every=1, progress_callback=None):
    """
    Calcula estadísticas estilo NBA por jugador usando una máquina de estados
    para detección precisa de tiros, rebotes, robos y asistencias.

    Máquina de estados de tiro:
      SHOT_IDLE -> SHOT_APPROACHING -> SHOT_NEAR_HOOP -> (resultado: FGM o FGA)

    Parámetros:
        player_map: dict de track_id_str -> {name, team, color}
        track_data: dict de track_id_int -> {frames, bbox_history, ...}
        ball_positions: lista de posiciones de la pelota
        hoop: dict con cx, cy, radius del aro
        fps: frames por segundo del video
        total_frames: total de frames del video
        w, h: dimensiones del frame
        sample_every: cada cuántos frames se muestreó
        progress_callback: función(0-100) para reportar progreso
    """

    # ── Inicializar estadísticas por jugador ──────────────────────────────
    player_stats = {}
    for tid_str, pinfo in player_map.items():
        player_stats[tid_str] = {
            "name": pinfo.get("name", f"Jugador {tid_str}"),
            "team": pinfo.get("team", "A"),
            "color": pinfo.get("color", "#3498db"),
            "PTS": 0, "AST": 0, "REB": 0, "OREB": 0, "DREB": 0,
            "FGM": 0, "FGA": 0, "FG_PCT": 0.0,
            "3PM": 0, "3PA": 0, "3P_PCT": 0.0,
            "FTM": 0, "FTA": 0, "FT_PCT": 0.0,
            "TO": 0, "STL": 0, "BLK": 0, "PF": 0,
            "MIN": 0.0,
            "PLUS_MINUS": 0,
        }

    # ── Tiempo en cancha (minutos) ─────────────────────────────────────────
    # Multiplicar por sample_every porque solo vemos 1 de cada N frames
    for tid_str in player_map:
        tid = int(tid_str)
        if tid in track_data:
            frames_vistos = len(track_data[tid]["frames"])
            minutos = round((frames_vistos * sample_every) / fps / 60, 1)
            player_stats[tid_str]["MIN"] = max(minutos, 0.1)

    # ── Si no hay datos suficientes, retornar stats vacías ────────────────
    if not ball_positions or not player_map or not track_data:
        return player_stats

    # ── Preparar índices de posición de pelota ────────────────────────────
    ball_by_frame = {b["frame"]: b for b in ball_positions}
    sorted_ball_frames = sorted(ball_by_frame.keys())

    if not sorted_ball_frames:
        return player_stats

    # ── Constantes de distancia (ajustadas para pickup filmado desde el piso) ─
    # La perspectiva de cámara baja distorsiona las distancias reales
    POSSESSION_DIST = h * 0.35       # Radio amplio para posesión (perspectiva baja)
    hoop_cx = hoop.get("cx", w / 2)
    hoop_cy = hoop.get("cy", h * 0.2)
    hoop_radius = hoop.get("radius", h * 0.08)
    HOOP_OUTER_R    = hoop_radius * 5.0    # Radio exterior amplio para "cerca del aro"
    HOOP_INNER_R    = HOOP_OUTER_R * 0.5   # Radio interior para "enceste"
    THREE_PT_DIST   = h * 0.55             # Línea de 3 puntos estimada

    print(f'📏 Stats engine configurado:')
    print(f'   Frame: {w}x{h}, FPS: {fps}, Sample: {sample_every}')
    print(f'   Aro estimado: ({hoop_cx:.0f}, {hoop_cy:.0f}), radio: {hoop_radius:.0f}')
    print(f'   POSSESSION_DIST: {POSSESSION_DIST:.0f}px')
    print(f'   HOOP_OUTER_R: {HOOP_OUTER_R:.0f}px, HOOP_INNER_R: {HOOP_INNER_R:.0f}px')
    print(f'   Detecciones pelota: {len(sorted_ball_frames)}')
    print(f'   Jugadores: {list(player_map.keys())}')

    # Debug: verificar que tracks tienen datos
    for tid_str in player_map:
        tid_int = int(tid_str)
        if tid_int in track_data:
            td = track_data[tid_int]
            n_frames = len(td.get('frames', []))
            n_bboxes = len(td.get('bbox_history', []))
            print(f'   Track {tid_str}: {n_frames} frames, {n_bboxes} bboxes')
        else:
            print(f'   ⚠️ Track {tid_str}: NO ENCONTRADO en track_data (claves: {list(track_data.keys())[:5]})')

    # ── Construir mapa rápido de posiciones de jugadores por frame ────────
    # Para cada frame, guardar la posición más cercana disponible de cada jugador
    def _get_player_pos_at_frame(tid, frame):
        """Obtiene la posición del jugador en un frame específico o el más cercano."""
        tid_int = int(tid)
        if tid_int not in track_data:
            return None
        bbox_hist = track_data[tid_int]["bbox_history"]
        if not bbox_hist:
            return None
        # Búsqueda binaria simplificada: encontrar el frame más cercano
        best = None
        best_diff = float('inf')
        for b in bbox_hist:
            diff = abs(b["frame"] - frame)
            if diff < best_diff:
                best_diff = diff
                best = b
        # Si la detección más cercana está a más de 2 segundos, no es válida
        # (era 1 segundo, ampliado para videos con tracking intermitente)
        if best_diff > fps * 2:
            return None
        return best

    # ── Secuencia de posesión ────────────────────────────────────────────
    possession_sequence = []
    prev_holder = None
    prev_holder_frame = None

    # ── Estado de la máquina de tiros ────────────────────────────────────
    shot_state = SHOT_IDLE
    shot_shooter = None           # Jugador que lanzó
    shot_shooter_pos = None       # Posición del lanzador al soltar la pelota
    shot_release_frame = None     # Frame en que se soltó el tiro
    last_passer = None            # Último jugador que pasó la pelota

    # ── Marcador por equipo (para plus/minus) ────────────────────────────
    team_score = {"A": 0, "B": 0}

    # ── Registro de tiros fallidos para rebotes ──────────────────────────
    pending_rebound = None  # (frame_miss, shooter_tid_str)

    # ── Contadores de debug ──────────────────────────────────────────────
    dbg_possessions = 0
    dbg_approaches = 0
    dbg_near_hoop = 0
    dbg_makes = 0
    dbg_misses = 0

    total_ball_frames = len(sorted_ball_frames)

    for idx, frame in enumerate(sorted_ball_frames):
        # Reportar progreso
        if progress_callback and idx % 50 == 0:
            progress_callback(int(idx / total_ball_frames * 100))

        ball = ball_by_frame[frame]
        bx, by = ball["x"], ball["y"]

        # ── Encontrar jugador más cercano a la pelota ────────────────────
        closest_tid_str = None
        closest_dist = float("inf")

        for tid_str in player_map:
            pos = _get_player_pos_at_frame(tid_str, frame)
            if pos is None:
                continue
            dist = math.hypot(bx - pos["cx"], by - pos["cy"])
            if dist < closest_dist:
                closest_dist = dist
                closest_tid_str = tid_str

        # ── Determinar poseedor ──────────────────────────────────────
        holder = closest_tid_str if closest_dist < POSSESSION_DIST else None
        possession_sequence.append((frame, holder))
        if holder is not None:
            dbg_possessions += 1

        # ── Detección de turnover / steal ────────────────────────────────
        if prev_holder and holder and holder != prev_holder:
            # Cambio de posesión
            if prev_holder_frame is not None:
                tiempo_entre = (frame - prev_holder_frame) / fps
                prev_team = player_stats.get(prev_holder, {}).get("team")
                curr_team = player_stats.get(holder, {}).get("team")

                # Si cambio rápido entre equipos distintos: turnover + steal
                if (tiempo_entre < 1.0 and prev_team and curr_team
                        and prev_team != curr_team):
                    player_stats[prev_holder]["TO"] += 1
                    player_stats[holder]["STL"] += 1

            last_passer = prev_holder

        # ── Rebote pendiente: buscar quién agarra la pelota ──────────────
        if pending_rebound is not None:
            miss_frame, shooter_str = pending_rebound
            frames_desde_miss = (frame - miss_frame) / fps
            if frames_desde_miss > 2.0:
                # Ya pasaron 2 segundos, cancelar búsqueda de rebote
                pending_rebound = None
            elif holder is not None:
                # Alguien agarró la pelota → rebote
                rebounder = holder
                player_stats[rebounder]["REB"] += 1
                # Clasificar como ofensivo o defensivo
                shooter_team = player_stats.get(shooter_str, {}).get("team")
                rebounder_team = player_stats.get(rebounder, {}).get("team")
                if shooter_team and rebounder_team:
                    if rebounder_team == shooter_team:
                        player_stats[rebounder]["OREB"] += 1
                    else:
                        player_stats[rebounder]["DREB"] += 1
                else:
                    player_stats[rebounder]["DREB"] += 1
                pending_rebound = None

        # ── Distancia de la pelota al aro ────────────────────────────────
        ball_to_hoop = math.hypot(bx - hoop_cx, by - hoop_cy)

        # ── Máquina de estados de tiro ───────────────────────────────────
        if shot_state == SHOT_IDLE:
            # Verificar si la pelota se mueve hacia el aro
            if ball_to_hoop < HOOP_OUTER_R * 1.5 and prev_holder is not None:
                # La pelota se acerca al aro y alguien la tenía
                shot_state = SHOT_APPROACHING
                shot_shooter = prev_holder
                shot_release_frame = frame
                pos = _get_player_pos_at_frame(shot_shooter, frame)
                shot_shooter_pos = pos
                dbg_approaches += 1

        elif shot_state == SHOT_APPROACHING:
            if ball_to_hoop < HOOP_OUTER_R:
                # Pelota entró en zona del aro
                shot_state = SHOT_NEAR_HOOP
                dbg_near_hoop += 1
            elif ball_to_hoop > HOOP_OUTER_R * 2.5:
                # La pelota se alejó demasiado, cancelar
                shot_state = SHOT_IDLE
                shot_shooter = None
                shot_shooter_pos = None

        elif shot_state == SHOT_NEAR_HOOP:
            if ball_to_hoop < HOOP_INNER_R:
                # ── ENCESTE (FGM) ────────────────────────────────────────
                if shot_shooter and shot_shooter in player_stats:
                    # Determinar si fue triple usando posición del LANZADOR
                    is_three = False
                    if shot_shooter_pos:
                        shooter_to_hoop = math.hypot(
                            shot_shooter_pos["cx"] - hoop_cx,
                            shot_shooter_pos["cy"] - hoop_cy
                        )
                        is_three = shooter_to_hoop > THREE_PT_DIST
                    else:
                        # Fallback: usar posición de la pelota al soltar
                        is_three = False

                    pts = 3 if is_three else 2
                    player_stats[shot_shooter]["FGM"] += 1
                    player_stats[shot_shooter]["FGA"] += 1
                    player_stats[shot_shooter]["PTS"] += pts
                    if is_three:
                        player_stats[shot_shooter]["3PM"] += 1
                        player_stats[shot_shooter]["3PA"] += 1
                    dbg_makes += 1
                    print(f'   🏀 ENCESTE: {player_stats[shot_shooter]["name"]} +{pts}pts (frame {frame})')

                    # ── Asistencia: último pasador distinto al lanzador ──
                    if (last_passer and last_passer != shot_shooter
                            and last_passer in player_stats):
                        last_passer_team = player_stats[last_passer]["team"]
                        shooter_team = player_stats[shot_shooter]["team"]
                        if last_passer_team == shooter_team:
                            player_stats[last_passer]["AST"] += 1

                    # ── Plus/Minus: actualizar para todos los jugadores ──
                    shooter_team = player_stats[shot_shooter]["team"]
                    team_score[shooter_team] = team_score.get(shooter_team, 0) + pts
                    for tid_s, ps in player_stats.items():
                        if ps["team"] == shooter_team:
                            ps["PLUS_MINUS"] += pts
                        else:
                            ps["PLUS_MINUS"] -= pts

                # Resetear estado de tiro
                shot_state = SHOT_IDLE
                shot_shooter = None
                shot_shooter_pos = None
                last_passer = None

            elif ball_to_hoop > HOOP_OUTER_R * 1.3:
                # ── TIRO FALLIDO (FGA sin FGM) ───────────────────────────
                if shot_shooter and shot_shooter in player_stats:
                    # Determinar si fue intento de triple
                    is_three = False
                    if shot_shooter_pos:
                        shooter_to_hoop = math.hypot(
                            shot_shooter_pos["cx"] - hoop_cx,
                            shot_shooter_pos["cy"] - hoop_cy
                        )
                        is_three = shooter_to_hoop > THREE_PT_DIST

                    player_stats[shot_shooter]["FGA"] += 1
                    if is_three:
                        player_stats[shot_shooter]["3PA"] += 1
                    dbg_misses += 1
                    print(f'   ❌ TIRO FALLIDO: {player_stats[shot_shooter]["name"]} (frame {frame})')

                    # Activar búsqueda de rebote
                    pending_rebound = (frame, shot_shooter)

                # Resetear estado de tiro
                shot_state = SHOT_IDLE
                shot_shooter = None
                shot_shooter_pos = None
                last_passer = None

        # ── Actualizar poseedor previo ───────────────────────────────────
        if holder is not None:
            prev_holder = holder
            prev_holder_frame = frame

    # ── Calcular porcentajes de tiro ──────────────────────────────────────
    for ps in player_stats.values():
        ps["FG_PCT"] = round(ps["FGM"] / ps["FGA"], 3) if ps["FGA"] > 0 else 0.0
        ps["3P_PCT"] = round(ps["3PM"] / ps["3PA"], 3) if ps["3PA"] > 0 else 0.0
        ps["FT_PCT"] = round(ps["FTM"] / ps["FTA"], 3) if ps["FTA"] > 0 else 0.0

    # ── Resumen de debug ────────────────────────────────────────────────
    print(f'\n📊 RESUMEN STATS ENGINE:')
    print(f'   Detecciones de posesión: {dbg_possessions}/{total_ball_frames} frames')
    print(f'   Transiciones APPROACHING: {dbg_approaches}')
    print(f'   Transiciones NEAR_HOOP: {dbg_near_hoop}')
    print(f'   Encestes (FGM): {dbg_makes}')
    print(f'   Tiros fallidos: {dbg_misses}')
    for tid_str, ps in player_stats.items():
        print(f'   {ps["name"]}: {ps["PTS"]}pts {ps["REB"]}reb {ps["AST"]}ast {ps["STL"]}stl {ps["TO"]}to')

    if progress_callback:
        progress_callback(100)

    return player_stats


# ══════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE VIDEO ANOTADO
# ══════════════════════════════════════════════════════════════════════════

def _generate_annotated_video(job_id, job, player_map, hoop, stats,
                              progress_callback=None):
    """
    Genera un video anotado con bounding boxes, nombres de jugadores,
    marcador en vivo y indicador de posesión.
    """
    try:
        model = get_model()
        ball_hoop_model = get_ball_hoop_model()
        video_path = job["video_path"]
        out_filename = f"{job_id}_annotated.mp4"
        out_path = str(PROC_DIR / out_filename)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        # ── Preparar colores por jugador ──────────────────────────────────
        color_map = {}
        for tid_str, pinfo in player_map.items():
            color_map[int(tid_str)] = _hex_to_bgr(pinfo.get("color", "#3498db"))

        # ── Calcular marcador por equipo ──────────────────────────────────
        score_a = sum(ps["PTS"] for ps in stats.values() if ps["team"] == "A")
        score_b = sum(ps["PTS"] for ps in stats.values() if ps["team"] == "B")

        # ── Constantes para dibujo ────────────────────────────────────────
        FUENTE = cv2.FONT_HERSHEY_SIMPLEX
        GROSOR_TEXTO = 2
        ESCALA_NOMBRE = 0.55
        ESCALA_MARCADOR = 0.7

        frame_idx = 0
        current_possession = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame, persist=True,
                classes=[PERSON_CLASS] if ball_hoop_model is not None else [PERSON_CLASS, SPORTS_BALL_CLASS],
                conf=0.35, verbose=False
            )

            ball_pos_this_frame = None

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                closest_to_ball_dist = float('inf')
                closest_to_ball_name = None

                for i, raw_cls_id in enumerate(boxes.cls.tolist()):
                    cls_id = int(raw_cls_id)
                    x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                    tid = int(boxes.id[i]) if boxes.id is not None else -1
                    tid_str = str(tid)

                    if cls_id == PERSON_CLASS and tid_str in player_map:
                        color = color_map.get(tid, (100, 200, 100))
                        name = player_map[tid_str].get("name", f"#{tid}")

                        # Dibujar bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Fondo del label con padding
                        text_size = cv2.getTextSize(name, FUENTE, ESCALA_NOMBRE, GROSOR_TEXTO)[0]
                        label_w = text_size[0] + 12
                        label_h = text_size[1] + 10
                        label_y1 = max(y1 - label_h - 2, 0)
                        label_y2 = max(y1 - 2, label_h)

                        # Rectángulo de fondo para el nombre
                        cv2.rectangle(
                            frame,
                            (x1, label_y1),
                            (x1 + label_w, label_y2),
                            color, -1
                        )
                        cv2.putText(
                            frame, name,
                            (x1 + 6, label_y2 - 4),
                            FUENTE, ESCALA_NOMBRE,
                            (255, 255, 255), GROSOR_TEXTO,
                            cv2.LINE_AA
                        )

                        # Verificar distancia a la pelota para posesión
                        if ball_pos_this_frame:
                            bcx, bcy = ball_pos_this_frame
                            pcx = (x1 + x2) / 2
                            pcy = (y1 + y2) / 2
                            dist = math.hypot(bcx - pcx, bcy - pcy)
                            if dist < closest_to_ball_dist:
                                closest_to_ball_dist = dist
                                closest_to_ball_name = name

                    elif cls_id == SPORTS_BALL_CLASS:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        ball_pos_this_frame = (cx, cy)
                        # Dibujar círculo naranja alrededor de la pelota
                        cv2.circle(frame, (cx, cy), 12, (0, 140, 255), 3)

                if ball_hoop_model is not None:
                    balls, hoops = _detect_basket_objects(ball_hoop_model, frame, frame_idx)
                    if balls:
                        ball = max(balls, key=lambda b: b.get("conf", 0))
                        cx, cy = int(ball["x"]), int(ball["y"])
                        ball_pos_this_frame = (cx, cy)
                        cv2.circle(frame, (cx, cy), 12, (0, 140, 255), 3)
                    for hoop_det in hoops:
                        if hoop_det.get("conf", 0) < 0.25:
                            continue
                        x1, y1 = int(hoop_det["x1"]), int(hoop_det["y1"])
                        x2, y2 = int(hoop_det["x2"]), int(hoop_det["y2"])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Segunda pasada para calcular posesión si hay pelota
                if ball_pos_this_frame:
                    bcx, bcy = ball_pos_this_frame
                    best_dist = float('inf')
                    best_name = None
                    for i, raw_cls_id in enumerate(boxes.cls.tolist()):
                        cls_id = int(raw_cls_id)
                        if cls_id != PERSON_CLASS:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                        tid = int(boxes.id[i]) if boxes.id is not None else -1
                        tid_str = str(tid)
                        if tid_str in player_map:
                            pcx = (x1 + x2) / 2
                            pcy = (y1 + y2) / 2
                            dist = math.hypot(bcx - pcx, bcy - pcy)
                            if dist < best_dist:
                                best_dist = dist
                                best_name = player_map[tid_str].get("name", f"#{tid}")
                    if best_dist < h * 0.25:
                        current_possession = best_name
                    else:
                        current_possession = None

            # ── Dibujar zona del aro ─────────────────────────────────────
            hcx = int(hoop.get("cx", w // 2))
            hcy = int(hoop.get("cy", int(h * 0.2)))
            hr = int(hoop.get("radius", 40))
            cv2.circle(frame, (hcx, hcy), hr, (0, 255, 255), 2)
            cv2.putText(
                frame, "ARO", (hcx - 15, hcy - 15),
                FUENTE, 0.5, (0, 255, 255), 2, cv2.LINE_AA
            )

            # ── Dibujar marcador (scoreboard) ────────────────────────────
            marcador_texto = f"Equipo A: {score_a}  |  Equipo B: {score_b}"
            text_size_m = cv2.getTextSize(marcador_texto, FUENTE, ESCALA_MARCADOR, 2)[0]
            marcador_x = (w - text_size_m[0]) // 2
            marcador_y = 35

            # Fondo semi-transparente del marcador
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (marcador_x - 15, 5),
                (marcador_x + text_size_m[0] + 15, marcador_y + 15),
                (0, 0, 0), -1
            )
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(
                frame, marcador_texto,
                (marcador_x, marcador_y),
                FUENTE, ESCALA_MARCADOR,
                (255, 255, 255), 2, cv2.LINE_AA
            )

            # ── Indicador de posesión ────────────────────────────────────
            if current_possession:
                posesion_texto = f"Posesion: {current_possession}"
                pos_size = cv2.getTextSize(posesion_texto, FUENTE, 0.5, 1)[0]
                pos_x = (w - pos_size[0]) // 2
                pos_y = marcador_y + 30
                cv2.putText(
                    frame, posesion_texto,
                    (pos_x, pos_y),
                    FUENTE, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA
                )

            out.write(frame)
            frame_idx += 1

            # Actualizar progreso del video
            if progress_callback and total_frames > 0 and frame_idx % 30 == 0:
                progress_callback(int(frame_idx / total_frames * 100))

        cap.release()
        out.release()

        if progress_callback:
            progress_callback(100)

        return f"/processed/{out_filename}"

    except Exception as e:
        print(f"❌ Error generando video anotado: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('\n🏀 Basketball Stats Tracker v2.0')
    print('📺 Abrí http://localhost:8080 en tu browser')
    print(f'🧠 Modelo: {MODEL_NAME}')
    print(f'🎬 ffmpeg: {"✅" if shutil.which("ffmpeg") else "❌ (video sin audio)"}')
    # Verificar disponibilidad de GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f'🚀 GPU: {torch.cuda.get_device_name(0)}')
        else:
            print('💻 Usando CPU')
    except Exception:
        print('💻 Usando CPU')
    print()
    uvicorn.run(app, host='0.0.0.0', port=8080, reload=False)
