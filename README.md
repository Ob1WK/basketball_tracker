# 🏀 Basketball Stats Tracker v2.0

Analizá tus pickups de básquet con visión computacional — 100% local, sin APIs pagas.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8s-Ultralytics-purple)
![License](https://img.shields.io/badge/License-MIT-orange)

---

## ¿Qué hace?

1. **Subís el video** → grabado desde el piso mirando de frente al aro
2. **YOLOv8s detecta automáticamente** jugadores, pelota y estima la posición del aro
3. **Identificás cada jugador** desde su thumbnail con nombre y equipo
4. **Obtenés las estadísticas** estilo NBA: PTS, REB, AST, FGM/FGA, 3PM/3PA, STL, +/-, etc.
5. **Video anotado** con bounding boxes, nombres, marcador y audio original

---

## ✨ Novedades v2.0

- 🔄 **Procesamiento asíncrono**: detect y analyze corren en background, no bloquean el servidor
- 📡 **Progreso en tiempo real**: barras de progreso con %, frames procesados y ETA via SSE
- 🧠 **Motor de stats mejorado**: máquina de estados para tiros, rebotes reales, +/-, turnovers bidireccionales
- 🎨 **UI premium**: diseño glassmorphism estilo NBA dashboard, animaciones, responsive
- 📊 **Tabla sorteable**: click en cualquier columna para ordenar
- 📄 **Export PDF**: descargá el box score como PDF estilizado
- 🔊 **Audio en video anotado**: merge automático con ffmpeg (si está disponible)
- 📺 **Scoreboard en video**: marcador Equipo A vs B overlay en el video anotado
- 🏥 **Health check**: endpoint /health con info de GPU, modelo y ffmpeg

---

## Instalación rápida

### Requisitos
- Python 3.9+
- ~800MB de espacio (modelo YOLOv8s + dependencias)
- GPU opcional pero recomendada para videos largos
- ffmpeg opcional (para audio en video anotado)

### Paso a paso

```bash
# 1. Ir a la carpeta del proyecto
cd basketball_tracker

# 2. Instalar dependencias (opción A: script automático)
python setup.py

# 2. Instalar dependencias (opción B: manual)
pip install ultralytics fastapi uvicorn[standard] opencv-python python-multipart numpy

# 3. Arrancar el servidor
python server.py

# 4. Abrir en el browser
# http://localhost:8080
```

---

## Uso

### 1. 📤 Subir Video
- Arrastrá el video o hacé click para seleccionarlo
- Formatos soportados: MP4, MOV, AVI, MKV
- Barra de progreso real durante la subida
- Recomendado: videos de hasta 10 minutos para tiempos razonables

### 2. 🔍 Detección
- Click en **"Iniciar Detección con YOLOv8"**
- Progreso en tiempo real: porcentaje, frames procesados, ETA
- YOLOv8s analiza ~5 frames por segundo del video
- Detecta automáticamente personas y pelotas

### 3. 👥 Identificar Jugadores
- Thumbnails de cada track detectado
- Poné el nombre del jugador y su equipo (A / B)
- Botón "Nombres rápidos" para autocompletar
- Elegí un color por jugador para el video anotado
- Toggle para omitir tracks irrelevantes (árbitros, espectadores)

### 4. 📊 Estadísticas
- Barra comparativa Team A vs Team B
- Box score completo estilo NBA con todas las stats
- Tabla sorteable (click en cualquier columna)
- Líderes resaltados en dorado
- Video anotado con bounding boxes, nombres y scoreboard
- Exportá en CSV o PDF

---

## Estadísticas incluidas

| Stat | Descripción |
|------|-------------|
| PTS  | Puntos |
| REB  | Rebotes totales |
| AST  | Asistencias |
| FGM/FGA | Tiros de campo anotados/intentados |
| FG%  | Porcentaje de tiro |
| 3PM/3PA | Triples anotados/intentados |
| 3P%  | Porcentaje de triples |
| FTM/FTA | Tiros libres |
| OREB/DREB | Rebotes ofensivos/defensivos |
| STL  | Robos |
| BLK  | Bloqueos |
| TO   | Pérdidas |
| +/-  | Plus/minus (diferencial de puntos) |
| MIN  | Minutos en cancha |

---

## Motor de Estadísticas v2.0

El motor usa una **máquina de estados** para detectar tiros con precisión:

```
IDLE → APPROACHING → NEAR_HOOP → SCORED (FGM) / MISSED (FGA)
```

### Mejoras sobre v1:
- **Tiros**: sin doble conteo, con transiciones claras de estado
- **3 puntos**: basado en posición del TIRADOR al soltar, no la pelota
- **Rebotes**: quién captura la pelota dentro de 2s post-miss
- **Asistencias**: último poseedor antes de un enceste
- **Turnovers**: cambio de posesión entre equipos (TO + STL bidireccional)
- **+/-**: diferencial de puntos mientras cada jugador está en cancha

---

## API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET`  | `/` | Sirve la UI |
| `POST` | `/upload` | Sube video → `{job_id, filename}` |
| `POST` | `/detect/{job_id}` | Inicia detección async → `{status, job_id}` |
| `GET`  | `/progress/{job_id}` | Stream SSE de progreso |
| `POST` | `/assign_players/{job_id}` | Asigna jugadores |
| `POST` | `/analyze/{job_id}` | Inicia análisis async → `{status, job_id}` |
| `GET`  | `/status/{job_id}` | Estado completo del job |
| `GET`  | `/health` | Info del sistema |
| `POST` | `/reset` | Limpia todos los jobs |

---

## Limitaciones y precisión

La detección es una **estimación** basada en visión computacional:

- **Tiros/puntos**: se infieren por la trayectoria de la pelota hacia el aro
- **Asistencias**: último poseedor antes de un enceste
- **Rebotes**: quién captura la pelota tras un tiro fallido
- **Bloqueos**: difíciles de detectar desde cámara frontal baja

Para mejor precisión:
- Video con buena iluminación
- Cámara elevada a ~1.2-1.5m del piso
- Jugadores con colores de remera distintos por equipo
- Resolución mínima 720p

---

## Hardware recomendado

| Hardware | Detección (10 min) | Video anotado |
|----------|-------------------|---------------|
| CPU moderno (i7/Ryzen 7) | 8-15 min | 10-20 min |
| GPU NVIDIA (GTX 1660+) | 1-3 min | 2-5 min |
| GPU NVIDIA (RTX 3080+) | < 1 min | 1-2 min |

Para usar GPU, asegurate de tener CUDA instalado:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Estructura del proyecto

```
basketball_tracker/
├── server.py          # Servidor FastAPI + motor de análisis v2
├── setup.py           # Script de instalación y verificación
├── README.md          # Este archivo
├── static/
│   └── index.html     # UI completa (single file, premium)
├── uploads/           # Videos subidos (temporales)
└── processed/         # Thumbnails y videos anotados
```

---

## Troubleshooting

### El servidor no arranca
```bash
# Verificar dependencias
python setup.py

# Verificar que el puerto 8000 esté libre
netstat -an | findstr 8000
```

### La detección es muy lenta
- Usá GPU si es posible (ver sección Hardware)
- Para más velocidad, cambiá `MODEL_NAME = 'yolov8n.pt'` en server.py (menos preciso)

### El video anotado no tiene audio
- Instalá ffmpeg: `winget install Gyan.FFmpeg` (Windows) o `brew install ffmpeg` (macOS)
- Reiniciá el servidor después de instalar

### Los tracks no se ven bien
- Ajustá la confianza mínima en `model.track(..., conf=0.35)` en server.py
- Videos con más iluminación dan mejores resultados

### Muchos tracks falsos (espectadores, árbitros)
- Es normal — simplemente no les asignes nombre en el paso 3
- Solo se analizan los jugadores con nombre asignado

---

## Modelos disponibles

Podés cambiar el modelo en server.py (`MODEL_NAME`):

| Modelo | Tamaño | Velocidad | Precisión |
|--------|--------|-----------|-----------|
| `yolov8n.pt` | 6 MB | ⚡⚡⚡ Rápido | ⭐⭐ Básica |
| `yolov8s.pt` | 22 MB | ⚡⚡ Medio | ⭐⭐⭐ Buena |
| `yolov8m.pt` | 50 MB | ⚡ Lento | ⭐⭐⭐⭐ Alta |

---

## FAQ

**¿Necesito internet?**
Solo para la primera descarga del modelo YOLOv8 (~22MB) y las fuentes de Google. Después todo es local.

**¿Soporta múltiples partidos?**
Sí, cada video genera un `job_id` independiente.

**¿Puedo cambiar el modelo YOLO?**
Sí, cambiá `MODEL_NAME` en `server.py`.

**¿El video anotado tiene audio?**
Sí, si tenés ffmpeg instalado. Si no, se genera sin audio (igual que antes).

**¿Puedo exportar las stats?**
Sí, en CSV y PDF desde la UI.

---

## Licencia

MIT — Usalo como quieras. 🏀
