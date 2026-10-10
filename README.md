# HRI/Vision Reto Candidates 2026

Sistema ROS 2 que identifica personas por cara, voz y movimiento de boca,
y responde sus preguntas usando un documento privado por persona (RAG local).
Todo corre en la laptop, sin APIs externas ni hardware adicional.

---

## Arquitectura

```
Camara → face_detection → face_tracker ──────────────────────┐
Microfono → audio_capture → VAD → transcription              ↓
                              └──→ voice_identity → identity_fusion → decision → RAG
Camara → lip_movement ────────────────────────────────────────┘
```

**Topicos principales:**
- `/camera/image_raw` — frames de camara (sensor_msgs/Image)
- `/vision/detections` — caras detectadas + embeddings (FaceDetectionArray)
- `/vision/tracked_faces` — caras con track_id estable e identidad (FaceDetectionArray)
- `/audio/raw_chunks` — chunks de audio 64ms (SpeechSegment)
- `/audio/speech_segments` — segmentos de voz completos (SpeechSegment)
- `/audio/transcriptions` — texto transcrito por Whisper (TranscriptionResult)
- `/audio/voice_identity` — identidad por voz ECAPA-TDNN (VoiceIdentity)
- `/audio/vad_active` — indicador de voz activa (std_msgs/Bool)
- `/fusion/lip_movement` — apertura de boca por track_id (LipMovementArray)
- `/fusion/identity_decision` — confianza fusionada y decision (FusedIdentity)
- `/decision/response` — respuesta final del sistema (std_msgs/String)

**Servicio:** `/rag/query` (RAGQuery.srv)

**Accion:** `/enroll` (EnrollSession.action)

---

## Instalacion desde cero

### 1. Prerequisitos del sistema
```bash
# Xcode Command Line Tools (macOS)
xcode-select --install

# Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Miniconda para Apple Silicon
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init zsh && source ~/.zshrc
conda install mamba -n base -c conda-forge -y
```

### 3. Ambiente ROS 2 Humble
```bash
mamba create -n ros2_hri -c robostack-humble -c conda-forge \
  ros-humble-desktop ros-humble-cv-bridge \
  compilers cmake pkg-config make ninja colcon-common-extensions \
  python=3.9 -y
conda activate ros2_hri
```

### 4. Dependencias Python
```bash
pip install insightface onnxruntime opencv-python sounddevice \
  faster-whisper speechbrain mediapipe chromadb fastapi uvicorn \
  requests PyMuPDF numpy torch silero-vad
```

> Nota: si pip instala `typing-3.7.4.3`, eliminalo:
> ```bash
> rm -f ~/miniconda3/envs/ros2_hri/lib/python3.9/site-packages/typing.py
> ```

### 5. Ollama y modelos LLM
```bash
brew install ollama

# en una terminal aparte
ollama serve

# descarga modelos (requiere ollama serve corriendo)
ollama pull llama3.2:1b        # LLM para generacion de respuestas
ollama pull nomic-embed-text   # modelo de embeddings para RAG
```

### 6. Descarga de modelos de vision y voz

Los modelos se descargan automaticamente la primera vez que corres el sistema:

- **InsightFace buffalo_m** (~270MB) — se descarga al primer uso de `face_detection_node`
- **ECAPA-TDNN SpeechBrain** (~100MB) — se descarga al primer uso de `voice_identity_node` o `enroll_voice.py`
- **Whisper tiny** (~75MB) — se descarga al primer uso de `transcription_node`
- **Silero VAD** (~2MB) — se descarga al primer uso de `vad_node`

Para pre-descargarlos antes de correr el sistema:
```bash
# buffalo_m (InsightFace)
python -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_m', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640,640))
print('buffalo_m listo')
"

# ECAPA-TDNN (SpeechBrain)
python -c "
from speechbrain.inference.speaker import EncoderClassifier
EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir='~/.speechbrain/spkrec-ecapa-voxceleb',
    run_opts={'device': 'cpu'}
)
print('ECAPA-TDNN listo')
"

# Whisper tiny
python -c "
from faster_whisper import WhisperModel
WhisperModel('tiny', device='cpu', compute_type='int8')
print('Whisper listo')
"
```

### 7. Compilar workspace
```bash
cd ~/ros2_hri_ws

# compila hri_interfaces (mensajes custom, necesario hacerlo primero)
bash build_interfaces.sh

# compila el resto
colcon build --base-paths src --build-base build --install-base install
```

---

## Depuracion con Foxglove Studio y ros2 bag

### Foxglove Studio para visualizar topicos en tiempo real

Foxglove Studio permite visualizar imagenes, graficas y mensajes de ROS 2 en el navegador.

```bash
# instalar el bridge
pip install foxglove-websocket
conda activate ros2_hri
pip install foxglove-websocket

# correr el bridge (con el sistema ya corriendo, en otra terminal)
ros2 run foxglove_bridge foxglove_bridge
```

Luego abrir https://app.foxglove.dev en el navegador y conectar a `ws://localhost:8765`.

Topicos utiles para visualizar:
- `/camera/image_raw` — video de camara
- `/vision/tracked_faces` — cajas de deteccion
- `/audio/transcriptions` — texto transcrito
- `/fusion/identity_decision` — confianza fusionada

### ros2 bag para grabar sesion y depurar despues

Graba todos los mensajes de una sesion y los reproduce sin necesitar camara ni microfono.

```bash
# grabar (con el sistema corriendo)
ros2 bag record \
  /camera/image_raw \
  /vision/detections \
  /vision/tracked_faces \
  /audio/transcriptions \
  /fusion/identity_decision \
  /decision/response \
  -o ~/ros2_hri_ws/data/sesion_bag

# reproducir (sin camara, para depurar)
ros2 bag play ~/ros2_hri_ws/data/sesion_bag
```

---

## Enrolamiento de personas

Sustituye `NombrePersona` por el nombre real (ej: `Gali`) y `/ruta/documento.txt`
por la ruta al PDF o TXT privado de esa persona.

### Opcion A — standalone (ANTES de arrancar el sistema, recomendado)

```bash
# paso 1: enrolar cara
# abre ventana de camara, presiona ESPACIO 25 veces mirando de frente
# tip: captura desde diferentes angulos y con diferente luz
python \
  ~/ros2_hri_ws/scripts/enroll_person.py --name NombrePersona

# paso 2: enrolar voz
# graba 5 muestras de 12 segundos, habla naturalmente en cada una
# tip: di tu nombre, describe tu proyecto, habla de cosas distintas
python \
  ~/ros2_hri_ws/scripts/enroll_voice.py --name NombrePersona

# paso 3: ingestar documento privado (requiere ollama serve corriendo)
bash ~/ros2_hri_ws/ingest_document.sh NombrePersona /ruta/documento.txt
```

Los embeddings se guardan en `~/ros2_hri_ws/data/enrolled/NombrePersona/`.

### Opcion B — ROS 2 action (CON el sistema ya corriendo)

Cara + voz + documento en un solo comando, con feedback de progreso en tiempo real:

```bash
ros2 action send_goal /enroll hri_interfaces/action/EnrollSession \
  "{person_name: 'NombrePersona', mode: 2, document_path: '/ruta/documento.txt'}"
```

Modos disponibles:
- `mode: 0` — solo cara (mira a la camara, acumula 25 capturas automaticamente)
- `mode: 1` — solo voz (graba 5 muestras de 12s)
- `mode: 2` — cara + voz + documento (recomendado para enrolamiento completo)

---

## Arrancar el sistema

```bash
# opcion principal
ros2 launch hri_fusion hri_system.launch.py

# opcion alternativa (bash)
bash ~/ros2_hri_ws/launch_all.sh
```

El sistema abre una ventana OpenCV con:
- Cajas alrededor de cada cara con nombre y confianza
- Barra de confianza fusionada por persona (verde/amarillo/rojo)
- Indicador de microfono activo
- Transcripcion en tiempo real
- Respuesta del sistema

Presiona **Q** para cerrar.

---

## Verificar que todo esta corriendo

```bash
# topicos activos
ros2 topic list

# nodos activos
ros2 node list

# acciones disponibles
ros2 action list
# debe mostrar: /enroll

# probar RAG manualmente
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"identity_name": "Gali", "question": "en que reto estoy?", "identity_confidence": 0.9}'
```

---

## Interfaces propias (hri_interfaces)

### Mensajes (.msg)

\*\*FaceDetection.msg\*\* deteccion de una cara
```
int32 track_id              # id estable del tracker entre frames
float32[4] bbox             # coordenadas [x1, y1, x2, y2]
float32 detection_confidence
float32[] embedding         # embedding CosFace 512-D (InsightFace buffalo_m)
string name                 # nombre si enrolado, "unknown" si no
float32 identity_confidence # similitud coseno contra enrolled (0-1)
bool is_known
```

**FaceDetectionArray.msg** — lista de detecciones en un frame
```
std_msgs/Header header
hri_interfaces/FaceDetection[] detections
```

**SpeechSegment.msg** — chunk o segmento de audio
```
std_msgs/Header header
float32[] audio_data        # muestras float32 a 16kHz
float32 sample_rate         # siempre 16000.0
float64 timestamp_start     # segundos desde epoch
float64 timestamp_end
```

**TranscriptionResult.msg** — resultado de Whisper
```
std_msgs/Header header
string text                 # texto transcrito
string detected_language    # codigo ISO (es, en, etc.)
float32 confidence          # probabilidad del idioma detectado
float64 timestamp_start
float64 timestamp_end
```

**VoiceIdentity.msg** — identidad por voz
```
std_msgs/Header header
string name                 # nombre identificado o "unknown"
float32 confidence          # similitud coseno ECAPA-TDNN (0-1)
bool is_known
float64 timestamp_start
float64 timestamp_end
```

**LipMovement.msg** — apertura de boca de una cara
```
int32 track_id
float32 mouth_openness      # ratio apertura/anchura (0=cerrada, 1=abierta)
bool is_speaking            # openness > umbral
builtin_interfaces/Time stamp
```

**LipMovementArray.msg** — lista de movimientos de boca en un frame
```
std_msgs/Header header
hri_interfaces/LipMovement[] lip_movements
```

**FusedIdentity.msg** — resultado de fusionar cara + voz + boca
```
std_msgs/Header header
int32 track_id
string name
float32 face_confidence     # contribucion de cara (peso 50%)
float32 voice_confidence    # contribucion de voz (peso 35%)
float32 lip_confidence      # contribucion de boca (peso 15%)
float32 fused_confidence    # confianza final (EMA suavizada)
uint8 decision              # 0=NEGAR, 1=PREGUNTAR, 2=RESPONDER
string decision_label       # "NEGAR" | "PREGUNTAR" | "RESPONDER"
string pending_question     # pregunta en espera de confirmacion
```

### Servicios (.srv)

**RAGQuery.srv** — consulta al RAG privado (identidad como permiso)
```
# request
string identity_name
float32 identity_confidence  # debe ser >= 0.55, de lo contrario se rechaza
string question
---
# response
bool authorized
string answer                # respuesta basada SOLO en el documento privado
string[] source_chunks       # fragmentos del documento usados
string denial_reason         # razon del rechazo si authorized=false
```

### Acciones (.action)

**EnrollSession.action** — enrolar una persona (cara + voz + documento)
```
# goal (lo que pides)
string person_name
uint8 mode              # 0=cara, 1=voz, 2=cara+voz+documento
string document_path    # ruta al archivo PDF o TXT privado
---
# result (cuando termina)
bool success
string message
float32 quality_score   # cohesion interna de embeddings (0=mala, 1=perfecta)
---
# feedback (progreso en tiempo real)
uint8 progress          # porcentaje 0-100
string status_message   # descripcion del paso actual
uint32 samples_collected
```

---

## Politica de decision (umbrales)

| Confianza fusionada | Decision | Accion del sistema |
|---|---|---|
| >= 0.75 | RESPONDER | Consulta RAG y responde con el documento |
| 0.45 - 0.74 | PREGUNTAR | Pide confirmacion verbal de identidad |
| < 0.45 | NEGAR | No puede verificar, rechaza responder |

---

## Anti-spoofing

- **Cara (foto/pantalla):** Eye Aspect Ratio (EAR) con landmarks 68 de InsightFace.
  Si no se detecta ningun parpadeo en 6 segundos → confianza de identidad = 0.
- **Voz (audio grabado):** coeficiente de variacion de energia RMS.
  Voz en vivo tiene variacion natural alta (CV > 0.15).
  Audio grabado reproducido tiene energia muy uniforme (CV < 0.15) → descartado.

---

## Pipeline RAG 100% local

| Componente | Herramienta | Modelo |
|---|---|---|
| Embeddings de documentos | Ollama | nomic-embed-text |
| Busqueda semantica | ChromaDB | — |
| Generacion de respuestas | Ollama | llama3.2:1b |
| Embeddings de voz | SpeechBrain | ECAPA-TDNN |
| Deteccion facial | InsightFace | buffalo_m (CosFace) |
| Transcripcion | faster-whisper | tiny |

Ningun fragmento de documentos privados ni preguntas salen de la laptop.
