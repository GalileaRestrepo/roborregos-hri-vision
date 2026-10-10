"""
proceso principal del sistema HRI — corre todos los nodos en un solo proceso
"""
import sys, os, threading, time, unicodedata

def ascii_safe(text):
    """quita acentos y caracteres especiales para que OpenCV pueda mostrar el texto"""
    # elimina signos de puntuacion especiales del español
    for ch in ['¿', '¡', '…', '"', '"', ''', ''', '«', '»']:
        text = text.replace(ch, '')
    # normaliza y quita diacriticos (acentos)
    nfkd = unicodedata.normalize('NFKD', text)
    result = ''.join(c for c in nfkd if not unicodedata.combining(c))
    # solo caracteres ASCII imprimibles
    return ''.join(c if ord(c) < 128 else '?' for c in result)

sys.path.insert(0, '/Users/agrm/ros2_hri_ws/install/hri_vision/lib/python3.9/site-packages')
sys.path.insert(0, '/Users/agrm/ros2_hri_ws/install/hri_interfaces/lib/python3.9/site-packages')
sys.path.insert(0, '/Users/agrm/ros2_hri_ws/install/hri_fusion/lib/python3.9/site-packages')
sys.path.insert(0, '/Users/agrm/ros2_hri_ws/install/hri_audio/lib/python3.9/site-packages')
sys.path.insert(0, '/Users/agrm/ros2_hri_ws/install/hri_rag/lib/python3.9/site-packages')
os.environ['AMENT_PREFIX_PATH'] = '/Users/agrm/ros2_hri_ws/install:/Users/agrm/miniconda3/envs/ros2_hri'

import rclpy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
from hri_interfaces.msg import FaceDetectionArray, TranscriptionResult, FusedIdentity
import cv2
import numpy as np

print('[HRI] cargando nodos de vision...', flush=True)
from hri_vision.camera_node import CameraNode
from hri_vision.face_detection_node import FaceDetectionNode
from hri_vision.face_tracker_node import FaceTrackerNode
from hri_vision.enroll_action_server import EnrollActionServer

print('[HRI] cargando nodos de audio...', flush=True)
from hri_audio.audio_capture_node import AudioCaptureNode
from hri_audio.vad_node import VadNode
from hri_audio.transcription_node import TranscriptionNode
from hri_audio.voice_identity_node import VoiceIdentityNode

print('[HRI] cargando nodos de fusion...', flush=True)
from hri_fusion.lip_movement_node import LipMovementNode
from hri_fusion.identity_fusion_node import IdentityFusionNode
from hri_fusion.decision_node import DecisionNode

COLORS = [
    (0,255,0),(255,128,0),(0,128,255),(255,0,255),(0,255,255),
    (0,255,128),(255,0,128),(128,255,0),(128,0,255),(255,255,0),
]
UNKNOWN_COLOR = (0, 0, 255)

# colores del overlay
OVERLAY_BG    = (30, 30, 30)
COLOR_GREEN   = (0, 220, 0)
COLOR_YELLOW  = (0, 200, 220)
COLOR_RED     = (0, 60, 220)
COLOR_WHITE   = (255, 255, 255)
COLOR_CYAN    = (220, 200, 0)


def draw_text_bg(frame, text, pos, font_scale=0.55, color=COLOR_WHITE, thickness=1):
    """dibuja texto con fondo oscuro"""
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x-4, y-th-6), (x+tw+4, y+4), OVERLAY_BG, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


class ViewerNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('viewer_node')
        self.bridge = CvBridge()
        self.latest_faces = []
        self.latest_frame = None
        self.color_map = {}

        # estado del overlay
        self.last_transcript = ''
        self.last_transcript_time = 0.0
        self.last_response = ''
        self.last_response_time = 0.0
        self.fused_by_track = {}
        self.vad_active = False

        self.create_subscription(Image, '/camera/image_raw', self.on_image, 10)
        self.create_subscription(FaceDetectionArray, '/vision/tracked_faces', self.on_faces, 10)
        self.create_subscription(TranscriptionResult, '/audio/transcriptions', self.on_transcript, 10)
        self.create_subscription(String, '/decision/response', self.on_response, 10)
        self.create_subscription(FusedIdentity, '/fusion/identity_decision', self.on_fused, 10)
        self.create_subscription(Bool, '/audio/vad_active', self.on_vad, 10)

    def on_faces(self, msg):
        self.latest_faces = msg.detections

    def on_image(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def on_transcript(self, msg):
        self.last_transcript = msg.text
        self.last_transcript_time = time.time()

    def on_response(self, msg):
        self.last_response = msg.data
        self.last_response_time = time.time()

    def on_fused(self, msg):
        self.fused_by_track[msg.track_id] = msg

    def on_vad(self, msg):
        self.vad_active = msg.data

    def wrap_text(self, text, max_chars):
        """divide texto largo en lineas"""
        lines = []
        while len(text) > max_chars:
            cut = text[:max_chars].rfind(' ')
            cut = cut if cut > 0 else max_chars
            lines.append(text[:cut].strip())
            text = text[cut:].strip()
        if text:
            lines.append(text)
        return lines

    def draw_panel(self, panel, y, label, value, label_color, value_color, font=0.62):
        """dibuja una fila label: value en el panel"""
        cv2.putText(panel, label, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, label_color, 1)
        y += 22
        for line in self.wrap_text(value, 28):
            cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, font, value_color, 1)
            y += 26
        return y + 6

    def get_frame(self):
        CAM_W, CAM_H = 960, 720
        PANEL_W = 320
        TOTAL_W = CAM_W + PANEL_W

        if self.latest_frame is None:
            blank = np.zeros((CAM_H, TOTAL_W, 3), dtype=np.uint8)
            cv2.putText(blank, 'Esperando camara...', (120, CAM_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            return blank

        frame = cv2.resize(self.latest_frame.copy(), (CAM_W, CAM_H))
        now = time.time()

        # --- cajas de caras sobre el video ---
        sx = CAM_W / self.latest_frame.shape[1]
        sy = CAM_H / self.latest_frame.shape[0]

        for face in self.latest_faces:
            x1, y1, x2, y2 = int(face.bbox[0]*sx), int(face.bbox[1]*sy), int(face.bbox[2]*sx), int(face.bbox[3]*sy)
            key = face.name if face.is_known else f'unknown_{face.track_id}'
            if key not in self.color_map:
                self.color_map[key] = COLORS[len(self.color_map) % len(COLORS)]
            color = self.color_map[key] if face.is_known else UNKNOWN_COLOR
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            label = f'{face.name} {face.identity_confidence:.0%}' if face.is_known else f'unknown #{face.track_id}'
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+4, y1), color, -1)
            cv2.putText(frame, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)

            fused = self.fused_by_track.get(face.track_id)
            if fused:
                dcol = {'RESPONDER': COLOR_GREEN, 'PREGUNTAR': COLOR_YELLOW, 'NEGAR': COLOR_RED}.get(fused.decision_label, COLOR_WHITE)
                fl = f'fusion {fused.fused_confidence:.0%}  {fused.decision_label}'
                (tw2,th2),_ = cv2.getTextSize(fl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y2+2), (x1+tw2+6, y2+th2+12), (20,20,20), -1)
                cv2.putText(frame, fl, (x1+3, y2+th2+6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dcol, 1)

        # --- panel lateral ---
        panel = np.zeros((CAM_H, PANEL_W, 3), dtype=np.uint8)
        panel[:] = (18, 18, 18)
        cv2.line(panel, (0, 0), (0, CAM_H), (60,60,60), 2)

        # titulo
        cv2.rectangle(panel, (0,0), (PANEL_W, 44), (35,35,35), -1)
        cv2.putText(panel, 'Reto HRI/Vision', (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_WHITE, 2)
        cv2.putText(panel, 'Candidates 2026', (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)
        cv2.line(panel, (0,44), (PANEL_W,44), (60,60,60), 1)

        y = 60

        # personas detectadas
        cv2.putText(panel, 'PERSONAS', (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1)

        y += 18
        cv2.line(panel, (14, y), (PANEL_W-14, y), (45,45,45), 1)
        y += 12

        if self.latest_faces:
            for face in self.latest_faces:
                color = self.color_map.get(face.name if face.is_known else f'unknown_{face.track_id}', UNKNOWN_COLOR)
                name = ascii_safe(face.name if face.is_known else f'unknown #{face.track_id}')
                cv2.putText(panel, name, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1)
                y += 22
                fused = self.fused_by_track.get(face.track_id)
                if fused:
                    dcol = {'RESPONDER': COLOR_GREEN, 'PREGUNTAR': COLOR_YELLOW, 'NEGAR': COLOR_RED}.get(fused.decision_label, COLOR_WHITE)
                    bar_w = int((PANEL_W - 28) * fused.fused_confidence)
                    cv2.rectangle(panel, (14, y), (PANEL_W-14, y+14), (45,45,45), -1)
                    cv2.rectangle(panel, (14, y), (14+bar_w, y+14), dcol, -1)
                    cv2.putText(panel, f'{fused.fused_confidence:.0%} {fused.decision_label}', (16, y+11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 1)
                    y += 20
                y += 8
        else:
            cv2.putText(panel, 'ninguna', (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,80,80), 1)
            y += 24

        y += 6
        cv2.line(panel, (14, y), (PANEL_W-14, y), (45,45,45), 1)
        y += 14

        # indicador de mic
        mic_color = COLOR_GREEN if self.vad_active else (80, 80, 80)
        mic_label = 'MIC  VOZ DETECTADA' if self.vad_active else 'MIC  en silencio'
        cv2.circle(panel, (24, y+1), 7, mic_color, -1)
        cv2.putText(panel, mic_label, (36, y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, mic_color, 1)
        y += 22
        cv2.line(panel, (14, y), (PANEL_W-14, y), (45,45,45), 1)
        y += 14

        # transcripcion
        cv2.putText(panel, 'ESCUCHANDO', (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1)
        y += 18
        cv2.line(panel, (14, y), (PANEL_W-14, y), (45,45,45), 1)
        y += 14

        if self.last_transcript and now - self.last_transcript_time < 8.0:
            alpha = max(0.3, 1.0 - (now - self.last_transcript_time) / 8.0)
            tcol = tuple(int(c * alpha) for c in COLOR_CYAN)
            for line in self.wrap_text(ascii_safe(f'"{self.last_transcript}"'), 30):
                cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, tcol, 1)
                y += 24
        else:
            cv2.putText(panel, '...', (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60,60,60), 1)
            y += 24

        y += 6
        cv2.line(panel, (14, y), (PANEL_W-14, y), (45,45,45), 1)
        y += 14

        # respuesta
        cv2.putText(panel, 'RESPUESTA', (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1)
        y += 18
        cv2.line(panel, (14, y), (PANEL_W-14, y), (45,45,45), 1)
        y += 14

        if self.last_response and now - self.last_response_time < 15.0:
            rcol = COLOR_GREEN if 'denegado' not in self.last_response.lower() else COLOR_RED
            for line in self.wrap_text(ascii_safe(self.last_response), 30)[:6]:
                cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rcol, 1)
                y += 24
        else:
            cv2.putText(panel, '...', (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60,60,60), 1)

        # junta camara + panel
        return np.hstack([frame, panel])


def main():
    rclpy.init()

    print('[HRI] iniciando nodos...', flush=True)

    nodes = [
        CameraNode(),
        FaceDetectionNode(),
        FaceTrackerNode(),
        EnrollActionServer(),
        AudioCaptureNode(),
        VadNode(),
        TranscriptionNode(),
        VoiceIdentityNode(),
        LipMovementNode(),
        IdentityFusionNode(),
        DecisionNode(),
        ViewerNode(),
    ]

    executor = MultiThreadedExecutor(num_threads=12)
    for node in nodes:
        executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print('[HRI] sistema completo corriendo — presiona Q para salir', flush=True)

    cv2.namedWindow('HRI Vision — RoBorregos', cv2.WINDOW_AUTOSIZE)
    viewer = nodes[-1]

    try:
        while rclpy.ok():
            frame = viewer.get_frame()
            cv2.imshow('Reto de HRI/Vision, RoBorregos', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
