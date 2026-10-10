import rclpy
from rclpy.node import Node
import numpy as np
import os
import glob
from collections import defaultdict, deque

# umbral de iou para matchear caras entre frames
IOU_THRESHOLD = 0.4

# similitud coseno minima para identificar a alguien
IDENTITY_THRESHOLD = 0.35

# frames sin ver antes de eliminar track
MAX_LOST_FRAMES = 30

# anti-spoofing: si no parpadea en N segundos -> posible foto
BLINK_WINDOW_SEC = 6.0
EAR_BLINK_THRESH = 0.22  # eye aspect ratio < este valor = ojo cerrado (parpadeo)


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def calc_ear(landmarks_3d, eye_indices):
    """eye aspect ratio usando 6 landmarks del ojo"""
    pts = landmarks_3d[eye_indices]
    # distancias verticales
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    # distancia horizontal
    h = np.linalg.norm(pts[0] - pts[3])
    if h == 0:
        return 1.0
    return (v1 + v2) / (2.0 * h)


# indices de landmarks 68 para ojos
LEFT_EYE_IDX  = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]


class Track:
    def __init__(self, track_id, bbox, embedding):
        self.track_id = track_id
        self.bbox = bbox
        self.embedding = embedding
        self.lost_frames = 0
        self.name = 'unknown'
        self.identity_confidence = 0.0
        self.is_known = False
        self.embedding_history = [embedding]

        # anti-spoofing: historial de EAR para detectar parpadeos
        self.ear_history = deque(maxlen=150)  # ~5s a 30fps
        self.blink_count = 0
        self.last_blink_time = 0.0
        self.first_seen_time = 0.0
        self.is_live = True  # se asume real hasta demostrar lo contrario


class FaceTrackerNode(Node):
    def __init__(self):
        super().__init__('face_tracker_node')

        from hri_interfaces.msg import FaceDetection, FaceDetectionArray
        self.FaceDetection = FaceDetection
        self.FaceDetectionArray = FaceDetectionArray

        self.subscription = self.create_subscription(
            FaceDetectionArray,
            '/vision/detections',
            self.on_detections,
            10
        )

        self.publisher = self.create_publisher(
            FaceDetectionArray,
            '/vision/tracked_faces',
            10
        )

        self.tracks = {}
        self.next_id = 0
        self.enrolled = self.load_enrolled_faces()

        self.get_logger().info(
            f'Tracker listo, {len(self.enrolled)} persona(s) enrolada(s): '
            f'{list(self.enrolled.keys())}'
        )

    def load_enrolled_faces(self):
        enrolled = {}
        data_dir = os.path.expanduser('~/ros2_hri_ws/data/enrolled')
        for person_dir in glob.glob(os.path.join(data_dir, '*')):
            name = os.path.basename(person_dir)
            emb_file = os.path.join(person_dir, 'face_embeddings.npy')
            if os.path.exists(emb_file):
                enrolled[name] = np.load(emb_file)
                self.get_logger().info(f'  Cargado: {name} ({len(enrolled[name])} muestras)')
        return enrolled

    def identify(self, embedding):
        if not self.enrolled or len(embedding) == 0:
            return 'unknown', 0.0
        best_name, best_score = 'unknown', -1.0
        for name, embeddings in self.enrolled.items():
            for e in embeddings:
                score = cosine_similarity(embedding, e)
                if score > best_score:
                    best_score = score
                    best_name = name
        if best_score < IDENTITY_THRESHOLD:
            return 'unknown', best_score
        return best_name, best_score

    def check_liveness_ear(self, track, ear, now):
        """detecta si la cara es real usando parpadeo (EAR)"""
        track.ear_history.append((now, ear))

        # detecta parpadeo: ear bajo el umbral
        if ear < EAR_BLINK_THRESH:
            if track.last_blink_time == 0 or (now - track.last_blink_time) > 0.3:
                track.blink_count += 1
                track.last_blink_time = now

        # evalua liveness despues de BLINK_WINDOW_SEC segundos
        if track.first_seen_time > 0 and (now - track.first_seen_time) > BLINK_WINDOW_SEC:
            track.is_live = track.blink_count >= 1
            if not track.is_live:
                self.get_logger().warn(
                    f'track {track.track_id}: sin parpadeo en {BLINK_WINDOW_SEC}s -> posible foto/pantalla'
                )

    def on_detections(self, msg):
        detections = msg.detections
        now = self.get_clock().now().nanoseconds / 1e9

        matched_track_ids = set()
        matched_det_indices = set()

        for det_idx, det in enumerate(detections):
            best_iou = IOU_THRESHOLD
            best_track_id = None

            for track_id, track in self.tracks.items():
                score = iou(det.bbox, track.bbox)
                if score > best_iou:
                    best_iou = score
                    best_track_id = track_id

            if best_track_id is not None:
                track = self.tracks[best_track_id]
                track.bbox = det.bbox
                track.lost_frames = 0

                if len(det.embedding) == 512:
                    track.embedding_history.append(det.embedding)
                    if len(track.embedding_history) > 10:
                        track.embedding_history.pop(0)
                    track.embedding = np.mean(track.embedding_history, axis=0).tolist()
                    if len(track.embedding_history) % 10 == 0:
                        name, conf = self.identify(track.embedding)
                        track.name = name
                        track.identity_confidence = conf
                        track.is_known = (name != 'unknown')

                # anti-spoofing: EAR viene empaquetado en identity_confidence
                # -1.0 = sin landmarks, cualquier otro valor = EAR calculado
                ear = det.identity_confidence
                if ear >= 0.0:
                    self.check_liveness_ear(track, ear, now)

                matched_track_ids.add(best_track_id)
                matched_det_indices.add(det_idx)

        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_indices and len(det.embedding) > 0:
                track_id = self.next_id
                self.next_id += 1
                name, conf = self.identify(det.embedding)
                track = Track(track_id, det.bbox, det.embedding)
                track.name = name
                track.identity_confidence = conf
                track.is_known = (name != 'unknown')
                track.first_seen_time = now
                self.tracks[track_id] = track

        to_delete = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_track_ids:
                track.lost_frames += 1
                if track.lost_frames > MAX_LOST_FRAMES:
                    to_delete.append(track_id)
        for tid in to_delete:
            del self.tracks[tid]

        array_msg = self.FaceDetectionArray()
        array_msg.header = msg.header

        for track in self.tracks.values():
            if track.lost_frames == 0:
                det = self.FaceDetection()
                det.track_id = track.track_id
                det.bbox = track.bbox
                det.embedding = list(track.embedding) if isinstance(track.embedding, np.ndarray) else track.embedding
                det.detection_confidence = 1.0 if track.is_live else 0.1  # baja confianza si no es live
                det.name = track.name
                # si no es live, baja la confianza de identidad para bloquear spoofing
                det.identity_confidence = track.identity_confidence if track.is_live else 0.0
                det.is_known = track.is_known and track.is_live
                array_msg.detections.append(det)

        self.publisher.publish(array_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FaceTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
