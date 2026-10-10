import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from hri_interfaces.msg import FaceDetectionArray, LipMovement, LipMovementArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict, deque

# apertura vertical/horizontal mínima para considerar boca abierta
LIP_OPEN_THRESHOLD = 0.02
# frames históricos para asociación con voz (~2s a 30fps)
HISTORY_FRAMES = 60

# índices de MediaPipe Face Mesh
UPPER_LIP   = 13
LOWER_LIP   = 14
LEFT_CORNER  = 78
RIGHT_CORNER = 308


class LipMovementNode(Node):
    def __init__(self):
        super().__init__('lip_movement_node')

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.bridge = CvBridge()

        # últimas detecciones del tracker para asociar track_id con bbox
        self.latest_tracked_faces = []

        # buffer temporal: {track_id: deque([(float_timestamp_s, openness)])}
        self.lip_history = defaultdict(lambda: deque(maxlen=HISTORY_FRAMES))

        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.on_frame, 10
        )
        self.tracked_sub = self.create_subscription(
            FaceDetectionArray, '/vision/tracked_faces', self.on_tracked_faces, 10
        )
        self.publisher = self.create_publisher(
            LipMovementArray, '/fusion/lip_movement', 10
        )

        self.get_logger().info('nodo de movimiento de boca listo')

    def on_tracked_faces(self, msg):
        self.latest_tracked_faces = msg.detections

    def on_frame(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        now = self.get_clock().now().nanoseconds / 1e9

        array_msg = LipMovementArray()
        array_msg.header = msg.header

        if not results.multi_face_landmarks:
            self.publisher.publish(array_msg)
            return

        for face_landmarks in results.multi_face_landmarks:
            openness = self._calc_openness(face_landmarks, h, w)
            mp_bbox = self._landmarks_to_bbox(face_landmarks, h, w)
            track_id = self._match_track(mp_bbox)

            if track_id < 0:
                continue

            self.lip_history[track_id].append((now, openness))

            lm = LipMovement()
            lm.track_id = track_id
            lm.mouth_openness = float(openness)
            lm.is_speaking = openness > LIP_OPEN_THRESHOLD
            lm.stamp = msg.header.stamp
            array_msg.lip_movements.append(lm)

        self.publisher.publish(array_msg)

    def _calc_openness(self, landmarks, h, w):
        """apertura normalizada por el ancho de boca (invariante a distancia)."""
        lm = landmarks.landmark
        upper = np.array([lm[UPPER_LIP].x * w,   lm[UPPER_LIP].y * h])
        lower = np.array([lm[LOWER_LIP].x * w,   lm[LOWER_LIP].y * h])
        left  = np.array([lm[LEFT_CORNER].x * w,  lm[LEFT_CORNER].y * h])
        right = np.array([lm[RIGHT_CORNER].x * w, lm[RIGHT_CORNER].y * h])

        vertical   = np.linalg.norm(lower - upper)
        horizontal = np.linalg.norm(right - left)

        if horizontal < 1e-3:
            return 0.0
        return float(min(vertical / horizontal, 1.0))

    def _landmarks_to_bbox(self, landmarks, h, w):
        """bbox [x1,y1,x2,y2] a partir de todos los landmarks de la cara."""
        xs = [lm.x * w for lm in landmarks.landmark]
        ys = [lm.y * h for lm in landmarks.landmark]
        return [min(xs), min(ys), max(xs), max(ys)]

    def _match_track(self, mp_bbox):
        """asocia la cara de mediapipe con el track_id del tracker por IoU."""
        best_id  = -1
        best_iou = 0.3  # mínimo de solapamiento requerido

        for face in self.latest_tracked_faces:
            score = _iou(mp_bbox, face.bbox)
            if score > best_iou:
                best_iou = score
                best_id  = face.track_id

        return best_id

    def get_lip_activity(self, track_id, t_start, t_end):
        """promedio de apertura de boca en la ventana [t_start, t_end] en segundos."""
        values = [v for (t, v) in self.lip_history[track_id] if t_start <= t <= t_end]
        return float(np.mean(values)) if values else 0.0


def _iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def main(args=None):
    rclpy.init(args=args)
    node = LipMovementNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
