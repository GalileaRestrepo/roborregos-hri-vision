import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from insightface.app import FaceAnalysis

# indices de landmarks 68 para calcular EAR (eye aspect ratio)
LEFT_EYE_IDX  = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]


def calc_ear(landmarks, eye_idx):
    """ear: 0 = ojo cerrado (parpadeo), 1 = ojo abierto"""
    pts = landmarks[eye_idx]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h  = np.linalg.norm(pts[0] - pts[3])
    if h == 0:
        return 1.0
    return float((v1 + v2) / (2.0 * h))


class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection_node')

        from hri_interfaces.msg import FaceDetection, FaceDetectionArray
        self.FaceDetection = FaceDetection
        self.FaceDetectionArray = FaceDetectionArray

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.on_image, 10
        )

        self.publisher = self.create_publisher(
            FaceDetectionArray, '/vision/detections', 10
        )

        self.bridge = CvBridge()

        self.get_logger().info('Cargando modelos de InsightFace...')
        self.app = FaceAnalysis(name='buffalo_m', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.get_logger().info('InsightFace listo y detectando caras en /vision/detections')

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)

        array_msg = self.FaceDetectionArray()
        array_msg.header = msg.header

        for face in faces:
            detection = self.FaceDetection()
            detection.bbox = face.bbox.astype(float).tolist()
            detection.detection_confidence = float(face.det_score)
            detection.embedding = face.embedding.astype(float).tolist() if face.embedding is not None else []
            detection.track_id = -1
            detection.name = 'unknown'
            detection.is_known = False

            # calcula EAR para anti-spoofing y lo mete en identity_confidence
            # (el tracker lo usara para detectar parpadeos)
            ear = -1.0
            if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
                lm = face.landmark_3d_68
                ear_l = calc_ear(lm, LEFT_EYE_IDX)
                ear_r = calc_ear(lm, RIGHT_EYE_IDX)
                ear = (ear_l + ear_r) / 2.0

            # empaqueta EAR en identity_confidence para pasar al tracker
            # -1.0 = sin landmarks disponibles
            detection.identity_confidence = float(ear)

            array_msg.detections.append(detection)

        self.publisher.publish(array_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
