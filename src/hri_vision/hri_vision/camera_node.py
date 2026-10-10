import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import importlib


try:
    cv2 = importlib.import_module('cv2')
except ImportError as exc:
    raise ImportError(
        'OpenCV no esta instalado'
    ) from exc


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # este nodo va a publicar imgs al topico /camera/image_raw
        # el 10 es el tamaño del buffer, osea cuantos msjs guarda si nadie lee
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)

        # este es el timer, llama a capture_frame() 30 veces x segundo
        self.timer = self.create_timer(1.0 / 30.0, self.capture_frame)

        # cvbridge convierte entre imgs de opencv y msjs de ROS
        self.bridge = CvBridge()

        # esto abre la cam. 0 = default cam 
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.get_logger().error('No se pudo abrir la cámara')
            return

        self.get_logger().info('Cámara abierta y publicando en /camera/image_raw a 30fps')

    def capture_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn('Frame vacío, reintentando...')
            return

        # convierte el frame (numpy array BGR) a un msj ROS
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

        # header con timestamp para sincronizar nodos
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'

        self.publisher.publish(msg)

    def destroy_node(self):
        # libera la cam cuando el nodo se cierra    
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)  
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
