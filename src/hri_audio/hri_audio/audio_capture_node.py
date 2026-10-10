import rclpy
from rclpy.node import Node
from hri_interfaces.msg import SpeechSegment
import sounddevice as sd
import numpy as np
import queue
import threading

# config del audio
SAMPLE_RATE = 16000 # 16kHz, basically lo que whisper y silero VAD esperan
CHUNK_MS = 64 # tamaño en milisegundos
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000) # = 480 muestras por chunk


class AudioCaptureNode(Node):
    def __init__(self):
        super().__init__('audio_capture_node')

        # publica chunks de audio crudos
        self.publisher = self.create_publisher(SpeechSegment, '/audio/raw_chunks', 100)

        # cola thread-safe: sounddevice captura en un hilo aparte, ROS 2 publica en el hilo principal
        self.audio_queue = queue.Queue()

        # abre el mic en un hilo separado
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1, # solo se necesita un canal
            dtype='float32',
            blocksize=CHUNK_SAMPLES,
            callback=self.audio_callback
        )
        self.stream.start()

        # timer que publica los chunks acumulados en la cola
        self.timer = self.create_timer(CHUNK_MS / 1000.0, self.publish_chunk)

        self.get_logger().info(
            f'Micrófono abierto, publicando chunks de {CHUNK_MS}ms a /audio/raw_chunks'
        )

    def audio_callback(self, indata, frames, time_info, status):
        # sounddevice llama esto automaticamente cada vez que llena un chunk
        # indata shape: (480, 1) - lo aplanamos a (480,)
        self.audio_queue.put(indata[:, 0].copy())

    def publish_chunk(self):
        # publica todos los chunks que se acumularon desde el ult timer
        while not self.audio_queue.empty():
            chunk = self.audio_queue.get()

            msg = SpeechSegment()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.audio_data = chunk.tolist()
            msg.sample_rate = float(SAMPLE_RATE)
            # timestamp_start y timestamp_end los llena el VAD cuando detecta voz
            msg.timestamp_start = 0.0
            msg.timestamp_end = 0.0

            self.publisher.publish(msg)

    def destroy_node(self):
        self.stream.stop()
        self.stream.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AudioCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
