import rclpy
from rclpy.node import Node
from hri_interfaces.msg import SpeechSegment
from std_msgs.msg import Bool
import numpy as np
import torch

# umbral: probabilidad de voz > 0.5 -> hay alguien hablando
VAD_THRESHOLD = 0.5

# cuantos chunks de silencio consecutivos antes de considerar que termino de hablar
# 64ms * 10 = 640ms de silencio -> fin de segmento
SILENCE_CHUNKS = 10

# Mínimo de chunks para publicar (evita segmentos demasiado cortos)
# 30ms * 10 = 300ms mínimo
MIN_SPEECH_CHUNKS = 10

SAMPLE_RATE = 16000


class VadNode(Node):
    def __init__(self):
        super().__init__('vad_node')

        # carga silero VAD desde torch.hub
        self.get_logger().info('Cargando Silero VAD...')
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.model.eval()  # modo inferencia, no entrenamiento
        self.get_logger().info('Silero VAD listo')

        # suscripcion a chunks crudos de audio
        self.subscription = self.create_subscription(
            SpeechSegment,
            '/audio/raw_chunks',
            self.on_chunk,
            100
        )

        # publica segmentos completos de habla
        self.publisher = self.create_publisher(
            SpeechSegment,
            '/audio/speech_segments',
            10
        )

        # publica si hay voz activa ahora mismo (para el indicador de mic)
        self.vad_active_pub = self.create_publisher(Bool, '/audio/vad_active', 10)

        # estado interno del detector
        self.speech_chunks = [] # chunks acumulados mientras hay voz
        self.silence_count = 0 # chunks de silencio consecutivos
        self.is_speaking = False # ¿está hablando alguien ahora mismo?
        self.segment_start = 0.0 # timestamp de cuando empezo a hablar

    def on_chunk(self, msg):
        audio = np.array(msg.audio_data, dtype=np.float32)

        # silero VAD v6 requiere exactamente 512 muestras a 16kHz
        # si el chunk es mas grande, se divide y se toma el max
        chunk_size = 512
        probs = []
        for i in range(0, len(audio), chunk_size):
            sub = audio[i:i+chunk_size]
            if len(sub) < chunk_size:
                sub = np.pad(sub, (0, chunk_size - len(sub)))
            tensor = torch.from_numpy(sub)
            with torch.no_grad():
                probs.append(self.model(tensor, SAMPLE_RATE).item())
        prob = max(probs) if probs else 0.0

        # publica estado de voz activa para el indicador del panel
        active_msg = Bool()
        active_msg.data = prob > VAD_THRESHOLD
        self.vad_active_pub.publish(active_msg)

        if prob > VAD_THRESHOLD:
            # hay voz
            if not self.is_speaking:
                # acaba de empezar a hablar
                self.is_speaking = True
                self.speech_chunks = []
                self.silence_count = 0
                now = self.get_clock().now().nanoseconds / 1e9
                self.segment_start = now
                self.get_logger().debug('Voz detectada - acumulando...')

            self.speech_chunks.append(audio)
            self.silence_count = 0  # resetea el contador de silencio

        else:
            # silencio
            if self.is_speaking:
                self.silence_count += 1
                self.speech_chunks.append(audio)  # incluye el silencio al final

                if self.silence_count >= SILENCE_CHUNKS:
                    # termino de hablar, publica el segmento si es suficientemente largo
                    self.is_speaking = False
                    if len(self.speech_chunks) >= MIN_SPEECH_CHUNKS:
                        self.publish_segment()
                    else:
                        self.get_logger().debug('Segmento muy corto, ignorando')
                    self.speech_chunks = []
                    self.silence_count = 0

    def publish_segment(self):
        # une todos los chunks en un solo array de audio
        full_audio = np.concatenate(self.speech_chunks)

        segment_end = self.get_clock().now().nanoseconds / 1e9
        duration = len(full_audio) / SAMPLE_RATE

        msg = SpeechSegment()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.audio_data = full_audio.tolist()
        msg.sample_rate = float(SAMPLE_RATE)
        msg.timestamp_start = self.segment_start
        msg.timestamp_end = segment_end

        self.publisher.publish(msg)
        self.get_logger().info(
            f'Segmento de voz publicado: {duration:.1f}s -> /audio/speech_segments'
        )


def main(args=None):
    rclpy.init(args=args)
    node = VadNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
