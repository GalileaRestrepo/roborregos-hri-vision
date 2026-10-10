import rclpy
from rclpy.node import Node
from hri_interfaces.msg import SpeechSegment, TranscriptionResult
import numpy as np
from faster_whisper import WhisperModel


class TranscriptionNode(Node):
    def __init__(self):
        super().__init__('transcription_node')

        # carga el modelo whisper "small" en CPU
        self.get_logger().info('Cargando Whisper small...')
        self.model = WhisperModel('tiny', device='cpu', compute_type='int8')
        self.get_logger().info('Whisper listo - transcribiendo en /audio/transcriptions')

        # recibe segmentos de voz completos del VAD
        self.subscription = self.create_subscription(
            SpeechSegment,
            '/audio/speech_segments',
            self.on_segment,
            10
        )

        # publica el texto transcrito
        self.publisher = self.create_publisher(
            TranscriptionResult,
            '/audio/transcriptions',
            10
        )

    def on_segment(self, msg):
        audio = np.array(msg.audio_data, dtype=np.float32)

        # whisper transcribe el audio y detecta el idioma automsticamente
        # beam_size=5: explora 5 caminos posibles de transcripcion
        segments, info = self.model.transcribe(
            audio,
            beam_size=5,
            language='es', # forzar español para evitar confusión si/see
            vad_filter=False # ya filtramos con silero VAD, no duplicar
        )

        # junta segmentos filtrando los de baja confianza (evita alucinaciones)
        segs_list = list(segments)
        good_segs = [s for s in segs_list if s.no_speech_prob < 0.6]
        text = ' '.join(seg.text.strip() for seg in good_segs).strip()

        # limpia signos que OpenCV no soporta
        for ch in ['¿', '¡', '…', '"', '"', ''', ''']:
            text = text.replace(ch, '')
        text = text.strip()

        if not text or len(text) < 2:
            return  # muy corto o vacio, probablemente ruido

        self.get_logger().info(f'Transcripcion: "{text}" ({info.language})')

        result = TranscriptionResult()
        result.header.stamp = self.get_clock().now().to_msg()
        result.text = text
        result.detected_language = info.language
        result.confidence = float(info.language_probability)
        result.timestamp_start = msg.timestamp_start
        result.timestamp_end = msg.timestamp_end

        self.publisher.publish(result)


def main(args=None):
    rclpy.init(args=args)
    node = TranscriptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
