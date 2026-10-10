import rclpy
from rclpy.node import Node
from hri_interfaces.msg import SpeechSegment, VoiceIdentity
import numpy as np
import os
import glob
import torch

# umbral de similitud coseno para identificar una voz
# ECAPA-TDNN es mas preciso que Resemblyzer, umbral mas alto
VOICE_THRESHOLD = 0.70

ECAPA_MODEL_DIR = '/Users/agrm/.speechbrain/spkrec-ecapa-voxceleb'
SAMPLE_RATE = 16000


def cosine_similarity(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class VoiceIdentityNode(Node):
    def __init__(self):
        super().__init__('voice_identity_node')

        # carga ECAPA-TDNN de speechbrain (mejor que Resemblyzer)
        self.get_logger().info('Cargando ECAPA-TDNN (SpeechBrain)...')
        from speechbrain.inference.speaker import EncoderClassifier
        self.encoder = EncoderClassifier.from_hparams(
            source='speechbrain/spkrec-ecapa-voxceleb',
            savedir=ECAPA_MODEL_DIR,
            run_opts={'device': 'cpu'}
        )
        self.get_logger().info('ECAPA-TDNN listo')

        self.enrolled = self.load_enrolled_voices()
        self.get_logger().info(
            f'{len(self.enrolled)} voz/voces enrolada(s): {list(self.enrolled.keys())}'
        )

        self.subscription = self.create_subscription(
            SpeechSegment,
            '/audio/speech_segments',
            self.on_segment,
            10
        )

        self.publisher = self.create_publisher(
            VoiceIdentity,
            '/audio/voice_identity',
            10
        )

    def load_enrolled_voices(self):
        enrolled = {}
        data_dir = os.path.expanduser('~/ros2_hri_ws/data/enrolled')
        for person_dir in glob.glob(os.path.join(data_dir, '*')):
            name = os.path.basename(person_dir)
            emb_file = os.path.join(person_dir, 'voice_embedding.npy')
            if os.path.exists(emb_file):
                enrolled[name] = np.load(emb_file)
                self.get_logger().info(f'  voz cargada: {name}')
        return enrolled

    def get_embedding(self, audio_np):
        """genera embedding ECAPA-TDNN de 192 dims"""
        # ECAPA espera tensor float32 de shape (1, N)
        tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = self.encoder.encode_batch(tensor)
        # shape: (1, 1, 192) -> (192,)
        return emb.squeeze().numpy()

    def identify_voice(self, embedding):
        if not self.enrolled:
            return 'unknown', 0.0

        best_name = 'unknown'
        best_score = -1.0

        for name, enrolled_emb in self.enrolled.items():
            score = cosine_similarity(embedding, enrolled_emb)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < VOICE_THRESHOLD:
            return 'unknown', best_score

        return best_name, best_score

    def is_live_audio(self, audio):
        """
        detecta si el audio es grabado/reproducido o voz real.
        audio grabado reproducido por altavoz tiene energia muy uniforme
        y pierde las variaciones naturales del habla en vivo.
        retorna True si parece audio en vivo.
        """
        # divide en ventanas de 100ms y calcula RMS de cada una
        window_size = int(0.1 * SAMPLE_RATE)
        windows = [audio[i:i+window_size] for i in range(0, len(audio)-window_size, window_size)]
        if len(windows) < 3:
            return True  # muy corto para analizar, asumir real

        rms_values = [np.sqrt(np.mean(w**2)) for w in windows if np.mean(w**2) > 0]
        if len(rms_values) < 3:
            return True

        # coef de variacion: audio real tiene CV alto (voz natural varia mucho)
        # audio grabado reproducido tiene CV bajo (energia muy uniforme)
        cv = np.std(rms_values) / (np.mean(rms_values) + 1e-9)

        # audio real CV > 0.3 tipicamente
        # audio grabado CV < 0.15 tipicamente
        is_live = cv > 0.15
        if not is_live:
            self.get_logger().warn(f'posible audio grabado detectado (CV={cv:.3f})')

        return is_live

    def on_segment(self, msg):
        audio = np.array(msg.audio_data, dtype=np.float32)

        # anti-spoofing: verifica que sea audio en vivo
        if not self.is_live_audio(audio):
            result = VoiceIdentity()
            result.header.stamp = self.get_clock().now().to_msg()
            result.name = 'spoofing_detected'
            result.confidence = 0.0
            result.is_known = False
            result.timestamp_start = msg.timestamp_start
            result.timestamp_end = msg.timestamp_end
            self.publisher.publish(result)
            return

        embedding = self.get_embedding(audio)
        name, confidence = self.identify_voice(embedding)

        self.get_logger().info(
            f'identidad de voz: {name} (confianza: {confidence:.2f})'
        )

        result = VoiceIdentity()
        result.header.stamp = self.get_clock().now().to_msg()
        result.name = name
        result.confidence = float(confidence)
        result.is_known = (name != 'unknown')
        result.timestamp_start = msg.timestamp_start
        result.timestamp_end = msg.timestamp_end

        self.publisher.publish(result)


def main(args=None):
    rclpy.init(args=args)
    node = VoiceIdentityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
