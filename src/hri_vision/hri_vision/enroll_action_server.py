import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from hri_interfaces.action import EnrollSession
from hri_interfaces.msg import FaceDetectionArray
import numpy as np
import os
import threading
import time
import sounddevice as sd
import torch

ENROLLED_DIR   = os.path.expanduser('~/ros2_hri_ws/data/enrolled')
SAMPLE_RATE    = 16000
FACE_SAMPLES   = 25
VOICE_SAMPLES  = 5
VOICE_DURATION = 12

MODE_FACE  = 0
MODE_VOICE = 1
MODE_BOTH  = 2


class EnrollActionServer(Node):
    def __init__(self):
        super().__init__('enroll_action_server')

        # suscripcion a detecciones — usa la camara ya abierta
        self.create_subscription(
            FaceDetectionArray,
            '/vision/detections',
            self._on_detections,
            10
        )

        self._action_server = ActionServer(
            self, EnrollSession, '/enroll', self.execute_callback
        )

        self._collecting = False
        self._embeddings = []
        self._lock = threading.Lock()

        self.get_logger().info('servidor de enrolamiento listo en /enroll')

    def _on_detections(self, msg):
        if not self._collecting:
            return
        for det in msg.detections:
            if len(det.embedding) == 512:
                with self._lock:
                    self._embeddings.append(list(det.embedding))

    async def execute_callback(self, goal_handle):
        name     = goal_handle.request.person_name
        mode     = goal_handle.request.mode
        doc_path = goal_handle.request.document_path

        self.get_logger().info(f'enrolando: {name} modo={mode}')
        person_dir = os.path.join(ENROLLED_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        feedback = EnrollSession.Feedback()
        result   = EnrollSession.Result()

        try:
            quality = 0.0

            if mode in (MODE_FACE, MODE_BOTH):
                quality = self._enroll_face(goal_handle, name, person_dir, feedback)

            if mode in (MODE_VOICE, MODE_BOTH):
                self._enroll_voice(goal_handle, name, person_dir, feedback)

            if doc_path:
                feedback.status_message = 'ingestando documento...'
                feedback.progress = 90
                goal_handle.publish_feedback(feedback)
                self._ingest_document(name, doc_path)

            feedback.progress = 100
            feedback.status_message = f'{name} enrolado'
            goal_handle.publish_feedback(feedback)

            result.success       = True
            result.message       = f'{name} enrolado correctamente'
            result.quality_score = float(quality)
            goal_handle.succeed()

        except Exception as e:
            self.get_logger().error(f'error: {e}')
            result.success       = False
            result.message       = str(e)
            result.quality_score = 0.0
            goal_handle.abort()

        return result

    def _enroll_face(self, goal_handle, name, person_dir, feedback):
        feedback.status_message = f'mira a la camara — necesito {FACE_SAMPLES} capturas'
        feedback.progress = 10
        goal_handle.publish_feedback(feedback)

        with self._lock:
            self._embeddings = []
            self._collecting = True

        while True:
            with self._lock:
                n = len(self._embeddings)
            if n >= FACE_SAMPLES:
                break
            feedback.samples_collected = n
            feedback.progress = int(10 + 45 * n / FACE_SAMPLES)
            feedback.status_message = f'cara: {n}/{FACE_SAMPLES}'
            goal_handle.publish_feedback(feedback)
            time.sleep(0.2)

        with self._lock:
            self._collecting = False
            embeddings = np.array(self._embeddings[:FACE_SAMPLES])

        np.save(os.path.join(person_dir, 'face_embeddings.npy'), embeddings)
        self.get_logger().info(f'cara enrolada: {embeddings.shape}')

        mean_emb = np.mean(embeddings, axis=0)
        sims = [float(np.dot(e, mean_emb) / (np.linalg.norm(e) * np.linalg.norm(mean_emb) + 1e-9))
                for e in embeddings]
        return float(np.mean(sims))

    def _enroll_voice(self, goal_handle, name, person_dir, feedback):
        from speechbrain.inference.speaker import EncoderClassifier
        encoder = EncoderClassifier.from_hparams(
            source='speechbrain/spkrec-ecapa-voxceleb',
            savedir='/Users/agrm/.speechbrain/spkrec-ecapa-voxceleb',
            run_opts={'device': 'cpu'}
        )
        embeddings = []
        for i in range(VOICE_SAMPLES):
            feedback.samples_collected = i
            feedback.status_message = f'voz: muestra {i+1}/{VOICE_SAMPLES} — habla {VOICE_DURATION}s'
            feedback.progress = int(60 + 30 * i / VOICE_SAMPLES)
            goal_handle.publish_feedback(feedback)
            time.sleep(1.0)

            audio = sd.rec(int(VOICE_DURATION * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()

            tensor = torch.tensor(audio[:, 0], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                emb = encoder.encode_batch(tensor).squeeze().numpy()
            embeddings.append(emb)

        final_emb = np.mean(embeddings, axis=0)
        final_emb /= np.linalg.norm(final_emb)
        np.save(os.path.join(person_dir, 'voice_embedding.npy'), final_emb)
        self.get_logger().info(f'voz enrolada: {final_emb.shape}')

    def _ingest_document(self, name, doc_path):
        import sys
        sys.path.insert(0, '/Users/agrm/ros2_hri_ws/install/hri_rag/lib/python3.9/site-packages')
        from hri_rag.document_ingestion import ingest_document
        ingest_document(name, doc_path)
        self.get_logger().info(f'documento de {name} ingestado')


def main(args=None):
    rclpy.init(args=args)
    node = EnrollActionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
