import rclpy
from rclpy.node import Node
from hri_interfaces.msg import FusedIdentity, TranscriptionResult
from hri_interfaces.srv import RAGQuery
from std_msgs.msg import String
import csv
import os
import time

THRESHOLD_RESPOND = 0.75
THRESHOLD_CONFIRM = 0.45

DECISION_DENY    = 0
DECISION_CONFIRM = 1
DECISION_RESPOND = 2

STATE_IDLE            = 'IDLE'
STATE_WAITING_CONFIRM = 'WAITING_CONFIRM'

YES_WORDS = {'sí', 'si', 'yes', 'correcto', 'claro', 'afirmativo', 'exacto', 'aja', 'ajá', 'yep', 'yeah', 'ok', 'okay', 'soy', 'confirmo'}
NO_WORDS  = {'no', 'nope', 'negativo', 'incorrecto'}

# segundos máximos esperando confirmación antes de negar
CONFIRM_TIMEOUT_SEC = 30.0
# segundos máximos entre que alguien habla y llega la transcripción
SPEAKING_TIMEOUT_SEC = 3.0


class DecisionNode(Node):
    def __init__(self):
        super().__init__('decision_node')

        # estado de cada track: state, pregunta pendiente, tiempo de espera
        self.track_states = {}

        # última FusedIdentity recibida por track_id
        self.latest_fused = {}

        # track que habló más recientemente y cuándo
        self.last_speaking_tid  = None
        self.last_speaking_time = 0.0

        self.fused_sub = self.create_subscription(
            FusedIdentity, '/fusion/identity_decision', self.on_fused_identity, 10
        )
        self.transcription_sub = self.create_subscription(
            TranscriptionResult, '/audio/transcriptions', self.on_transcription, 10
        )

        self.response_pub = self.create_publisher(String, '/decision/response', 10)

        self.rag_client = self.create_client(RAGQuery, '/rag/query')

        self.create_timer(1.0, self.check_timeouts)

        self.log_path = os.path.expanduser('~/ros2_hri_ws/data/decision_log.csv')
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._init_log()

        self.get_logger().info('nodo de decisión listo')

    # ------------------------------------------------------------------ setup

    def _init_log(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'timestamp', 'track_id', 'name',
                    'face_conf', 'voice_conf', 'lip_conf', 'fused_conf',
                    'decision', 'question', 'ground_truth'
                ])

    def _log(self, fused: FusedIdentity, decision: str, question: str = ''):
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                time.time(), fused.track_id, fused.name,
                f'{fused.face_confidence:.4f}', f'{fused.voice_confidence:.4f}',
                f'{fused.lip_confidence:.4f}', f'{fused.fused_confidence:.4f}',
                decision, question, ''  # ground_truth se anota manualmente
            ])

    # ---------------------------------------------------------------- callbacks

    def on_fused_identity(self, msg: FusedIdentity):
        self.latest_fused[msg.track_id] = msg

        if msg.track_id not in self.track_states:
            self.track_states[msg.track_id] = {
                'state':            STATE_IDLE,
                'pending_question': '',
                'confirm_time':     0.0,
            }

        # registra quién está hablando ahora mismo
        if msg.voice_confidence > 0.05:
            self.last_speaking_tid  = msg.track_id
            self.last_speaking_time = self.get_clock().now().nanoseconds / 1e9

    def on_transcription(self, msg: TranscriptionResult):
        text = msg.text.strip()
        now  = self.get_clock().now().nanoseconds / 1e9

        tid = self._find_current_speaker(now)
        if tid is None:
            self.get_logger().warn('transcripción sin hablante identificado — ignorando')
            return

        fused = self.latest_fused.get(tid)
        if fused is None:
            return

        state_info = self.track_states[tid]

        if state_info['state'] == STATE_WAITING_CONFIRM:
            self._handle_confirmation(tid, fused, text.lower(), state_info)
        else:
            self._handle_new_query(tid, fused, text)

    # ---------------------------------------------------------------- lógica

    def _find_current_speaker(self, now: float):
        if self.last_speaking_tid is None:
            return None
        if now - self.last_speaking_time > SPEAKING_TIMEOUT_SEC:
            return None
        return self.last_speaking_tid

    def _handle_new_query(self, tid: int, fused: FusedIdentity, question: str):
        decision = fused.decision

        if decision == DECISION_RESPOND:
            self.get_logger().info(
                f'[{fused.name}] conf={fused.fused_confidence:.2f} → RESPONDER'
            )
            self._log(fused, 'RESPONDER', question)
            self._call_rag(fused.name, fused.fused_confidence, question)

        elif decision == DECISION_CONFIRM:
            self.get_logger().info(
                f'[{fused.name}] conf={fused.fused_confidence:.2f} → PREGUNTAR confirmación'
            )
            self._log(fused, 'PREGUNTAR', question)
            self.track_states[tid]['state']            = STATE_WAITING_CONFIRM
            self.track_states[tid]['pending_question'] = question
            self.track_states[tid]['confirm_time']     = self.get_clock().now().nanoseconds / 1e9
            self._publish(f'¿Eres {fused.name}? Confirma tu identidad.')

        else:
            self.get_logger().info(
                f'[{fused.name}] conf={fused.fused_confidence:.2f} → NEGAR acceso'
            )
            self._log(fused, 'NEGAR', question)
            self._publish(f'No pude identificarte con suficiente confianza ({fused.fused_confidence:.0%}). Por favor acercate a la camara.')

    def _handle_confirmation(self, tid: int, fused: FusedIdentity, text: str, state_info: dict):
        # elimina puntuacion para que "si," y "si." coincidan con YES_WORDS
        import re
        clean = re.sub(r'[^\w\s]', '', text)
        words = set(clean.split())
        pending = state_info['pending_question']

        if words & YES_WORDS:
            self.get_logger().info(f'[{fused.name}] confirmó identidad → RESPONDER')
            self._log(fused, 'CONFIRMADO', pending)
            self.track_states[tid]['state'] = STATE_IDLE
            self._call_rag(fused.name, fused.fused_confidence, pending)

        elif words & NO_WORDS:
            self.get_logger().info(f'[{fused.name}] negó identidad → NEGAR')
            self._log(fused, 'NEGADO_POR_USUARIO', pending)
            self.track_states[tid]['state'] = STATE_IDLE
            self._publish('Entendido. No puedo procesar tu solicitud.')

        else:
            self._publish(f'No entendí. ¿Eres {fused.name}? Responde sí o no.')

    def check_timeouts(self):
        now = self.get_clock().now().nanoseconds / 1e9
        for tid, state_info in self.track_states.items():
            if state_info['state'] == STATE_WAITING_CONFIRM:
                if now - state_info['confirm_time'] > CONFIRM_TIMEOUT_SEC:
                    self.get_logger().info(
                        f'track {tid}: timeout de confirmación → NEGAR'
                    )
                    state_info['state'] = STATE_IDLE
                    self._publish('No recibi confirmacion a tiempo. Intentalo de nuevo.')

    # ---------------------------------------------------------------- RAG

    def _call_rag(self, name: str, confidence: float, question: str):
        if not self.rag_client.service_is_ready():
            self.get_logger().warn('servicio RAG no disponible')
            self._publish('El servicio de respuestas no está disponible.')
            return

        req = RAGQuery.Request()
        req.identity_name       = name
        req.identity_confidence = confidence
        req.question            = question

        future = self.rag_client.call_async(req)
        future.add_done_callback(lambda f: self._on_rag_response(f, name))

    def _on_rag_response(self, future, name: str):
        try:
            result = future.result()
            if result.authorized and result.answer:
                self._publish(f'{name}: {result.answer}')
            elif not result.authorized:
                self._publish(result.denial_reason or 'No tengo permiso para responder esa consulta.')
            else:
                self._publish('No encontré información relevante para tu pregunta.')
        except Exception as e:
            self.get_logger().error(f'error en respuesta RAG: {e}')
            self._publish('Error al consultar el documento.')

    def _publish(self, text: str):
        msg = String()
        msg.data = text
        self.response_pub.publish(msg)
        self.get_logger().info(f'→ {text}')


def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
