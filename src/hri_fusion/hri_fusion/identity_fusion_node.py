import rclpy
from rclpy.node import Node
from hri_interfaces.msg import FaceDetectionArray, VoiceIdentity, LipMovementArray, FusedIdentity
from collections import defaultdict, deque
import numpy as np

FACE_WEIGHT = 0.50
VOICE_WEIGHT = 0.30
LIP_WEIGHT   = 0.20

WINDOW_SIZE = 20   # frames para el promedio deslizante
EMA_ALPHA   = 0.3  # suavizado exponencial

# ventana hacia atrás para asociar voz con cara (en segundos)
VOICE_WINDOW_SEC = 1.5

# umbrales — reflejados en decision_node
THRESHOLD_RESPOND = 0.75
THRESHOLD_CONFIRM = 0.45

DECISION_DENY    = 0
DECISION_CONFIRM = 1
DECISION_RESPOND = 2


class IdentityFusionNode(Node):
    def __init__(self):
        super().__init__('identity_fusion_node')

        self.tracking_data = defaultdict(lambda: {
            'name':       'unknown',
            'is_known':   False,
            'face_conf':  deque(maxlen=WINDOW_SIZE),
            'voice_conf': deque(maxlen=WINDOW_SIZE),
            'lip_conf':   deque(maxlen=WINDOW_SIZE),
            'fused_conf': 0.0,
            'last_seen':  0.0,
        })

        # historial de lip activity: {track_id: deque([(time_s, openness)])}
        self.lip_history = defaultdict(lambda: deque(maxlen=60))

        self.face_sub = self.create_subscription(
            FaceDetectionArray, '/vision/tracked_faces', self.on_tracked_faces, 10
        )
        self.voice_sub = self.create_subscription(
            VoiceIdentity, '/audio/voice_identity', self.on_voice_identity, 10
        )
        self.lip_sub = self.create_subscription(
            LipMovementArray, '/fusion/lip_movement', self.on_lip_movement, 10
        )

        self.decision_pub = self.create_publisher(
            FusedIdentity, '/fusion/identity_decision', 10
        )

        self.timer = self.create_timer(0.1, self.process_fusion)

        self.get_logger().info('nodo de fusión de identidad listo')

    def on_tracked_faces(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        for face in msg.detections:
            tid = face.track_id
            d = self.tracking_data[tid]
            d['name']     = face.name
            d['is_known'] = face.is_known
            d['face_conf'].append(face.identity_confidence)
            d['last_seen'] = now

    def on_lip_movement(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        for lm in msg.lip_movements:
            self.lip_history[lm.track_id].append((now, lm.mouth_openness))
            if lm.is_speaking:
                self.tracking_data[lm.track_id]['lip_conf'].append(lm.mouth_openness)

    def on_voice_identity(self, msg):
        """asigna la voz al track correcto usando boca + nombre de cara."""
        # si se detecto spoofing, ignorar
        if msg.name == 'spoofing_detected':
            self.get_logger().warn('audio grabado detectado — ignorando segmento')
            return

        t_end   = msg.timestamp_end
        t_start = t_end - VOICE_WINDOW_SEC

        # estrategia 1: si la voz identifico a alguien, busca el track con ese nombre
        if msg.is_known and msg.name != 'unknown':
            for tid, d in self.tracking_data.items():
                if d['name'] == msg.name:
                    d['voice_conf'].append(msg.confidence)
                    self.get_logger().info(
                        f'voz "{msg.name}" ({msg.confidence:.2f}) → track {tid} (por nombre)'
                    )
                    return

        # estrategia 2: busca cara con mayor actividad de boca en la ventana
        best_tid      = None
        best_activity = 0.0

        for tid, history in self.lip_history.items():
            values = [v for (t, v) in history if t_start <= t <= t_end]
            if values:
                activity = float(np.mean(values))
                if activity > best_activity:
                    best_activity = activity
                    best_tid = tid

        # estrategia 3: fallback al track mas reciente
        if best_tid is None and self.tracking_data:
            best_tid = max(self.tracking_data.items(), key=lambda x: x[1]['last_seen'])[0]

        if best_tid is not None:
            self.tracking_data[best_tid]['voice_conf'].append(msg.confidence)
            self.get_logger().info(
                f'voz "{msg.name}" ({msg.confidence:.2f}) → track {best_tid}'
                f' (boca: {best_activity:.3f})'
            )

    def process_fusion(self):
        now = self.get_clock().now().nanoseconds / 1e9

        for track_id, d in self.tracking_data.items():
            if not d['face_conf']:
                continue
            # ignora tracks que llevan más de 3s sin verse
            if now - d['last_seen'] > 3.0:
                continue

            face_avg  = float(np.mean(d['face_conf']))
            voice_avg = float(np.mean(d['voice_conf'])) if d['voice_conf'] else 0.0
            lip_avg   = float(np.mean(d['lip_conf']))   if d['lip_conf']   else 0.0

            # si no hay voz enrolada, redistribuye pesos a cara y boca
            if voice_avg == 0.0:
                fused = face_avg * 0.85 + lip_avg * 0.15
            else:
                fused = face_avg * FACE_WEIGHT + voice_avg * VOICE_WEIGHT + lip_avg * LIP_WEIGHT
            d['fused_conf'] = EMA_ALPHA * fused + (1 - EMA_ALPHA) * d['fused_conf']

            conf = d['fused_conf']
            if conf >= THRESHOLD_RESPOND:
                decision, label = DECISION_RESPOND, 'RESPONDER'
            elif conf >= THRESHOLD_CONFIRM:
                decision, label = DECISION_CONFIRM, 'PREGUNTAR'
            else:
                decision, label = DECISION_DENY, 'NEGAR'

            out = FusedIdentity()
            out.header.stamp     = self.get_clock().now().to_msg()
            out.track_id         = track_id
            out.name             = d['name']
            out.face_confidence  = face_avg
            out.voice_confidence = voice_avg
            out.lip_confidence   = lip_avg
            out.fused_confidence = d['fused_conf']
            out.decision         = decision
            out.decision_label   = label

            self.decision_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = IdentityFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
