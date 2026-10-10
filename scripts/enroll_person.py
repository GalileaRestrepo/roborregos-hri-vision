"""
script de enrolamiento, esto se corre 1 vez por persona antes de arrancar el sistema
"""
import argparse
import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis


ENROLLED_DIR = os.path.expanduser('~/ros2_hri_ws/data/enrolled')
SAMPLES_NEEDED = 25


def enroll(name):
    person_dir = os.path.join(ENROLLED_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    print(f'\nEnrolando a: {name}')
    print('Cargando InsightFace...')

    app = FaceAnalysis(name='buffalo_m', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: No se pudo abrir la cámara')
        return

    embeddings = []
    print(f'\nMírate a la cámara. Se necesitan {SAMPLES_NEEDED} capturas.')
    print('Presiona ESPACIO para capturar, Q para salir.\n')

    while len(embeddings) < SAMPLES_NEEDED:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = app.get(rgb)

        display = frame.copy()

        if len(faces) == 0:
            cv2.putText(display, 'No se ve ninguna cara', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif len(faces) > 1:
            cv2.putText(display, 'Hay mas de una cara, quedate solo', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        else:
            face = faces[0]
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            progress = f'Capturas: {len(embeddings)}/{SAMPLES_NEEDED}'
            cv2.putText(display, progress, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, 'ESPACIO para capturar', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        display = cv2.resize(display, (1280, 720))
        cv2.imshow(f'Enrolando a {name}', display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print('Cancelado.')
            break

        if key == ord(' ') and len(faces) == 1:
            emb = faces[0].embedding
            if emb is not None:
                embeddings.append(emb)
                print(f'  Captura {len(embeddings)}/{SAMPLES_NEEDED} guardada')

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < SAMPLES_NEEDED:
        print(f'\nSolo se capturaron {len(embeddings)} muestras. Necesitas {SAMPLES_NEEDED}.')
        return

    # se guardan todos los embeddings como un array numpy
    emb_array = np.array(embeddings)
    save_path = os.path.join(person_dir, 'face_embeddings.npy')
    np.save(save_path, emb_array)

    print(f'\nEnrolamiento completo.')
    print(f'Guardado en: {save_path}')
    print(f'Shape: {emb_array.shape}  ({SAMPLES_NEEDED} muestras x 512 dimensiones)')
    print(f'\nAhora enrola tu voz:')
    print(f'  python ~/ros2_hri_ws/scripts/enroll_voice.py --name {name}')
    print(f'\nLuego arranca el sistema:')
    print(f'  bash ~/ros2_hri_ws/launch_all.sh')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enrola una persona en el sistema HRI')
    parser.add_argument('--name', required=True, help='Nombre de la persona (ej: "Gali")')
    args = parser.parse_args()
    enroll(args.name)
