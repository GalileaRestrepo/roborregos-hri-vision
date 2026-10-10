"""
enrolamiento de voz con ECAPA-TDNN (SpeechBrain)
uso: python scripts/enroll_voice.py --name gali
"""
import argparse
import os
import numpy as np
import sounddevice as sd
import torch

ENROLLED_DIR = os.path.expanduser('~/ros2_hri_ws/data/enrolled')
ECAPA_MODEL_DIR = '/Users/agrm/.speechbrain/spkrec-ecapa-voxceleb'
SAMPLE_RATE = 16000
DURATION_SEC = 12
SAMPLES_NEEDED = 5  # mas muestras = embedding mas robusto


def get_embedding(encoder, audio_np):
    tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = encoder.encode_batch(tensor)
    return emb.squeeze().numpy()


def enroll_voice(name):
    person_dir = os.path.join(ENROLLED_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    print(f'\nEnrolando voz de: {name}')
    print('Cargando ECAPA-TDNN...')
    from speechbrain.inference.speaker import EncoderClassifier
    encoder = EncoderClassifier.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb',
        savedir=ECAPA_MODEL_DIR,
        run_opts={'device': 'cpu'}
    )
    print('✓ Modelo listo\n')

    embeddings = []

    for i in range(SAMPLES_NEEDED):
        input(f'Muestra {i+1}/{SAMPLES_NEEDED}: presiona ENTER y habla {DURATION_SEC}s (cuenta algo, di tu nombre, habla natural)...')
        print('Grabando...', end='', flush=True)

        audio = sd.rec(
            int(DURATION_SEC * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print(' listo')

        audio_1d = audio[:, 0]
        emb = get_embedding(encoder, audio_1d)
        embeddings.append(emb)
        print(f'  embedding shape: {emb.shape}, norm: {np.linalg.norm(emb):.3f}')

    # promedia y normaliza
    final_emb = np.mean(embeddings, axis=0)
    final_emb = final_emb / np.linalg.norm(final_emb)

    save_path = os.path.join(person_dir, 'voice_embedding.npy')
    np.save(save_path, final_emb)

    print(f'\n✓ Enrolamiento completo')
    print(f'  guardado en: {save_path}')
    print(f'  shape: {final_emb.shape}  ({SAMPLES_NEEDED} muestras promediadas)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    enroll_voice(args.name)
