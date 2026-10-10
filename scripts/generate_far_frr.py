"""
genera la curva FAR vs FRR a partir del decision_log.csv

uso:
    python scripts/generate_far_frr.py
    python scripts/generate_far_frr.py --demo   # genera datos simulados

el csv tiene columna ground_truth: 1=genuino, 0=impostor
"""
import csv
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # no necesita ventana GUI

LOG_PATH = os.path.expanduser('~/ros2_hri_ws/data/decision_log.csv')
OUT_PATH = os.path.expanduser('~/ros2_hri_ws/data/far_frr_curve.png')


def load_data(path):
    """lee csv y retorna (scores, labels)"""
    scores = []
    labels = []

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = row.get('ground_truth', '').strip()
            if gt not in ('0', '1'):
                continue  # fila sin anotacion, skip
            scores.append(float(row['fused_conf']))
            labels.append(int(gt))

    return np.array(scores), np.array(labels)


def generate_demo_data():
    """
    genera datos simulados para ver como se ve la curva
    genuinos: confianza alta (media 0.78)
    impostores: confianza baja (media 0.35)
    """
    np.random.seed(42)
    n = 200

    genuinos   = np.clip(np.random.normal(0.78, 0.12, n), 0, 1)
    impostores = np.clip(np.random.normal(0.35, 0.12, n), 0, 1)

    scores = np.concatenate([genuinos, impostores])
    labels = np.concatenate([np.ones(n), np.zeros(n)])

    return scores, labels


def calc_far_frr(scores, labels, thresholds):
    """calcula FAR y FRR para cada umbral"""
    far_list = []
    frr_list = []

    genuinos   = scores[labels == 1]
    impostores = scores[labels == 0]

    for t in thresholds:
        # FAR: impostores aceptados / total impostores
        fa  = np.sum(impostores >= t)
        far = fa / len(impostores) if len(impostores) > 0 else 0.0

        # FRR: genuinos rechazados / total genuinos
        fr  = np.sum(genuinos < t)
        frr = fr / len(genuinos) if len(genuinos) > 0 else 0.0

        far_list.append(far)
        frr_list.append(frr)

    return np.array(far_list), np.array(frr_list)


def find_eer(thresholds, far, frr):
    """encuentra el EER (equal error rate) donde FAR ~ FRR"""
    diff = np.abs(far - frr)
    idx  = np.argmin(diff)
    eer  = (far[idx] + frr[idx]) / 2
    return thresholds[idx], eer


def plot_curve(thresholds, far, frr, eer_threshold, eer, is_demo=False):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(thresholds, far, label='FAR (impostores aceptados)', color='red',   linewidth=2)
    ax.plot(thresholds, frr, label='FRR (genuinos rechazados)',  color='blue',  linewidth=2)
    ax.axvline(eer_threshold, color='green', linestyle='--', linewidth=1.5,
               label=f'EER = {eer:.1%} (umbral={eer_threshold:.2f})')

    ax.set_xlabel('Umbral de confianza', fontsize=12)
    ax.set_ylabel('Tasa de error', fontsize=12)
    title = 'Curva FAR vs FRR — Sistema HRI/Vision'
    if is_demo:
        title += '\n(datos simulados)'
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f'curva guardada en: {OUT_PATH}')
    print(f'EER = {eer:.1%} en umbral = {eer_threshold:.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true',
                        help='usar datos simulados en vez del CSV')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    if args.demo:
        print('usando datos simulados...')
        scores, labels = generate_demo_data()
        is_demo = True
    else:
        if not os.path.exists(LOG_PATH):
            print(f'no encontre el CSV en {LOG_PATH}')
            print('corre con --demo para ver la curva con datos simulados')
            print('o anota ground_truth (1=genuino, 0=impostor) en el CSV y vuelve a correr')
            return
        scores, labels = load_data(LOG_PATH)
        if len(scores) == 0:
            print('el CSV no tiene filas con ground_truth anotado (1 o 0)')
            print('corre con --demo para ver la curva con datos simulados')
            return
        print(f'{len(scores)} muestras cargadas ({int(labels.sum())} genuinos, {int((1-labels).sum())} impostores)')
        is_demo = False

    thresholds = np.linspace(0, 1, 500)
    far, frr   = calc_far_frr(scores, labels, thresholds)
    eer_t, eer = find_eer(thresholds, far, frr)

    plot_curve(thresholds, far, frr, eer_t, eer, is_demo=is_demo)


if __name__ == '__main__':
    main()
