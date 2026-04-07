"""
neoveo_muzzle_pipeline / 02_train_detector.py
==============================================
Stage 2 – Treina YOLO11n para detectar a região do muzzle
em fotos frontais completas de bovinos.

Dataset: Ahmed et al. 2024 (Zenodo 10535934)
  ~8 000 imagens de face inteira com bbox YOLO real do muzzle,
  459 indivíduos. Gerado por 01_prepare_dataset.py em
  data/detection/ (imagens + labels + muzzle.yaml).

Requer:
  pip install ultralytics>=8.3 torch torchvision

GPU recomendada (treina em ~20 min numa RTX 4070).
Sem GPU roda em CPU mas demora várias horas.
"""

from ultralytics import YOLO
from pathlib import Path
import shutil
import torch

# ── Config ──────────────────────────────────────────────────────────────────
YAML_PATH   = Path("data/detection/muzzle.yaml")
MODEL_BASE  = "yolo11n.pt"          # nano: mais rápido, bom para demo
                                    # Use "yolo11s.pt" para melhor acurácia
EPOCHS      = 100
IMGSZ       = 640
BATCH       = 32                    # RTX 4070 12 GB suporta 32; reduza para 16 se der OOM
PROJECT     = "runs/detect"
RUN_NAME    = "neoveo_muzzle_v1"
DEVICE      = "0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ── Treino ──────────────────────────────────────────────────────────────────
def train():
    print(f"[INFO] Dispositivo: {DEVICE}")
    print(f"[INFO] Modelo base: {MODEL_BASE}")
    print(f"[INFO] YAML: {YAML_PATH}")

    model = YOLO(MODEL_BASE)

    results = model.train(
        data      = str(YAML_PATH),
        epochs    = EPOCHS,
        imgsz     = IMGSZ,
        batch     = BATCH,
        device    = DEVICE,
        project   = PROJECT,
        name      = RUN_NAME,
        # Augmentações relevantes para muzzle bovino
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        flipud    = 0.0,            # focinho tem orientação definida
        fliplr    = 0.5,
        mosaic    = 0.5,
        mixup     = 0.1,
        # Early stopping
        patience  = 20,
        # Salva melhor checkpoint
        save      = True,
        save_period = 10,
    )

    # Usa o save_dir real do YOLO (evita bug de path duplicado em algumas versões)
    actual_best = Path(results.save_dir) / "weights" / "best.pt"
    canonical   = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"

    if actual_best.exists() and actual_best != canonical:
        canonical.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(actual_best, canonical)
        print(f"\n[OK] Treino concluído.")
        print(f"     Salvo em  : {actual_best}")
        print(f"     Copiado para: {canonical}")
    else:
        print(f"\n[OK] Treino concluído.")
        print(f"     Melhor modelo: {canonical}")

    return canonical


def validate(model_path: Path):
    print(f"\n[INFO] Validando {model_path} ...")
    model = YOLO(str(model_path))
    metrics = model.val(data=str(YAML_PATH), device=DEVICE)
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision:{metrics.box.p.mean():.4f}")
    print(f"  Recall:   {metrics.box.r.mean():.4f}")
    return metrics


if __name__ == "__main__":
    best = train()
    validate(best)
