"""
neoveo_muzzle_pipeline / 01_prepare_dataset.py
================================================
Stage 1 – Prepara os dois datasets para as duas tarefas:

  A) Detecção de muzzle  → Ahmed et al. 2024 (Zenodo 10535934)
     Fotos frontais de face inteira com anotações YOLO de bbox do muzzle.
     Estrutura esperada:
       DETECTION_ROOT/
         images/   (*.jpg)
         labels/   (*.txt, formato YOLO: class cx cy w h)

  B) Re-ID por embedding → Li et al. 2022 (Zenodo 6324361)
     Crops isolados de muzzle organizados por identidade.
     Estrutura esperada:
       REID_ROOT/
         cattle_0100/  (12-15 imagens .jpg)
         cattle_0200/
         ...

Saída:
  data/
    detection/          ← YOLO format (images + labels)
      images/train/
      images/val/
      labels/train/
      labels/val/
      muzzle.yaml
    reid/               ← pastas por identidade
      train/
      val/
      test/
"""

import shutil
import random
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
DETECTION_ROOT = Path("DetectionDataset")          # Roboflow (padrão) ou Ahmed 2024 — ver start_training.sh
REID_ROOT_SRC  = Path("BeefCattle_Muzzle_Individualized")  # Li et al. 2022 — Zenodo 6324361
OUTPUT_ROOT    = Path("data")
SEED           = 42

# Split detecção: 85% train / 15% val (sobre as imagens já anotadas)
DET_SPLIT      = 0.85

# Split re-ID: 70% train / 15% val / 15% test  (por identidade, não por imagem)
REID_SPLIT     = (0.70, 0.15, 0.15)

random.seed(SEED)


def get_cattle_dirs(root: Path) -> list[Path]:
    dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"[INFO] Encontrados {len(dirs)} indivíduos em {root}")
    return dirs


def prepare_detection(det_src: Path, det_out: Path, split: float):
    """
    Prepara formato YOLO para detectar muzzle.

    Auto-detecta o formato de entrada:
      - Roboflow (pré-dividido): det_src/train/images/ + det_src/valid/images/
      - Flat (Ahmed 2024):       det_src/images/ + det_src/labels/  → split automático
    """
    img_train = det_out / "images" / "train"
    img_val   = det_out / "images" / "val"
    lbl_train = det_out / "labels" / "train"
    lbl_val   = det_out / "labels" / "val"
    for p in [img_train, img_val, lbl_train, lbl_val]:
        p.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png"}

    if (det_src / "train").exists():
        # ── Formato Roboflow (já pré-dividido) ──────────────────────────────
        print("[DET] Formato Roboflow detectado (train/valid pré-divididos)")
        counts = {"train": 0, "valid": 0}
        for subset_name, out_img_dir, out_lbl_dir in [
            ("train", img_train, lbl_train),
            ("valid", img_val,   lbl_val),
        ]:
            src_imgs = det_src / subset_name / "images"
            src_lbls = det_src / subset_name / "labels"
            if not src_imgs.exists():
                continue
            for img_path in src_imgs.rglob("*"):
                if img_path.suffix.lower() not in img_exts:
                    continue
                shutil.copy2(img_path, out_img_dir / img_path.name)
                lbl_src = src_lbls / (img_path.stem + ".txt")
                if lbl_src.exists():
                    shutil.copy2(lbl_src, out_lbl_dir / lbl_src.name)
                counts[subset_name] += 1
        print(f"[DET] {counts['train']} train | {counts['valid']} val")

    else:
        # ── Formato flat (Ahmed 2024) — split manual ─────────────────────────
        print("[DET] Formato flat (images/ + labels/) detectado")
        img_dir = det_src / "images"
        lbl_dir = det_src / "labels"
        if not img_dir.exists():
            raise FileNotFoundError(
                f"Nem 'train/' nem 'images/' encontrado em {det_src}.\n"
                "Verifique a estrutura do DetectionDataset/."
            )
        all_images = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in img_exts])
        if not all_images:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em {img_dir}")
        random.shuffle(all_images)
        n_train = int(len(all_images) * split)
        train_imgs = all_images[:n_train]
        val_imgs   = all_images[n_train:]
        missing = 0
        for subset_imgs, out_img_dir, out_lbl_dir in [
            (train_imgs, img_train, lbl_train),
            (val_imgs,   img_val,   lbl_val),
        ]:
            for img_path in subset_imgs:
                shutil.copy2(img_path, out_img_dir / img_path.name)
                lbl_src = lbl_dir / (img_path.stem + ".txt")
                if not lbl_src.exists():
                    candidates = list(lbl_dir.rglob(img_path.stem + ".txt"))
                    lbl_src = candidates[0] if candidates else lbl_src
                if lbl_src.exists():
                    shutil.copy2(lbl_src, out_lbl_dir / lbl_src.name)
                else:
                    missing += 1
        if missing:
            print(f"[WARN] {missing} imagens sem label correspondente")
        print(f"[DET] {len(train_imgs)} train | {len(val_imgs)} val")

    yaml_path = det_out / "muzzle.yaml"
    yaml_path.write_text(f"""# Neoveo.ai — Cattle Muzzle Detection
path: {det_out.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  0: muzzle
""")
    print(f"[DET] YAML salvo em {yaml_path}")
    return yaml_path


def prepare_reid(reid_src: Path, reid_out: Path, splits: tuple):
    """
    Prepara estrutura de pastas por identidade para re-ID (Li et al. 2022).
    Divide os INDIVÍDUOS (não imagens) em train/val/test.
    """
    cattle_dirs = get_cattle_dirs(reid_src)

    n = len(cattle_dirs)
    n_train = int(n * splits[0])
    n_val   = int(n * splits[1])

    shuffled = cattle_dirs.copy()
    random.shuffle(shuffled)

    subsets = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }

    for subset_name, dirs in subsets.items():
        subset_root = reid_out / subset_name
        for cattle_dir in dirs:
            dst = subset_root / cattle_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for img in cattle_dir.glob("*"):
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    shutil.copy2(img, dst / img.name)

        counts = sum(
            len(list((reid_out / subset_name / d.name).glob("*")))
            for d in dirs
        )
        print(f"[REID] {subset_name}: {len(dirs)} identidades | {counts} imagens")


def main():
    if not DETECTION_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset de detecção não encontrado em '{DETECTION_ROOT}'.\n"
            "Execute start_training.sh ou extraia manualmente para DetectionDataset/."
        )
    if not REID_ROOT_SRC.exists():
        raise FileNotFoundError(
            f"Dataset de re-ID não encontrado em '{REID_ROOT_SRC}'. "
            "Baixe Li et al. 2022 (Zenodo 6324361) e ajuste REID_ROOT_SRC."
        )

    det_root  = OUTPUT_ROOT / "detection"
    reid_root = OUTPUT_ROOT / "reid"

    print("\n── Stage A: Detection dataset (Ahmed 2024) ──")
    yaml_path = prepare_detection(DETECTION_ROOT, det_root, DET_SPLIT)

    print("\n── Stage B: Re-ID dataset (Li 2022) ──")
    prepare_reid(REID_ROOT_SRC, reid_root, REID_SPLIT)

    print(f"\n[OK] Dataset pronto em '{OUTPUT_ROOT}/'")
    print(f"     YOLO yaml: {yaml_path}")


if __name__ == "__main__":
    main()
