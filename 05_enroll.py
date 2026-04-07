"""
neoveo_muzzle_pipeline / 05_enroll.py
=======================================
Stage 5 – Cadastro (enrollment) de um novo animal na galeria.

Não requer re-treino. O modelo já aprendeu features discriminativas
de muzzle com ArcFace — um novo animal é apenas um novo vetor na galeria.

Workflow:
  1. YOLO11 detecta e cropa o muzzle em cada foto
  2. EfficientNet extrai embedding por foto
  3. Mean pooling dos embeddings → 1 vetor de 512 dims por animal
  4. Vetor salvo em runs/reid/gallery_custom.pkl com o ID informado

  Na inferência (04_inference.py), a galeria base e a custom são
  carregadas e mergeadas automaticamente.

Uso:
  # Cadastrar com pasta de fotos (recomendado: 3–5 fotos)
  python 05_enroll.py --id "Nelore_042" --source fotos/boi_42/

  # Cadastrar com foto única
  python 05_enroll.py --id "Nelore_042" --source boi_42.jpg

  # Listar animais cadastrados
  python 05_enroll.py --list

  # Remover um animal
  python 05_enroll.py --remove "Nelore_042"
"""

import argparse
import pickle
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from muzzle_model import MuzzleEmbedder

# ── Config ──────────────────────────────────────────────────────────────────
DETECTOR_WEIGHTS = Path("runs/detect/neoveo_muzzle_v1/weights/best.pt")
EMBEDDER_WEIGHTS = Path("runs/reid/neoveo_reid_v1/best_reid.pth")
GALLERY_CUSTOM   = Path("runs/reid/gallery_custom.pkl")

EMBED_DIM  = 512
IMGSZ      = 224
DET_CONF   = 0.30   # threshold menor para enrollment (queremos capturar o muzzle)
DEVICE     = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Transform ────────────────────────────────────────────────────────────────
embed_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMGSZ, IMGSZ)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_models():
    detector = YOLO(str(DETECTOR_WEIGHTS))
    embedder = MuzzleEmbedder(embed_dim=EMBED_DIM).to(DEVICE)
    embedder.load_state_dict(torch.load(EMBEDDER_WEIGHTS, map_location=DEVICE))
    embedder.eval()
    return detector, embedder


def load_custom_gallery() -> dict:
    if GALLERY_CUSTOM.exists():
        with open(GALLERY_CUSTOM, "rb") as f:
            return pickle.load(f)
    return {"embeddings": torch.zeros(0, EMBED_DIM), "labels": [], "classes": []}


def save_custom_gallery(gallery: dict):
    GALLERY_CUSTOM.parent.mkdir(parents=True, exist_ok=True)
    with open(GALLERY_CUSTOM, "wb") as f:
        pickle.dump(gallery, f)


@torch.no_grad()
def extract_embedding(detector, embedder, img_bgr: np.ndarray) -> torch.Tensor | None:
    """
    Detecta o muzzle na imagem, cropa e extrai embedding.
    Retorna None se nenhum muzzle for detectado.
    """
    results = detector(img_bgr, conf=DET_CONF, verbose=False)[0]
    if len(results.boxes) == 0:
        return None

    # Usa a detecção com maior confiança
    best_box = max(results.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
    crop = img_bgr[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor   = embed_tf(crop_rgb).unsqueeze(0).to(DEVICE)
    return embedder(tensor)   # (1, EMBED_DIM)


def enroll(animal_id: str, source: str):
    """Cadastra um novo animal na galeria custom."""
    source_path = Path(source)

    # Coleta imagens
    if source_path.is_dir():
        img_paths = sorted([p for p in source_path.rglob("*") if p.suffix.lower() in IMG_EXTS])
    elif source_path.suffix.lower() in IMG_EXTS:
        img_paths = [source_path]
    else:
        raise ValueError(f"Source inválido: {source}. Use uma imagem ou pasta de imagens.")

    if not img_paths:
        raise ValueError(f"Nenhuma imagem encontrada em: {source}")

    print(f"[INFO] Dispositivo: {DEVICE}")
    print(f"[INFO] Carregando modelos ...")
    detector, embedder = load_models()

    # Extrai embeddings
    embeddings = []
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Não foi possível ler: {img_path.name}")
            continue

        emb = extract_embedding(detector, embedder, img)
        if emb is None:
            print(f"[WARN] Muzzle não detectado em: {img_path.name}")
        else:
            embeddings.append(emb)
            print(f"  ✓ {img_path.name}")

    if not embeddings:
        print("[ERRO] Nenhum muzzle detectado nas imagens. Verifique as fotos e tente novamente.")
        return

    # Mean pooling + L2-norm → 1 vetor representativo
    stacked    = torch.cat(embeddings, dim=0)          # (N, 512)
    mean_emb   = stacked.mean(dim=0, keepdim=True)     # (1, 512)
    mean_emb   = F.normalize(mean_emb, dim=1)          # L2-norm

    # Carrega galeria e verifica duplicata
    gallery = load_custom_gallery()
    if animal_id in gallery["classes"]:
        print(f"[WARN] '{animal_id}' já está cadastrado. Atualizando embedding ...")
        idx = gallery["classes"].index(animal_id)
        gallery["embeddings"][idx] = mean_emb.squeeze(0).cpu()
    else:
        new_idx = len(gallery["classes"])
        gallery["classes"].append(animal_id)
        gallery["labels"].append(new_idx)
        new_emb_cpu = mean_emb.squeeze(0).cpu().unsqueeze(0)
        if gallery["embeddings"].shape[0] == 0:
            gallery["embeddings"] = new_emb_cpu
        else:
            gallery["embeddings"] = torch.cat([gallery["embeddings"], new_emb_cpu], dim=0)

    save_custom_gallery(gallery)
    print(f"\n[OK] '{animal_id}' cadastrado com {len(embeddings)} foto(s) de {len(img_paths)} fornecida(s).")
    print(f"     Galeria custom: {len(gallery['classes'])} animal(is) em {GALLERY_CUSTOM}")


def list_enrolled():
    """Lista todos os animais na galeria custom."""
    gallery = load_custom_gallery()
    if not gallery["classes"]:
        print("[INFO] Galeria custom vazia. Nenhum animal cadastrado ainda.")
        return
    print(f"[INFO] {len(gallery['classes'])} animal(is) cadastrado(s):")
    for i, name in enumerate(gallery["classes"]):
        print(f"  [{i+1:03d}] {name}")


def remove_enrolled(animal_id: str):
    """Remove um animal da galeria custom."""
    gallery = load_custom_gallery()
    if animal_id not in gallery["classes"]:
        print(f"[ERRO] '{animal_id}' não encontrado na galeria custom.")
        return

    idx = gallery["classes"].index(animal_id)
    gallery["classes"].pop(idx)
    gallery["labels"].pop(idx)
    # Reconstrói labels sequenciais
    gallery["labels"] = list(range(len(gallery["classes"])))
    # Remove linha do tensor
    all_rows = list(range(gallery["embeddings"].shape[0]))
    all_rows.pop(idx)
    if all_rows:
        gallery["embeddings"] = gallery["embeddings"][all_rows]
    else:
        gallery["embeddings"] = torch.zeros(0, EMBED_DIM)

    save_custom_gallery(gallery)
    print(f"[OK] '{animal_id}' removido da galeria custom.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neoveo — Enrollment de bovino na galeria")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id",     metavar="ID",
                       help="ID do animal a cadastrar (ex: 'Nelore_042')")
    group.add_argument("--list",   action="store_true",
                       help="Lista animais cadastrados na galeria custom")
    group.add_argument("--remove", metavar="ID",
                       help="Remove um animal da galeria custom pelo ID")

    parser.add_argument("--source", metavar="PATH",
                        help="Imagem ou pasta de imagens para enrollment (obrigatório com --id)")
    args = parser.parse_args()

    if args.list:
        list_enrolled()
    elif args.remove:
        remove_enrolled(args.remove)
    elif args.id:
        if not args.source:
            parser.error("--source é obrigatório ao usar --id")
        enroll(args.id, args.source)
