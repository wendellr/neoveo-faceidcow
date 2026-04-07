"""
neoveo_muzzle_pipeline / 04_inference.py
=========================================
Stage 4 – Inferência completa: imagem OU vídeo.

Pipeline:
  Input → YOLO11 detecta ROI do muzzle
        → crop do muzzle
        → EfficientNet extrai embedding
        → cosine similarity contra galeria
        → retorna identidade + score

Uso:
  # Imagem
  python 04_inference.py --source foto.jpg

  # Vídeo / webcam
  python 04_inference.py --source video.mp4
  python 04_inference.py --source 0          # webcam

  # Pasta de imagens
  python 04_inference.py --source data/reid/test/Cattle_001/

Requer:
  pip install ultralytics torch torchvision timm opencv-python
"""

import argparse
import pickle
import json
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
GALLERY_PATH     = Path("runs/reid/neoveo_reid_v1/embeddings_val.pkl")
GALLERY_CUSTOM   = Path("runs/reid/gallery_custom.pkl")   # galeria de enrollment
LABEL_MAP_PATH   = Path("runs/reid/neoveo_reid_v1/label_map.json")

EMBED_DIM        = 512
IMGSZ            = 224
DET_CONF         = 0.40            # confiança mínima do detector
REID_THRESHOLD   = 0.55            # cosine sim mínima para aceitar ID
                                   # abaixo disso → "Desconhecido"
DEVICE           = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ── Transform para o embedder ────────────────────────────────────────────────
embed_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMGSZ, IMGSZ)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_models():
    print("[INFO] Carregando detector YOLO11 ...")
    detector = YOLO(str(DETECTOR_WEIGHTS))

    print("[INFO] Carregando embedder EfficientNet-B0 ...")
    embedder = MuzzleEmbedder(embed_dim=EMBED_DIM).to(DEVICE)
    embedder.load_state_dict(torch.load(EMBEDDER_WEIGHTS, map_location=DEVICE))
    embedder.eval()

    print("[INFO] Carregando galeria de embeddings ...")
    with open(GALLERY_PATH, "rb") as f:
        gallery = pickle.load(f)

    # Merge com galeria custom de enrollment (se existir)
    if GALLERY_CUSTOM.exists():
        with open(GALLERY_CUSTOM, "rb") as f:
            custom = pickle.load(f)
        gallery["embeddings"] = torch.cat([gallery["embeddings"], custom["embeddings"]], dim=0)
        gallery["labels"]     = gallery["labels"] + custom["labels"]
        gallery["classes"]    = gallery["classes"] + custom["classes"]
        print(f"       +{len(custom['labels'])} animais da galeria custom")

    label_map = json.loads(LABEL_MAP_PATH.read_text())

    gallery["embeddings"] = gallery["embeddings"].to(DEVICE)
    print(f"       {len(gallery['labels'])} embeddings | "
          f"{len(gallery['classes'])} identidades")

    return detector, embedder, gallery, label_map


@torch.no_grad()
def embed_crop(embedder: MuzzleEmbedder, crop_bgr: np.ndarray) -> torch.Tensor:
    """Extrai embedding L2-normalizado de um crop BGR (OpenCV)."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor   = embed_tf(crop_rgb).unsqueeze(0).to(DEVICE)
    return embedder(tensor)   # shape (1, EMBED_DIM)


def identify(query_emb: torch.Tensor, gallery: dict,
             label_map: dict, threshold: float) -> tuple[str, float]:
    """Cosine similarity contra toda a galeria, retorna melhor match."""
    sims = F.cosine_similarity(query_emb, gallery["embeddings"])
    best_idx  = sims.argmax().item()
    best_sim  = sims[best_idx].item()
    best_label_idx = gallery["labels"][best_idx]
    cattle_id = label_map.get(str(best_label_idx),
                              gallery["classes"][best_label_idx])

    if best_sim < threshold:
        return "Desconhecido", best_sim
    return cattle_id, best_sim


def draw_result(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                cattle_id: str, sim: float, det_conf: float) -> np.ndarray:
    color = (0, 200, 0) if cattle_id != "Desconhecido" else (0, 0, 220)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{cattle_id}  sim={sim:.2f}  det={det_conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def process_frame(frame: np.ndarray, detector, embedder,
                  gallery, label_map) -> np.ndarray:
    results = detector(frame, conf=DET_CONF, verbose=False)[0]
    boxes = results.boxes

    # Fallback: se nenhuma detecção, trata a imagem inteira como muzzle crop
    # (útil para imagens já pre-cropadas, como as do dataset Li et al.)
    if len(boxes) == 0:
        h, w = frame.shape[:2]
        emb = embed_crop(embedder, frame)
        cattle_id, sim = identify(emb, gallery, label_map, REID_THRESHOLD)
        print(f"  → {cattle_id}  sim={sim:.3f}")
        frame = draw_result(frame, 0, 0, w, h, cattle_id, sim, 0.0)
        return frame

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det_conf = float(box.conf[0])

        # Garante crop válido
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue

        emb = embed_crop(embedder, crop)
        cattle_id, sim = identify(emb, gallery, label_map, REID_THRESHOLD)
        print(f"  → {cattle_id}  sim={sim:.3f}  det={det_conf:.2f}")
        frame = draw_result(frame, x1, y1, x2, y2, cattle_id, sim, det_conf)

    return frame


def run(source: str):
    detector, embedder, gallery, label_map = load_models()

    # Detecta se é imagem ou vídeo/webcam
    source_path = Path(source)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if source_path.is_dir():
        images = [p for p in sorted(source_path.rglob("*"))
                  if p.suffix.lower() in img_exts]
        print(f"[INFO] Processando {len(images)} imagens de {source_path}")
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            print(f"  {img_path.name}")
            out = process_frame(frame, detector, embedder, gallery, label_map)
            cv2.imshow(f"Neoveo — {img_path.name}", out)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

    elif source_path.suffix.lower() in img_exts:
        frame = cv2.imread(source)
        out   = process_frame(frame, detector, embedder, gallery, label_map)
        out_path = source_path.with_stem(source_path.stem + "_result")
        cv2.imwrite(str(out_path), out)
        print(f"[OK] Resultado salvo em {out_path}")
        cv2.imshow("Neoveo — Muzzle Re-ID", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        # Vídeo ou webcam
        cap_src = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(cap_src)

        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir: {source}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if source_path.exists():
            out_path = source_path.with_stem(source_path.stem + "_result")
            writer   = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (width, height)
            )
        else:
            writer = None

        print("[INFO] Processando vídeo — pressione 'q' para sair.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_frame = process_frame(frame, detector, embedder, gallery, label_map)
            cv2.imshow("Neoveo — Muzzle Re-ID", out_frame)
            if writer:
                writer.write(out_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer:
            writer.release()
            print(f"[OK] Vídeo salvo em {out_path}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neoveo — Cattle Muzzle Re-ID"
    )
    parser.add_argument("--source", required=True,
                        help="Imagem, pasta, vídeo ou índice de webcam (0)")
    args = parser.parse_args()
    run(args.source)
