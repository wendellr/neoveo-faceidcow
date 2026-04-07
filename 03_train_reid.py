"""
neoveo_muzzle_pipeline / 03_train_reid.py
==========================================
Stage 3 – Treina modelo de re-ID individual por embedding.

Arquitetura:
  Backbone: EfficientNet-B0 (pré-treinado ImageNet)
  Head:     ArcFace Loss  → embeddings de 512 dims
  Métrica:  Cosine Similarity na inferência

Por que ArcFace?
  É o estado da arte para re-ID biométrico (face recognition humans).
  Forçou o estado da arte para muzzle bovino em Li et al. 2022 também.
  Gera embeddings discriminativos mesmo com poucas imagens por animal.

Requer:
  pip install ultralytics torch torchvision timm

Saída:
  runs/reid/neoveo_reid_v1/
    best_reid.pth         ← weights do backbone+head
    embeddings_val.pkl    ← galeria de embeddings (val set)
    label_map.json        ← Cattle_001 → idx
"""

import json
import pickle
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from muzzle_model import MuzzleEmbedder

# ── Config ──────────────────────────────────────────────────────────────────
REID_ROOT    = Path("data/reid")
PROJECT      = Path("runs/reid/neoveo_reid_v1")
EPOCHS       = 80
BATCH        = 32
LR           = 3e-4
EMBED_DIM    = 512
IMGSZ        = 224
DEVICE       = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
NUM_WORKERS  = 4 if DEVICE == "cuda" else 0

PROJECT.mkdir(parents=True, exist_ok=True)


# ── ArcFace Loss ─────────────────────────────────────────────────────────────
class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.
    s=64, m=0.5 são os valores padrão do paper original.
    """
    def __init__(self, in_features: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine   = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return F.cross_entropy(output, labels)


# ── Transforms ───────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMGSZ + 32, IMGSZ + 32)),
    transforms.RandomCrop(IMGSZ),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMGSZ, IMGSZ)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Treino ───────────────────────────────────────────────────────────────────
def train():
    train_ds = datasets.ImageFolder(REID_ROOT / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(REID_ROOT / "val",   transform=val_tf)

    num_classes = len(train_ds.classes)
    print(f"[INFO] {num_classes} identidades | "
          f"{len(train_ds)} imgs train | {len(val_ds)} imgs val")

    # Salva mapeamento label
    label_map = {v: k for k, v in train_ds.class_to_idx.items()}
    (PROJECT / "label_map.json").write_text(json.dumps(label_map, indent=2))

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    model     = MuzzleEmbedder(embed_dim=EMBED_DIM).to(DEVICE)
    arcface   = ArcFaceLoss(EMBED_DIM, num_classes).to(DEVICE)

    optimizer = AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_rank1 = 0.0
    best_margin = -2.0
    log_path  = PROJECT / "training_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_rank1,pos_sim,neg_sim,margin\n")

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train(); arcface.train()
        train_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            embs = model(imgs)
            loss = arcface(embs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # Val: Rank-1 accuracy (leave-one-out nearest neighbor)
        model.eval()
        all_embs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_dl:
                embs = model(imgs.to(DEVICE)).cpu()
                all_embs.append(embs)
                all_labels.extend(labels.tolist())
        all_embs   = torch.cat(all_embs)          # (N, 512)
        all_labels = torch.tensor(all_labels)

        # Cosine similarity matrix
        sim_matrix = torch.mm(all_embs, all_embs.t())
        sim_matrix.fill_diagonal_(-1.0)           # exclui self-match
        pred_labels = all_labels[sim_matrix.argmax(dim=1)]
        rank1 = (pred_labels == all_labels).float().mean().item()

        # Cosine margin: pos_sim (mesmo animal) vs neg_sim (animais diferentes)
        sim_full = torch.mm(all_embs, all_embs.t())
        same_mask = all_labels.unsqueeze(0) == all_labels.unsqueeze(1)
        same_mask.fill_diagonal_(False)           # exclui diagonal
        diff_mask = ~same_mask
        diff_mask.fill_diagonal_(False)
        pos_sim = sim_full[same_mask].mean().item()
        neg_sim = sim_full[diff_mask].mean().item()
        margin  = pos_sim - neg_sim

        scheduler.step()

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{rank1:.4f},{pos_sim:.4f},{neg_sim:.4f},{margin:.4f}\n")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"train_loss={train_loss:.4f} | val_rank1={rank1:.4f} | "
                  f"pos={pos_sim:.3f} neg={neg_sim:.3f} margin={margin:.3f}")

        if rank1 > best_rank1 or (rank1 == best_rank1 and margin > best_margin):
            best_rank1  = rank1
            best_margin = margin
            torch.save(model.state_dict(), PROJECT / "best_reid.pth")

        # Salva sempre o último epoch (train_loss ainda melhora mesmo com rank1 saturado)
        torch.save(model.state_dict(), PROJECT / "last_reid.pth")

    print(f"\n[OK] Melhor val Rank-1: {best_rank1:.4f} | Melhor margin: {best_margin:.4f}")
    print(f"     Pesos (melhor)       : {PROJECT / 'best_reid.pth'}")
    print(f"     Pesos (último epoch) : {PROJECT / 'last_reid.pth'}")
    return model


def build_gallery(model: MuzzleEmbedder):
    """Gera a galeria de embeddings do val set para usar na inferência."""
    val_ds = datasets.ImageFolder(REID_ROOT / "val", transform=val_tf)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False,
                        num_workers=NUM_WORKERS)
    model.eval()
    all_embs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_dl:
            embs = model(imgs.to(DEVICE)).cpu()
            all_embs.append(embs)
            all_labels.extend(labels.tolist())

    gallery = {
        "embeddings": torch.cat(all_embs),
        "labels":     all_labels,
        "classes":    val_ds.classes,
    }
    out = PROJECT / "embeddings_val.pkl"
    with open(out, "wb") as f:
        pickle.dump(gallery, f)
    print(f"[OK] Galeria salva em {out} ({len(all_labels)} embeddings)")


if __name__ == "__main__":
    trained_model = train()
    # Recarrega melhor checkpoint para gerar galeria
    best_model = MuzzleEmbedder(embed_dim=EMBED_DIM).to(DEVICE)
    best_model.load_state_dict(torch.load(PROJECT / "best_reid.pth",
                                          map_location=DEVICE))
    build_gallery(best_model)
