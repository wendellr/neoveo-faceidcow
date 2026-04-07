#!/bin/bash
# ============================================================
# Neoveo — Muzzle Re-ID Pipeline
# Script de setup e treino para Ubuntu + RTX 4070 (CUDA)
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "=================================================="
echo " Neoveo — Cattle Muzzle Re-ID Training"
echo " Dir: $PROJECT_DIR"
echo "=================================================="

# ── 1. Verificar Python ─────────────────────────────────────
echo ""
echo "[1/7] Verificando Python..."
if ! command -v python3 &>/dev/null; then
    echo "[ERRO] Python3 não encontrado. Instale com: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi
python3 --version

# ── 2. Criar e ativar venv ──────────────────────────────────
echo ""
echo "[2/7] Configurando virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "      .venv criado."
else
    echo "      .venv já existe, reutilizando."
fi
source .venv/bin/activate

# ── 3. Instalar dependências ────────────────────────────────
echo ""
echo "[3/7] Instalando dependências..."
pip install --upgrade pip -q

# Instala PyTorch cu124 apenas se CUDA não estiver disponível no venv atual
CUDA_OK=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$CUDA_OK" != "True" ]; then
    echo "      Instalando PyTorch cu124 (compatível com driver CUDA 12.8)..."
    pip install "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124 -q
else
    echo "      PyTorch com CUDA já presente, pulando reinstalação."
fi

# Restante das dependências (torch já instalado, não será sobrescrito)
pip install -r requirements.txt -q
echo "      Dependências instaladas."

# ── 4. Baixar modelo YOLO11n ────────────────────────────────
echo ""
echo "[4/7] Verificando modelo YOLO11n..."
if [ ! -f "yolo11n.pt" ]; then
    echo "      Baixando yolo11n.pt..."
    python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
    echo "      yolo11n.pt baixado."
else
    echo "      yolo11n.pt já presente, pulando."
fi

# ── 5. Baixar datasets ─────────────────────────────────────
echo ""
echo "[5/7] Verificando datasets..."

# ── 5a. Li et al. 2022 — Re-ID (Zenodo 6324361) ────────────
REID_DIR="BeefCattle_Muzzle_Individualized"
REID_URL="https://zenodo.org/records/6324361/files/BeefCattle_Muzzle_database.zip?download=1"
REID_ZIP="BeefCattle_Muzzle_database.zip"

if [ ! -d "$REID_DIR" ]; then
    echo "      [Li 2022] Baixando dataset re-ID (~600 MB) do Zenodo..."
    wget -q --show-progress -O "$REID_ZIP" "$REID_URL"
    echo "      [Li 2022] Extraindo..."
    unzip -q "$REID_ZIP"
    EXTRACTED=$(unzip -Z1 "$REID_ZIP" | head -1 | cut -d'/' -f1)
    if [ "$EXTRACTED" != "$REID_DIR" ]; then
        mv "$EXTRACTED" "$REID_DIR"
    fi
    rm "$REID_ZIP"
    echo "      [Li 2022] Dataset re-ID pronto em $REID_DIR/"
else
    echo "      [Li 2022] Dataset re-ID já presente, pulando download."
fi

# ── 5b. Ahmed et al. 2024 — Detecção (Zenodo 10535934) ────
DET_DIR="AhmedMuzzle2024"
DET_URL="https://zenodo.org/records/10535934/files/INDIVIDUAL%20SUBJECTS%20Data.zip?download=1"
DET_ZIP="INDIVIDUAL_SUBJECTS_Data.zip"   # nome local sem espaços

if [ ! -d "$DET_DIR" ]; then
    echo "      [Ahmed 2024] Baixando dataset de detecção (~13 GB) do Zenodo..."
    wget --show-progress -O "$DET_ZIP" "$DET_URL"
    echo "      [Ahmed 2024] Extraindo (pode demorar alguns minutos)..."
    # O zip extrai para "INDIVIDUAL SUBJECTS Data/" — renomeia para AhmedMuzzle2024/
    unzip -q "$DET_ZIP"
    EXTRACTED=$(unzip -Z1 "$DET_ZIP" | head -1 | cut -d'/' -f1)
    if [ "$EXTRACTED" != "$DET_DIR" ]; then
        mv "$EXTRACTED" "$DET_DIR"
    fi
    rm -f "$DET_ZIP"
    echo "      [Ahmed 2024] Dataset de detecção pronto em $DET_DIR/"
    echo "      [Ahmed 2024] Verifique que existe $DET_DIR/images/ e $DET_DIR/labels/"
else
    echo "      [Ahmed 2024] Dataset de detecção já presente, pulando download."
fi

# ── 6. Verificar CUDA ──────────────────────────────────────
echo ""
echo "[6/7] Verificando GPU..."
python3 - <<'EOF'
import torch
print(f"      PyTorch   : {torch.__version__}")
print(f"      CUDA disp.: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"      GPU       : {torch.cuda.get_device_name(0)}")
    print(f"      VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("      [WARN] CUDA não disponível — treino rodará em CPU (muito lento).")
EOF

# ── 7. Treino ──────────────────────────────────────────────
echo ""
echo "[7/7] Iniciando pipeline de treino..."
echo "--------------------------------------------------"

echo ""
echo ">>> Stage 1: Preparando dataset..."
python3 01_prepare_dataset.py

echo ""
echo ">>> Stage 2: Treinando detector YOLO11 (muzzle detection)..."
python3 02_train_detector.py

echo ""
echo ">>> Stage 3: Treinando re-ID EfficientNet-B0 + ArcFace..."
python3 03_train_reid.py

echo ""
echo "=================================================="
echo " Treino concluído!"
echo " Pesos do detector : runs/detect/neoveo_muzzle_v1/weights/best.pt"
echo " Pesos do re-ID    : runs/reid/neoveo_reid_v1/best_reid.pth"
echo " Galeria base      : runs/reid/neoveo_reid_v1/embeddings_val.pkl"
echo ""
echo " Para baixar os pesos de volta para o Mac:"
echo "   rsync -av usuario@servidor:$(pwd)/runs/ ./runs/"
echo "=================================================="
