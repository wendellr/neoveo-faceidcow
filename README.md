# Neoveo.ai — Cattle Muzzle Re-ID Pipeline

Pipeline de identificação biométrica individual de bovinos por padrão de focinho (muzzle print), usando YOLO11 para detecção de ROI e EfficientNet-B0 + ArcFace para re-ID por embedding.

## Início rápido (servidor Ubuntu + GPU)

Este é um repositório **privado**. É necessário um Personal Access Token (PAT) do GitHub.

**Gerar o token (só uma vez):**
GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
→ Generate new token → marcar **`repo`** → copiar o token

```bash
# 1. Configurar identidade git (só na primeira vez)
git config --global user.name "Seu Nome"
git config --global user.email "seu@email.com"

# 2. Clonar com o PAT na URL (repositório privado)
git clone https://SEU_TOKEN@github.com/wendellr/neoveo-faceidcow.git
cd neoveo-faceidcow
chmod +x start_training.sh

# 3. Salvar o token para futuros git pull (opcional)
git remote set-url origin https://SEU_TOKEN@github.com/wendellr/neoveo-faceidcow.git

# 4. Rodar — instala dependências, baixa datasets e treina tudo
./start_training.sh
```

O script `start_training.sh` cuida automaticamente de:
- Criar `.venv` e instalar PyTorch cu124 (compatível com driver CUDA 12.8+)
- Baixar dataset de **detecção** (Roboflow por padrão, ou Ahmed 2024)
- Baixar **Li et al. 2022** (~600 MB) para re-ID
- Preparar `data/detection/` e `data/reid/`
- Treinar YOLO11n → EfficientNet-B0 + ArcFace

### Configuração obrigatória antes de rodar

Abra `start_training.sh` e edite o bloco de configuração no topo:

```bash
# Dataset de DETECÇÃO — escolha uma opção:
#   "roboflow"  → rápido (~200 MB), ideal para validar o pipeline
#   "ahmed"     → produção (~13 GB), qualidade máxima
DETECTION_SOURCE="roboflow"

# Se DETECTION_SOURCE="roboflow":
#   1. Acesse https://universe.roboflow.com
#   2. Busque: cattle muzzle  (ou cow nose, bovine muzzle)
#   3. Dataset → Download → YOLOv11 → curl → copie a URL
ROBOFLOW_ZIP_URL=""   # cole a URL aqui
```

> **Ahmed 2024:** se preferir o dataset completo, mude para `DETECTION_SOURCE="ahmed"` — o script baixa os 13 GB automaticamente.

Após o treino, baixe os pesos para o Mac via **iTerm2 (`it2dl`)**:
```bash
# No servidor — empacota apenas os pesos (pequeno, ~50 MB)
cd ~/neoveo-faceidcow
zip -r runs_weights.zip runs/detect/neoveo_muzzle_v1/weights/best.pt \
                        runs/reid/neoveo_reid_v1/best_reid.pth \
                        runs/reid/neoveo_reid_v1/embeddings_val.pkl

# Ainda no servidor — envia para o Mac pelo iTerm2
it2dl runs_weights.zip
```
O arquivo aparece em `~/Downloads/` no Mac. Depois:
```bash
# No Mac — extrai mantendo estrutura runs/
cd ~/Downloads && unzip runs_weights.zip -d /caminho/para/neoveo-faceidcow/
```

## Fluxo de trabalho Mac ↔ Servidor

```
Mac      →  edita código  →  git push
Servidor →  git pull      →  ./start_training.sh
Servidor →  zip pesos     →  it2dl arquivo.zip  →  Mac
```

Para atualizar somente o código sem re-treinar:
```bash
# No servidor
cd neoveo-faceidcow
git pull
```

Os datasets e pesos em `runs/` ficam **só no servidor** (estão no `.gitignore`)
— o `git pull` nunca sobrescreve o que foi treinado.

---

## Datasets

**Li et al. 2022 — Beef Cattle Muzzle/Noseprint Database**
- Download: https://zenodo.org/records/6324361
- 4.923 imagens · 268 bovinos · ~12 imgs/animal
- Usado para: **re-ID** (EfficientNet-B0 + ArcFace)
- Pasta esperada: `BeefCattle_Muzzle_Individualized/`

**Roboflow Universe — detecção (opção rápida, recomendada para validar)**
- Acesse: https://universe.roboflow.com
- Busque: `cattle muzzle`, `cow nose`, `bovine muzzle`
- Formato de download: **YOLOv11** (já vem com train/valid split e labels)
- Tamanho: ~200 MB · tempo de treino: ~3 min na RTX 4070
- Estrutura gerada: `DetectionDataset/train/` + `DetectionDataset/valid/`

**Ahmed et al. 2024 — opção produção**
- Download: https://zenodo.org/records/10535934/files/INDIVIDUAL%20SUBJECTS%20Data.zip?download=1
- ~13 GB · fotos frontais completas · 459 bovinos · anotações YOLO reais
- Usado para: **detecção** (YOLO11n) em produção
- Estrutura esperada: `DetectionDataset/images/` + `DetectionDataset/labels/`

## Estrutura do projeto

```
neoveo_muzzle_pipeline/
├── 01_prepare_dataset.py   ← organiza dados para detecção e re-ID
├── 02_train_detector.py    ← treina YOLO11n detector de muzzle
├── 03_train_reid.py        ← treina EfficientNet-B0 + ArcFace
├── 04_inference.py         ← inferência em imagem/vídeo/webcam
└── requirements.txt
```

## Instalação

```bash
pip install -r requirements.txt
```

## Execução passo a passo

### 1. Preparar dados
```bash
python 01_prepare_dataset.py
```
Gera `data/detection/` (YOLO format) e `data/reid/` (pastas por identidade).

### 2. Treinar detector de muzzle
```bash
python 02_train_detector.py
```
- Modelo base: `yolo11n.pt` (baixado automaticamente)
- Saída: `runs/detect/neoveo_muzzle_v1/weights/best.pt`
- Tempo estimado: ~10 min (GPU RTX 4000) / ~2h (CPU)

### 3. Treinar re-ID por embedding
```bash
python 03_train_reid.py
```
- Backbone: EfficientNet-B0 (ImageNet pretrained via timm)
- Loss: ArcFace (s=64, m=0.5)
- Embeddings: 512 dims, L2-normalized
- Saída: `runs/reid/neoveo_reid_v1/best_reid.pth`
- Tempo estimado: ~20 min (GPU) / ~4h (CPU)

### 4. Inferência
```bash
# Imagem única
python 04_inference.py --source minha_vaca.jpg

# Pasta de imagens (navegação com qualquer tecla, sai com 'q')
python 04_inference.py --source data/reid/test/Cattle_001/

# Vídeo
python 04_inference.py --source fazenda.mp4

# Webcam
python 04_inference.py --source 0
```

## Arquitetura do pipeline

```
Input (imagem/frame)
    │
    ▼
YOLO11n ──► detecta ROI do muzzle (bbox)
    │
    ▼
Crop do muzzle (resize 224×224)
    │
    ▼
EfficientNet-B0 backbone
    │
    ▼
Linear(1280→512) + BN + L2-norm
    │                           ┌─────────────────────┐
    ▼                           │  Galeria (val set)  │
Cosine Similarity ◄─────────────│  embeddings + IDs   │
    │                           └─────────────────────┘
    ▼
ID do animal + score (threshold 0.55)
    │
    ▼
Overlay na imagem/vídeo
```

## Configurações-chave

| Parâmetro | Valor | Onde alterar |
|---|---|---|
| Confiança detector | 0.40 | `04_inference.py → DET_CONF` |
| Threshold re-ID | 0.55 | `04_inference.py → REID_THRESHOLD` |
| Epochs detector | 50 | `02_train_detector.py → EPOCHS` |
| Epochs re-ID | 80 | `03_train_reid.py → EPOCHS` |
| Dimensão embedding | 512 | `03_train_reid.py → EMBED_DIM` |
| Batch size | 32 | `03_train_reid.py → BATCH` |

## Próximos passos para produção

1. **Anotar imagens de campo** com muzzle em foto inteira (não cropped) → re-treinar detector
2. **Coletar imagens de Nelore/Zebu** para fine-tuning do embedder
3. **Exportar para ONNX** para deploy no backend Modal:
   ```python
   model = YOLO("best.pt")
   model.export(format="onnx")
   ```
4. **Quantizar** com TensorRT ou OpenVINO para edge (Jetson Orin Nano)

## Referência

Li, G., Erickson, G.E., Xiong, Y. (2022). Individual Beef Cattle Identification Using
Muzzle Images and Deep Learning Techniques. *Animals*, 12(11), 1453.
https://doi.org/10.3390/ani12111453
