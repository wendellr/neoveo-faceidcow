# Neoveo.ai — Cattle Muzzle Re-ID Pipeline

Pipeline de identificação biométrica individual de bovinos por padrão de focinho (muzzle print), usando YOLO11 para detecção de ROI e EfficientNet-B0 + ArcFace para re-ID por embedding.

## Dataset

**Li et al. 2022 — Beef Cattle Muzzle/Noseprint Database**
- Download: https://zenodo.org/records/6324361
- 4.923 imagens · 268 bovinos · ~12 imgs/animal
- Descompacte como `BeefCattle_Muzzle_database/` na raiz do projeto

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
