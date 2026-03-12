# SegRENE - Segmentazione Rene da Ultrasuoni

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.6+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Non--Commercial-green.svg)](#licenza)

Pipeline completa per la **segmentazione multiclasse** di immagini ecografiche renali utilizzando reti neurali deep learning (Res-UNet con Attention Gates).

---

## 📋 Indice

- [Panoramica](#panoramica)
- [Struttura del Progetto](#struttura-del-progetto)
- [Classi di Segmentazione](#classi-di-segmentazione)
- [Dataset](#dataset)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Architettura della Rete](#architettura-della-rete)
- [Script e Strumenti](#script-e-strumenti)
- [Output e Risultati](#output-e-risultati)
- [Riferimenti](#riferimenti)

---

## 🎯 Panoramica

Questo progetto implementa una pipeline end-to-end per la segmentazione semantica multiclasse di immagini ecografiche renali. Il sistema è basato sul dataset **Open Kidney Ultrasound** e utilizza un'architettura **Res-UNet con Attention Gates** per identificare e segmentare le diverse strutture anatomiche del rene.

### Caratteristiche Principali

- 🔬 **Segmentazione multiclasse** (5 classi anatomiche)
- 🧠 **Res-UNet con Attention Gates** per focus sulle strutture rilevanti
- 📊 **Deep Supervision** durante il training per gradienti migliori
- 🎨 **Data Augmentation avanzata** (elastic transform, grid distortion, cutout)
- 📈 **Loss functions combinate** (Focal Loss + Dice Loss + Lovasz-Softmax)
- 📉 **Metriche complete** (DSC, Hausdorff Distance, Precision, Recall)

---

## 📁 Struttura del Progetto

```
SegRENE/
│
├── README.md                              # Questo file
├── prepare_dataset copia.py               # Script per preparare il dataset
├── analyze_logo.py                        # Analisi colori del logo
├── Rete_segmentation_Kidney copia.ipynb   # Notebook principale training/inference
│
├── kidneyUS-main copia/                   # Dataset e strumenti originali
│   ├── README.md                          # Documentazione dataset originale
│   ├── requirements.txt                   # Dipendenze Python
│   ├── labels/                            # Annotazioni e maschere
│   │   ├── reviewed_labels_1.csv          # Annotazioni sonografo 1
│   │   ├── reviewed_labels_2.csv          # Annotazioni sonografo 2
│   │   ├── reviewed_masks_1/              # Maschere sonografo 1
│   │   │   ├── capsule/                   # Maschere capsula renale
│   │   │   └── regions/                   # Maschere regioni interne
│   │   └── reviewed_masks_2/              # Maschere sonografo 2
│   │       ├── capsule/
│   │       └── regions/
│   └── src/                               # Script di utilità
│       ├── clean_segmentation.py          # Pulizia maschere
│       ├── compare_mutual_information.py  # Confronto mutual information
│       ├── segmentation_evaluation.py     # Valutazione segmentazione
│       ├── annotation_analysis/           # Analisi annotazioni
│       │   ├── interrater_dsc.py          # DSC inter-annotatore
│       │   ├── intrarater_dsc.py          # DSC intra-annotatore
│       │   ├── interrater_hd.py           # Hausdorff inter-annotatore
│       │   ├── intrarater_hd.py           # Hausdorff intra-annotatore
│       │   ├── cohens_kappa.py            # Cohen's Kappa
│       │   ├── overlay_masks.py           # Sovrapposizione maschere
│       │   ├── create_masks.py            # Creazione maschere da CSV
│       │   └── ...                        # Altri script di analisi
│       ├── echotools/                     # Strumenti per DICOM
│       │   ├── dicom.py                   # Elaborazione DICOM
│       │   └── clean_dcm_images.py        # Pulizia immagini DICOM
│       └── tools/                         # Utility generiche
│           ├── basic_utils.py             # Funzioni base
│           ├── metric_utils.py            # Metriche (DSC, HD, etc.)
│           ├── image_utils.py             # Manipolazione immagini
│           ├── distribution_utils.py      # Analisi distribuzioni
│           ├── size_tools.py              # Strumenti dimensioni
│           └── variability_shared.py      # Funzioni condivise
│
├── dataset_segmentation/                  # Dataset preparato (generato)
│   ├── train/                             # Set di training
│   │   ├── images/                        # Immagini US
│   │   ├── masks_rgb/                     # Maschere colorate
│   │   └── masks_class/                   # Maschere con indici classe
│   ├── val/                               # Set di validazione
│   └── test/                              # Set di test
│
└── outputs copia/                         # Risultati del training
    ├── training_curves.png                # Curve di apprendimento
    ├── confusion_matrix.png               # Matrice di confusione
    ├── test_predictions_multiclass.png    # Predizioni sul test set
    ├── postprocessing_comparison.png      # Confronto post-processing
    └── final_results_optimized.png        # Risultati finali ottimizzati
```

---

## 🎨 Classi di Segmentazione

Il modello segmenta **5 classi** anatomiche:

| Classe | Nome | Colore RGB | Descrizione |
|--------|------|------------|-------------|
| 0 | **Background** | `(0, 0, 0)` Nero | Sfondo dell'immagine |
| 1 | **Capsule** | `(180, 80, 80)` Rosso scuro | Contorno esterno del rene |
| 2 | **Central Echo Complex (CEC)** | `(80, 80, 180)` Blu | Zona centrale (pelvi renale) |
| 3 | **Medulla** | `(80, 180, 80)` Verde | Zona intermedia (piramidi renali) |
| 4 | **Cortex** | `(180, 180, 80)` Giallo | Zona esterna interna |

![Anatomia Rene](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Kidney_PioM.png/220px-Kidney_PioM.png)

---

## 📊 Dataset

### Open Kidney Ultrasound Dataset

Il dataset originale contiene:
- **500+ immagini** ecografiche B-mode addominali
- **Annotazioni fine-grained** di 2 esperti sonografi
- **Varietà di vendor**: Philips, GE, Acuson, Siemens, Toshiba, SonoSite
- **Contesti reali**: ICU, bedside, pazienti con malattia renale cronica

### Formato Maschere Originali

Le maschere sono organizzate in:
- `capsule/`: Maschera binaria della capsula renale (contorno)
- `regions/`: Maschera multiclasse delle regioni interne
  - 1 = Central Echo Complex
  - 2 = Medulla  
  - 3 = Cortex

### Dataset Preparato

Dopo l'esecuzione di `prepare_dataset copia.py`:
- Maschere combinate in formato multiclasse (0-4)
- Split automatico: **Train (70%)**, **Val (15%)**, **Test (15%)**
- Visualizzazioni di esempio generate

---

## 🛠 Installazione

### Prerequisiti

- Python 3.8+
- macOS / Linux / Windows
- GPU consigliata (supporto MPS per Apple Silicon)

### Setup Ambiente

```bash
# Clona il repository
cd /Users/ada/Desktop/SegRENE

# Crea virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Installa dipendenze
pip install -r "kidneyUS-main copia/requirements.txt"

# Dipendenze aggiuntive per il training
pip install torch torchvision scikit-learn tqdm matplotlib
```

### Dipendenze Principali

| Pacchetto | Versione | Utilizzo |
|-----------|----------|----------|
| PyTorch | ≥1.6.0 | Deep Learning framework |
| OpenCV | ≥4.5.4 | Elaborazione immagini |
| NumPy | ≥1.19.2 | Operazioni numeriche |
| scikit-learn | ≥1.1.2 | Split dataset, metriche |
| MedPy | ≥0.4.0 | Metriche mediche (HD) |
| Pillow | ≥9.2.0 | I/O immagini |
| matplotlib | ≥3.3.1 | Visualizzazioni |

---

## 🚀 Utilizzo

### 1. Preparazione Dataset

```bash
python "prepare_dataset copia.py"
```

Questo script:
- ✅ Combina maschere capsule + regions
- ✅ Crea split train/val/test
- ✅ Genera maschere RGB colorate
- ✅ Salva statistiche delle classi
- ✅ Crea visualizzazioni di esempio

### 2. Training del Modello

Apri il notebook Jupyter:

```bash
jupyter notebook "Rete_segmentation_Kidney copia.ipynb"
```

Oppure esegui le celle in VS Code.

#### Iperparametri Principali

```python
IMG_SIZE = 256          # Dimensione immagini
BATCH_SIZE = 8          # Batch size
LR = 1e-3               # Learning rate
N_EPOCHS = 50           # Epoche di training
PATIENCE = 15           # Early stopping patience
NUM_CLASSES = 5         # Numero classi

# Pesi per classi sbilanciate
CLASS_WEIGHTS = [0.1, 2.5, 4.0, 7.0, 12.0]
```

### 3. Valutazione

Il notebook include:
- Calcolo metriche per classe (DSC, IoU, Precision, Recall)
- Matrice di confusione
- Visualizzazione predizioni
- Confronto con ground truth

---

## 🧠 Architettura della Rete

### Res-UNet con Attention Gates

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT (1, 256, 256)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  ENCODER                                                     │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │ResBlock │──▶│ResBlock │──▶│ResBlock │──▶│ResBlock │──▶   │
│  │  64     │   │  128    │   │  256    │   │  512    │      │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘     │
└───────│─────────────│─────────────│─────────────│───────────┘
        │             │             │             │
        │    Skip     │    Skip     │    Skip     │    Skip
        │  Connection │  Connection │  Connection │  Connection
        │             │             │             │
┌───────│─────────────│─────────────│─────────────│───────────┐
│       ▼             ▼             ▼             ▼           │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │Attention│   │Attention│   │Attention│   │Attention│     │
│  │  Gate   │   │  Gate   │   │  Gate   │   │  Gate   │     │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘     │
│       │             │             │             │           │
│  DECODER                                                     │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │ResBlock │◀──│ResBlock │◀──│ResBlock │◀──│ResBlock │◀──   │
│  │  64     │   │  128    │   │  256    │   │  512    │      │
│  └────┬────┘   └────┴────┘   └────┴────┘   └────┴────┘     │
└───────│─────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────┐
│                   OUTPUT (5, 256, 256)                       │
│              [Background, Capsule, CEC, Medulla, Cortex]     │
└─────────────────────────────────────────────────────────────┘
```

### Componenti Chiave

1. **ResBlock**: Blocchi residuali con skip connection per miglior propagazione del gradiente
2. **Attention Gates**: Filtrano le features irrilevanti nelle skip connections
3. **Deep Supervision**: Output ausiliari a diverse risoluzioni durante il training

### Loss Function Combinata

```python
Total_Loss = 0.4 × Focal_Loss + 0.3 × Dice_Loss + 0.3 × Lovasz_Softmax
```

- **Focal Loss**: Gestisce lo sbilanciamento delle classi
- **Dice Loss**: Ottimizza l'overlap tra predizione e GT
- **Lovasz-Softmax**: Ottimizza direttamente l'IoU

---

## 📜 Script e Strumenti

### Script Principali

| File | Descrizione |
|------|-------------|
| `prepare_dataset copia.py` | Prepara il dataset per il training |
| `Rete_segmentation_Kidney copia.ipynb` | Pipeline completa di training e valutazione |
| `analyze_logo.py` | Analizza i colori del logo KidneyUS |

### Strumenti di Analisi (`kidneyUS-main copia/src/`)

#### Valutazione Segmentazione
| Script | Funzione |
|--------|----------|
| `segmentation_evaluation.py` | Valutazione completa (DSC, HD, overlay) |
| `clean_segmentation.py` | Pulizia e raffinamento maschere |

#### Analisi Annotazioni (`annotation_analysis/`)
| Script | Funzione |
|--------|----------|
| `interrater_dsc.py` | DSC tra annotatori diversi |
| `intrarater_dsc.py` | DSC stesso annotatore (ripetizioni) |
| `interrater_hd.py` | Hausdorff Distance inter-annotatore |
| `cohens_kappa.py` | Cohen's Kappa per accordo |
| `overlay_masks.py` | Sovrapposizione maschere su immagini |
| `create_masks.py` | Genera maschere da annotazioni CSV |

#### Utility (`tools/`)
| Script | Funzione |
|--------|----------|
| `metric_utils.py` | DSC, Hausdorff, Precision, Recall |
| `image_utils.py` | Manipolazione e overlay immagini |
| `basic_utils.py` | Funzioni di utilità base |
| `variability_shared.py` | Funzioni condivise per variabilità |

---

## 📈 Output e Risultati

### File Generati

```
outputs copia/
├── training_curves.png          # Loss e metriche durante training
├── confusion_matrix.png         # Matrice di confusione per classe
├── test_predictions_multiclass.png  # Esempi di predizioni
├── postprocessing_comparison.png    # Effetto post-processing
└── final_results_optimized.png      # Risultati finali
```

### Metriche Valutate

- **Dice Similarity Coefficient (DSC)**: Overlap tra predizione e GT
- **Intersection over Union (IoU/Jaccard)**: Rapporto intersezione/unione
- **Hausdorff Distance (HD)**: Distanza massima tra contorni
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)

---

## 📚 Riferimenti

### Paper e Risorse

1. **Open Kidney Dataset**: [GitHub - rsingla92/kidneyUS](https://github.com/rsingla92/kidneyUS)
2. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
3. **Attention U-Net**: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas" (2018)
4. **Lovasz-Softmax**: Berman et al., "The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-over-Union Measure" (CVPR 2018)
5. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)

### Citazione Dataset

```bibtex
@misc{kidneyUS,
  author = {Singla, Ramandeep},
  title = {The Open Kidney Ultrasound Data Set},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rsingla92/kidneyUS}}
}
```

---

## ⚖️ Pesi del Modello

I **pesi pre-addestrati** del modello non sono inclusi nel repository in quanto troppo pesanti per essere caricati su GitHub.

Per ottenere i pesi del modello, contattami direttamente.

---

## 📄 Licenza

Questo progetto utilizza il dataset Open Kidney che è disponibile **solo per uso non commerciale**. 

L'approvazione istituzionale è stata ricevuta (H21-02375).

---

## 👥 Contatti

Per domande o contributi, apri una issue nel repository.

---

*Ultimo aggiornamento: Marzo 2026*
