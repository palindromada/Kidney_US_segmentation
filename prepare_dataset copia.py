"""
Script per preparare il dataset di segmentazione del rene da ultrasuoni.

CLASSI:
- 0: Background - NERO
- 1: Capsule - VERDE  
- 2: Central Echo Complex - BLU
- 3: Medulla - GIALLO
- 4: Cortex - ROSSO
"""

import os
import json
import csv
import shutil
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Percorsi principali
BASE_DIR = Path("/Users/ada/Desktop/SegRENE")
IMAGES_DIR = BASE_DIR / "kidneyUS_images_25_june_2025"
LABELS_DIR = BASE_DIR / "kidneyUS-main" / "labels"
MASKS_DIR_1 = LABELS_DIR / "reviewed_masks_1"
CSV_FILE_1 = LABELS_DIR / "reviewed_labels_1.csv"

# Directory di output
OUTPUT_DIR = BASE_DIR / "dataset_segmentation"

# Colori RGB per visualizzazione (ispirati al logo KidneyUS)
COLORS_RGB = {
    0: (0, 0, 0),          # Background - NERO
    1: (180, 80, 80),      # Capsule - ROSSO SCURO/MARRONE (contorno esterno)
    2: (80, 80, 180),      # Central Echo Complex - BLU (zona centrale)
    3: (80, 180, 80),      # Medulla - VERDE (zona intermedia)
    4: (180, 180, 80),     # Cortex - GIALLO/VERDE CHIARO (zona esterna interna)
}


def maybe_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def create_masks(capsule_path, regions_path, filename):
    """
    Combina le maschere capsule e regions in una singola maschera multiclasse.
    
    IMPORTANTE: L'ordine di applicazione e':
    1. Prima la capsule (contorno esterno del rene)
    2. Poi le regioni interne (CEC, Medulla, Cortex) che sovrascrivono
    
    Maschere regions originali: CEC=1, Medulla=2, Cortex=3
    Output finale: Background=0, Capsule=1, CEC=2, Medulla=3, Cortex=4
    """
    capsule_file = capsule_path / filename
    regions_file = regions_path / filename
    
    mask_class = None
    
    # Determina dimensioni
    if regions_file.exists():
        regions_mask = cv2.imread(str(regions_file), cv2.IMREAD_GRAYSCALE)
        if regions_mask is not None:
            height, width = regions_mask.shape
            mask_class = np.zeros((height, width), dtype=np.uint8)
    elif capsule_file.exists():
        capsule_mask = cv2.imread(str(capsule_file), cv2.IMREAD_GRAYSCALE)
        if capsule_mask is not None:
            height, width = capsule_mask.shape
            mask_class = np.zeros((height, width), dtype=np.uint8)
    
    if mask_class is None:
        return None, None
    
    # PASSO 1: Applica PRIMA la capsule (area esterna del rene)
    if capsule_file.exists():
        capsule_mask = cv2.imread(str(capsule_file), cv2.IMREAD_GRAYSCALE)
        if capsule_mask is not None:
            mask_class[capsule_mask == 1] = 1  # Capsule
    
    # PASSO 2: Applica le regioni INTERNE (sovrascrivono la capsule dove necessario)
    if regions_file.exists():
        regions_mask = cv2.imread(str(regions_file), cv2.IMREAD_GRAYSCALE)
        if regions_mask is not None:
            # CEC: era 1 -> diventa 2
            mask_class[regions_mask == 1] = 2
            # Medulla: era 2 -> diventa 3
            mask_class[regions_mask == 2] = 3
            # Cortex: era 3 -> diventa 4
            mask_class[regions_mask == 3] = 4
    
    # Crea maschera RGB colorata
    height, width = mask_class.shape
    mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in COLORS_RGB.items():
        mask_rgb[mask_class == class_id] = color
    
    return mask_rgb, mask_class


def visualize_sample(image_path, mask_rgb, mask_class, save_path=None):
    """Visualizza immagine, maschera RGB colorata e overlay"""
    img = cv2.imread(str(image_path))
    if img is None:
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crea overlay
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    alpha = 0.5
    overlay = cv2.addWeighted(img, 1-alpha, mask_bgr, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title("Immagine Originale")
    axes[0].axis('off')
    
    axes[1].imshow(mask_rgb)
    axes[1].set_title("Maschera RGB Colorata")
    axes[1].axis('off')
    
    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    # Legenda con colori del logo
    legend_labels = ['Background', 'Capsule', 'CEC', 'Medulla', 'Cortex']
    legend_colors = [(0,0,0), (180/255,80/255,80/255), (80/255,80/255,180/255), (80/255,180/255,80/255), (180/255,180/255,80/255)]
    patches = [plt.Rectangle((0,0), 1, 1, fc=c) for c in legend_colors]
    fig.legend(patches, legend_labels, loc='lower center', ncol=5)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def prepare_dataset(test_size=0.15, val_size=0.15, random_state=42):
    print("=" * 60)
    print("PREPARAZIONE DATASET SEGMENTAZIONE RENE")
    print("=" * 60)
    
    # Rimuovi vecchio dataset
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    # Crea directory
    for split in ['train', 'val', 'test']:
        maybe_mkdir(OUTPUT_DIR / split / "images")
        maybe_mkdir(OUTPUT_DIR / split / "masks_rgb")
        maybe_mkdir(OUTPUT_DIR / split / "masks_class")
    
    # Trova immagini con maschere
    capsule_dir = MASKS_DIR_1 / "capsule"
    regions_dir = MASKS_DIR_1 / "regions"
    
    valid_images = []
    for f in IMAGES_DIR.glob("*.png"):
        if f.name.startswith('.'):
            continue
        if (capsule_dir / f.name).exists() or (regions_dir / f.name).exists():
            valid_images.append(f.name)
    
    valid_images = sorted(valid_images)
    print(f"\nImmagini con maschere: {len(valid_images)}")
    
    # Split
    train_val, test_files = train_test_split(valid_images, test_size=test_size, random_state=random_state)
    train_files, val_files = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=random_state)
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_files)}")
    print(f"  Val:   {len(val_files)}")
    print(f"  Test:  {len(test_files)}")
    
    class_stats = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    splits = [('train', train_files), ('val', val_files), ('test', test_files)]
    
    for split_name, file_list in splits:
        print(f"\nProcesso {split_name}...")
        
        for filename in tqdm(file_list, desc=f"  {split_name}"):
            # Crea maschere
            mask_rgb, mask_class = create_masks(capsule_dir, regions_dir, filename)
            
            if mask_class is None:
                continue
            
            # Copia immagine
            src_img = IMAGES_DIR / filename
            dst_img = OUTPUT_DIR / split_name / "images" / filename
            shutil.copy2(src_img, dst_img)
            
            # Salva maschere
            dst_rgb = OUTPUT_DIR / split_name / "masks_rgb" / filename
            dst_class = OUTPUT_DIR / split_name / "masks_class" / filename
            
            cv2.imwrite(str(dst_rgb), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(dst_class), mask_class)
            
            # Statistiche
            for label in range(5):
                class_stats[label] += int(np.sum(mask_class == label))
    
    # Statistiche
    total = sum(class_stats.values())
    print("\n" + "=" * 60)
    print("STATISTICHE CLASSI")
    print("=" * 60)
    class_names = ['Background', 'Capsule', 'CEC', 'Medulla', 'Cortex']
    for label, count in class_stats.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {label}: {class_names[label]:15s}: {count:>12,} pixels ({pct:>5.2f}%)")
    
    # Visualizzazioni
    print("\nSalvataggio esempi...")
    viz_dir = OUTPUT_DIR / "visualizations"
    maybe_mkdir(viz_dir)
    
    for i, filename in enumerate(train_files[:10]):
        img_path = OUTPUT_DIR / "train" / "images" / filename
        rgb_path = OUTPUT_DIR / "train" / "masks_rgb" / filename
        class_path = OUTPUT_DIR / "train" / "masks_class" / filename
        
        mask_rgb = cv2.imread(str(rgb_path))
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask_class = cv2.imread(str(class_path), cv2.IMREAD_GRAYSCALE)
        
        if mask_rgb is not None:
            visualize_sample(img_path, mask_rgb, mask_class, viz_dir / f"example_{i+1}.png")
    
    # Info JSON
    info = {
        'classes': {
            '0': {'name': 'Background', 'color_rgb': [0, 0, 0]},
            '1': {'name': 'Capsule', 'color_rgb': [0, 255, 0]},
            '2': {'name': 'Central Echo Complex', 'color_rgb': [0, 0, 255]},
            '3': {'name': 'Medulla', 'color_rgb': [255, 255, 0]},
            '4': {'name': 'Cortex', 'color_rgb': [255, 0, 0]},
        },
        'splits': {
            'train': len(train_files),
            'val': len(val_files), 
            'test': len(test_files)
        },
        'class_pixels': {class_names[k]: int(v) for k, v in class_stats.items()}
    }
    
    with open(OUTPUT_DIR / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n{'='*60}")
    print("DATASET SALVATO!")
    print(f"{'='*60}")
    print(f"Percorso: {OUTPUT_DIR}")
    print("\nLEGENDA COLORI (stile KidneyUS logo):")
    print("  NERO           -> Background (0)")
    print("  ROSSO SCURO    -> Capsule (1)")
    print("  BLU            -> Central Echo Complex (2)")
    print("  VERDE          -> Medulla (3)")
    print("  GIALLO/VERDE   -> Cortex (4)")


if __name__ == "__main__":
    prepare_dataset()
    print("\nCOMPLETATO!")
