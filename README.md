# Few-Shot CV Research — DTD & BSCD-FSL Benchmark

> Projet de recherche en classification d'images par few-shot learning.  
> Objectif : battre le SOTA sur EuroSAT (BSCD-FSL) et analyser les limites du ViT-B sur DTD.

---

## Vue d'ensemble

Ce projet explore deux directions complémentaires :

1. **DTD (Describable Textures Dataset)** — analyse des limites du ViT-B pour la reconnaissance de textures, baseline à 76% (Split 1), SOTA à 84.4%
2. **BSCD-FSL / EuroSAT** — cross-domain few-shot learning, entraîné sur miniImageNet uniquement, testé sur images satellites, SOTA à ~93.52% (CPLSR, 2025)

---


## Datasets

### miniImageNet (méta-entraînement)
- **Taille** : ~200 Mo
- **Classes** : 100 (64 train / 16 val / 20 test)
- **Images** : 600 par classe, 84×84 px
- **Source** : [Kaggle](https://www.kaggle.com/datasets/arjunashok33/miniimagenet)
- **Split** : Ravi & Larochelle (2017)

```bash
python scripts/download_miniimagenet.py --data_dir ./data
```

### EuroSAT RGB (test BSCD-FSL)
- **Taille** : ~174 Mo
- **Classes** : 10 (AnnualCrop, Forest, Highway, Industrial, ...)
- **Images** : 27 000 images satellite Sentinel-2, 64×64 px
- **Source** : [HuggingFace](https://huggingface.co/datasets/blanchon/EuroSAT_RGB)
- **Protocole** : 5-way 5-shot, 600 épisodes, IC 95%

```bash
python scripts/install_datasets.py --only eurosat --data_dir ./data
```

### DTD — Describable Textures Dataset
- **Taille** : ~600 Mo
- **Classes** : 47 textures (crystalline, woven, braided, ...)
- **Images** : 5 640 images (120 par classe)
- **Source** : [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- **Splits** : 10 folds officiels

---

## Scores de référence

### DTD — Classification supervisée

| Modèle | Accuracy (Split 1) | SOTA (2025) |
|---|---|---|
| ResNet-50 | ~68% | — |
| **ViT-B/16** | **75.98%** | — |
| ViT-L/16 | ~80% | — |
| Chaotic Contrastive | — | **84.4%** |

> **Résultat obtenu** : ViT-B → Val 74.29%, **Test 75.98%** (Split 1)

### BSCD-FSL — EuroSAT 5-way 5-shot

| Méthode | Backbone | 1-shot | 5-shot |
|---|---|---|---|
| Baseline fine-tuning | ResNet-10 | ~49% | ~64% |
| DINO + ProtoNet | ViT-S | ~79% | ~88% |
| PMF | ViT-S | ~80% | ~89% |
| StyleAdv | ViT-S | ~81% | ~90% |
| IM-DCL | ViT | 77.14% | 89.47% |
| **CPLSR** | **ViT-S** | — | **~93.52%** |

> **Protocole** : méta-entraînement sur miniImageNet uniquement (pas de données médicales ni satellite)

---

## Analyse des résultats DTD

### Classes difficiles (< 65%)

Le ViT-B échoue sur les textures **amorphes et à forte variance intra-classe** :

| Classe | Score | Problème |
|---|---|---|
| woven | 42% | Confondu avec braided / fibrous |
| blotchy | 45% | Distribution spatiale aléatoire |
| bumpy / marbled | 50% | Forte variabilité intra-classe |
| pitted / stained | 52% | Pas de pattern régulier |
| grid | 57% | Confondu avec crosshatched |

### Classes faciles (≥ 88%)

Textures **géométriques régulières** bien captées par l'attention globale du ViT :
crystalline (98%), cobwebbed (98%), zigzagged (95%), frilly (95%), chequered (95%)

### Hypothèse

> L'attention globale du ViT-B est insuffisante pour les textures amorphes. Une approche multi-échelle (features locales + fréquentielles) permettrait de traiter les classes <65%.

---

## Références clés

### Benchmarks

- **BSCD-FSL** : Guo et al., *A Broader Study of Cross-Domain Few-Shot Learning*, ECCV 2020 — [arXiv:1912.07200](https://arxiv.org/abs/1912.07200) · [Code IBM](https://github.com/IBM/cdfsl-benchmark)
- **DTD** : Cimpoi et al., *Describing Textures in the Wild*, CVPR 2014 — [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- **EuroSAT** : Helber et al., *EuroSAT: A Novel Dataset*, IEEE JSTARS 2019 — [arXiv:1709.00029](https://arxiv.org/abs/1709.00029)

### SOTA à battre

- **CPLSR** (EuroSAT SOTA, ~93.52% 5-shot) — Paeedeh et al., 2025 — [arXiv:2507.15243](https://arxiv.org/abs/2507.15243) · [Code](https://github.com/Naeem-Paeedeh/CPLSR)
- **Chaotic Contrastive** (DTD SOTA, 84.4%) — Mai 2025

### Méthodes comparées

- **DINO** : Caron et al., *Emerging Properties in Self-Supervised ViTs*, ICCV 2021
- **PMF** : Hu et al., *Pushing the Limits of Simple Pipelines for Few-Shot Learning*, CVPR 2022
- **StyleAdv** : Fu et al., *StyleAdv: Meta Style Adversarial Training for Cross-Domain FSL*, CVPR 2023
- **IM-DCL** : *Enhancing Information Maximization with Distance-Aware Contrastive Learning*, 2024 — [arXiv:2403.01966](https://arxiv.org/abs/2403.01966)

---

## Installation

```bash
# Cloner le repo
git clone <votre-repo>
cd projet

# Installer les dépendances
pip install torch torchvision datasets pillow tqdm kaggle

# Télécharger les datasets
python scripts/download_miniimagenet.py --data_dir ./data
python scripts/install_datasets.py --only eurosat --data_dir ./data

# Vérifier l'installation
python scripts/install_datasets.py --verify --data_dir ./data
```

---

## Lancer les expériences

```bash
# Benchmark DTD (ViT-B + ResNet-50)
python scripts/dtd_benchmark.py --data_dir ./data/dtd

# Baseline BSCD-FSL sur EuroSAT (à venir)
python scripts/baseline_bscd.py \
    --train_dir ./data/miniImageNet/train \
    --test_dir ./data/EuroSAT/images \
    --n_way 5 --n_shot 5 --n_episodes 600
```

---

## Résultats actuels

| Expérience | Statut | Score |
|---|---|---|
| DTD ViT-B baseline | ✅ Terminé | **75.98%** test |


---

## Pistes de contribution

1. **Fréquentielle** — adapter les features dans l'espace fréquentiel (FFT) avant le few-shot test, inspiré de FAP/Wave-SAN
2. **Multi-échelle locale** — extraire des features patch-level pour les textures amorphes (DTD woven/blotchy)
3. **Test-time adaptation légère** — adapter uniquement les Coalescent Projections du ViT sans données cibles en entraînement

---

*Dernière mise à jour : Mai 2026*
