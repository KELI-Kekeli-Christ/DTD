# Few-Shot CV Research DTD 

> Projet de recherche en classification d'images par few-shot learning.  

---

## Vue d'ensemble

Ce projet explore deux directions complémentaires :

1. **DTD (Describable Textures Dataset)** — analyse des limites du ViT-B pour la reconnaissance de textures, baseline à 76% (Split 1), SOTA à 84.4%

---


## Datasets

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
> 
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

- **DTD** : Cimpoi et al., *Describing Textures in the Wild*, CVPR 2014 — [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

### SOTA à battre

- **Chaotic Contrastive** (DTD SOTA, 84.4%) — Mai 2025

### Méthodes comparées

- **DINO** : Caron et al., *Emerging Properties in Self-Supervised ViTs*, ICCV 2021
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
