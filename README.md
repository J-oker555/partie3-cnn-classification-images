# Partie 3 - CNN, classification d'images et transfer learning

Projet issu du sujet `partie3_cnn_classification_images.pdf`.

Objectif final : construire un notebook complet de classification binaire d'images avec trois iterations comparables :

1. CNN from scratch avec overfitting visible.
2. CNN avec data augmentation et Dropout.
3. MobileNetV2 en transfer learning puis fine-tuning.

Le livrable attendu contient le notebook, les courbes matplotlib, le meilleur modele `.keras`, un export `.tflite`, un tableau de comparaison et des commits atomiques par phase.

## Installation locale

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copier `.env.example` vers `.env` si vous voulez lancer les scripts localement.

## Structure

```text
notebooks/
  partie3_cnn_classification_images.ipynb
src/cnn_image_classification/
  config.py
  data.py
  models.py
  training.py
  plots.py
  export.py
docs/
  issues.md
```

Les dossiers `data/`, `logs/`, `models/`, `outputs/` et `exports/` sont volontairement ignores par Git.

## Dataset

Le sujet demande un dataset binaire d'images disponible sur Kaggle ou HuggingFace, telechargeable rapidement, avec au moins 1 000 images par classe.

Structure finale attendue :

```text
data/
  train/
    classe_a/
    classe_b/
  val/
    classe_a/
    classe_b/
```

Par defaut, les exemples utilisent `cat` vs `dog`, mais les constantes `CLASS_A` et `CLASS_B` permettent de changer de sujet.

Preparation locale a partir de deux dossiers bruts :

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\prepare_binary_dataset.py `
  --raw-class-a raw\cat `
  --raw-class-b raw\dog `
  --class-a cat `
  --class-b dog `
  --output-root data

.\.venv\Scripts\python.exe scripts\inspect_binary_dataset.py `
  --data-root data `
  --class-a cat `
  --class-b dog `
  --output outputs\sample_grid.png

.\.venv\Scripts\python.exe scripts\check_dataset_pipeline.py `
  --data-root data `
  --img-size 128 `
  --batch-size 32

.\.venv\Scripts\python.exe scripts\check_cnn_scratch_architecture.py `
  --img-size 128

.\.venv\Scripts\python.exe scripts\train_cnn_scratch.py `
  --data-root data `
  --img-size 128 `
  --batch-size 32 `
  --epochs 20

.\.venv\Scripts\python.exe scripts\preview_data_augmentation.py `
  --data-root data `
  --img-size 128 `
  --output outputs\augmentation_grid.png

.\.venv\Scripts\python.exe scripts\train_cnn_augmented.py `
  --data-root data `
  --img-size 128 `
  --batch-size 32 `
  --epochs 20 `
  --dropout 0.4
```

## Phases de developpement

Les phases a traiter sont documentees dans [docs/issues.md](docs/issues.md). Chaque phase doit produire un commit clair, sans `git add .`, avec staging explicite.

## Artefacts attendus

- `notebooks/partie3_cnn_classification_images.ipynb`
- `outputs/curves_cnn_scratch.png`
- `outputs/curves_cnn_augmente_dropout.png`
- `outputs/comparison_tp1_tp2.png`
- `outputs/comparison_all.png`
- `models/model_scratch.keras`
- `models/model_augmented.keras`
- `models/model_tl.keras`
- `exports/<class_a>_vs_<class_b>_mobilenet.tflite`
