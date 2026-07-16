# Issues GitHub a creer

Ces issues suivent les phases du PDF. Chaque issue doit etre traitee sur une branche dediee, avec un commit atomique et un message clair.

## Issue 1 - Setup initial du repo et environnement

Objectif : initialiser le repo, documenter l'installation, creer l'environnement Python et poser le notebook de travail.

Critere d'acceptation :
- README present et coherent avec le sujet.
- `.gitignore` exclut `.venv`, `data/`, `logs/`, `models/`, `outputs/`, `exports/` et secrets Kaggle.
- `requirements.txt` liste TensorFlow, Kaggle, matplotlib, pandas, numpy.
- Notebook cree dans `notebooks/`.
- Premier commit : `setup initial`.

## Issue 2 - Phase 1.1 : setup Colab et organisation dataset

Objectif : configurer Kaggle/Colab, telecharger le dataset, organiser `train/` et `val/` par classe, afficher des exemples.

Critere d'acceptation :
- Variables `CLASS_A`, `CLASS_B`, `DATA_ROOT` definies.
- Split 80/20 reproductible avec seed 42.
- Au moins 500 images train par classe.
- Grille 2x3 sauvegardee ou affichee.
- Commit : `phase 1.1 setup dataset`.

## Issue 3 - Phase 1.2 : preprocessing, normalisation et batching

Objectif : charger les images avec `image_dataset_from_directory`, normaliser et optimiser le pipeline.

Critere d'acceptation :
- `IMG_SIZE` et `BATCH_SIZE` definis.
- `train_ds` en shuffle avec seed 42, `val_ds` sans shuffle.
- `Rescaling(1./255)`, `cache()` et `prefetch(AUTOTUNE)` appliques.
- Shapes et min/max du premier batch verifies.
- Commit : `phase 1.2 dataset pipeline`.

## Issue 4 - Phase 1.3 : architecture CNN from scratch

Objectif : construire le CNN simple 3 blocs Conv/Pool, Flatten, Dense, sortie sigmoid.

Critere d'acceptation :
- `build_cnn_scratch(input_shape)` implemente.
- `model.summary()` montre 3 blocs Conv2D + Flatten + 2 Dense.
- Shape Flatten et nombre de parametres Dense calcules dans le notebook.
- Commit : `phase 1.3 architecture CNN`.

## Issue 5 - Phase 1.4 : training scratch et diagnostic overfitting

Objectif : entrainer le CNN scratch, logger TensorBoard, sauvegarder les courbes.

Critere d'acceptation :
- Compilation `adam`, `binary_crossentropy`, `accuracy`.
- TensorBoard dans `logs/scratch/`.
- EarlyStopping `val_loss`, patience 5, `restore_best_weights=True`.
- `history_scratch`, `training_time_scratch`, courbes loss/accuracy.
- Diagnostic de l'epoch de divergence.
- Commit : `phase 1.4 training scratch`.

## Issue 6 - Phase 2.1 : pipeline data augmentation

Objectif : ajouter les couches Keras d'augmentation et verifier visuellement les transformations.

Critere d'acceptation :
- `RandomFlip("horizontal")`, `RandomRotation`, `RandomZoom`.
- Augmentation active uniquement en training.
- Grille 3x3 d'une meme image augmentee sauvegardee.
- Commit : `phase 2.1 data augmentation pipeline`.

## Issue 7 - Phase 2.2 : CNN augmente avec Dropout

Objectif : entrainer un nouveau CNN avec augmentation et Dropout.

Critere d'acceptation :
- `build_cnn_augmented(input_shape)` implemente.
- Dropout place apres Flatten, taux autour de 0.4.
- Training avec memes hyperparametres que TP1.
- `history_augmented`, `training_time_augmented`, courbes sauvegardees.
- Commit : `phase 2.2 dropout retraining`.

## Issue 8 - Phase 2.3 : comparaison TP1 vs TP2

Objectif : comparer courbes, temps, parametres et accuracy entre scratch et augmente.

Critere d'acceptation :
- Graphe `comparison_tp1_tp2.png`.
- `val_accuracy` max, temps et parametres affiches.
- Paragraphe Markdown d'interpretation.
- Commit : `phase 2.3 comparaison scratch vs augmente`.

## Issue 9 - Phase 3.1 : MobileNetV2 et freezing

Objectif : reconstruire le pipeline 160x160 et charger MobileNetV2 frozen.

Critere d'acceptation :
- `IMG_SIZE_TL = (160, 160)`.
- `preprocess_input` MobileNetV2 utilise sans `Rescaling`.
- `base_model.trainable = False`.
- Tete custom avec GAP, Dense(128), Dropout(0.3), Dense(1 sigmoid).
- Summary avec base non entrainable et sortie `(None, 1)`.
- Commit : `phase 3.1 MobileNetV2 chargement`.

## Issue 10 - Phase 3.2 : entrainement tete MobileNetV2

Objectif : entrainer uniquement la tete pendant 10 epochs.

Critere d'acceptation :
- Compilation Adam `learning_rate=1e-3`.
- Logs TensorBoard dans `logs/transfer/`.
- `history_tl_head`, `training_time_head`.
- `val_accuracy` cible superieure a 90% si le dataset le permet.
- Commit : `phase 3.2 head training`.

## Issue 11 - Phase 3.3 : fine-tuning partiel

Objectif : degeler les 20% dernieres couches et reentrainer avec LR reduit.

Critere d'acceptation :
- `fine_tune_at = int(len(base_model.layers) * 0.8)`.
- 80% premieres couches gelees.
- Recompilation Adam `learning_rate=1e-4`.
- EarlyStopping sur `val_accuracy`, patience 5.
- `history_tl_finetune`, `training_time_finetune`.
- Commit : `phase 3.3 fine-tuning`.

## Issue 12 - Phase 3.4 : tableau cross-modeles

Objectif : sauvegarder les trois modeles et produire le tableau comparatif.

Critere d'acceptation :
- `model_scratch.keras`, `model_augmented.keras`, `model_tl.keras`.
- Tableau 3 lignes avec `val_acc`, `params`, `temps`, `taille`.
- Graphe `comparison_all.png`.
- Paragraphe Markdown d'interpretation performance/cout.
- Commit : `phase 3.4 comparaison cross-modeles`.

## Issue 13 - Phase 3.5 : export TFLite et exploration

Objectif : exporter le meilleur modele et lancer au moins une exploration.

Critere d'acceptation :
- Fichier `.tflite` produit avec nom base sur les classes.
- Taille `.keras` vs `.tflite` et facteur de compression affiches.
- Inference TFLite executee sur une image de validation.
- Au moins une direction exploree : quantization INT8, architecture alternative, precision, autre dataset.
- Limite hors distribution documentee dans le README ou notebook.
- Commit : `phase 3.5 export tflite`.
