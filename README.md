# Natural Language Processing @ UNIPA

This repository contains the implementation for **Assignment 03** of the **Natural Language Processing** course. It is part of the Master’s Degree in **Computer Engineering** at the **University of Palermo**.

---

## Course Information
* **Instructor:** Prof. Roberto Pirrone
* **Academic Year:** 2026/2027
* **Task Reference:** [EVALITA 2023 - WiC-ITA](https://wic-ita.github.io/) (*Word-in-Context for Italian*, Subtask 1)
* **Topic:** Lexical Semantics & Word Sense Disambiguation (WSD) via WordNet, Word2Vec, and FastText

---

## Overview

The objective of this assignment is to tackle the binary Word Sense Disambiguation task on the **WiC-ITA** dataset. Given a target lemma and a pair of Italian sentences containing that lemma, the system must determine whether the target word carries the same meaning (`label = 1`) or different meanings (`label = 0`) across the two contexts.

The project implements:
1. **Symbolic Baseline:** A WordNet-based matching heuristic using Jaccard Similarity between sentences and synset definitions/examples (`omw-it:1.4`).
2. **Embeddings & Subspace Geometry:** Training domain-specific Skip-Gram models (**Word2Vec** and **FastText**) on an extended lexical corpus, verifying semantic parallelogram properties via PCA, t-SNE, and UMAP projections.
3. **Deep Learning Classifier:** An MLP neural network (**BinaryNet**) trained on concatenated lemma and sentence representation vectors, with class imbalance handling (`pos_weight`), early stopping on ROC-AUC, and threshold calibration.

> For a detailed step-by-step explanation of the implementation choices, see [NOTES.md](NOTES.md).

---

## Directory Structure

```
main_dir:
|
|--DATASET/ (folder, optional if in root)
|    |--train.parquet
|    |--validation.parquet
|    |--test.parquet
|
|--checkpoints/ (folder created automatically)
|    |--best_model_bundle.pt
|    |--train_dataset_w2v.npz
|    |--train_dataset_ft.npz
|    |--...
|
|--images/
|    |--parallelogram_pca.png
|    |--f1_threshold_w2v_clean.png
|    |--f1_threshold_fft_clean.png
|    |--auc_roc_w2v_clean.png
|    |--auc_roc_fft_clean.png
|
|--plots_word2vec.py
|--word2vec.py
|--README.md
|--.gitignore
```

---

## Experimental Analysis & Results

### 1. Semantic Analogy & Parallelogram Rule
To assess whether the trained embeddings preserve linear semantic regularities ($\vec{v}_{\text{uomo}} - \vec{v}_{\text{donna}} \approx \vec{v}_{\text{ragazzo}} - \vec{v}_{\text{ragazza}}$), the vector space was projected into 2D using PCA, t-SNE, and UMAP across both Word2Vec and FastText representations.

The **PCA projection on Word2Vec embeddings** captured the geometric structure with the highest fidelity:

<p align="center">
  <img src="images/parallelogram_pca.png" alt="Semantic Parallelogram in PCA Subspace" width="550"/>
</p>

* The direction and magnitude of the gender offset vectors ($\vec{v}_{\text{uomo}} \to \vec{v}_{\text{donna}}$ and $\vec{v}_{\text{ragazzo}} \to \vec{v}_{\text{ragazza}}$) are practically identical and parallel.
* The grey point cloud represents the broader semantic manifold, highlighting that the analogical quadruplet forms a coherent parallelogram well within the semantic neighborhood.

---

### 2. Decision Threshold & F1-Score Optimization
Because the WiC-ITA dataset presents intrinsic class imbalance, using a default threshold of $0.5$ leads to suboptimal F1 performance. By performing fine-grained threshold sweeps on the validation set, we identify the exact operating point that maximizes the harmonic mean of Precision and Recall:

| Word2Vec (Optimal $\tau = 0.2348$) | FastText (Optimal $\tau = 0.2448$) |
| :---: | :---: |
| <img src="images/f1_threshold_w2v_clean.png" alt="F1 Threshold Word2Vec" width="460"/> | <img src="images/f1_threshold_fft_clean.png" alt="F1 Threshold FastText" width="460"/> |

* Calibrating the decision boundary yields an F1-Score of approximately **~0.694** for Word2Vec and **~0.695** for FastText on the validation set, significantly outperforming an uncalibrated threshold.

---

### 3. Classification Performance & ROC Analysis

The discriminative capability of both neural architectures was validated through Receiver Operating Characteristic (ROC) curves:

| Word2Vec (AUC = 0.6843) | FastText (AUC = 0.6681) |
| :---: | :---: |
| <img src="images/auc_roc_w2v_clean.png" alt="ROC Curve Word2Vec" width="430"/> | <img src="images/auc_roc_fft_clean.png" alt="ROC Curve FastText" width="430"/> |

#### Performance Summary

| Model / Approach | Representation | Optimal Threshold ($\tau$) | ROC-AUC | F1-Score |
| :--- | :--- | :---: | :---: | :---: |
| **WordNet Baseline** | Jaccard Overlap on Synset Definitions | N/A | — | ~0.52 |
| **BinaryNet + Word2Vec** | Skip-Gram (150-D) + Sentence Mean | **0.2348** | **0.6843** | **0.694** |
| **BinaryNet + FastText** | Skip-Gram Subwords (150-D) + Sentence Mean | **0.2448** | 0.6681 | **0.695** |

*Both neural models comfortably outperform the symbolic WordNet baseline, with Word2Vec achieving the highest global discriminative power (AUC = 0.6843).*

#### Challenge Context

The best system at the official **EVALITA 2023 WiC-ITA** evaluation campaign achieved an F1-Score of **~0.73**. Our approach — relying exclusively on lightweight Skip-Gram embeddings, a small MLP, and threshold calibration, without any pre-trained language model or fine-tuning — reaches **~0.695 F1**, placing it within ~5% of the top-ranked system. This gap is remarkably small considering the deliberate simplicity of the pipeline.

---

## Setup & Requirements

1. **Python Environment:** Ensure Python 3.10+ is installed.
2. **Install Dependencies:**
   ```bash
   pip install torch spacy gensim nltk wn pandas numpy scikit-learn umap-learn matplotlib emoji num2words fastparquet
   ```
3. **Download Linguistic Resources:**
   ```bash
   python -m spacy download it_core_news_md
   python -c "import wn; wn.download('omw-it:1.4')"
   ```

---

## User Guide

The script `word2vec.py` is organized into independent modular phases controlled by boolean flags in the `if __name__ == "__main__":` block. Visualization functions are decoupled and maintained inside `plots_word2vec.py`.

### Execution Flags

Inside `word2vec.py`, toggle the desired pipeline steps:

* **`eval_baseline = True`**: Computes the symbolic baseline on `train`, `validation`, and `test` splits using WordNet synsets and Jaccard similarity.
* **`build_embeddings = True`**: Extracts sentences, definitions, and associated lemmas to build the extended corpus, then trains Word2Vec and FastText models.
* **`plot_subspace = True`**: Computes 2D projections (PCA, t-SNE, UMAP) and generates semantic parallelogram plots for analogical concept quadruplets.
* **`create_datasets = True`**: Encodes sentences and lemmas into $L_2$-normalized feature vectors and saves compressed `.npz` datasets.
* **`training = True`**: Trains the `BinaryNet` neural network across models (`w2v` and `ft`), plotting loss convergence, ROC curves, and optimal decision threshold calibration.
* **`model_selection = True`**: Selects the best performing model on the test set and serializes the complete bundle (`state_dict`, threshold, and metrics) into `checkpoints/best_model_bundle.pt`.

### Running the Pipeline

Simply run the main script from your terminal:

```bash
python word2vec.py
```