import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, roc_curve
import umap


def calculate_projections(vectors_w2v: np.ndarray, vectors_ft: np.ndarray) -> dict:
    projections = {}
    print("<- Calcolo proiezioni PCA, t-SNE, UMAP ->")
    projections['PCA'] = PCA(n_components=2).fit_transform(vectors_w2v)
    projections['t-SNE'] = TSNE(n_components=2, metric='cosine', init='pca', random_state=177).fit_transform(vectors_w2v)
    projections['UMAP'] = umap.UMAP(n_components=2, n_neighbors=30, min_dist=1, metric='cosine',
                                    random_state=177).fit_transform(vectors_w2v)

    projections['PCA_f'] = PCA(n_components=2).fit_transform(vectors_ft)
    projections['t-SNE_f'] = TSNE(n_components=2, metric='cosine', init='pca', random_state=177).fit_transform(vectors_ft)
    projections['UMAP_f'] = umap.UMAP(n_components=2, n_neighbors=30, min_dist=1, metric='cosine',
                                      random_state=177).fit_transform(vectors_ft)
    return projections


def plot_words_in_subspace(projections: dict, words_w2v: list, words_ft: list, concepts: list = None) -> None:
    if concepts is None:
        concepts = [
            ['ragazzo', 'ragazza', 'uomo', 'donna'],
            ['alto', 'basso', 'caldo', 'freddo'],
            ['soldato', 'generale', 'impiegato', 'direttore']
        ]

    for target_words in concepts:
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))
        fig.suptitle(f"Confronto Parallelogramma: {' vs '.join(target_words)}", fontsize=18, fontweight='bold')

        axes = axes.flatten()

        for ax, (nome_metodo, vettori_2d) in zip(axes, projections.items()):
            current_words = words_ft if '_f' in nome_metodo else words_w2v

            ax.scatter(vettori_2d[:, 0], vettori_2d[:, 1], alpha=0.03, c='lightgray')

            coords = {}
            for w in target_words:
                if w not in current_words:
                    continue
                idx = current_words.index(w)
                coords[w] = (vettori_2d[idx, 0], vettori_2d[idx, 1])

                ax.scatter(coords[w][0], coords[w][1], color='red', s=80, zorder=5)
                ax.annotate(w, (coords[w][0], coords[w][1]), fontsize=8, fontweight='bold',
                            xytext=(8, 5), textcoords='offset points')

            if len(coords) == 4:
                w_a, w_b, w_c, w_d = target_words
                ax.annotate("", xy=coords[w_b], xytext=coords[w_a],
                            arrowprops=dict(arrowstyle="->", color="blue", lw=2, alpha=0.6))
                ax.annotate("", xy=coords[w_d], xytext=coords[w_c],
                            arrowprops=dict(arrowstyle="->", color="blue", lw=2, alpha=0.6))

                ax.plot([coords[w_a][0], coords[w_c][0]], [coords[w_a][1], coords[w_c][1]], 'k--', alpha=0.5)
                ax.plot([coords[w_b][0], coords[w_d][0]], [coords[w_b][1], coords[w_d][1]], 'k--', alpha=0.5)

            ax.set_title(f"{nome_metodo}", fontsize=12, fontweight='semibold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()


def plot_training_loss(train_losses: list, val_losses: list, best_epoch: int, model_name: str = "") -> None:
    plt.figure(figsize=(10, 6))
    epoche_effettive = range(1, len(train_losses) + 1)
    plt.plot(epoche_effettive, train_losses, label='Train Loss', color='#2980b9', linewidth=2)
    plt.plot(epoche_effettive, val_losses, label='Validation Loss', color='#e67e22', linewidth=2)
    plt.axvline(x=best_epoch, color='#c0392b', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
    plt.scatter(best_epoch, val_losses[best_epoch - 1], color='#c0392b', s=100, zorder=5)

    title = f"Andamento della Loss ({model_name.upper()})" if model_name else "Andamento della Loss durante l'Addestramento"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Epoche", fontsize=11)
    plt.ylabel("Loss (BCEWithLogits)", fontsize=11)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, auc_score: float, model_name: str = "") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#d35400', lw=2.5, label=f'Curva ROC (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='#2c3e50', lw=1.8, linestyle='--', label='Classificatore Casuale')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=11)
    plt.ylabel('True Positive Rate (TPR)', fontsize=11)
    title = f'Receiver Operating Characteristic (ROC) - {model_name.upper()}' if model_name else 'Receiver Operating Characteristic (ROC)'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", frameon=True)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_thresholds_and_f1(y_true: np.ndarray, y_prob: np.ndarray, best_threshold: float, best_f1: float, model_name: str = "") -> None:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], label='Precision', color='#2980b9', linestyle='--', linewidth=1.8)
    plt.plot(thresholds, recalls[:-1], label='Recall', color='#27ae60', linestyle='--', linewidth=1.8)
    plt.plot(thresholds, f1_scores[:-1], label='F1-Score', color='#8e44ad', linewidth=2.5)

    plt.axvline(x=best_threshold, color='#c0392b', linestyle=':', linewidth=2,
                label=f'Optimal Threshold ({best_threshold:.4f})')
    plt.scatter(best_threshold, best_f1, color='#c0392b', s=100, zorder=5)

    title = f"Precision, Recall e F1-Score vs Soglia ({model_name.upper()})" if model_name else "Andamento di Precision, Recall e F1-Score al variare della Soglia"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Soglia di Decisione (Threshold)", fontsize=11)
    plt.ylabel("Valore Metrica", fontsize=11)
    plt.legend(loc='lower left', frameon=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def compare_models_performance(results: dict) -> None:
    models = list(results.keys())
    metrics = ['auc', 'f1', 'precision', 'recall']
    metric_labels = ['ROC-AUC', 'F1-Score', 'Precision', 'Recall']

    x = np.arange(len(metric_labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    for i, model_name in enumerate(models):
        values = [results[model_name][m] for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        rects = plt.bar(x + offset, values, width, label=model_name.upper(), color=colors[i % len(colors)], alpha=0.85)

        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.3f}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.ylabel('Valore Metrica', fontsize=11)
    plt.title('Confronto Prestazioni Modelli sul Test Set', fontsize=14, fontweight='bold')
    plt.xticks(x, metric_labels, fontsize=11)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
