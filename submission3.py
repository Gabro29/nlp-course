import copy
import os
import random
import re
import unicodedata
import emoji
import nltk
from nltk.corpus import stopwords
from num2words import num2words
import pandas as pd
import numpy as np
import spacy
import wn

from gensim.models import Word2Vec, FastText
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from plots_submission3 import (
    calculate_projections,
    plot_words_in_subspace,
    plot_training_loss,
    plot_roc_curve,
    plot_thresholds_and_f1,
    compare_models_performance
)



def set_seed(seed: int = 177) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_dataset_path(filename: str) -> str:
    return os.path.join("DATASET", filename)



def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def get_stopset() -> set:
    nltk.download('stopwords', quiet=True)
    nltk_stops = set(stopwords.words('italian'))
    custom_words = {'no', 'si', 'ok', 'okay'}

    da_salvare = {'più', 'piu', 'faccia', 'non', 'senza', 'nessun', 'nessuno', 'niente', 'né', 'stato', 'stati'}
    raw_stops = (nltk_stops | custom_words) - da_salvare

    return raw_stops | {strip_accents(w) for w in raw_stops}


nlp = spacy.load("it_core_news_md", disable=["parser", "ner"])
stop_set = get_stopset()
da_salvare = {'più', 'piu', 'faccia', 'non', 'senza', 'nessun', 'nessuno', 'niente', 'né', 'stato', 'stati'}


def clean_and_tokenize(text: str) -> list:
    if not isinstance(text, str):
        return []

    RE_NUMBERS = re.compile(r'\b\d+\b')
    RE_LAUGHS = re.compile(r'\b(?:[aeiou]*[hjk][aeiou]+[hjk]?|[aeiou]+[hjk][aeiou]*)+\b', re.IGNORECASE)
    RE_REPEATED = re.compile(r'(.)\1{2,}')
    RE_MONOCHAR = re.compile(r'\b([a-zA-Z])\1+\b')

    text = text.lower()
    text = emoji.replace_emoji(text, replace=' ')

    text = RE_NUMBERS.sub(lambda m: num2words(int(m.group(0)), lang='it'), text)
    text = RE_LAUGHS.sub(' ', text)
    text = RE_MONOCHAR.sub(' ', text)
    text = RE_REPEATED.sub(r'\1', text)

    doc = nlp(text)
    valid_tokens = []

    for t in doc:
        if t.is_punct or t.is_space or t.like_url:
            continue

        t_lemma = t.lemma_.strip()
        is_valid_pos = t.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'AUX']
        is_saved_word = t.text in da_salvare or t.lemma_ in da_salvare

        if is_valid_pos or is_saved_word:
            if t_lemma not in stop_set and strip_accents(t_lemma) not in stop_set:
                valid_tokens.append(t_lemma)

    return valid_tokens


def jaccard_similarity(set_a: set, set_b: set) -> float:
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(intersection) / len(union)


def calculate_jaccard_pred(dataset: pd.DataFrame, it_wn) -> list:
    predictions = []

    for idx, row in dataset.iterrows():
        lemma = row['lemma']
        s1_set = set(clean_and_tokenize(row['sentence1']))
        s2_set = set(clean_and_tokenize(row['sentence2']))

        synsets = it_wn.synsets(lemma)
        definitions = [ss.definition() for ss in synsets if ss.definition()]

        if not definitions:
            predictions.append(0)
            continue

        best_s1_idx = -1
        best_s1_score = -1.0
        best_s2_idx = -1
        best_s2_score = -1.0

        for i, def_text in enumerate(definitions):
            def_set = set(clean_and_tokenize(def_text))
            score1 = jaccard_similarity(s1_set, def_set)
            score2 = jaccard_similarity(s2_set, def_set)

            if score1 > best_s1_score:
                best_s1_score = score1
                best_s1_idx = i

            if score2 > best_s2_score:
                best_s2_score = score2
                best_s2_idx = i

        if best_s1_idx == best_s2_idx:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions


def run_baseline_evaluation(dataset_files: dict = None) -> dict:
    if dataset_files is None:
        dataset_files = {
            "train": resolve_dataset_path("train.parquet"),
            "val": resolve_dataset_path("validation.parquet"),
            "test": resolve_dataset_path("test.parquet")
        }

    it_wn = wn.Wordnet("omw-it:1.4")
    results = {}

    for split_name, file_path in dataset_files.items():
        if not os.path.exists(file_path):
            print(f"File {file_path} non trovato. Salto {split_name}.")
            continue

        df = pd.read_parquet(file_path, engine="fastparquet")
        predictions = calculate_jaccard_pred(df, it_wn)
        df['predicted_label'] = predictions

        f1 = f1_score(df['label'], df['predicted_label'], average='binary')
        prec = precision_score(df['label'], df['predicted_label'], average='binary', zero_division=0)
        rec = recall_score(df['label'], df['predicted_label'], average='binary', zero_division=0)

        results[split_name] = {'f1': f1, 'precision': prec, 'recall': rec}
        print(f"<- Baseline [{split_name.upper()}] -> F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

    return results


def create_corpus(dataset_path: str) -> list:
    dataset = pd.read_parquet(dataset_path, engine="fastparquet")
    corpus = []

    for s1, s2 in zip(dataset["sentence1"], dataset["sentence2"]):
        s1_clean = clean_and_tokenize(s1)
        s2_clean = clean_and_tokenize(s2)
        if s1_clean:
            corpus.append(s1_clean)
        if s2_clean:
            corpus.append(s2_clean)

    for lemma in dataset["lemma"].unique():
        synsets = wn.synsets(lemma, lang="it")
        for syn in synsets:
            defn = syn.definition()
            if defn:
                defn_clean = clean_and_tokenize(defn)
                if defn_clean:
                    corpus.append(defn_clean)

            if hasattr(syn, "examples"):
                for esempio in syn.examples():
                    es_clean = clean_and_tokenize(esempio)
                    if es_clean:
                        corpus.append(es_clean)

            lemmi_limpidi = []
            try:
                lemmi_raw = syn.lemmas()
                for l in lemmi_raw:
                    lemma_str = l.name if hasattr(l, 'name') else str(l)
                    lemma_str = lemma_str.replace('_', ' ')
                    tokens = clean_and_tokenize(lemma_str)
                    lemmi_limpidi.extend(tokens)
            except Exception:
                pass

            if len(lemmi_limpidi) > 1:
                corpus.append(lemmi_limpidi)

    return corpus


def get_w2v(train_corpus: list, vector_size: int) -> tuple:
    w2v_model = Word2Vec(
        sentences=train_corpus,
        vector_size=vector_size,
        window=5,
        min_count=1,
        epochs=30,
        workers=4,
        sg=1
    )
    words_w2v = list(w2v_model.wv.index_to_key)
    return w2v_model, words_w2v, w2v_model.wv[words_w2v]


def get_fastT(train_corpus: list, vector_size: int) -> tuple:
    fft_model = FastText(
        sentences=train_corpus,
        vector_size=vector_size,
        window=5,
        min_count=2,
        sg=1,
        epochs=30,
        workers=4,
        word_ngrams=1
    )
    words_fft = list(fft_model.wv.index_to_key)
    return fft_model, words_fft, fft_model.wv[words_fft]


def train_embedding_models(train_path: str = None, vector_size: int = 150) -> tuple:
    if train_path is None:
        train_path = resolve_dataset_path("train.parquet")

    print(f"<- Costruzione del corpus da: {train_path} ->")
    train_corpus = create_corpus(train_path)
    print(f"<- Corpus creato con {len(train_corpus)} sequenze ->")

    print("<- Addestramento Word2Vec (Skip-Gram) ->")
    w2v_model, words_w2v, vectors_w2v = get_w2v(train_corpus, vector_size)

    print("<- Addestramento FastText (Skip-Gram) ->")
    fft_model, words_fft, vectors_ft = get_fastT(train_corpus, vector_size)

    return w2v_model, fft_model, words_w2v, words_fft, vectors_w2v, vectors_ft


def run_subspace_analysis(vectors_w2v: np.ndarray, vectors_ft: np.ndarray,
                          words_w2v: list, words_fft: list) -> None:
    projections = calculate_projections(vectors_w2v, vectors_ft)
    plot_words_in_subspace(projections, words_w2v, words_fft)


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v, ord=2, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps)


def get_average_embedding(tokens: list, model, vector_size: int) -> np.ndarray:
    valid_words = [word for word in tokens if word in model.wv]
    if not valid_words:
        return np.zeros(vector_size)
    embeddings = [model.wv[word] for word in valid_words]
    return np.mean(embeddings, axis=0)


def create_embeddings_for(dataset_name: str, path_tosave: str, embedding_model, vector_size: int) -> None:
    dataset = pd.read_parquet(dataset_name, engine="fastparquet")
    dataset_embeddings = []
    vector_size = embedding_model.vector_size

    for index, row in dataset.iterrows():
        tokens_lemma = clean_and_tokenize(row["lemma"])
        tokens_s1 = clean_and_tokenize(row["sentence1"])
        tokens_s2 = clean_and_tokenize(row["sentence2"])

        emb_lemma = get_average_embedding(tokens_lemma, embedding_model, vector_size)
        emb_s1 = get_average_embedding(tokens_s1, embedding_model, vector_size)
        emb_s2 = get_average_embedding(tokens_s2, embedding_model, vector_size)

        emb_lemma_norm = l2_normalize(emb_lemma)
        emb_s1_norm = l2_normalize(emb_s1)
        emb_s2_norm = l2_normalize(emb_s2)

        emb_sentences_mean = (emb_s1_norm + emb_s2_norm) / 2.0

        final_embedding = np.concatenate([emb_lemma_norm, emb_sentences_mean])
        dataset_embeddings.append(final_embedding)

    assert np.array(dataset_embeddings).shape[0] == dataset["label"].values.shape[0]
    os.makedirs(os.path.dirname(path_tosave) or '.', exist_ok=True)
    np.savez_compressed(path_tosave, X=np.array(dataset_embeddings), y=dataset["label"].values)


def build_all_feature_datasets(w2v_model, fft_model, vector_size: int = 150,
                               checkpoints_dir: str = "checkpoints") -> None:
    os.makedirs(checkpoints_dir, exist_ok=True)
    dataset_files = {
        "train": resolve_dataset_path("train.parquet"),
        "val": resolve_dataset_path("validation.parquet"),
        "test": resolve_dataset_path("test.parquet")
    }

    for split_name, file_path in dataset_files.items():
        path_w2v = os.path.join(checkpoints_dir, f"{split_name}_dataset_w2v.npz")
        path_w2v_local = f"{split_name}_dataset_w2v.npz"
        print(f"<- Creazione feature W2V per [{split_name}] ->")
        create_embeddings_for(file_path, path_w2v, w2v_model, vector_size)
        create_embeddings_for(file_path, path_w2v_local, w2v_model, vector_size)

        path_ft = os.path.join(checkpoints_dir, f"{split_name}_dataset_ft.npz")
        path_ft_local = f"{split_name}_dataset_ft.npz"
        print(f"<- Creazione feature FastText per [{split_name}] ->")
        create_embeddings_for(file_path, path_ft, fft_model, vector_size)
        create_embeddings_for(file_path, path_ft_local, fft_model, vector_size)


def create_dataloaders(type_model: str, batch_size: int = 32, checkpoints_dir: str = "checkpoints") -> tuple:
    def find_file(prefix):
        p1 = f"{prefix}_dataset_{type_model}.npz"
        p2 = os.path.join(checkpoints_dir, f"{prefix}_dataset_{type_model}.npz")
        return p1 if os.path.exists(p1) else p2

    data_train = np.load(find_file('train'))
    X_train, y_train = data_train['X'], data_train['y']

    data_val = np.load(find_file('val'))
    X_val, y_val = data_val['X'], data_val['y']

    data_test = np.load(find_file('test'))
    X_test, y_test = data_test['X'], data_test['y']

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    val_dataset   = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
    test_dataset  = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl, y_train


class EarlyStoppingAUC:
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.max_validation_auc = -float('inf')
        self.best_model_weights = None
        self.best_epoch = 1

    def __call__(self, validation_auc: float, model: nn.Module, epoch: int) -> None:
        if (validation_auc - self.min_delta) <= self.max_validation_auc:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.max_validation_auc = validation_auc
            self.counter = 0
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch


class BinaryNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.7),

            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def train_and_evaluate(train_dl, val_dl, y_train_current, input_dim: int,
                       epoches: int = 200, lr: float = 3e-4,
                       model_name: str = "", make_plots: bool = True) -> tuple:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BinaryNet(input_dim=input_dim).to(device)

    num_pos = (y_train_current == 1).sum()
    num_neg = (y_train_current == 0).sum()
    peso_positivi = num_neg / max(num_pos, 1)
    pos_weight_tensor = torch.tensor([peso_positivi], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    stopper = EarlyStoppingAUC(patience=10, min_delta=0.001)

    train_losses, val_losses = [], []

    for epoch in range(1, epoches + 1):
        model.train()
        loss_epoch_train = 0.0
        for X, y in train_dl:
            X = X.to(device)
            y = y.to(device).float().unsqueeze(1) if y.ndim == 1 else y.to(device).float()

            optimizer.zero_grad()
            logits = model(X)
            batch_loss = criterion(logits, y)
            batch_loss.backward()
            optimizer.step()
            loss_epoch_train += batch_loss.item()

        train_loss = loss_epoch_train / len(train_dl)
        train_losses.append(train_loss)

        model.eval()
        loss_epoch_val = 0.0
        y_true_net, y_prob_net = [], []

        with torch.no_grad():
            for X, y in val_dl:
                X = X.to(device)
                y = y.to(device).float().unsqueeze(1) if y.ndim == 1 else y.to(device).float()

                logits = model(X)
                batch_loss = criterion(logits, y)
                loss_epoch_val += batch_loss.item()

                probs = torch.sigmoid(logits)
                y_true_net.extend(y.cpu().numpy().flatten())
                y_prob_net.extend(probs.cpu().numpy().flatten())

        val_loss = loss_epoch_val / len(val_dl)
        val_losses.append(val_loss)

        current_auc = roc_auc_score(y_true_net, y_prob_net)
        print(f"Epoca {epoch:03d} | Loss Train: {train_loss:.4f} | Loss Val: {val_loss:.4f} | AUC Val: {current_auc:.4f}")

        scheduler.step(current_auc)
        stopper(current_auc, model, epoch)
        if stopper.early_stop:
            print(f"<- Early stopping attivato all'epoca {epoch}. Miglior AUC: {stopper.max_validation_auc:.4f} (epoca migliore: {stopper.best_epoch}) ->")
            break

    if stopper.best_model_weights is not None:
        model.load_state_dict(stopper.best_model_weights)

    best_epoch = stopper.best_epoch

    model.eval()
    y_true_best, y_prob_best = [], []
    with torch.no_grad():
        for X, y in val_dl:
            X = X.to(device)
            y = y.to(device).float().unsqueeze(1) if y.ndim == 1 else y.to(device).float()
            logits = model(X)
            probs = torch.sigmoid(logits)
            y_true_best.extend(y.cpu().numpy().flatten())
            y_prob_best.extend(probs.cpu().numpy().flatten())

    precisions, recalls, thresholds = precision_recall_curve(y_true_best, y_prob_best)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_threshold_idx]
    val_auc = roc_auc_score(y_true_best, y_prob_best)

    print(f"<- Miglior F1-score su validation ({model_name.upper()}): {best_f1:.4f} a soglia {best_threshold:.4f} ->")

    if make_plots:
        plot_training_loss(train_losses, val_losses, best_epoch, model_name=model_name)
        plot_roc_curve(np.array(y_true_best), np.array(y_prob_best), val_auc, model_name=model_name)
        plot_thresholds_and_f1(np.array(y_true_best), np.array(y_prob_best), best_threshold, best_f1,
                               model_name=model_name)

    return model, best_threshold, {'train_losses': train_losses, 'val_losses': val_losses, 'val_auc': val_auc}


def evaluate_on_test(model: nn.Module, test_dl, device: str = None, threshold: float = None) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    y_true, y_probs = [], []

    with torch.no_grad():
        for X, y in test_dl:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)
            y_true.extend(y.cpu().numpy().flatten())
            y_probs.extend(probs.cpu().numpy().flatten())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    auc = roc_auc_score(y_true, y_probs)

    if threshold is None:
        threshold = 0.5

    y_pred = (y_probs >= threshold).astype(float)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)

    print("\n" + "=" * 45)
    print("          RISULTATI SUL TEST SET             ")
    print("=" * 45)
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"Soglia    : {threshold:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print("=" * 45)

    return {'auc': auc, 'f1': f1, 'precision': precision, 'recall': recall, 'threshold': threshold}


def run_training_pipeline(models_to_test: list = None, vector_size: int = 150,
                          batch_size: int = 32, epoches: int = 200, lr: float = 3e-4,
                          make_plots: bool = True) -> dict:
    if models_to_test is None:
        models_to_test = ['w2v', 'ft']

    input_dim = vector_size * 2
    training_results = {}

    for type_model in models_to_test:
        print(f"\n=================== ADDESTRAMENTO MODELLO: {type_model.upper()} ===================")
        train_dl, val_dl, test_dl, y_train_current = create_dataloaders(type_model, batch_size=batch_size)
        best_model, best_threshold, history = train_and_evaluate(
            train_dl=train_dl,
            val_dl=val_dl,
            y_train_current=y_train_current,
            input_dim=input_dim,
            epoches=epoches,
            lr=lr,
            model_name=type_model,
            make_plots=make_plots
        )
        test_metrics = evaluate_on_test(best_model, test_dl, threshold=best_threshold)

        training_results[type_model] = {
            'model': best_model,
            'threshold': best_threshold,
            'test_metrics': test_metrics,
            'history': history,
            'input_dim': input_dim
        }

    if make_plots and len(training_results) > 1:
        compare_dict = {m: res['test_metrics'] for m, res in training_results.items()}
        compare_models_performance(compare_dict)

    return training_results


def save_best_model_bundle(training_results: dict, checkpoints_dir: str = "checkpoints") -> str:
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_model_name = max(training_results.keys(), key=lambda k: training_results[k]['test_metrics']['auc'])
    best_info = training_results[best_model_name]

    bundle = {
        'model_name': best_model_name,
        'state_dict': best_info['model'].state_dict(),
        'threshold': best_info['threshold'],
        'test_metrics': best_info['test_metrics'],
        'input_dim': best_info['input_dim']
    }

    bundle_path = os.path.join(checkpoints_dir, "best_model_bundle.pt")
    torch.save(bundle, bundle_path)

    print(f"\n<- Miglior modello selezionato: {best_model_name.upper()} (ROC-AUC: {best_info['test_metrics']['auc']:.4f}) ->")
    print(f"<- Bundle salvato con successo in: {bundle_path} ->")
    return bundle_path



if __name__ == "__main__":

    set_seed(177)
    vector_size = 150
    batch_size = 32
    epoches = 200
    lr = 3e-4


    eval_baseline = True
    if eval_baseline:
        baseline_results = run_baseline_evaluation()


    build_embeddings = True
    if build_embeddings:
        w2v_model, fft_model, words_w2v, words_fft, vectors_w2v, vectors_ft = train_embedding_models(
            vector_size=vector_size
        )


    plot_subspace = False
    if plot_subspace:
        run_subspace_analysis(vectors_w2v, vectors_ft, words_w2v, words_fft)


    create_datasets = True
    if create_datasets:
        build_all_feature_datasets(w2v_model, fft_model, vector_size=vector_size)


    training = True
    if training:
        training_results = run_training_pipeline(
            models_to_test=['w2v', 'ft'],
            vector_size=vector_size,
            batch_size=batch_size,
            epoches=epoches,
            lr=lr,
            make_plots=True
        )


    model_selection = True
    if model_selection:
        best_bundle_path = save_best_model_bundle(training_results)
