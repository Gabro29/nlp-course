# Implementation Notes

This file provides a detailed, step-by-step explanation of the main implementation choices made throughout the project.

---

# Baseline

The Jaccard similarity requires computing the intersection over union (IoU) between two token sets.

The core idea is to treat WordNet as a ground truth source: the words appearing in the synset definitions and examples define the meaning of the current target word. For example:

11 synsets found for the word `elemento` (element):
* **Synset 1:** `omw-it-05868954-n`
  <br>Definition: *each of the elements that contribute to characterizing a phenomenon or an abstract entity*
  <br>Associated lemmas: `['ingrediente', 'componente', 'elemento', 'fattore']`
* **Synset 2:** `omw-it-13934465-n`
  <br>Definition: *condition of ease*
  <br>Associated lemmas: `['elemento']`
* **Synset 3:** `omw-it-09909060-n`
  <br>Definition: *a person who is out of the ordinary or has original, bizarre behavior*
  <br>Associated lemmas: `['tipo', 'elemento', 'personaggio', 'sagoma', 'eccentrico', 'originale', 'numero']`

This gives us multiple definitions, i.e., multiple ground-truth strings.

For each ground-truth string, we compute the Jaccard similarity against `sentence1` and then against `sentence2`.

`sentence1` will achieve a maximum similarity score with one specific definition. The same applies to `sentence2`.

The label assigned to the record will be `1` if both `sentence1` and `sentence2` achieve their maximum similarity match with the **same** ground-truth string. Otherwise it is `0`.

---

# Word2Vec

The goal is to build one large list of all available sentences to use as a training corpus.

For a single dataset split, the available sources per record are:
* `sentence1`
* `sentence2`
* The target word, i.e., the `lemma`
* Definitions provided by WordNet
* Usage examples provided by WordNet
* Related lemmas for the target word provided by WordNet

Concrete example for the lemma `elemento` (element):

* *In particolare, l'elemento più scandaloso del provvedimento che ci accingiamo a votare è rappresentato dagli articoli 7 e 8 su cui, questa mattina, abbiamo condotto, come opposizione, una lunga battaglia di buon senso.* → `sentence1` from the dataset
* *Il Fondo mostra di apprezzare "certi elementi di maggiore flessibilità del patto di stabilita', mentre si rammarica per la vaghezza di altre".* → `sentence2` from the dataset
* `elemento` → target word (lemma)
* *each of the elements that contribute to characterizing a phenomenon or an abstract entity* → WordNet definition
* *a person who is out of the ordinary or has original, bizarre behavior* → WordNet definition
* `ingrediente`, `componente`, `elemento`, `fattore` → WordNet associated lemmas

All of these strings, after preprocessing and tokenization, become token sequences that are added to the training corpus. The rationale is straightforward: the more contexts we feed to the model, the richer and more accurate the resulting word vector representations will be.

---
