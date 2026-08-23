# BERT-BiLSTM Marine Incident Classifier

Implementation of the BERT + BiLSTM text-classification architecture from:

> Zhao, Z.; Liu, X.; Feng, L.; Grifoll, M.; Feng, H. (2025). *Causation
> Analysis of Marine Traffic Accidents Using Deep Learning Approaches: A Case
> Study from China's Coasts.* Systems, 13(4), 284.
> https://doi.org/10.3390/systems13040284
> (full text: [`../Causation_Analysis_of_Marine_Traffic_Accidents_Usi.pdf`](../Causation_Analysis_of_Marine_Traffic_Accidents_Usi.pdf))

applied to the [`baker-street/maib-incident-reports-5K`](https://huggingface.co/datasets/baker-street/maib-incident-reports-5K)
dataset of UK MAIB marine incident narratives.

## Fidelity to the paper

The architecture and every hyperparameter below are taken directly from the
paper's Table 9 ("Model parameter settings", covering its BERT + BiLSTM
configuration) and Section 4.1 / Figure 7 for the data split and epoch count:

| Setting | Paper value | Source |
|---|---|---|
| BiLSTM hidden size | 128 | Table 9 ("LSTM layer 128") |
| Learning rate | 1e-6 (single rate for all params, AdamW) | Table 9 |
| L2 weight decay | 0.05 | Table 9 |
| L1 regularization | 5e-10 | Table 9 |
| Gradient clipping (max norm) | 2.75 | Table 9 |
| Batch size | 32 | Table 9 |
| Dropout layers | 2 (bracketing the BiLSTM) | Table 9 ("Dropout Layer: 2") |
| Activation (BiLSTM branch) | Mish | Table 9 |
| Optimizer | AdamW | Table 9 |
| Loss | Cross-entropy (paper's output layer is Softmax; Section 3.1.1) | Section 3.1.1 |
| Epochs to convergence | ~20 | Figure 7 caption |
| Train / val / test split | 70% / 15% / 15% | Section 4.1 |

Everything in that table is also exposed as a CLI flag on `train.py` in case
you want to deviate from the paper.

**What's *not* reproduced here, and why:**

- **The paper's dataset differs from this project's dataset.** The paper
  trains on ~25,930 examples across **32 causal-factor classes** (e.g.
  "Improper operation", "Negligent lookout", "Mismanagement by the
  shipowner"), sourced from Chinese and international (GISIS) accident
  reports and rebalanced with text augmentation (synonym replacement, back
  translation, AI generation — Section 4.1). This project instead uses the
  ~5.8k-example, single-label **incident-type** classification in
  `baker-street/maib-incident-reports-5K` (Collision, Grounding/Stranding,
  Fire/Explosion, Accident to person(s), Damage/Loss of Equipment, ...),
  with no augmentation. The architecture and hyperparameters are reproduced
  faithfully, but don't expect the same absolute accuracy (the paper reports
  89.8% on its own dataset) on this smaller, differently-labeled one.
- **The paper's exact BERT variant** is not named in the text (Section 3.1.1
  describes the WordPiece/embedding/Transformer mechanics generically). This
  project uses `bert-base-uncased`, a reasonable default for the MAIB
  dataset's English text.
- **The Apriori association-rule mining stage** (Section 3.2 / 3.3): after
  classification, the paper mines frequent co-occurring *causal factors*
  across reports (support/confidence/lift thresholds tuned in Section 4.4) to
  reveal multi-factor interactions. That stage operates on the paper's
  multi-label causal-factor tags, which the MAIB incident-type dataset
  doesn't have, so it isn't implemented here. Ask if you'd like it added on
  top of the classifier's predictions.

## Architecture

```
input text
  -> BERT tokenizer (bert-base-uncased)
  -> BERT encoder            (contextual token embeddings, last_hidden_state)
  -> Dropout
  -> Bidirectional LSTM, 128 units  (models sequential dependencies over the tokens)
  -> concat(final forward, final backward hidden states)
  -> Mish activation
  -> Dropout
  -> Linear -> softmax over incident-type classes
```

See `model.py` (`BertBiLSTMClassifier`), which mirrors Figure 1 of the paper.

## Setup

With [uv](https://docs.astral.sh/uv/) (recommended — resolves and installs
into a project-local `.venv` from `pyproject.toml`/`uv.lock`):

```bash
uv sync
```

Or with plain `pip`:

```bash
pip install -r requirements.txt
```

## Train

```bash
uv run train.py
# or, without uv:
python train.py
```

Default hyperparameters (override any via CLI flags — run `python train.py -h`;
see the table above for where each one comes from):

| Hyperparameter        | Default             |
|------------------------|---------------------|
| Pretrained encoder     | `bert-base-uncased` |
| Max sequence length    | 128                 |
| BiLSTM hidden size     | 128 (x2 for bidirectional) |
| BiLSTM layers          | 1                   |
| Dropout                | 0.3                 |
| Batch size             | 32                  |
| Epochs                 | 20                  |
| Learning rate          | 1e-6 (AdamW)        |
| L2 weight decay        | 0.05                |
| L1 penalty             | 5e-10               |
| Gradient clipping      | max norm 2.75       |
| Train/val/test split   | 70% / 15% / 15% (stratified) |

Training prints per-epoch loss/accuracy/macro-F1 for train and validation,
saves the best checkpoint (by validation macro-F1) to `checkpoints/`, and
finishes with a full `sklearn.classification_report` on the held-out test
split.

Note: the paper's learning rate (1e-6) is unusually low for BERT
fine-tuning (typical recipes use 1e-5–5e-5) — its own Figure 7 shows this
is compensated for by training ~20 epochs rather than the usual 3-4.

## Predict

```bash
uv run predict.py --checkpoint-dir checkpoints \
    "A bulk carrier ran aground after losing steering control in heavy weather."
# or, without uv:
python predict.py --checkpoint-dir checkpoints \
    "A bulk carrier ran aground after losing steering control in heavy weather."
```

## Notebook

`BERT_BiLSTM_Marine_Incident_Classification.ipynb` walks through the same
pipeline end-to-end in a single notebook (data loading/splitting, model
definition, training loop, test evaluation, checkpoint saving, and
inference) — handy for running on Colab/Kaggle/Jupyter with a GPU instead
of the CLI scripts below.

## Files

- `dataset.py` — loads the dataset from the Hub, cleans text, encodes labels,
  builds stratified train/val/test splits, and defines the `torch.Dataset`.
- `model.py` — the `BertBiLSTMClassifier` module.
- `train.py` — training/evaluation loop.
- `predict.py` — inference on new narratives with a saved checkpoint.
- `BERT_BiLSTM_Marine_Incident_Classification.ipynb` — notebook version of
  the full pipeline.
- `pyproject.toml` / `uv.lock` — project metadata and pinned dependencies for
  [uv](https://docs.astral.sh/uv/); `requirements.txt` covers plain `pip`.
