# BERT-BiLSTM Marine Incident Classifier

Implementation of the BERT + BiLSTM text-classification architecture from:

> Xu et al. (2025). *Causation Analysis of Marine Traffic Accidents Using
> Deep Learning Approaches: A Case Study from China's Coasts.* Systems,
> 13(4), 284. https://doi.org/10.3390/systems13040284

applied to the [`baker-street/maib-incident-reports-5K`](https://huggingface.co/datasets/baker-street/maib-incident-reports-5K)
dataset of UK MAIB marine incident narratives.

## Important caveat

Both `mdpi.com` and `researchgate.net` were unreachable from this environment,
so the paper's exact hyperparameters (learning rate, batch size, LSTM hidden
size, epoch count, train/val/test split ratio, etc.) could not be confirmed
from the full text — only the architecture description (BERT encoder → BiLSTM
→ dense classifier, combined with Apriori association-rule mining for
factor-interaction analysis) was available via search summaries and the
abstract. This implementation:

- Faithfully reproduces the **BERT + BiLSTM classification architecture**.
- Uses **standard, widely-used BERT fine-tuning defaults** in place of the
  paper's unconfirmed hyperparameters (documented below). Every one of them
  is a CLI flag on `train.py`, so they can be corrected in seconds once the
  paper's exact values are available.
- Does **not** implement the paper's Apriori association-rule mining stage.
  That stage operates on causal-factor tags extracted from accident reports
  (a different, multi-label annotation scheme) to find co-occurring risk
  factors — it's a separate analysis step from the classifier itself, and
  the MAIB dataset here ships single-label *incident types*
  (Collision, Grounding/Stranding, Fire/Explosion, Accident to person(s),
  Damage/Loss of Equipment, ...) rather than the paper's causal-factor
  categories. Ask if you'd like that mining stage added on top of the
  classifier's predictions.

## Architecture

```
input text
  -> BERT tokenizer (bert-base-uncased)
  -> BERT encoder            (contextual token embeddings, last_hidden_state)
  -> Bidirectional LSTM      (models sequential dependencies over the tokens)
  -> concat(final forward, final backward hidden states)
  -> Dropout
  -> Linear -> softmax over incident-type classes
```

See `model.py` (`BertBiLSTMClassifier`).

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Default hyperparameters (override any via CLI flags — run `python train.py -h`):

| Hyperparameter        | Default             |
|------------------------|---------------------|
| Pretrained encoder     | `bert-base-uncased` |
| Max sequence length    | 128                 |
| BiLSTM hidden size     | 256 (x2 for bidirectional) |
| BiLSTM layers          | 1                   |
| Dropout                | 0.3                 |
| Batch size             | 16                  |
| Epochs                 | 4                   |
| BERT learning rate     | 2e-5 (AdamW)        |
| Head learning rate     | 1e-3 (AdamW)        |
| LR schedule            | linear warmup (10%) + linear decay |
| Gradient clipping      | max norm 1.0        |
| Train/val/test split   | 80% / 10% / 10% (stratified) |

Training prints per-epoch loss/accuracy/macro-F1 for train and validation,
saves the best checkpoint (by validation macro-F1) to `checkpoints/`, and
finishes with a full `sklearn.classification_report` on the held-out test
split.

## Predict

```bash
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
