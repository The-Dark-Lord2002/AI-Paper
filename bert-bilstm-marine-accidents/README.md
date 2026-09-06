# BERT-BiLSTM Marine Incident Classifier

Implementation of the full three-part method from:

> Zhao, Z.; Liu, X.; Feng, L.; Grifoll, M.; Feng, H. (2025). *Causation
> Analysis of Marine Traffic Accidents Using Deep Learning Approaches: A Case
> Study from China's Coasts.* Systems, 13(4), 284.
> https://doi.org/10.3390/systems13040284
> (full text: [`../Causation_Analysis_of_Marine_Traffic_Accidents_Usi.pdf`](../Causation_Analysis_of_Marine_Traffic_Accidents_Usi.pdf))

applied to the [`baker-street/maib-incident-reports-5K`](https://huggingface.co/datasets/baker-street/maib-incident-reports-5K)
dataset of UK MAIB marine incident narratives.

## What's implemented

| Paper component | What it does | Here |
|---|---|---|
| **BERT** (baseline, Table 9 "BERT" column) | Plain BERT + dense head classifier, no BiLSTM | `baseline_model.py` + `train_bert_baseline.py` |
| **BERT + BiLSTM** (Table 9 "BERT + BiLSTM" column) | The paper's proposed model | `model.py` + `train.py` |
| **Comparison** (the paper's own Table 10 ablation) | Does BiLSTM actually help? | `compare_models.py` |
| **Apriori** (Section 3.2/3.3, Section 4.4) | Mines which causal factors co-occur | `causal_factors.py` + `apriori_analysis.py` |

All three of the paper's components now have a runnable counterpart. The
one place this project necessarily deviates from the paper is *what feeds*
Apriori — see [Apriori](#apriori-mining-causal-factor-associations) below
for exactly why and what that means for the results.

## Fidelity to the paper

### BERT + BiLSTM (`model.py`, `train.py`)

Taken directly from the paper's Table 9 ("Model parameter settings",
"BERT + BiLSTM" column) and Section 4.1 / Figure 7 for the data split and
epoch count:

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

### Plain BERT baseline (`baseline_model.py`, `train_bert_baseline.py`)

Taken from Table 9's first column — the ablation the paper itself trains
to show what BiLSTM adds (Table 10: 88.7% vs. 89.8% on the paper's data):

| Setting | Paper value | Source |
|---|---|---|
| Hidden layer width | 512 | Table 9 ("BERT layer 512") |
| Learning rate | 1e-6 (AdamW) | Table 9 |
| L2 weight decay | 0.05 | Table 9 |
| L1 regularization | 1e-8 | Table 9 |
| Gradient clipping (max norm) | 2.35 | Table 9 |
| Batch size | 32 | Table 9 |
| Dropout layers | 3 | Table 9 |
| Activation | GELU | Table 9 |
| Optimizer | AdamW | Table 9 |

Table 9 names the hidden width (512) and dropout-layer count (3) but not
their exact arrangement; `baseline_model.py`'s docstring explains the
(reasonable, but not paper-specified) 2-dense-layer head this project built
around those two numbers.

Everything in both tables is also a CLI flag on the matching training
script, in case you want to deviate further.

**What's *not* reproduced, and why:**

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
- **What Apriori mines from is reconstructed, not the paper's own labels** —
  see the dedicated section below.

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
The plain-BERT baseline (`baseline_model.py`) replaces everything from
"Dropout" onward with a 2-layer, 512-unit GELU MLP head straight on BERT's
pooled `[CLS]` output — see that file's docstring.

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

## Train BERT + BiLSTM

```bash
uv run train.py
# or, without uv:
python train.py
```

Default hyperparameters (override any via CLI flags — run `python train.py -h`;
see the fidelity table above for where each one comes from):

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

## Train the plain-BERT baseline

```bash
uv run train_bert_baseline.py
# or, without uv:
python train_bert_baseline.py
```

Same dataset, same split (same `--seed`), same training loop as `train.py`
— just without the BiLSTM stage, and with Table 9's "BERT" column
hyperparameters as defaults (hidden width 512, dropout×3, L1 1e-8, gradient
clip 2.35, GELU). Saves to `checkpoints_bert_baseline/` by default so it
doesn't collide with `train.py`'s `checkpoints/`.

## Compare BERT vs. BERT + BiLSTM

Once both are trained:

```bash
uv run compare_models.py
```

Reloads both checkpoints, evaluates both on the *same* held-out test split
(reconstructed from `--seed`/`--val-size`/`--test-size`, which default to
matching both training scripts), and prints/saves a side-by-side accuracy /
macro-F1 / weighted-F1 / per-class-F1 comparison to `comparison_report.md`
— the same kind of ablation the paper itself runs in Table 10, just on this
project's dataset instead of the paper's.

### Running out of GPU memory?

The paper trained on an RTX 4060 Ti (8GB VRAM) at batch size 32. On a
smaller GPU, `RuntimeError: CUDA out of memory` at that batch size is
expected, not a bug. Two independent knobs help, on **both** training
scripts:

- **Mixed precision is on by default on CUDA** (roughly halves activation
  memory); disable it with `--no-amp` only if you need exact fp32 training.
- **Gradient accumulation** lets you shrink the per-step batch while keeping
  the same *effective* batch size the paper used — e.g. on a 4GB GPU:

  ```bash
  uv run train.py --batch-size 4 --grad-accum-steps 8
  uv run train_bert_baseline.py --batch-size 4 --grad-accum-steps 8
  ```

  This still averages gradients over 32 examples before each optimizer step
  (matching Table 9), it just never holds more than 4 examples' activations
  in memory at once. Drop `--batch-size` further (and raise
  `--grad-accum-steps` to match) if it still doesn't fit; also try
  `--max-length 64` to shrink attention memory further.

## Predict

```bash
uv run predict.py --checkpoint-dir checkpoints \
    "A bulk carrier ran aground after losing steering control in heavy weather."
# or, without uv:
python predict.py --checkpoint-dir checkpoints \
    "A bulk carrier ran aground after losing steering control in heavy weather."
```

## Apriori: mining causal-factor associations

The paper's third component (Section 3.2 "Apriori Algorithm", Section 4.4
"Apriori Association Results") doesn't classify anything — it takes the
causal factors a report was tagged with and mines which factors tend to
occur *together*, using the standard association-rule measures:

```
Support(X ∪ Y)  = P(a report has both X and Y)
Confidence(X⇒Y) = Support(X ∪ Y) / Support(X)
Lift(X⇒Y)       = Confidence(X⇒Y) / Support(Y)
```

**The catch this project has to work around:** the paper mines its own
hand-labeled, *multi-label* causal-factor tags (a report can be tagged
"Improper operation" **and** "Negligent lookout" **and** "Rough sea state").
The MAIB dataset here only has a single *incident-type* label per report
(what happened, not why) — there's nothing to mine co-occurrence between.

So `causal_factors.py` heuristically re-derives multi-label causal-factor
tags per report by **keyword/phrase matching against the paper's own 32
causal-factor category names** (its Table 10 lists all 32 verbatim — e.g.
"Negligent lookout", "Work through fatigue", "Equipment failure",
"Mismanagement by the shipowner or company"). A report matching phrases like
*"failed to keep a proper lookout"* gets tagged `Negligent lookout`; one
mentioning *"long hours"* / *"exhausted"* gets tagged `Work through
fatigue`; and so on — see the `CAUSAL_FACTOR_KEYWORDS` dict in that file for
all 32 categories' keyword lists.

**This is not the paper's method — be upfront about that.** The paper's
tags came from a human expert reading each report; keyword matching will
miss paraphrased mentions (false negatives) and occasionally fire on
incidental word choice (false positives). Treat `apriori_analysis.py`'s
output as a demonstration that the pipeline works end-to-end and a
plausible-looking set of associations, not as a validated re-annotation of
this dataset or a number to quote as reproducing the paper's Tables 5–8.

Run it with:

```bash
uv run apriori_analysis.py
# or narrow/widen the rule set:
uv run apriori_analysis.py --min-support 0.01 --min-confidence 0.1 --output-csv rules.csv
```

Defaults (`--min-support 0.008 --min-confidence 0.15`) match the paper's own
general-rule thresholds (Section 4.4); pass `--min-support 0.01
--min-confidence 0.1` to match its looser causal-chain thresholds instead.
Output: how many reports got tagged with at least one factor, the frequent
itemsets found, and the top rules sorted by confidence and by lift (mirroring
the paper's Tables 5–8), optionally saved to CSV.

## Notebooks

`BERT_BiLSTM_Marine_Incident_Classification.ipynb` walks through the
BERT + BiLSTM pipeline end-to-end in a single notebook (data
loading/splitting, model definition, training loop, test evaluation,
checkpoint saving, and inference) — handy for running on Colab/Kaggle/Jupyter
with a GPU instead of the CLI scripts. It doesn't cover the plain-BERT
baseline, comparison, or Apriori stage.

`Thesis_Progress_Full_Pipeline.ipynb` covers all four components in one
notebook — plain BERT baseline, BERT + BiLSTM, the Table-10-style comparison,
and Apriori — by importing this project's own modules (`dataset.py`,
`model.py`, `baseline_model.py`, `causal_factors.py`, `training_utils.py`),
so run it from inside this directory. It has a `FAST_DEMO` config flag
(5 epochs, batch size 8 with gradient accumulation to an effective batch of
32 — sized for a 4GB laptop GPU) for a quick, genuine end-to-end check;
flip it off for the paper-faithful 20-epoch run once you have time. Ends
with one summary table covering all four components and a
`thesis_progress_summary.md` export.

## Files

- `dataset.py` — loads the dataset from the Hub, cleans text, encodes labels,
  builds stratified train/val/test splits (with a fallback to a random split
  for any class too rare to stratify), and defines the `torch.Dataset`.
- `model.py` — the `BertBiLSTMClassifier` module (the paper's proposed model).
- `baseline_model.py` — the `BertClassifier` module (the paper's plain-BERT
  ablation).
- `training_utils.py` — the shared training/evaluation loop (`run_epoch`) and
  `set_seed`, used by both training scripts.
- `train.py` — trains `BertBiLSTMClassifier`.
- `train_bert_baseline.py` — trains `BertClassifier`.
- `compare_models.py` — evaluates both trained checkpoints on the same test
  split and reports the comparison.
- `causal_factors.py` — keyword-based multi-label causal-factor tagging
  (see the Apriori section above for why this exists and its limits).
- `apriori_analysis.py` — mines association rules from those tags with
  Apriori (via `mlxtend`).
- `predict.py` — inference on new narratives with a saved
  `BertBiLSTMClassifier` checkpoint.
- `BERT_BiLSTM_Marine_Incident_Classification.ipynb` — notebook version of
  the BERT + BiLSTM pipeline.
- `pyproject.toml` / `uv.lock` — project metadata and pinned dependencies for
  [uv](https://docs.astral.sh/uv/); `requirements.txt` covers plain `pip`.
