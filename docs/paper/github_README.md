# Classification of Marine Incident Reports Using a BERT–BiLSTM–Attention Framework with LLM-Based Data Augmentation

Code, results and data for the paper

> M. M. Arab, H. Mashayekhi, M. Khodabakhsh, "Classification of Marine Incident Reports Using a BERT–BiLSTM–Attention Framework with LLM-Based Data Augmentation," *(venue to be added after acceptance)*.

Faculty of Computer Engineering, Shahrood University of Technology, Shahrood, Iran.
Corresponding author: Hoda Mashayekhi (hmashayekhi@shahroodut.ac.ir).

## Overview

The most safety-critical marine incident types (fire, flooding, capsizing) are also the rarest, so a classifier tuned for accuracy can largely ignore them. The proposed framework combines

- a fine-tuned **BERT** encoder (`bert-base-uncased`) for the maritime vocabulary in context,
- a **BiLSTM** (128 units per direction) that follows the sequence of events in a report,
- an **attention** layer that pools the BiLSTM states and shows which words a decision rests on,
- an **imbalance-handling module**: class-weighted losses (`wce`, `focal`, `cb`), oversampling, and paraphrasing of rare-class training reports by an LLM (`Qwen2.5-1.5B-Instruct`) with a label-consistency filter (a TF-IDF + SVM trained on the real reports must assign each rewrite to the class of its source).

All BERT-based models share one training recipe and are trained with two seeds. Statistical significance is tested with the exact McNemar test and a paired bootstrap.

## Main results

Test set: 865 real reports. Mean ± standard deviation over seeds 1 and 2. Critical recall is the mean recall of Fire, Flooding and Capsizing.

| Configuration | Macro-F1 | Accuracy | Critical recall |
|---|---|---|---|
| TF-IDF + SVM | 0.885 | 0.914 | 0.827 |
| BERT | 0.898 ± 0.007 | 0.927 ± 0.005 | 0.873 ± 0.008 |
| BERT + BiLSTM (architecture of Zhao et al., 2025) | 0.901 ± 0.004 | 0.928 ± 0.001 | 0.851 ± 0.040 |
| BERT + BiLSTM + attention (Att) | 0.905 ± 0.004 | 0.931 ± 0.002 | 0.845 ± 0.016 |
| Att + wce | 0.914 ± 0.006 | 0.935 ± 0.003 | 0.868 ± 0.016 |
| **Att + wce + LLM augmentation (proposed)** | **0.915 ± 0.002** | **0.935 ± 0.002** | 0.868 ± 0.016 |

The proposed framework is significantly better than the SVM baseline (bootstrap p = 0.005). The differences between the BERT-based configurations are not statistically significant on this test set. The full table, with all losses, oversampling and larger BiLSTMs, is in `maib-thesis/reports/main_table.md`.

## Data

- Source: the public MAIB incident dataset on Hugging Face, [`baker-street/maib-incident-reports-5K`](https://huggingface.co/datasets/baker-street/maib-incident-reports-5K) (5,768 short narratives of the UK Marine Accident Investigation Branch, labelled with the main event type). It is downloaded automatically by `step1_prepare_data.py`.
- Cleaning: one exact duplicate and two classes with fewer than 20 reports (Hull Failure, Non-accidental Event) are removed. 5,762 reports in nine classes remain.
- Split: one stratified 70/15/15 split (`SPLIT_SEED = 42`), shared by all models.
- `maib-thesis/data/augmented.csv`: the 1,068 LLM rewrites used in the paper. They are included because LLM generation cannot be reproduced exactly. Only training reports were rewritten; validation and test sets contain real reports only.

## Repository structure

```
maib-thesis/
  config.py                 all settings (models, learning rates, seeds, experiment list)
  step1_prepare_data.py     download, clean, split
  step2_tfidf_svm.py        TF-IDF + SVM baseline (plain, oversampled, augmented)
  models.py                 bert, bert_bilstm, bert_bilstm_att
  losses.py                 ce, wce, focal, cb
  step3_train.py            train and evaluate one configuration
  step4_run_experiments.py  run all configurations
  step5_report.py           result tables, per-class recall, confusion matrix
  step6_attention.py        attention analysis
  step7_augment.py          LLM paraphrasing of rare-class reports with the label-consistency filter
  step8_significance.py     McNemar test and paired bootstrap on the saved predictions
  common.py                 data loading and metrics
  MAIB_Pipeline.ipynb       all steps in one notebook
  results/                  one JSON per run: settings, training history, test predictions
  reports/                  tables and figures used in the paper
```

## Installation

Python 3.10–3.13. A CUDA GPU is recommended; the paper's runs used a GTX 1650 Ti (4 GB).

With [uv](https://docs.astral.sh/uv/):

```bash
cd maib-thesis
uv sync
```

Then prefix each command below with `uv run`. Without uv:

```bash
cd maib-thesis
pip install -r requirements.txt
```

## Reproducing the paper

The tables and significance tests can be rebuilt from the saved runs in `results/` without any training:

```bash
python step1_prepare_data.py
python step5_report.py
python step8_significance.py
```

To retrain everything (more than 30 GPU hours on a GTX 1650 Ti):

```bash
python step1_prepare_data.py                                # data (seconds)
python step2_tfidf_svm.py                                   # SVM baselines (seconds)
python step4_run_experiments.py --seeds 1 2                 # architectures and losses
python step4_run_experiments.py --oversample --seeds 1 2    # oversampling
python step7_augment.py                                     # LLM paraphrasing (downloads about 3 GB once)
python step2_tfidf_svm.py                                   # SVM on augmented data
python step4_run_experiments.py --augment --seeds 1 2       # LLM augmentation
python step4_run_experiments.py --bilstm-sizes --seeds 1 2  # larger BiLSTMs
python step5_report.py
python step6_attention.py
python step8_significance.py
```

Interrupted runs can be restarted with the same command; runs whose result file exists are skipped.

## Citation

```bibtex
@inproceedings{arab2026marine,
  author    = {Arab, Mohammad Mahdi and Mashayekhi, Hoda and Khodabakhsh, Maryam},
  title     = {Classification of Marine Incident Reports Using a {BERT}--{BiLSTM}--Attention Framework with {LLM}-Based Data Augmentation},
  booktitle = {(to be added after acceptance)},
  year      = {2026}
}
```

## License

To be added. Until a license file is present, the code is not licensed for reuse. The MAIB dataset and Qwen2.5 keep their own licenses.
