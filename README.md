# AI-Paper

Implementations of deep-learning architectures from published papers, applied
to real datasets.

## Projects

### [bert-bilstm-marine-accidents](bert-bilstm-marine-accidents/)

A BERT + BiLSTM text classifier, based on the architecture from Xu et al.
(2025), *"Causation Analysis of Marine Traffic Accidents Using Deep Learning
Approaches: A Case Study from China's Coasts"* (Systems, 13(4):284), fine-tuned
on the [`baker-street/maib-incident-reports-5K`](https://huggingface.co/datasets/baker-street/maib-incident-reports-5K)
dataset to classify UK MAIB marine incident narratives by incident type
(Collision, Grounding/Stranding, Fire/Explosion, Accident to person(s),
Damage/Loss of Equipment, ...).

Includes CLI training/inference scripts (`train.py`, `predict.py`) and an
equivalent Jupyter notebook (`BERT_BiLSTM_Marine_Incident_Classification.ipynb`).
See that project's own [README](bert-bilstm-marine-accidents/README.md) for
setup, usage, architecture details, and a caveat on which hyperparameters are
confirmed from the paper versus standard defaults.
