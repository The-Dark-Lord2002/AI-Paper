# AI-Paper

Implementations of deep-learning architectures from published papers, applied
to real datasets.

## Projects

### [bert-bilstm-marine-accidents](bert-bilstm-marine-accidents/)

A BERT + BiLSTM text classifier, based on the architecture and hyperparameters
from Zhao et al. (2025), *"Causation Analysis of Marine Traffic Accidents
Using Deep Learning Approaches: A Case Study from China's Coasts"* (Systems,
13(4):284) — full text: [`Causation_Analysis_of_Marine_Traffic_Accidents_Usi.pdf`](Causation_Analysis_of_Marine_Traffic_Accidents_Usi.pdf) —
fine-tuned on the [`baker-street/maib-incident-reports-5K`](https://huggingface.co/datasets/baker-street/maib-incident-reports-5K)
dataset to classify UK MAIB marine incident narratives by incident type
(Collision, Grounding/Stranding, Fire/Explosion, Accident to person(s),
Damage/Loss of Equipment, ...).

Runs entirely from one Jupyter notebook
(`Thesis_Progress_Full_Pipeline.ipynb`) covering all three parts of the
paper's method: plain-BERT baseline, BERT+BiLSTM, the Table-10-style
comparison, and Apriori causal-factor mining. See that project's own
[README](bert-bilstm-marine-accidents/README.md) for setup, usage,
architecture details, and exactly which hyperparameters are confirmed from
the paper's Table 9 versus this project's own dataset-driven choices.
