"""
All settings of the project, in one place.
Every step script reads its settings from here, so changing a number here changes it everywhere.
"""

# ---------------------------------------------------------------- data (proposal, step 1)
DATASET = "baker-street/maib-incident-reports-5K"   # Hugging Face dataset id
SPLIT_SEED = 42                # the train/val/test split is made ONCE with this seed and reused by every model
MIN_REPORTS_PER_CLASS = 20     # classes with fewer reports are dropped (too few to learn from or to test on)

DATA_DIR = "data"              # step 1 writes train.csv / val.csv / test.csv here
RESULTS_DIR = "results"        # one JSON file per trained model
REPORTS_DIR = "reports"        # tables and figures for the thesis

# Safety-critical rare classes named in the proposal. Their recall is reported separately.
CRITICAL_CLASSES = ["Fire / Explosion", "Capsizing / Listing", "Flooding / Foundering"]

# ---------------------------------------------------------------- BERT models (proposal, steps 2-4)
BERT_NAME = "bert-base-uncased"
MAX_TOKENS = 192               # reports are short (median ~60 words); about 2% are longer and get cut
LSTM_HIDDEN = 128              # BiLSTM units per direction, as in the reference paper
LSTM_LAYERS = 1                # stacked BiLSTM layers, as in the reference paper
ATTENTION_DIM = 128            # size of the hidden layer inside the attention scorer

EFFECTIVE_BATCH = 32           # reports per weight update, as in the reference paper
BATCH_SIZE = 8                 # reports per forward pass; fits a 4 GB GPU. 32 // 8 = 4 accumulation steps
EVAL_BATCH_SIZE = 64

EPOCHS = 5                     # maximum passes over the training set
PATIENCE = 2                   # stop early if validation macro-F1 has not improved for this many epochs
LR_BERT = 2e-5                 # learning rate for the pre-trained BERT weights (small: only adjust them)
LR_HEAD = 1e-3                 # learning rate for the new layers on top (BiLSTM, attention, classifier)
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1          # first 10% of updates: learning rate grows from 0 to its full value

FOCAL_GAMMA = 2.0              # focal loss: how strongly easy examples are down-weighted (Lin et al. 2017)
CB_BETA = 0.999                # class-balanced loss: "effective number" parameter (Cui et al. 2019)

SEEDS = [1, 2, 3]              # every configuration is trained 3 times; we report mean +- std

# ---------------------------------------------------------------- the experiment table
# (model, loss)                       proposal step
EXPERIMENTS = [
    ("bert",            "ce"),        # 2b  BERT with a simple classifier
    ("bert_bilstm",     "ce"),        # 2c  reference paper: BERT + BiLSTM
    ("bert_bilstm_att", "ce"),        # 3   proposed: BERT + BiLSTM + attention
    ("bert_bilstm_att", "wce"),       # 4a  + inverse-frequency weighted cross-entropy
    ("bert_bilstm_att", "focal"),     # 4b  + focal loss
    ("bert_bilstm_att", "cb"),        # 4c  + class-balanced (effective number) weights
]

# The configuration the thesis proposes. Step 5 draws its confusion matrix. Differences smaller than the
# seed-to-seed noise should not decide which model a figure shows, so this is fixed, not "the best mean".
MAIN_CONFIG = "bert_bilstm_att + wce"

# Oversampling (supervisor's request): instead of changing the loss, show the model the rare classes more often.
# Training reports are drawn with probability 1 / (size of their class), so every class is seen about equally
# often. The epoch length stays the same (same number of updates as every other run), only the mix changes.
# Combined with plain "ce" on purpose: oversampling AND a weighted loss would correct the imbalance twice.
# Run with:  python step4_run_experiments.py --oversample
OVERSAMPLE_EXPERIMENTS = [
    ("bert_bilstm_att", "ce"),        # compare with bert_bilstm_att + ce (nothing) and + wce (weighted loss)
]

# Bigger BiLSTM (supervisor's question): more units per direction and/or more stacked layers.
# 384 units per direction give 2 x 384 = 768 numbers per word, the same size as BERT's output.
# Each (units, layers) pair is trained as bert_bilstm_att + BILSTM_SIZE_LOSS and compared with the
# default (128, 1) of the main table.
# Run with:  python step4_run_experiments.py --bilstm-sizes
BILSTM_SIZES = [(384, 1), (128, 2), (384, 2)]
BILSTM_SIZE_LOSS = "wce"       # the best loss of the main table

# Data augmentation with a language model (supervisor's request, step 7).
# Copying reports (oversampling) adds no new information. Instead, a small instruction-tuned language model
# rewrites real TRAINING reports of the rare classes in new words. Validation and test stay 100% real.
AUGMENT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"   # Apache-2.0, fits a 4 GB GPU in float16
AUGMENT_TARGET = 400           # every class with fewer training reports is filled up to this many
AUGMENT_BATCH = 8              # reports rewritten at the same time on the GPU
AUGMENT_LABEL_CHECK = True     # keep a rewrite only if a TF-IDF + SVM trained on the real reports still
                               # puts it in the class of its original (LAMBADA-style filter)
# Trained on real + rewritten reports. Run with:  python step4_run_experiments.py --augment
AUGMENT_EXPERIMENTS = [
    ("bert_bilstm_att", "ce"),        # compare with + ce (nothing), + ce [oversample] (copies) and + wce
    ("bert_bilstm_att", "wce"),       # does augmentation still help on top of the best loss?
]

# Optional: the reference paper's exact learning rate (1e-6 for every layer). Needs many more epochs.
# Run with:  python step4_run_experiments.py --paper-settings
PAPER_SETTINGS = dict(lr_bert=1e-6, lr_head=1e-6, epochs=30, patience=5, tag="paper_lr")
