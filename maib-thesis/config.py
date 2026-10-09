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

# Optional: the reference paper's exact learning rate (1e-6 for every layer). Needs many more epochs.
# Run with:  python step4_run_experiments.py --paper-settings
PAPER_SETTINGS = dict(lr_bert=1e-6, lr_head=1e-6, epochs=30, patience=5, tag="paper_lr")
