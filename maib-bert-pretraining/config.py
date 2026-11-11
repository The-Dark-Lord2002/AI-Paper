"""
All settings of the domain-adaptation experiment, in one place.

The experiment has two stages:
  1. UNSUPERVISED: continue the pre-training of BERT with masked language modelling (MLM) on maritime texts
     that are NOT part of the labelled dataset (no labels needed)                     steps 2 and 3
  2. SUPERVISED: fine-tune the adapted BERT on the labelled MAIB dataset with exactly the recipe of the
     paper, and compare it with the original BERT                                      steps 4 to 6
"""

# ---------------------------------------------------------------- labelled dataset (stage 2)
# Identical to the paper (maib-thesis/config.py), so the results of both projects can be compared.
DATASET = "baker-street/maib-incident-reports-5K"   # Hugging Face dataset id
DATASET_FILE = "maib-incident-reports-dataset.jsonl"  # its one data file
SPLIT_SEED = 42                # the train/val/test split is made ONCE with this seed
MIN_REPORTS_PER_CLASS = 20     # classes with fewer reports are dropped

DATA_DIR = "data"              # step 1 writes train.csv / val.csv / test.csv here
RESULTS_DIR = "results"        # one JSON file per fine-tuned model
REPORTS_DIR = "reports"        # summaries and comparison tables

CRITICAL_CLASSES = ["Fire / Explosion", "Capsizing / Listing", "Flooding / Foundering"]

# Test reports per class in the paper. Step 1 checks that it rebuilt exactly this split.
EXPECTED_TEST_COUNTS = {
    "Accident to person(s)": 250, "Damage / Loss Of Equipment": 206, "Loss Of Control": 124,
    "Grounding / Stranding": 84, "Contact": 61, "Collision": 49, "Fire / Explosion": 48,
    "Flooding / Foundering": 28, "Capsizing / Listing": 15,
}

# ---------------------------------------------------------------- unlabelled corpus (stage 1, step 2)
SOURCES_DIR = "corpus_sources" # put the files of the other dataset here: .csv / .xlsx / .txt / .pdf
CORPUS_DIR = "corpus"          # step 2 writes the cleaned texts here, one text per line

# Columns of a .csv / .xlsx whose text is used, joined in this order, as in the labelled dataset
# ("Short Description" + "Description" of the MAIB data portal). Matched ignoring case, spaces and "_".
TEXT_COLUMNS = ["short_description", "description"]
MIN_WORDS = 5                  # shorter texts are dropped
CHUNK_WORDS = 120              # longer texts (e.g. PDF reports) are cut into pieces of about this many words

# A corpus text is removed if it is a labelled report or a near-copy of one (see step 2).
OVERLAP_NGRAM = 8              # compare word 8-grams
OVERLAP_THRESHOLD = 0.5        # removed if >= 50% of the 8-grams of either text are shared
MLM_VAL_FRACTION = 0.05        # share of the corpus held out to measure the MLM loss

# ---------------------------------------------------------------- MLM pre-training (stage 1, step 3)
BASE_MODEL = "bert-base-uncased"   # the starting point, the same BERT as in the paper
MODELS_DIR = "models"              # adapted models are saved here (about 440 MB each)

MLM_PROBABILITY = 0.15         # share of word-pieces hidden in every text (Devlin et al., 2019)
MLM_MAX_TOKENS = 192           # same length limit as the fine-tuning
MLM_BATCH_SIZE = 8             # fits a 4 GB GPU
MLM_EFFECTIVE_BATCH = 32       # 32 // 8 = 4 accumulation steps
MLM_LR = 5e-5
MLM_WEIGHT_DECAY = 0.01
MLM_WARMUP_FRACTION = 0.06
# How long stage 1 runs. The number of epochs is NOT chosen by hand: training continues until the MLM loss on the
# validation reports has not improved for MLM_PATIENCE epochs (early stopping), and the best epoch is kept.
# MLM_EPOCHS is only a ceiling, set high enough that early stopping, not the ceiling, ends the run.
# If a run still ends at the ceiling, step 3 says so ("not converged") and the ceiling should be raised.
#   external            ~69,000 texts: one epoch is long (~6 h on a GTX 1650 Ti) and BERT sees a lot per epoch,
#                       so a few epochs are enough and a patience of 2 avoids wasting ~12 h after the best epoch
#   task                 ~4,000 texts: one epoch is short (~25 min), so many passes are needed and affordable
#   external-then-task   DAPT first, then TAPT starting from the DAPT model (the best order, Gururangan et al.)
#   external+task        both corpora mixed in one run (optional)
MLM_EPOCHS = {"external": 8, "task": 50, "external-then-task": 50, "external+task": 8}
MLM_PATIENCE = {"external": 2, "task": 3, "external-then-task": 3, "external+task": 2}
MLM_SEED = 42

# ---------------------------------------------------------------- fine-tuning (stage 2, step 4)
# Identical to the paper (maib-thesis/config.py); only the encoder changes.
MAX_TOKENS = 192
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
ATTENTION_DIM = 128

EFFECTIVE_BATCH = 32
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 64

EPOCHS = 5
PATIENCE = 2
LR_BERT = 2e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1

FOCAL_GAMMA = 2.0
CB_BETA = 0.999

# 3 seeds: with 2, the standard deviation is a guess and the significance tests have little power.
# Seeds 1 and 2 of the original BERT are the paper's runs; seed 3 is trained here.
SEEDS = [1, 2, 3]

# Configurations fine-tuned with every encoder (step 5): plain BERT and the paper's main model.
FINETUNE_CONFIGS = [
    ("bert",            "ce"),
    ("bert_bilstm_att", "wce"),
]
