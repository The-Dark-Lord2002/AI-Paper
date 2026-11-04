"""
The whole experiment with one command, in the right order. Each step is the script of the same name and can
also be run on its own (README.md). Finished work is skipped, so after an interruption just run the command
again.

  stage 1 (self-supervised): BERT reads unlabelled maritime text with masked words   -> models/bert-maib-*
     task      the TRAINING reports of the labelled dataset (TAPT; no download needed)
     external  older MAIB investigation reports and Safety Digests, up to 2020 (DAPT)
  stage 2 (supervised): every BERT is fine-tuned on the labelled TRAINING reports and tested ONCE on the
     labelled TEST reports, exactly as in the paper                                  -> results/
  comparison: original BERT against the adapted ones, with significance tests       -> reports/comparison.md

  uv run python run_pipeline.py                         # everything (downloads if corpus_sources/ is empty)
  uv run python run_pipeline.py --corpora task          # only TAPT: no other dataset needed
  uv run python run_pipeline.py --skip-base             # original-BERT results copied from the paper (README)
  uv run python run_pipeline.py --max-texts 20000 --mlm-epochs 3      # a shorter stage 1 on a big corpus
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import config

parser = argparse.ArgumentParser()
parser.add_argument("--corpora", nargs="+", choices=["task", "external", "external+task"], default=["task", "external"],
                    help="stage-1 corpora, one adapted BERT each (default: task and external)")
parser.add_argument("--skip-base", action="store_true", help="do not fine-tune the original BERT (copied from the paper)")
parser.add_argument("--download", action="store_true", help="run step 0 even if corpus_sources/ already has files")
parser.add_argument("--since", type=int, default=1990, help="step 0: first year to download")
parser.add_argument("--max-reports", type=int, help="step 0: at most this many investigation reports")
parser.add_argument("--max-texts", type=int, help="step 3: at most this many texts of the other dataset")
parser.add_argument("--mlm-epochs", type=int, default=config.MLM_EPOCHS, help="step 3: maximum epochs")
args = parser.parse_args()

START = time.time()


def run(script, *script_args):
    command = [sys.executable, script, *map(str, script_args)]
    print(f"\n{'=' * 78}\n>>> {' '.join(command[1:])}    ({(time.time() - START) / 3600:.1f} h so far)\n{'=' * 78}",
          flush=True)
    subprocess.run(command, check=True)


def has_sources():
    return any(p.suffix.lower() in (".pdf", ".csv", ".xlsx", ".txt") for p in Path(config.SOURCES_DIR).rglob("*"))


def model_dir(corpus):
    return Path(config.MODELS_DIR) / ("bert-maib-" + corpus.replace("+", "-"))


def main():
    needs_external = any("external" in c for c in args.corpora)
    encoders = (0 if args.skip_base else 1) + len(args.corpora)
    runs = encoders * len(config.FINETUNE_CONFIGS) * len(config.SEEDS)
    print(f"Plan: stage 1 on {args.corpora}; stage 2: {runs} fine-tuning runs "
          f"(about {runs * 75 / 60:.0f} h on a GTX 1650 Ti, less on a bigger GPU).")

    # ------------------------------------------------------------------ data
    if needs_external and (args.download or not has_sources()):
        extra = ["--since", args.since] + (["--max-reports", args.max_reports] if args.max_reports else [])
        run("step0_download_sources.py", *extra)
    if not (Path(config.DATA_DIR) / "test.csv").exists():
        run("step1_prepare_data.py")
    summary = json.loads((Path(config.REPORTS_DIR) / "data_summary.json").read_text())
    if not summary.get("same_test_set_as_paper"):
        raise SystemExit("The test set is not the paper's (see step 1). Stopped: the comparison would not be fair.")
    run("step2_build_corpus.py")
    if needs_external and not (Path(config.CORPUS_DIR) / "external_train.txt").read_text().strip():
        raise SystemExit("No text of the other dataset is left after step 2. Check corpus_sources/ and "
                         "reports/corpus_summary.json, or run with --corpora task.")

    # ------------------------------------------------------------------ stage 1: self-supervised
    for corpus in args.corpora:
        if (model_dir(corpus) / "config.json").exists():
            print(f"\n[skip] stage 1 on '{corpus}': {model_dir(corpus)} already exists (delete it to train again)")
            continue
        extra = ["--epochs", args.mlm_epochs] + (["--max-texts", args.max_texts] if args.max_texts else [])
        run("step3_pretrain_mlm.py", "--corpus", corpus, *extra)

    # ------------------------------------------------------------------ stage 2: supervised, and comparison
    run("step5_run_all.py", "--encoders", *[model_dir(c) for c in args.corpora],
        *(["--skip-base"] if args.skip_base else []))
    run("step6_compare.py")
    print(f"\nDone in {(time.time() - START) / 3600:.1f} h. Results: {config.REPORTS_DIR}/comparison.md")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(f"\nStopped: a step failed (exit code {error.returncode}); its messages are above. "
                 "Fix the cause and run the same command again: finished work is skipped.")
