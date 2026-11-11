"""
The whole experiment with one command, in the right order. Each step is the script of the same name and can
also be run on its own (README.md). Finished work is skipped and an interrupted run continues after its last
finished epoch, so after any stop (Ctrl+C, power cut, closing the lid) just run the SAME command again.

  stage 1 (self-supervised): BERT reads unlabelled maritime text with masked words   -> models/bert-maib-*
     task                the TRAINING reports of the labelled dataset (TAPT; no download needed)
     external            older MAIB investigation reports and Safety Digests, up to 2020 (DAPT)
     external-then-task  DAPT first, then TAPT starting from the DAPT model
  stage 2 (supervised): every BERT is fine-tuned on the labelled TRAINING reports and tested ONCE on the
     labelled TEST reports, exactly as in the paper                                  -> results/
  comparison: original BERT against the adapted ones, with significance tests       -> reports/comparison.md

Order: the original BERT first, then for each adapted BERT its stage 1 and right after it its stage 2, and the
comparison table is rebuilt after each one. So reports/comparison.md already shows the finished encoders while
the next one is still training.

  python run_pipeline.py                         # the full, principled experiment (README, section 5)
  python run_pipeline.py --corpora task          # only TAPT: no other dataset needed
  python run_pipeline.py --skip-base             # do not train the original BERT (copied from the paper)
  python run_pipeline.py --max-texts 20000 --mlm-epochs 3      # a shorter, NOT converged stage 1 (a quick test)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import config

parser = argparse.ArgumentParser()
parser.add_argument("--corpora", nargs="+", choices=list(config.MLM_EPOCHS),
                    default=["task", "external", "external-then-task"],
                    help="stage-1 corpora, one adapted BERT each (default: task, external, external-then-task)")
parser.add_argument("--skip-base", action="store_true", help="do not fine-tune the original BERT (copied from the paper)")
parser.add_argument("--download", action="store_true", help="run step 0 even if corpus_sources/ already has files")
parser.add_argument("--rebuild-corpus", action="store_true", help="run step 2 even if the corpus is up to date")
parser.add_argument("--since", type=int, default=1990, help="step 0: first year to download")
parser.add_argument("--max-reports", type=int, help="step 0: at most this many investigation reports")
parser.add_argument("--max-texts", type=int, help="step 3: at most this many texts of the other dataset "
                                                  "(default: all of them)")
parser.add_argument("--mlm-epochs", type=int, help="step 3: maximum epochs for every corpus "
                                                   "(default: config.MLM_EPOCHS, with early stopping)")
args = parser.parse_args()

START = time.time()


def run(script, *script_args):
    """Run one step as its own process; stop the pipeline if it fails."""
    command = [sys.executable, script, *map(str, script_args)]
    print(f"\n{'=' * 78}\n>>> {' '.join(command[1:])}    ({(time.time() - START) / 3600:.1f} h so far)\n{'=' * 78}",
          flush=True)
    subprocess.run(command, check=True)


def source_files():
    return [p for p in Path(config.SOURCES_DIR).rglob("*") if p.suffix.lower() in (".pdf", ".csv", ".xlsx", ".txt")]


def model_dir(corpus):
    return Path(config.MODELS_DIR) / ("bert-maib-" + corpus.replace("+", "-"))


def corpus_up_to_date(needs_external):
    """Step 2 takes ~20 min on the full download; skip it if its output is newer than every input file."""
    summary = Path(config.REPORTS_DIR) / "corpus_summary.json"
    outputs = [Path(config.CORPUS_DIR) / "task_train.txt"]
    if needs_external:
        outputs += [Path(config.CORPUS_DIR) / "external_train.txt", Path(config.CORPUS_DIR) / "external_val.txt"]
    if not summary.exists() or not all(p.exists() for p in outputs):
        return False
    built = summary.stat().st_mtime
    inputs = source_files() + [Path(config.DATA_DIR) / "train.csv", Path("config.py"), Path("step2_build_corpus.py")]
    return all(p.stat().st_mtime < built for p in inputs if p.exists())


def main():
    # external-then-task starts from the DAPT model, so 'external' must be trained before it
    corpora = sorted(args.corpora, key=lambda c: c == "external-then-task")
    if "external-then-task" in corpora and "external" not in corpora and not (model_dir("external") / "config.json").exists():
        corpora.insert(0, "external")
        print("[note] external-then-task needs the DAPT model: 'external' is trained first")
    needs_external = any("external" in c for c in corpora)

    encoders = ([] if args.skip_base else [config.BASE_MODEL]) + [str(model_dir(c)) for c in corpora]
    runs = len(encoders) * len(config.FINETUNE_CONFIGS) * len(config.SEEDS)
    print(f"Plan: stage 1 on {corpora} (each until early stopping); stage 2: up to {runs} fine-tuning runs of "
          f"~75 min on a GTX 1650 Ti (finished runs are skipped). Seeds {config.SEEDS}.")

    # ------------------------------------------------------------------ data
    if needs_external and (args.download or not source_files()):
        extra = ["--since", args.since] + (["--max-reports", args.max_reports] if args.max_reports else [])
        run("step0_download_sources.py", *extra)
    if not (Path(config.DATA_DIR) / "test.csv").exists():
        run("step1_prepare_data.py")
    summary = json.loads((Path(config.REPORTS_DIR) / "data_summary.json").read_text())
    if not summary.get("same_test_set_as_paper"):
        raise SystemExit("The test set is not the paper's (see step 1). Stopped: the comparison would not be fair.")
    if args.rebuild_corpus or not corpus_up_to_date(needs_external):
        run("step2_build_corpus.py")
    else:
        print("\n[skip] step 2: the corpus is up to date (use --rebuild-corpus to build it again)")
    if needs_external and not (Path(config.CORPUS_DIR) / "external_train.txt").read_text().strip():
        raise SystemExit("No text of the other dataset is left after step 2. Check corpus_sources/ and "
                         "reports/corpus_summary.json, or run with --corpora task.")

    # ------------------------------------------------------------------ original BERT: stage 2 only
    if not args.skip_base:
        run("step5_run_all.py", "--encoders", config.BASE_MODEL)
        run("step6_compare.py")

    # ------------------------------------------------------------------ every adapted BERT: stage 1, then stage 2
    for corpus in corpora:
        if (model_dir(corpus) / "config.json").exists():
            print(f"\n[skip] stage 1 on '{corpus}': {model_dir(corpus)} already exists "
                  f"(move it away to train it again, see archive_previous_run.py)")
        else:
            extra = (["--epochs", args.mlm_epochs] if args.mlm_epochs else []) + \
                    (["--max-texts", args.max_texts] if args.max_texts and "external" in corpus.split("+") else [])
            run("step3_pretrain_mlm.py", "--corpus", corpus, *extra)
        run("step5_run_all.py", "--encoders", model_dir(corpus), "--skip-base")
        run("step6_compare.py")                     # the table grows after every finished encoder

    print(f"\nDone in {(time.time() - START) / 3600:.1f} h. Results: {config.REPORTS_DIR}/comparison.md")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(f"\nStopped: a step failed (exit code {error.returncode}); its messages are above. "
                 "Fix the cause and run the same command again: finished work is skipped.")
