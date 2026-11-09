#!/usr/bin/env bash
# Pack the project for Kaggle / Colab as maib-work.zip (next to the project folder).
# Included: code, data/ (the paper's split), corpus/ (the cleaned texts, so nothing is downloaded or rebuilt),
#           reports/, results/ (the original-BERT results copied from the paper) and any unfinished *.resume.pt.
# Left out: .venv/ (Kaggle and Colab have their own Python), corpus_sources/ (1.5 GB of PDFs, no longer needed
#           once corpus/ exists), archive/ and caches.
# Finished adapted models in models/ are included, so a run started on the laptop continues in the cloud;
# pass --no-models to leave them out (smaller upload; they are then trained again in the cloud).
#
#   bash cloud/make_upload_zip.sh               # run inside maib-bert-pretraining/
#   bash cloud/make_upload_zip.sh --no-models
set -euo pipefail
cd "$(dirname "$0")/.."                                    # the project folder
[ -f corpus/task_train.txt ] || { echo "corpus/ is missing: run step2_build_corpus.py (or the pipeline) first"; exit 1; }
name=$(basename "$PWD")
excludes=("$name/.venv/*" "$name/corpus_sources/*" "$name/archive/*" "*/__pycache__/*" "*.resume.pt.tmp" "$name/*.log")
if [ "${1:-}" = "--no-models" ]; then excludes+=("$name/models/*"); fi
cd ..
rm -f maib-work.zip
zip -r -q maib-work.zip "$name" -x "${excludes[@]}"
echo "Wrote $(pwd)/maib-work.zip ($(du -h maib-work.zip | cut -f1))"
