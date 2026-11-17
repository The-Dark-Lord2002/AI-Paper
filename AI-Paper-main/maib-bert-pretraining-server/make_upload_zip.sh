#!/usr/bin/env bash
# Pack this server project as maib-server.zip (in the repository folder), ready to upload to Kaggle, Colab or a
# rented GPU server. The laptop folder is only READ, never changed.
#
# What goes in, besides the code of this folder:
#   data/  and reports/{data,corpus}_summary.json   from ../maib-bert-pretraining (the paper's split, checked by step 1)
#   corpus/                                          from ../maib-bert-pretraining (the cleaned texts, ~100 MB), so the
#                                                    server neither downloads the 1.5 GB of PDFs nor rebuilds the corpus
#   results/bert__ce__seed{1,2}.json and bert_bilstm_att__wce__seed{1,2}.json
#                                                    from ../maib-thesis/results (the paper's original-BERT runs), so
#                                                    the server only trains seed 3 of the original BERT
# Anything missing is left out with a note; the server then rebuilds it (data and corpus need internet there).
#
#   bash make_upload_zip.sh          # run inside maib-bert-pretraining-server/
set -euo pipefail
cd "$(dirname "$0")"
here=$PWD
name=$(basename "$here")
laptop="$here/../maib-bert-pretraining"
paper="$here/../maib-thesis/results"

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/$name"

# the code of this folder (no local outputs or environments)
tar -C "$here" --exclude=.venv --exclude=__pycache__ --exclude=./models --exclude=./archive --exclude='*.log' \
    --exclude='*.zip' --exclude='*.resume.pt*' -cf - . | tar -C "$stage/$name" -xf -

copy() {  # copy <from> <to-inside-project> <description>
  if [ -e "$1" ]; then mkdir -p "$(dirname "$stage/$name/$2")"; cp -r "$1" "$stage/$name/$2"; echo "  + $3"
  else echo "  - $3: not found ($1), the server will rebuild it"; fi
}
echo "Adding data from the other folders (read only):"
copy "$laptop/data"                         data                             "labelled split (data/)"
copy "$laptop/reports/data_summary.json"    reports/data_summary.json        "split check (data_summary.json)"
copy "$laptop/corpus"                       corpus                           "cleaned corpus (corpus/)"
copy "$laptop/reports/corpus_summary.json"  reports/corpus_summary.json      "corpus summary"
for run in bert__ce__seed1 bert__ce__seed2 bert_bilstm_att__wce__seed1 bert_bilstm_att__wce__seed2; do
  copy "$paper/$run.json" "results/$run.json" "paper result $run"
done

cd "$stage"
rm -f "$here/../maib-server.zip"
zip -r -q "$here/../maib-server.zip" "$name"
echo "Wrote $(cd "$here/.." && pwd)/maib-server.zip ($(du -h "$here/../maib-server.zip" | cut -f1))"
