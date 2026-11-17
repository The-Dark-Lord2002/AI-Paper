"""
STEP 2 - Build the UNLABELLED text corpus for masked language modelling (stage 1).

Put the files of the other dataset in corpus_sources/ (sub-folders are read too):
  *.csv / *.xlsx  e.g. the occurrences table of the MAIB data portal. The columns of config.TEXT_COLUMNS
                  are joined, as in the labelled dataset ("short description" + "description").
                  Any label column is ignored: stage 1 needs no labels.
  *.txt           texts separated by a blank line
  *.pdf           e.g. MAIB Safety Digests or investigation reports (needs: pip install pypdf)

Why the overlap check matters: the labelled dataset is what stage 2 is tested on. If the corpus contained a
test report, BERT would have read that report before being tested on it (even without its label), and the
test score would be too optimistic. So every corpus text that is a labelled report, or a near-copy of one,
is removed. ALL labelled reports (train, val and test) are used for this check, so the corpus is a truly
separate dataset.

Run (after step 1):
  python step2_build_corpus.py
  python step2_build_corpus.py --years 2010-2020           # only rows of these years (needs a date column)
  python step2_build_corpus.py --text-columns "Short Description" "Description"

Output:
  corpus/external_train.txt, corpus/external_val.txt   the other dataset, one text per line (for DAPT)
  corpus/task_train.txt                                the TRAINING reports of the labelled dataset (for TAPT)
  reports/corpus_summary.json
"""
import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import config
from common import load_split, save_json, write_lines

parser = argparse.ArgumentParser()
parser.add_argument("--sources", default=config.SOURCES_DIR, help="folder with the corpus files")
parser.add_argument("--text-columns", nargs="+", default=config.TEXT_COLUMNS,
                    help="columns of a .csv/.xlsx joined into one text")
parser.add_argument("--years", help="keep only rows of these years, e.g. 2010-2020 (.csv/.xlsx with a date column)")
args = parser.parse_args()


def column_key(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def tidy(text):
    return " ".join(str(text).split())


def chunks(text):
    """Cut a long text into pieces of about CHUNK_WORDS words, at sentence ends where possible."""
    words = text.split()
    if len(words) <= 1.5 * config.CHUNK_WORDS:
        return [text]
    pieces, current = [], []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        current += sentence.split()
        if len(current) >= config.CHUNK_WORDS:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    out = []
    for piece in pieces:
        w = piece.split()
        if len(w) <= 2 * config.CHUNK_WORDS:
            out.append(piece)
        else:                       # a very long "sentence" (e.g. a table read from a PDF): cut by word count
            out += [" ".join(w[i:i + config.CHUNK_WORDS]) for i in range(0, len(w), config.CHUNK_WORDS)]
    return out


def year_range(text):
    first, _, last = text.partition("-")
    return int(first), int(last or first)


def read_table(path):
    """Texts of one .csv/.xlsx file: the text columns joined, optionally filtered by year."""
    if path.suffix == ".csv":
        try:
            df = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    else:
        df = pd.read_excel(path)                     # needs openpyxl
    by_key = {column_key(c): c for c in df.columns}
    columns = [by_key[column_key(c)] for c in args.text_columns if column_key(c) in by_key]
    if not columns:
        print(f"[skip] {path.name}: none of the text columns {args.text_columns} found.\n"
              f"       Its columns are: {list(df.columns)}\n       Choose them with --text-columns")
        return []
    if args.years:
        date_columns = [c for c in df.columns if "date" in column_key(c) or "year" in column_key(c)]
        if not date_columns:
            raise SystemExit(f"--years given, but {path.name} has no date or year column: {list(df.columns)}")
        values = df[date_columns[0]]
        years = values if "year" in column_key(date_columns[0]) else pd.to_datetime(values, errors="coerce", dayfirst=True).dt.year
        first, last = year_range(args.years)
        df = df[pd.to_numeric(years, errors="coerce").between(first, last)]
    texts = df[columns].fillna("").astype(str).agg(" ".join, axis=1)
    print(f"{path.name}: {len(df)} rows, text from {columns}")
    return texts.tolist()


def read_pdf(path):
    """
    The running text of a PDF report or digest, without the parts that are not prose: running headers and
    footers (lines repeated on many pages), page numbers, tables of numbers, table-of-contents lines
    ("1.2 Narrative ........ 4"), web and e-mail addresses.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise SystemExit("Reading PDF files needs pypdf:  uv add pypdf  (or: pip install pypdf)")
    try:
        pages = [page.extract_text() or "" for page in PdfReader(path).pages]
    except Exception as error:                      # a damaged or encrypted file: skip it, keep going
        print(f"[skip] {path.name}: cannot be read ({error})")
        return []
    page_lines = [[line.strip() for line in page.splitlines() if line.strip()] for page in pages]
    # a running header or footer is a line at the top or bottom of many pages; body text is never removed this way
    edges = Counter(line for lines in page_lines for line in set(lines[:3] + lines[-3:]))
    repeated = {line for line, n in edges.items() if n >= max(3, 0.3 * len(pages))}
    kept = []
    for lines in page_lines:
        for line in lines:
            if not line or line in repeated:
                continue
            visible = line.replace(" ", "")
            if sum(c.isalpha() for c in visible) < 0.6 * len(visible):   # page numbers, tables of figures
                continue
            if re.search(r"\.{4,}|https?://|www\.|\S+@\S+", line):        # contents lines, addresses
                continue
            kept.append(line)
    text = re.sub(r"(?<=[a-z])-\s+(?=[a-z])", "", " ".join(kept))   # join words hyphenated at a line end
    print(f"{path.name}: {len(pages)} pages, {len(text.split())} words of running text")
    return [text]


def read_txt(path):
    parts = re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="replace"))
    print(f"{path.name}: {len(parts)} texts")
    return parts


READERS = {".csv": read_table, ".xlsx": read_table, ".xls": read_table, ".pdf": read_pdf, ".txt": read_txt}

# ----------------------------------------------------------------------------- 1. read every source file
files = sorted(p for p in Path(args.sources).rglob("*") if p.suffix.lower() in READERS)
if not files:
    print(f"[note] no .csv/.xlsx/.txt/.pdf files in {args.sources}/: only the task corpus (TAPT) is built.\n"
          "       For the other dataset (DAPT) run step0_download_sources.py first; see README.md.")
raw = []
per_source = Counter()
for path in files:
    folder = path.parent.name if path.parent != Path(args.sources) else path.name
    for text in READERS[path.suffix.lower()](path):
        for piece in chunks(tidy(text)):
            raw.append((folder, piece))
            per_source[folder] += 1

# ----------------------------------------------------------------------------- 2. clean
summary = {"files": len(files), "texts_read": len(raw)}
texts, seen = [], set()
short = duplicates = 0
for source, text in raw:
    if len(text.split()) < config.MIN_WORDS:
        short += 1
        continue
    key = text.lower()
    if key in seen:
        duplicates += 1
        continue
    seen.add(key)
    texts.append((source, text))
summary.update({"removed_short": short, "removed_duplicates": duplicates})

# ----------------------------------------------------------------------------- 3. remove the labelled reports
train, val, test, _ = load_split()
labelled = pd.concat([train, val, test])["text"].tolist()


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(tokens):
    n = config.OVERLAP_NGRAM
    return {hash(tuple(tokens[i:i + n])) for i in range(len(tokens) - n + 1)}


labelled_exact = {" ".join(words(t)) for t in labelled}
labelled_ngrams = [ngrams(words(t)) for t in labelled]
index = defaultdict(list)                           # n-gram -> labelled reports that contain it
for doc_id, grams in enumerate(labelled_ngrams):
    for g in grams:
        index[g].append(doc_id)

kept, exact, near = [], 0, 0
kept_per_source = Counter()
for source, text in texts:
    tokens = words(text)
    if " ".join(tokens) in labelled_exact:
        exact += 1
        continue
    grams = ngrams(tokens)
    shared = Counter(doc_id for g in grams for doc_id in index.get(g, ()))
    # near-copy: most of the corpus text is in one labelled report, or most of that report is in the text
    if any(n >= config.OVERLAP_THRESHOLD * min(len(grams), len(labelled_ngrams[d])) for d, n in shared.items()):
        near += 1
        continue
    kept.append(text)
    kept_per_source[source] += 1
summary.update({"removed_labelled_exact": exact, "removed_labelled_near_copy": near, "kept_texts": len(kept)})

# ----------------------------------------------------------------------------- 4. save
random.Random(config.MLM_SEED).shuffle(kept)
n_val = max(1, round(config.MLM_VAL_FRACTION * len(kept))) if len(kept) > 1 else 0   # 0 if no other dataset
corpus = Path(config.CORPUS_DIR)
write_lines(kept[n_val:], corpus / "external_train.txt")
write_lines(kept[:n_val], corpus / "external_val.txt")
write_lines(train["text"].tolist(), corpus / "task_train.txt")    # TRAINING reports only, never val or test

summary.update({
    "texts_per_source": dict(per_source),
    "kept_per_source": dict(kept_per_source),
    "external_train_texts": len(kept) - n_val,
    "external_val_texts": n_val,
    "external_words": sum(len(t.split()) for t in kept),
    "task_train_texts": len(train),
    "task_train_words": int(train["text"].str.split().str.len().sum()),
})
save_json(summary, Path(config.REPORTS_DIR) / "corpus_summary.json")

print(f"\nRead {len(raw)} texts; removed {short} short, {duplicates} duplicates, "
      f"{exact} labelled reports and {near} near-copies of labelled reports.")
print(f"Corpus: {len(kept)} texts, {summary['external_words']:,} words -> {corpus}/external_train.txt "
      f"({len(kept) - n_val}) and external_val.txt ({n_val})")
print(f"Task corpus: {len(train)} training reports -> {corpus}/task_train.txt")
if exact + near > 0.5 * max(1, len(texts)):
    print("[note] most source texts were already in the labelled dataset: the source covers the same years.")
if 0 < len(kept) < 1000:
    print("[note] the corpus is small; more text (e.g. MAIB Safety Digests as PDF) usually helps stage 1.")
