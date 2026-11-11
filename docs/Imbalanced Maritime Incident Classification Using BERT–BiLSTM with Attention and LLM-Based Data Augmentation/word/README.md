# Word version of the paper (`../main.docx`)

Built from `main.tex` so that both versions have the same text, numbers and references:

```bash
python prep.py .. body.tex                      # resolve citations [n] and references, inline the tables
pandoc body.tex -f latex -t docx -o body.docx   # text, tables and equations (as Word equations)
python build_docx.py .. ../main.docx            # IEEE layout: letter, two columns, Times New Roman
```

`build_docx.py` needs `python-docx`; `fig1.png` is Fig. 1 (TikZ) rendered from `../main.pdf` at 600 dpi, and
`fig2.png` is rendered from `../figures/confusion_matrix.pdf` (`pdftoppm -r 600 -png -singlefile`).
`main_docx_preview.pdf` is how LibreOffice lays out `main.docx` (6 pages); Word may differ slightly.
After editing `main.tex`, rebuild with the three commands above; small edits can also be made in Word directly.
