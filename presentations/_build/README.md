# Build scripts for the report, slides and study guide

Not part of the research code. They read the results of `maib-thesis/` and rebuild the files in
`../1405-7-19_progress-meeting/`. Run them from this folder after `step5_report.py` and `step6_attention.py`.

```bash
npm install docx pptxgenjs @fontsource/vazirmatn       # once

M=../../maib-thesis
OUT=../1405-7-19_progress-meeting
python3 build_data.py $M out                           # aggregates results -> out/data.json + figures
node build_docx.js out $M $OUT/report_fa.docx          # Word report
node build_pptx.js out $M $OUT/slides_fa.pptx          # slides

python3 build_guide.py out/fig_results.png $M/reports/confusion_matrix.png      # study guide -> guide.html
chromium --headless --no-pdf-header-footer --print-to-pdf=$OUT/meeting_prep_fa.pdf guide.html
```

The report and slides take every number from `results/*.json`. The study guide's text is written in
`build_guide.py` itself, so update it by hand when the results change.
