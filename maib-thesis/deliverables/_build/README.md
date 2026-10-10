# Build scripts for the report and slides

Not part of the research code. They read `../../results/*.json` and `../../reports/*` and rebuild
`report_fa.docx` and `slides_fa.pptx`.

```bash
python3 build_data.py ../.. out          # aggregates results -> out/data.json + two figures
node build_docx.js out ../.. ../report_fa.docx   # needs the npm package "docx"
node build_pptx.js out ../.. ../slides_fa.pptx   # needs the npm package "pptxgenjs"
```
